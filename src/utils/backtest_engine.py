import backtrader as bt
import pandas as pd
from datetime import datetime
from loguru import logger
from .plot import Plot
import sys

class AShareCommissionInfo(bt.CommInfoBase):
    """A股交易费用"""
    params = (
        ('commission', 0.00025),     # 佣金费率 0.025%
        ('stocklike', True),        # 股票模式
        ('commtype', bt.CommInfoBase.COMM_PERC),  # 按百分比收取
    )

    def _getcommission(self, size, price, pseudoexec):
        """计算交易费用"""
        value = abs(size) * price
        
        # 计算佣金（买入和卖出都收取）
        commission = value * self.p.commission
        
        return commission

class BacktestEngine:
    def __init__(self, strategy_class, data_feed, cash=100000.0, commission=0.00025, strategy_params=None):
        """初始化回测引擎"""
        self.cerebro = bt.Cerebro()
        self.cerebro.broker.setcash(cash)
        self.cerebro.broker.setcommission(commission=commission)
        
        # 添加数据源
        try:
            if isinstance(data_feed, list):
                # 如果是数据源列表，添加所有数据源
                for feed in data_feed:
                    self.cerebro.adddata(feed)
            else:
                # 如果是单个数据源，直接添加
                self.cerebro.adddata(data_feed)
        except Exception as e:
            logger.warning(f"添加数据源时出错: {str(e)}")
            # 如果出错，尝试不带ts_code参数添加
            if hasattr(data_feed, 'params'):
                data_feed.params.pop('ts_code', None)
            self.cerebro.adddata(data_feed)
            
        # 添加策略和参数
        if strategy_params:
            self.cerebro.addstrategy(strategy_class, **strategy_params)
        else:
            self.cerebro.addstrategy(strategy_class)
            
        # 添加分析器
        self.cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', riskfreerate=0.02)
        self.cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        self.cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
        self.cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
        self.cerebro.addanalyzer(bt.analyzers.VWR, _name='vwr')  # 波动率加权收益
        self.cerebro.addanalyzer(bt.analyzers.SQN, _name='sqn')  # 系统质量指数
        
        # 为每个数据源添加单独的交易记录分析器
        if isinstance(data_feed, list):
            for feed in data_feed:
                self.cerebro.addanalyzer(bt.analyzers.Transactions, _name=f'txn_{feed._name}')
        else:
            self.cerebro.addanalyzer(bt.analyzers.Transactions, _name='txn')
        
        self.trades = []  # 存储交易记录
        
    def run(self):
        """运行回测"""
        results = self.cerebro.run()
        
        # 打印回测结果的详细信息
        for result in results:
            logger.info(f"回测结果:")
            logger.info(f"  策略名称: {result.__class__.__name__}")
            logger.info(f"  参数设置:")
            for name, value in result.params._getitems():
                logger.info(f"    {name}: {value}")
            logger.info(f"  最终资金: {result.broker.getvalue():.2f}")
            logger.info(f"  持仓情况:")
            for data in result.datas:
                position = result.getposition(data)
                if position.size != 0:
                    logger.info(f"    {data._name}: {position.size}股")

        self.strategy = results[0]
        
        # 获取所有数据源的交易记录
        all_transactions = {}
        if isinstance(self.strategy.datas, list):
            for data in self.strategy.datas:
                txn_name = f'txn_{data._name}'
                # 使用 getattr 来安全地获取分析器
                analyzer = getattr(self.strategy.analyzers, txn_name, None)
                if analyzer is not None:
                    logger.info(f"获取 {data._name} 的交易记录")
                    transactions = analyzer.get_analysis()
                    # 修改合并逻辑，确保不会覆盖相同日期的交易记录
                    for date, txns in transactions.items():
                        if date not in all_transactions:
                            all_transactions[date] = []
                        all_transactions[date].extend(txns)
                    logger.info(f"{data._name} 交易记录数量: {len(transactions)}")
        else:
            # 对于单个数据源的情况
            analyzer = getattr(self.strategy.analyzers, 'txn', None)
            if analyzer is not None:
                all_transactions = analyzer.get_analysis()
                logger.info(f"交易记录数量: {len(all_transactions)}")
        
        logger.info(f"交易记录: {all_transactions}")
        
        # 整理交易记录
        for date, txns in all_transactions.items():
            for txn in txns:
                # 处理日期时间
                if isinstance(date, datetime):
                    trade_date = date
                else:
                    try:
                        trade_date = bt.num2date(date)
                    except:
                        trade_date = datetime.fromordinal(int(date))
                
                # 计算手续费
                size = abs(txn[0])
                price = abs(txn[1])
                value = size * price
                is_buy = txn[0] > 0
                
                # 计算各项费用
                commission = value * 0.00025
                
                # 获取ETF代码
                etf_code = None
                if hasattr(txn, 'data') and hasattr(txn.data, '_name'):
                    etf_code = txn.data._name
                elif isinstance(txn, tuple) and len(txn) > 2 and hasattr(txn[2], '_name'):
                    etf_code = txn[2]._name
                
                trade = {
                    '时间': trade_date,
                    '类型': '买入' if is_buy else '卖出',
                    '价格': price,
                    '数量': size,
                    '成交额': value,
                    '手续费': commission,
                    '总费用': commission,
                    'ETF代码': etf_code
                }
                self.trades.append(trade)
        
        # 转换为DataFrame并按时间排序
        if self.trades:
            self.trades_df = pd.DataFrame(self.trades)
            self.trades_df = self.trades_df.sort_values('时间')
            
            # 计算累计收益
            self.trades_df['累计收益'] = 0.0
            running_pnl = 0
            position = 0
            cost_basis = 0
            
            for idx, row in self.trades_df.iterrows():
                if row['类型'] == '买入':
                    position += row['数量']
                    cost_basis = (cost_basis * (position - row['数量']) + 
                                row['成交额']) / position if position > 0 else 0
                else:
                    pnl = (row['价格'] - cost_basis) * row['数量']
                    running_pnl += pnl
                    position -= row['数量']
                    
                self.trades_df.at[idx, '累计收益'] = running_pnl
            
            # 打印交易记录
            logger.info("=== 交易记录 ===")
            for _, trade in self.trades_df.iterrows():
                logger.info(
                    f"{trade['时间'].strftime('%Y-%m-%d')} | "
                    f"{trade['类型']} | "
                    f"价格: {trade['价格']:.3f} | "
                    f"数量: {trade['数量']:.0f} | "
                    f"成交额: {trade['成交额']:.2f} | "
                    f"手续费: {trade['手续费']:.2f} | "
                    f"累计收益: {trade['累计收益']:.2f}"
                )
        
        analysis = self._get_analysis(self.strategy)
        logger.info("=== 回测统计 ===")
        logger.info(f"总收益率: {analysis['total_return']:.2%}")
        logger.info(f"夏普比率: {analysis['sharpe_ratio']:.2f}")
        logger.info(f"最大回撤: {analysis['max_drawdown']:.2%}")
        logger.info(f"胜率: {analysis['win_rate']:.2%}")
        logger.info(f"盈亏比: {analysis['profit_factor']:.2f}")
        logger.info(f"系统质量指数(SQN): {analysis['sqn']:.2f}")
        
        return analysis
    
    def plot(self, **kwargs):
        """使用Plotly绘制交互式回测结果"""
        fig = Plot(self.strategy).plot()
        return fig
        
    def _get_analysis(self, strategy):
        """获取回测分析结果"""
        analysis = {}
        
        # 计算总收益率
        analysis['total_return'] = (strategy.broker.getvalue() / strategy.broker.startingcash) - 1
        
        # 获取夏普比率
        sharpe_analysis = strategy.analyzers.sharpe.get_analysis()
        analysis['sharpe_ratio'] = sharpe_analysis.get('sharperatio', 0) or 0
        
        # 获取最大回撤
        drawdown_analysis = strategy.analyzers.drawdown.get_analysis()
        analysis['max_drawdown'] = drawdown_analysis.get('max', {}).get('drawdown', 0) / 100
        
        # 获取交易统计
        trade_analysis = strategy.analyzers.trades.get_analysis()
        
        # 总交易次数
        total_closed = trade_analysis.get('total', {}).get('closed', 0)
        analysis['total_trades'] = total_closed
        
        # 计算胜率
        total_won = trade_analysis.get('won', {}).get('total', 0)
        analysis['win_rate'] = total_won / total_closed if total_closed > 0 else 0
        
        # 计算盈亏比
        won_pnl = trade_analysis.get('won', {}).get('pnl', {}).get('total', 0)
        lost_pnl = abs(trade_analysis.get('lost', {}).get('pnl', {}).get('total', 1))
        analysis['profit_factor'] = won_pnl / lost_pnl if lost_pnl != 0 else 0
        
        # 计算平均收益
        pnl_net = trade_analysis.get('pnl', {}).get('net', {}).get('total', 0)
        analysis['avg_trade_return'] = pnl_net / total_closed if total_closed > 0 else 0
        
        # 获取SQN
        sqn_analysis = strategy.analyzers.sqn.get_analysis()
        analysis['sqn'] = sqn_analysis.get('sqn', 0)
        
        # 计算手续费统计
        total_commission = 0
        
        # 使用策略的_orders记录
        if hasattr(strategy, '_orders') and strategy._orders:
            # 使用集合来存储已处理的订单，避免重复
            processed_orders = set()
            
            for order in strategy._orders:
                if not order.executed.size:  # 跳过未执行的订单
                    continue
                    
                # 创建订单的唯一标识 - 使用数据源的时间而不是订单执行时间
                order_id = f"{order.data._name}_{order.data.datetime[0]}_{order.executed.size}_{order.executed.price}"
                if order_id in processed_orders:
                    continue
                    
                processed_orders.add(order_id)
                
                # 获取交易日期 - 使用数据源的时间
                trade_date = bt.num2date(order.data.datetime[0])
                
                # 获取订单信息
                size = order.executed.size
                price = order.executed.price
                value = abs(size) * price
                is_buy = order.isbuy()
                
                # 计算佣金
                commission = value * 0.00025
                total_commission += commission
                
                # 获取交易原因和资产信息
                if hasattr(order, 'info') and order.info:
                    trade_info = order.info.get('reason', '未知原因')
                    total_value = order.info.get('total_value', None)
                    position_value = order.info.get('position_value', None)
                    etf_code = order.info.get('etf_code', order.data._name if hasattr(order, 'data') else None)
                else:
                    trade_info = '未知原因'
                    total_value = None
                    position_value = None
                    etf_code = order.data._name if hasattr(order, 'data') else None
                
                # 添加交易记录
                if is_buy:
                    self.trades.append({
                        'time': trade_date,
                        'direction': 'Long',
                        'price': price,
                        'size': size,
                        'avg_price': price,  # 买入时均价就是成交价
                        'pnl': 0,
                        'return': 0,
                        'reason': trade_info,
                        'total_value': total_value,
                        'position_value': position_value,
                        'etf_code': etf_code
                    })
                else:
                    # 尝试找到对应的买入记录以计算盈亏
                    entry_price = None
                    for t in reversed(self.trades):
                        if t['direction'] == 'Long' and t['etf_code'] == etf_code:
                            entry_price = t['price']
                            break
                    
                    if entry_price:
                        pnl = (price - entry_price) * size
                        ret = (price - entry_price) / entry_price if entry_price > 0 else 0
                    else:
                        pnl = 0
                        ret = 0
                    
                    self.trades.append({
                        'time': trade_date,
                        'direction': 'Short',
                        'price': price,
                        'size': abs(size),
                        'avg_price': entry_price if entry_price else 0,
                        'pnl': -pnl,
                        'return': ret,
                        'reason': trade_info,
                        'total_value': total_value,
                        'position_value': position_value,
                        'etf_code': etf_code
                    })
        # 如果策略没有_orders记录，则使用Transactions分析器
        elif all_transactions:
            # 临时存储开仓信息
            position = {
                'size': 0,
                'value': 0,
                'avg_price': 0
            }
            running_pnl = 0  # 用于累计盈亏
            
            for date, txns in sorted(all_transactions.items()):  # 确保按日期排序
                for txn in txns:
                    # 确保txn是一个列表或元组
                    if not isinstance(txn, (list, tuple)) or len(txn) < 2:
                        continue
                    
                    size = txn[0]
                    price = txn[1]
                    
                    # 转换日期格式
                    if isinstance(date, datetime):
                        trade_date = date
                    else:
                        trade_date = bt.num2date(date)
                    
                    if size > 0:  # 买入
                        # 更新持仓信息
                        position['size'] += size
                        position['value'] += price * size
                        position['avg_price'] = position['value'] / position['size'] if position['size'] > 0 else 0
                        
                        self.trades.append({
                            'time': trade_date,
                            'direction': 'Long',
                            'price': price,
                            'size': size,
                            'avg_price': position['avg_price'],
                            'pnl': 0,
                            'return': 0,
                            'reason': '未知原因',
                            'total_value': None,
                            'position_value': None,
                            'etf_code': None
                        })
                    else:  # 卖出
                        size = abs(size)
                        # 计算这次卖出的盈亏
                        pnl = (price - position['avg_price']) * size
                        ret = (price - position['avg_price']) / position['avg_price'] if position['avg_price'] > 0 else 0
                        running_pnl += pnl
                        
                        self.trades.append({
                            'time': trade_date,
                            'direction': 'Short',
                            'price': price,
                            'size': size,
                            'avg_price': position['avg_price'],
                            'pnl': pnl,
                            'return': ret,
                            'reason': '未知原因',
                            'total_value': None,
                            'position_value': None,
                            'etf_code': None
                        })
                        
                        # 更新持仓
                        position['size'] -= size
                        if position['size'] > 0:
                            position['value'] = position['avg_price'] * position['size']
                        else:
                            position['size'] = 0
                            position['value'] = 0
                            position['avg_price'] = 0
        else:
            logger.warning("无法获取交易记录，既没有策略的_orders列表，也没有Transactions分析器数据")
        
        # 检查是否有交易记录
        if not self.trades:
            logger.warning("回测期间没有产生任何交易")
            analysis['trades'] = pd.DataFrame()
            analysis['total_pnl'] = 0
            return analysis
        
        # 按时间排序
        self.trades.sort(key=lambda x: x['time'])
        
        # 转换为DataFrame
        trades_df = pd.DataFrame(self.trades)
        
        # 计算总盈亏（在格式化之前）
        total_pnl = trades_df['pnl'].astype(float).sum()
        
        # 格式化数据
        trades_df['time'] = pd.to_datetime(trades_df['time']).dt.strftime('%Y-%m-%d')
        trades_df['price'] = trades_df['price'].map(lambda x: '{:.3f}'.format(x) if x is not None else '')
        trades_df['avg_price'] = trades_df['avg_price'].map(lambda x: '{:.4f}'.format(x) if x is not None else '')
        trades_df['return'] = trades_df['return'].map(lambda x: '{:.2%}'.format(x) if x is not None else '')
        trades_df['pnl'] = trades_df['pnl'].map(lambda x: '{:.2f}'.format(x) if x is not None else '')
        trades_df['size'] = trades_df['size'].astype(int)
        
        # 处理可能为None的字段
        trades_df['total_value'] = trades_df['total_value'].map(lambda x: '{:.2f}'.format(x) if x is not None else '')
        trades_df['position_value'] = trades_df['position_value'].map(lambda x: '{:.2f}'.format(x) if x is not None else '')
        
        # 转换方向
        trades_df['direction'] = trades_df['direction'].map({'Long': '买入', 'Short': '卖出'})
        
        # 添加ETF代码列
        if 'etf_code' in trades_df.columns:
            # 重命名列并选择要显示的列
            display_df = trades_df[['time', 'direction', 'etf_code', 'price', 'avg_price', 'size', 'total_value', 'position_value', 'pnl', 'return', 'reason']]
            display_df = display_df.rename(columns={
                'time': '交易时间',
                'direction': '方向',
                'etf_code': 'ETF代码',
                'price': '成交价',
                'avg_price': '持仓均价',
                'size': '数量',
                'total_value': '总资产',
                'position_value': '持仓市值',
                'pnl': '盈亏',
                'return': '收益率',
                'reason': '交易原因'
            })
        else:
            # 重命名列并选择要显示的列
            display_df = trades_df[['time', 'direction', 'price', 'avg_price', 'size', 'total_value', 'position_value', 'pnl', 'return', 'reason']]
            display_df = display_df.rename(columns={
                'time': '交易时间',
                'direction': '方向',
                'price': '成交价',
                'avg_price': '持仓均价',
                'size': '数量',
                'total_value': '总资产',
                'position_value': '持仓市值',
                'pnl': '盈亏',
                'return': '收益率',
                'reason': '交易原因'
            })
        
        analysis['trades'] = display_df
        analysis['total_pnl'] = total_pnl  # 添加总盈亏到分析结果中
        analysis['total_commission'] = total_commission  # 添加佣金总额
        analysis['total_cost'] = 0  # 添加总费用
        
        return analysis 