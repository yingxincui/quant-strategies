import backtrader as bt
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from loguru import logger
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
        self.cerebro.addanalyzer(bt.analyzers.Transactions, _name='txn')  # 交易记录
        
        self.trades = []  # 存储交易记录
        
    def run(self):
        """运行回测"""
        results = self.cerebro.run()
        self.strategy = results[0]
        
        # 获取交易记录
        transactions = self.strategy.analyzers.txn.get_analysis()
        
        # 整理交易记录
        for date, txns in transactions.items():
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
                
                trade = {
                    '时间': trade_date,
                    '类型': '买入' if is_buy else '卖出',
                    '价格': price,
                    '数量': size,
                    '成交额': value,
                    '手续费': commission,
                    '总费用': commission
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
        # 获取策略数据
        data = self.strategy.data
        trailing_stop = self.strategy.trailing_stop
        
        # 将数据转换为numpy数组，处理日期时间
        dates = [datetime.fromordinal(int(d)).strftime('%Y-%m-%d') for d in data.datetime.array]
        opens = np.array(data.open.array)
        highs = np.array(data.high.array)
        lows = np.array(data.low.array)
        closes = np.array(data.close.array)
        volumes = np.array(data.volume.array)
        
        # 检查trailing_stop是否为None
        if trailing_stop is not None:
            trailing_stop_vals = np.array(trailing_stop.trailing_stop.array)
        else:
            # 如果trailing_stop为None，创建一个全为0的数组
            trailing_stop_vals = np.zeros_like(closes)
        
        # 创建DataFrame以便处理数据
        df = pd.DataFrame({
            'date': dates,
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': volumes,
            'trailing_stop': trailing_stop_vals
        })
        
        # 移除volume为0的行（非交易日）
        df = df[df['volume'] > 0].copy()
        
        # 创建子图
        fig = make_subplots(rows=2, cols=1, 
                           shared_xaxes=True,
                           vertical_spacing=0.03, 
                           row_heights=[0.7, 0.3])
        
        # 添加K线图
        fig.add_trace(go.Candlestick(
            x=df['date'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='K线',
            increasing_line_color='#ff0000',  # 上涨为红色
            decreasing_line_color='#00ff00',  # 下跌为绿色
        ), row=1, col=1)
        
        # 添加追踪止损线
        # 过滤掉追踪止损为0的点
        valid_stops = df[df['trailing_stop'] > 0].copy()
        if not valid_stops.empty:
            fig.add_trace(go.Scatter(
                x=valid_stops['date'],
                y=valid_stops['trailing_stop'],
                name='追踪止损',
                line=dict(color='#f1c40f', dash='dash')
            ), row=1, col=1)
        
        # 添加买卖点标记
        if hasattr(self, 'trades_df') and not self.trades_df.empty:
            # 获取买入点
            buy_points = self.trades_df[self.trades_df['类型'] == '买入']
            if not buy_points.empty:
                fig.add_trace(go.Scatter(
                    x=buy_points['时间'].dt.strftime('%Y-%m-%d'),
                    y=buy_points['价格'],
                    mode='markers+text',
                    marker=dict(symbol='triangle-up', size=15, color='#2ecc71'),
                    text=[f"买入\n价格:{price:.2f}\n数量:{size}" 
                          for price, size in zip(buy_points['价格'], buy_points['数量'])],
                    textposition="top center",
                    name='买入点',
                    hoverinfo='text'
                ), row=1, col=1)
            
            # 获取卖出点
            sell_points = self.trades_df[self.trades_df['类型'] == '卖出']
            if not sell_points.empty:
                fig.add_trace(go.Scatter(
                    x=sell_points['时间'].dt.strftime('%Y-%m-%d'),
                    y=sell_points['价格'],
                    mode='markers+text',
                    marker=dict(symbol='triangle-down', size=15, color='#e74c3c'),
                    text=[f"卖出\n价格:{price:.2f}\n收益:{profit:.2f}" 
                          for price, profit in zip(sell_points['价格'], sell_points['累计收益'])],
                    textposition="bottom center",
                    name='卖出点',
                    hoverinfo='text'
                ), row=1, col=1)
        
        # 添加成交量图
        colors = ['#ff0000' if close >= open_ else '#00ff00' 
                 for close, open_ in zip(df['close'], df['open'])]
        
        fig.add_trace(go.Bar(
            x=df['date'],
            y=df['volume'],
            name='成交量',
            marker_color=colors,
            marker=dict(
                color=colors,
                line=dict(color=colors, width=1)
            ),
            hovertemplate='日期: %{x}<br>成交量: %{y:,.0f}<extra></extra>'
        ), row=2, col=1)
        
        # 更新布局
        fig.update_layout(
            title={'text': '回测结果', 'font': {'family': 'Arial'}},
            yaxis_title={'text': '价格', 'font': {'family': 'Arial'}},
            yaxis2_title={'text': '成交量', 'font': {'family': 'Arial'}},
            xaxis_rangeslider_visible=False,
            height=800,
            template='plotly_white',
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                font=dict(family='Arial')
            ),
            # 优化X轴显示
            xaxis=dict(
                type='category',
                rangeslider=dict(visible=False),
                showgrid=False,  # 移除X轴网格线
                gridwidth=1,
                gridcolor='lightgrey'
            ),
            xaxis2=dict(
                type='category',
                rangeslider=dict(visible=False),
                showgrid=False,  # 移除X轴网格线
                gridwidth=1,
                gridcolor='lightgrey'
            ),
            bargap=0,  # 设置柱状图之间的间隔为0
            bargroupgap=0  # 设置柱状图组之间的间隔为0
        )
        
        # 更新Y轴格式
        fig.update_yaxes(
            title_text="价格", 
            title_font=dict(family='Arial'),
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgrey',
            row=1, 
            col=1
        )
        fig.update_yaxes(
            title_text="成交量", 
            title_font=dict(family='Arial'),
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgrey',
            row=2, 
            col=1
        )
        
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
        
        # 从交易记录中统计各项费用
        txn_analysis = strategy.analyzers.txn.get_analysis()
        
        # 检查交易记录是否为空
        if not txn_analysis:
            logger.warning("回测期间没有产生任何交易")
            analysis['trades'] = pd.DataFrame()
            analysis['total_pnl'] = 0
            analysis['total_commission'] = 0
            analysis['total_cost'] = 0
            return analysis
            
        for date, txns in txn_analysis.items():
            for txn in txns:
                if len(txn) >= 2:  # 确保有手续费信息
                    size = abs(txn[0])
                    price = abs(txn[1])
                    value = price * size
                    
                    # 计算佣金 (万分之2.5)
                    commission = value * 0.00025
                    
                    total_commission += commission
                    
                    logger.info(f"交易费用明细 - 成交金额: {value:.2f}, 佣金: {commission:.2f}")
        
        # 添加费用统计到分析结果
        analysis['total_commission'] = total_commission  # 纯佣金
        analysis['total_cost'] = total_commission  # 总费用
        
        # 添加交易记录
        trades = []
        try:
            # 使用transactions分析器获取交易记录
            txn_analysis = strategy.analyzers.txn.get_analysis()
            
            # 如果没有交易记录，返回空DataFrame
            if not txn_analysis:
                display_df = pd.DataFrame()
                total_pnl = 0
                analysis['trades'] = display_df
                analysis['total_pnl'] = total_pnl
                return analysis
                
            # 临时存储开仓信息
            position = {
                'size': 0,
                'value': 0,
                'avg_price': 0
            }
            running_pnl = 0  # 用于累计盈亏
            
            for date, txns in sorted(txn_analysis.items()):  # 确保按日期排序
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
                        
                        # 获取交易原因
                        trade_info = None
                        total_value = None
                        
                        # 直接使用策略当前的trade_reason
                        for order in strategy._orders:
                            if (order.executed.size == size and 
                                abs(order.executed.price - price) < 0.000001 and 
                                order.isbuy()):
                                trade_info = order.info.get('reason') if hasattr(order, 'info') else None
                                total_value = order.info.get('total_value') if hasattr(order, 'info') else None
                                break
                        
                        if not trade_info:
                            logger.debug("未找到交易原因，使用默认值")
                            trade_info = '未知原因'
                        
                        trades.append({
                            'time': trade_date,
                            'direction': 'Long',
                            'price': price,
                            'size': size,
                            'avg_price': position['avg_price'],
                            'pnl': 0,
                            'return': 0,
                            'reason': trade_info,
                            'total_value': total_value,  # 总资产（含现金）
                            'position_value': order.info.get('position_value') if hasattr(order, 'info') else 0  # 持仓市值
                        })
                    else:  # 卖出
                        size = abs(size)
                        # 计算这次卖出的盈亏
                        pnl = (price - position['avg_price']) * size
                        ret = (price - position['avg_price']) / position['avg_price'] if position['avg_price'] > 0 else 0
                        running_pnl += pnl
                        
                        # 获取交易原因
                        trade_info = None
                        total_value = None
                        
                        # 直接使用策略当前的trade_reason
                        for order in strategy._orders:
                            if (order.executed.size == -size and 
                                abs(order.executed.price - price) < 0.000001 and 
                                not order.isbuy()):
                                trade_info = order.info.get('reason') if hasattr(order, 'info') else None
                                total_value = order.info.get('total_value') if hasattr(order, 'info') else None
                                break
                        
                        if not trade_info:
                            logger.debug("未找到交易原因，使用默认值")
                            trade_info = '未知原因'
                        
                        trades.append({
                            'time': trade_date,
                            'direction': 'Short',
                            'price': price,
                            'size': size,
                            'avg_price': position['avg_price'],
                            'pnl': pnl,
                            'return': ret,
                            'reason': trade_info,
                            'total_value': total_value,  # 总资产（含现金）
                            'position_value': order.info.get('position_value') if hasattr(order, 'info') else 0  # 持仓市值
                        })
                        
                        # 更新持仓
                        position['size'] -= size
                        if position['size'] > 0:
                            position['value'] = position['avg_price'] * position['size']
                        else:
                            position['size'] = 0
                            position['value'] = 0
                            position['avg_price'] = 0
            
            # 按时间排序
            trades.sort(key=lambda x: x['time'])
            
            # 转换为DataFrame
            trades_df = pd.DataFrame(trades)
            
            # 计算总盈亏（在格式化之前）
            total_pnl = trades_df['pnl'].astype(float).sum()
            
            # 格式化数据
            trades_df['time'] = pd.to_datetime(trades_df['time']).dt.strftime('%Y-%m-%d')
            trades_df['price'] = trades_df['price'].map('{:.3f}'.format)
            trades_df['avg_price'] = trades_df['avg_price'].map('{:.4f}'.format)
            trades_df['return'] = trades_df['return'].map('{:.2%}'.format)
            trades_df['pnl'] = trades_df['pnl'].map('{:.2f}'.format)
            trades_df['size'] = trades_df['size'].astype(int)
            trades_df['total_value'] = trades_df['total_value'].map('{:.2f}'.format)
            trades_df['position_value'] = trades_df['position_value'].map('{:.2f}'.format)
            
            # 转换方向
            trades_df['direction'] = trades_df['direction'].map({'Long': '买入', 'Short': '卖出'})
            
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
            
        except Exception as e:
            logger.warning(f"处理交易记录时出错: {str(e)}")
            logger.warning(f"交易记录内容: {txn_analysis}")
            display_df = pd.DataFrame()  # 返回空DataFrame
            total_pnl = 0
        
        analysis['trades'] = display_df
        analysis['total_pnl'] = total_pnl  # 添加总盈亏到分析结果中
        
        return analysis 