import numpy as np
from datetime import datetime, time, date
import pandas as pd
import backtrader as bt
from loguru import logger

class Analysis:
    def __init__(self):
        pass

    def _get_analysis(self, engine, strategy):
        """获取回测分析结果"""
        analysis = {}
        
        # 在计算总收益率之前，检查是否是双账户策略
        if hasattr(strategy.broker, 'total_returns'):
            # 使用双账户策略中计算的总收益率
            analysis['total_return'] = strategy.broker.total_returns
            analysis['etf_return'] = strategy.broker.etf_returns
            analysis['future_return'] = strategy.broker.future_returns
            analysis['etf_value'] = strategy.broker.etf_value
            analysis['future_value'] = strategy.broker.future_value
            analysis['is_dual_account'] = True
            
            logger.info(f"双账户策略 - ETF账户: {analysis['etf_value']:.2f} ({analysis['etf_return']:.2%}), "
                      f"期货账户: {analysis['future_value']:.2f} ({analysis['future_return']:.2%}), "
                      f"总收益率: {analysis['total_return']:.2%}")
        else:
            # 传统单账户策略
            analysis['total_return'] = (strategy.broker.getvalue() / strategy.broker.startingcash) - 1
            analysis['is_dual_account'] = False
        
        # 计算年化收益率
        start_date = bt.num2date(strategy.data.datetime[-len(strategy.data.datetime)+1])
        end_date = bt.num2date(strategy.data.datetime[0])
        # 确保结束日期是当天的收盘时间
        end_date = end_date.replace(hour=15, minute=0, second=0, microsecond=0)
        days = (end_date - start_date).days
        logger.info(f"回测开始日期: {start_date}, 回测结束日期: {end_date}, 回测天数: {days}，总收益率: {analysis['total_return']}")
        if days > 0:
            # 使用252个交易日作为一年
            years = days / 252
            analysis['annualized_return'] = (1 + analysis['total_return']) ** (1/years) - 1
        else:
            analysis['annualized_return'] = 0
        
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
                    avg_cost = order.info.get('avg_cost', None)
                    etf_code = order.info.get('etf_code', order.data._name if hasattr(order, 'data') else None)
                    # 获取交易日期
                    trade_date = order.info.get('execution_date', bt.num2date(order.data.datetime[0]))
                else:
                    trade_info = '未知原因'
                    total_value = None
                    position_value = None
                    avg_cost = None
                    etf_code = order.data._name if hasattr(order, 'data') else None
                    trade_date = bt.num2date(order.data.datetime[0])
                
                # 添加交易记录
                if is_buy:
                    # 检查是否有期货交易的盈亏记录
                    trade_pnl = order.info.get('pnl', 0) if hasattr(order, 'info') and order.info else 0
                    trade_return = order.info.get('return', 0) if hasattr(order, 'info') and order.info else 0
                    
                    engine.trades.append({
                        'time': trade_date,
                        'direction': 'Long',
                        'price': price,
                        'size': size,
                        'avg_price': avg_cost if avg_cost else 0,  # 买入时均价就是成交价
                        'pnl': trade_pnl,  # 使用订单中记录的盈亏，特别是对期货交易
                        'return': trade_return,  # 使用订单中记录的收益率
                        'reason': trade_info,
                        'total_value': total_value,
                        'position_value': position_value,
                        'etf_code': etf_code
                    })
                else:
                    # 使用持仓均价计算盈亏，或从订单信息中获取
                    trade_pnl = order.info.get('pnl', 0) if hasattr(order, 'info') and order.info else 0
                    trade_return = order.info.get('return', 0) if hasattr(order, 'info') and order.info else 0
                    
                    # 如果没有预先计算的盈亏，尝试计算（针对ETF交易）
                    if trade_pnl == 0 and avg_cost:
                        pnl = (price - avg_cost) * size
                        ret = (price - avg_cost) / avg_cost
                        trade_pnl = -pnl
                        trade_return = ret
                    
                    engine.trades.append({
                        'time': trade_date,
                        'direction': 'Short',
                        'price': price,
                        'size': abs(size),
                        'avg_price': avg_cost if avg_cost else 0,
                        'pnl': trade_pnl,
                        'return': trade_return,
                        'reason': trade_info,
                        'total_value': total_value,
                        'position_value': position_value,
                        'etf_code': etf_code
                    })
        
        # 检查是否有交易记录
        if not engine.trades:
            logger.warning("回测期间没有产生任何交易")
            analysis['trades'] = pd.DataFrame()
            analysis['total_pnl'] = 0
            return analysis
        
        # 按时间排序 - 确保所有时间都是相同类型
        # 先将所有日期类型转换为datetime类型
        for trade in engine.trades:
            if isinstance(trade['time'], date) and not isinstance(trade['time'], datetime):
                # 如果是date类型，转换为datetime类型（设置时间为当天15:00:00）
                trade['time'] = datetime.combine(trade['time'], time(15, 0, 0))

        # 现在所有的时间都是datetime类型，可以安全排序
        engine.trades.sort(key=lambda x: x['time'])
        
        # 转换为DataFrame
        trades_df = pd.DataFrame(engine.trades)
        
        # 确保数值列为数字类型
        trades_df['pnl'] = pd.to_numeric(trades_df['pnl'], errors='coerce')
        trades_df['return'] = pd.to_numeric(trades_df['return'], errors='coerce')
        
        # 计算总盈亏（在格式化之前）
        total_pnl = trades_df['pnl'].sum()
        
        # 格式化数据
        trades_df['time'] = pd.to_datetime(trades_df['time']).dt.strftime('%Y-%m-%d')
        trades_df['price'] = trades_df['price'].map(lambda x: '{:.3f}'.format(x) if x is not None else '')
        trades_df['avg_price'] = trades_df['avg_price'].map(lambda x: '{:.4f}'.format(x) if x is not None else '')
        
        # 收益率格式化 - 百分比显示
        trades_df['return'] = trades_df['return'].apply(
            lambda x: '{:.2%}'.format(x) if pd.notnull(x) else ''
        )
        
        trades_df['pnl'] = trades_df['pnl'].apply(
            lambda x: '{:.2f}'.format(x) if pd.notnull(x) else ''
        )
        
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
        analysis['total_dividend'] = getattr(strategy, 'total_dividend', 0)  # 添加分红总额
        
        return analysis  # 添加返回语句