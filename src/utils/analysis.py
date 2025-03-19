import numpy as np
from datetime import datetime
import pandas as pd
import backtrader as bt
from loguru import logger

class Analysis:
    def __init__(self):
        pass

    def _get_analysis(self, engine, strategy):
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
                    # 获取交易日期
                    trade_date = order.info.get('execution_date', bt.num2date(order.data.datetime[0]))
                else:
                    trade_info = '未知原因'
                    total_value = None
                    position_value = None
                    etf_code = order.data._name if hasattr(order, 'data') else None
                    trade_date = bt.num2date(order.data.datetime[0])
                
                # 添加交易记录
                if is_buy:
                    engine.trades.append({
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
                    for t in reversed(engine.trades):
                        if t['direction'] == 'Long' and t['etf_code'] == etf_code:
                            entry_price = t['price']
                            break
                    
                    if entry_price:
                        pnl = (price - entry_price) * size
                        ret = (price - entry_price) / entry_price if entry_price > 0 else 0
                    else:
                        pnl = 0
                        ret = 0
                    
                    engine.trades.append({
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
        
        # 检查是否有交易记录
        if not engine.trades:
            logger.warning("回测期间没有产生任何交易")
            analysis['trades'] = pd.DataFrame()
            analysis['total_pnl'] = 0
            return analysis
        
        # 按时间排序
        engine.trades.sort(key=lambda x: x['time'])
        
        # 转换为DataFrame
        trades_df = pd.DataFrame(engine.trades)
        
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
        analysis['total_dividend'] = getattr(strategy, 'total_dividend', 0)  # 添加分红总额
        
        return analysis  # 添加返回语句