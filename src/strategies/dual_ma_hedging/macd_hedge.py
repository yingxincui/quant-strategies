import backtrader as bt
from loguru import logger

class MACDHedge:
    def __init__(self, strategy):
        self.strategy = strategy
        self.enabled = False
        self.hedge_position = None
        self.hedge_entry_price = None
        self.hedge_order = None
        self.hedge_contract_code = None
        self.hedge_entry_date = None  # 添加入场日期记录
        
        # 初始化MACD指标
        self.macd = bt.indicators.MACD(
            self.strategy.data.close,
            period_me1=12,
            period_me2=26,
            period_signal=9
        )
        
    def enable(self):
        """启用MACD对冲功能"""
        self.enabled = True
        logger.info("启用MACD对冲功能")
        
    def disable(self):
        """禁用MACD对冲功能"""
        self.enabled = False
        logger.info("禁用MACD对冲功能")
        
    def on_death_cross(self):
        """在MACD零轴上方死叉时开空期货"""
        if not self.enabled:
            return
            
        if self.hedge_position is not None or self.hedge_order is not None:
            logger.info("已有对冲仓位或对冲订单，不再开仓")
            return
            
        # 检查MACD是否在零轴上方
        if self.macd.macd[0] <= 0:
            logger.info("MACD不在零轴上方，不开仓")
            return
            
        # 检查是否形成死叉（MACD线下穿信号线）
        if self.macd.macd[0] > self.macd.signal[0] or self.macd.macd[-1] <= self.macd.signal[-1]:
            logger.info("未形成死叉，不开仓")
            return
            
        try:
            # 计算ATR止盈止损价格
            current_atr = self.strategy.atr[0]
            stop_loss = self.strategy.data1.close[0] + (current_atr * self.strategy.p.atr_loss_multiplier)
            take_profit = self.strategy.data1.close[0] - (current_atr * self.strategy.p.atr_profit_multiplier)
            
            # 开空豆粕期货
            hedge_size = self.strategy.p.hedge_contract_size
            
            # 检查期货账户资金是否足够
            future_price = self.strategy.data1.close[0]
            margin_requirement = future_price * hedge_size * self.strategy.p.future_contract_multiplier * self.strategy.p.m_margin_ratio
            
            if margin_requirement > self.strategy.future_cash:
                logger.warning(f"期货账户资金不足，需要{margin_requirement:.2f}，当前可用{self.strategy.future_cash:.2f}")
                # 根据可用资金调整手数
                adjusted_size = int(self.strategy.future_cash / (future_price * self.strategy.p.future_contract_multiplier * self.strategy.p.m_margin_ratio))
                if adjusted_size < 1:
                    logger.error("期货账户资金不足以开仓一手")
                    return
                hedge_size = adjusted_size
                logger.info(f"已调整对冲手数为: {hedge_size}")
            
            self.hedge_order = self.strategy.sell(data=self.strategy.data1, size=hedge_size)
            
            if self.hedge_order:
                # 记录入场价格和合约代码
                self.hedge_entry_price = self.strategy.data1.close[0]
                # 确保获取正确的合约代码
                current_date = self.strategy.data1.datetime.datetime(0)
                if hasattr(self.strategy.data1, 'contract_mapping') and current_date in self.strategy.data1.contract_mapping:
                    self.hedge_contract_code = self.strategy.data1.contract_mapping[current_date]
                else:
                    # 如果无法获取映射，使用数据名称
                    self.hedge_contract_code = self.strategy.data1._name
                self.hedge_entry_date = self.strategy.data.datetime.date(0)  # 记录入场日期
                
                # 计算保证金
                margin = self.hedge_entry_price * hedge_size * self.strategy.p.future_contract_multiplier * self.strategy.p.m_margin_ratio
                
                # 从期货账户扣除保证金
                pre_cash = self.strategy.future_cash
                self.strategy.future_cash -= margin
                
                logger.info(f"开仓扣除保证金 - 之前: {pre_cash:.2f}, 扣除: {margin:.2f}, 之后: {self.strategy.future_cash:.2f}")
                
                # 记录交易信息
                self.hedge_order.info = {
                    'reason': f"MACD死叉开空 - MACD: {self.macd.macd[0]:.2f}, Signal: {self.macd.signal[0]:.2f}",
                    'margin': margin,
                    'future_cash': self.strategy.future_cash,
                    'execution_date': self.hedge_entry_date,
                    'total_value': self.strategy.future_cash,
                    'position_value': abs(margin),
                    'position_ratio': margin / self.strategy.future_cash if self.strategy.future_cash > 0 else 0,
                    'etf_code': self.hedge_contract_code,
                    'pnl': 0,
                    'return': 0,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'avg_cost': self.hedge_entry_price
                }
                
                logger.info(f"MACD死叉开空 - 合约: {self.hedge_contract_code}, 价格: {self.hedge_entry_price:.2f}, 数量: {hedge_size}手, "
                          f"止损价: {stop_loss:.2f}, 止盈价: {take_profit:.2f}")
                
        except Exception as e:
            logger.error(f"MACD死叉开空失败: {str(e)}")
            
    def check_exit(self):
        """检查是否需要平仓"""
        if not self.enabled or not self.hedge_position:
            return
            
        current_price = self.strategy.data1.close[0]
        current_atr = self.strategy.atr[0]
        
        # 计算ATR止盈止损价格
        stop_loss = self.hedge_entry_price + (current_atr * self.strategy.p.atr_loss_multiplier)
        take_profit = self.hedge_entry_price - (current_atr * self.strategy.p.atr_profit_multiplier)
        
        # 检查是否触发止盈止损
        if current_price >= stop_loss or current_price <= take_profit:
            if self.hedge_order is None:  # 确保没有未完成订单
                self.hedge_order = self.strategy.close(data=self.strategy.data1)
                reason = "触发止盈" if current_price <= take_profit else "触发止损"
                logger.info(f"MACD死叉对冲{reason} - 当前价格: {current_price:.2f}, {reason}价: {take_profit if current_price <= take_profit else stop_loss:.2f}")
                
    def on_order_completed(self, order):
        """处理订单完成事件"""
        if not self.enabled:
            return
            
        if order.status in [order.Completed]:
            if order.isbuy():  # 买入豆粕期货（平空）
                # 确保有对应的入场价格
                if self.hedge_entry_price is None or self.hedge_contract_code is None:
                    logger.error("平仓时找不到入场价格或合约代码，跳过处理")
                    return
                
                # 记录平仓前的合约信息，用于日志
                entry_price = self.hedge_entry_price
                contract_code = self.hedge_contract_code
                entry_date = self.hedge_entry_date
                
                # 记录交易日期和价格
                trade_date = self.strategy.data.datetime.date(0)
                trade_price = order.executed.price
                
                # 先重置持仓相关变量，防止重复平仓
                self.hedge_position = None
                self.hedge_order = None
                self.hedge_entry_price = None
                self.hedge_contract_code = None
                self.hedge_entry_date = None
                self.hedge_target_profit = None
                
                # 计算对冲盈亏（空仓：入场价 - 平仓价）
                hedge_profit = (entry_price - trade_price) * self.strategy.p.hedge_contract_size * self.strategy.p.future_contract_multiplier
                
                # 减去开平仓手续费
                total_fee = self.strategy.p.hedge_fee * self.strategy.p.hedge_contract_size * 2
                net_profit = hedge_profit - total_fee
                
                # 归还保证金并添加盈亏到期货账户
                margin_returned = entry_price * self.strategy.p.hedge_contract_size * self.strategy.p.future_contract_multiplier * self.strategy.p.m_margin_ratio
                
                # 记录更新前的资金
                pre_cash = self.strategy.future_cash
                
                # 更新期货账户资金
                self.strategy.future_cash += (margin_returned + net_profit)
                
                # 记录资金变动
                logger.info(f"平仓资金变动 - 之前: {pre_cash:.2f}, 返还保证金: {margin_returned:.2f}, 盈亏: {net_profit:.2f}, 之后: {self.strategy.future_cash:.2f}")
                
                # 更新期货账户最高净值
                self.strategy.future_highest_value = max(self.strategy.future_highest_value, self.strategy.future_cash)
                
                # 计算期货账户回撤
                future_drawdown = (self.strategy.future_highest_value - self.strategy.future_cash) / self.strategy.future_highest_value if self.strategy.future_highest_value > 0 else 0
                
                # 计算收益率
                return_pct = (hedge_profit / (entry_price * self.strategy.p.hedge_contract_size * self.strategy.p.future_contract_multiplier)) * 100
                
                # 更新订单信息
                order.info.update({
                    'pnl': hedge_profit,
                    'return': return_pct,
                    'total_value': self.strategy.future_cash,
                    'position_value': 0,  # 平仓后持仓价值为0
                    'avg_cost': entry_price,
                    'etf_code': contract_code,  # 确保使用原始合约代码
                    'execution_date': trade_date,  # 确保使用当前交易日期
                    'reason': f"MACD死叉对冲平仓 - 合约: {contract_code}, 入场日期: {entry_date}, 入场价: {entry_price:.2f}, 平仓价: {trade_price:.2f}, 收益率: {return_pct:.2f}%"
                })
                
                logger.info(f"MACD死叉对冲平仓 - 日期: {trade_date}, 合约: {contract_code}, 价格: {trade_price:.2f}, 盈利: {hedge_profit:.2f}, "
                          f"手续费: {total_fee:.2f}, 净盈利: {net_profit:.2f}, "
                          f"期货账户余额: {self.strategy.future_cash:.2f}, 回撤: {future_drawdown:.2%}, "
                          f"收益率: {return_pct:.2f}%")
                
            else:  # 卖出豆粕期货（开空）
                # 记录对冲持仓
                self.hedge_position = order
                
                # 更新订单信息
                order.info.update({
                    'total_value': self.strategy.future_cash,
                    'position_value': abs(order.info['margin']),
                    'avg_cost': order.executed.price,
                    'etf_code': self.hedge_contract_code  # 确保合约代码正确
                })
                
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.hedge_order = None
            logger.warning(f'MACD死叉对冲订单失败 - 状态: {order.getstatusname()}')
            
    def on_strategy_stop(self):
        """策略结束时平掉所有持仓"""
        if not self.enabled or not self.hedge_position:
            return
            
        try:
            if self.hedge_order is None:  # 确保没有未完成订单
                self.hedge_order = self.strategy.close(data=self.strategy.data1)
                logger.info("策略结束，平掉MACD对冲持仓")
        except Exception as e:
            logger.error(f"策略结束时平仓失败: {str(e)}") 