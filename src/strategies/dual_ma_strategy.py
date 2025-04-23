import backtrader as bt
from loguru import logger
from src.indicators.trailing_stop import TrailingStop
import math

class DualMAStrategy(bt.Strategy):
    params = (
        ('fast_period', 5),      # 快速移动平均线周期
        ('slow_period', 13),      # 慢速移动平均线周期
        ('trail_percent', 2.0),   # 追踪止损百分比
        ('risk_ratio', 0.02),     # 单次交易风险比率
        ('max_drawdown', 0.15),   # 最大回撤限制
        ('price_limit', 0.10),    # 涨跌停限制(10%)
        ('enable_trailing_stop', False),  # 是否启用追踪止损
        ('atr_loss_multiplier', 1.0),  # ATR倍数
        ('atr_profit_multiplier', 1.0),  # ATR倍数
        ('atr_period', 14),       # ATR周期
        ('enable_death_cross', False),  # 是否启用死叉卖出信号
    )

    def __init__(self):
        # 获取ETF代码
        self.etf_code = None
        try:
            # 首先尝试从数据源的params属性获取
            if hasattr(self.data, 'params') and hasattr(self.data.params, 'ts_code'):
                self.etf_code = self.data.params.ts_code
            # 然后尝试从数据源的其他属性获取
            elif hasattr(self.data, 'ts_code'):
                self.etf_code = self.data.ts_code
            # 最后尝试从数据源的_name属性获取
            elif hasattr(self.data, '_name'):
                self.etf_code = self.data._name
        except Exception as e:
            logger.warning(f"获取ETF代码时出错: {str(e)}")
            
        # 如果没有找到ETF代码，使用默认名称
        if not self.etf_code:
            self.etf_code = "ETF_1"
            
        # 设置数据源的名称
        self.data._name = self.etf_code
        
        # 移动平均线指标
        self.fast_ma = bt.indicators.SMA(
            self.data.close, period=self.p.fast_period)
        self.slow_ma = bt.indicators.SMA(
            self.data.close, period=self.p.slow_period)
        self.crossover = bt.indicators.CrossOver(self.fast_ma, self.slow_ma)
        
        # 5日线趋势判断
        self.ma5 = bt.indicators.SMA(self.data.close, period=5)
        
        # 追踪止损指标（根据参数决定是否启用）
        self.trailing_stop = None
        if self.p.enable_trailing_stop:
            self.trailing_stop = TrailingStop(self.data, trailing=self.p.trail_percent/100.0)
            self.trailing_stop._owner = self
        
        # ATR指标
        self.atr = bt.indicators.ATR(self.data, period=self.p.atr_period)
        
        # 记录最高净值，用于计算回撤
        self.highest_value = self.broker.getvalue()
        
        # 用于跟踪订单和持仓
        self.order = None
        self.entry_price = None  # 记录入场价格
        self.trade_reason = None  # 记录交易原因
        self._orders = []  # 记录所有订单
        
        # 持仓管理
        self.current_position_ratio = 0.0  # 当前仓位比例
        self.avg_cost = None  # 平均持仓成本
        
        # T+1交易限制
        self.buy_dates = set()  # 记录买入日期
        
        # 记录上次清仓日期
        self.last_close_date = None
        
        logger.info(f"策略初始化完成 - 参数: 快线={self.p.fast_period}, 慢线={self.p.slow_period}, 追踪止损={self.p.trail_percent}%, 风险比例={self.p.risk_ratio:.2%}, 最大回撤={self.p.max_drawdown:.2%}")
        
    def round_shares(self, shares):
        """将股数调整为100的整数倍"""
        return math.floor(shares / 100) * 100
        
    def check_price_limit(self, price):
        """检查是否触及涨跌停"""
        prev_close = self.data.close[-1]
        upper_limit = prev_close * (1 + self.p.price_limit)
        lower_limit = prev_close * (1 - self.p.price_limit)
        return lower_limit <= price <= upper_limit
        
    def calculate_trade_size(self, price):
        """计算可交易的股数（考虑资金、手续费和100股整数倍）"""
        cash = self.broker.getcash()
        
        # 预留更多手续费和印花税缓冲
        cash = cash * 0.95  # 预留5%的资金作为手续费缓冲
        
        # 计算风险金额（使用总资产的一定比例）
        total_value = self.broker.getvalue()
        risk_amount = total_value * self.p.risk_ratio
        
        # 使用ATR计算每股风险
        current_atr = self.atr[0]  # 当前ATR值
        # 使用1.5倍ATR作为止损距离，这个系数可以根据需要调整
        risk_per_share = current_atr * 1.5
        
        # 如果ATR过小，使用传统的百分比止损
        min_risk = price * (self.p.trail_percent/100.0)
        risk_per_share = max(risk_per_share, min_risk)
        
        # 根据风险计算的股数
        risk_size = risk_amount / risk_per_share if risk_per_share > 0 else 0
        
        # 根据可用资金计算的股数
        cash_size = cash / price
        
        # 取较小值并调整为100股整数倍
        shares = min(risk_size, cash_size)
        shares = self.round_shares(shares)
        
        # 再次验证金额是否超过可用资金
        if shares * price > cash:
            shares = self.round_shares(cash / price)
            
        logger.info(f"计算持仓 - ATR: {current_atr:.2f}, 每股风险: {risk_per_share:.2f}, 总风险金额: {risk_amount:.2f}, 计算股数: {shares}")
        
        return shares if shares >= 100 else 0

    def next(self):
        # 重置交易原因（在每个新的交易周期开始时）
        self.trade_reason = None
        
        # 如果有未完成的订单，不执行新的交易
        if self.order:
            return
            
        # 计算当前回撤
        current_value = self.broker.getvalue()
        self.highest_value = max(self.highest_value, current_value)
        drawdown = (self.highest_value - current_value) / self.highest_value
        
        # 如果回撤超过限制，不开新仓
        if drawdown > self.p.max_drawdown:
            if self.position:
                self.trade_reason = f"触发最大回撤限制 ({drawdown:.2%})"
                self.close()
                logger.info(f"触发最大回撤限制 - 当前回撤: {drawdown:.2%}, 限制: {self.p.max_drawdown:.2%}")
            return
            
        # 检查是否触及涨跌停
        if not self.check_price_limit(self.data.close[0]):
            return
            
        current_price = self.data.close[0]
        
        # 强制更新指标
        if self.position:
            if self.p.enable_trailing_stop:
                self.trailing_stop.next()
        
        if not self.position:  # 没有持仓
            # 判断5日线趋势：当前值大于前一个值表示上升趋势
            ma5_trend_up = self.ma5[0] > self.ma5[-1]
            
            # 判断是否收阴线
            is_red_candle = self.data.close[0] < self.data.open[0]
            
            # 判断是否放巨量（量能比前一日大两倍）
            volume_spike = self.data.volume[0] > self.data.volume[-1] * 2
            
            if self.crossover > 0 and ma5_trend_up and not is_red_candle and not volume_spike:  # 金叉且5日线趋势向上，且不是阴线，且不是巨量
                shares = self.calculate_trade_size(current_price)
                
                if shares >= 100:  # 确保至少有100股
                    self.trade_reason = f"快线上穿慢线 ({self.p.fast_period}日均线上穿{self.p.slow_period}日均线)"
                    self.order = self.buy(size=shares)
                    if self.order:
                        # 记录买入日期和价格
                        self.buy_dates.add(self.data.datetime.date())
                        self.entry_price = current_price
                        logger.info(f"买入信号 - 数量: {shares}, 价格: {current_price:.2f}, 可用资金: {self.broker.getcash():.2f}, 风险比例: {self.p.risk_ratio:.2%}")
                        
        else:  # 有持仓
            # 检查是否可以卖出（T+1规则）
            current_date = self.data.datetime.date()
            if current_date in self.buy_dates:
                return
                
            # 计算ATR止盈止损价格
            current_atr = self.atr[0]
            stop_loss = self.entry_price - (current_atr * self.p.atr_loss_multiplier)
            take_profit = self.entry_price + (current_atr * self.p.atr_profit_multiplier)
            
            # 获取追踪止损价格（如果启用）
            trailing_stop_price = None
            if self.p.enable_trailing_stop:
                trailing_stop_price = self.trailing_stop[0]
            
            logger.info(f"持仓检查 - 今天日期: {current_date}, 当前价格: {current_price:.2f}, ATR止损: {stop_loss:.2f}, ATR止盈: {take_profit:.2f}")
            
            if self.p.enable_death_cross and self.crossover < 0:  # 死叉，卖出信号
                self.trade_reason = f"快线下穿慢线 ({self.p.fast_period}日均线下穿{self.p.slow_period}日均线)"
                self.order = self.close()
                if self.order:
                    logger.info(f"卖出信号 - 价格: {current_price:.2f}")
            
            # ATR止损检查
            elif current_price < stop_loss:
                self.trade_reason = f"触发ATR止损 (止损价: {stop_loss:.2f})"
                self.order = self.close()
                if self.order:
                    logger.info(f"ATR止损触发 - 当前价格: {current_price:.2f}, 止损价: {stop_loss:.2f}")
            
            # ATR止盈检查
            elif current_price > take_profit:
                self.trade_reason = f"触发ATR止盈 (止盈价: {take_profit:.2f})"
                self.order = self.close()
                if self.order:
                    logger.info(f"ATR止盈触发 - 当前价格: {current_price:.2f}, 止盈价: {take_profit:.2f}")
            
            # 追踪止损检查（如果启用）
            elif self.p.enable_trailing_stop and current_price < trailing_stop_price:
                self.trade_reason = f"触发追踪止损 (止损价: {trailing_stop_price:.2f})"
                self.order = self.close()
                if self.order:
                    logger.info(f"追踪止损触发 - 当前价格: {current_price:.2f}, 止损价: {trailing_stop_price:.2f}, 最高价: {self.trailing_stop.max_price:.2f}")

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
            
        if order.status in [order.Completed]:
            if order.isbuy():
                # 买入订单执行后立即重置追踪止损，使用实际成交价
                if self.p.enable_trailing_stop:
                    self.trailing_stop.reset(price=order.executed.price)
                
                # 更新平均成本 - 买入时更新
                if self.avg_cost is None:
                    # 首次买入
                    self.avg_cost = order.executed.price
                else:
                    # 计算新的平均成本
                    current_position = self.position.size - order.executed.size
                    current_value = current_position * self.avg_cost
                    new_value = order.executed.size * order.executed.price
                    total_position = current_position + order.executed.size
                    self.avg_cost = (current_value + new_value) / total_position
                    logger.info(f"当前持仓量: {current_position}, 当前持仓市值: {current_value}, 新买入市值: {new_value}, 总持仓量: {total_position}")

                logger.info(f"买入平均成本: {self.avg_cost:.4f}")
                # 记录入场价格用于计算止盈
                self.entry_price = order.executed.price  # 保留最后一次买入价格
                
                # 更新当前持仓比例
                self.current_position_ratio = self.get_position_value_ratio()
                
                order.info = {'reason': self.trade_reason}  # 记录交易原因
                # 计算总资产和持仓市值 - 使用最新价格
                current_price = self.data.close[0]
                position_value = self.position.size * current_price if self.position else 0
                total_value = self.broker.getcash() + position_value
                
                order.info['total_value'] = total_value  # 记录总资产（含现金）
                order.info['position_value'] = position_value  # 记录持仓市值
                order.info['position_ratio'] = self.current_position_ratio  # 记录持仓比例
                order.info['avg_cost'] = self.avg_cost  # 记录平均成本
                order.info['etf_code'] = self.etf_code  # 添加ETF代码
                order.info['execution_date'] = self.data.datetime.date(0)  # 添加执行日期
                self._orders.append(order)  # 添加到订单列表
                logger.info(f'买入执行 - 价格: {order.executed.price:.2f}, 数量: {order.executed.size}, '
                          f'仓位比例: {self.current_position_ratio:.2%}, 平均成本: {self.avg_cost:.2f}, 原因: {self.trade_reason}')
            else:
                # 卖出 - 更新持仓相关指标
                # 记录卖出前的平均成本（用于日志记录）
                last_avg_cost = self.avg_cost
                
                if not self.position or self.position.size == 0:  # 如果全部平仓
                    self.entry_price = None
                    self.avg_cost = None
                    if self.p.enable_trailing_stop:
                        self.trailing_stop.stop_tracking()
                    # 记录清仓日期
                    self.last_close_date = self.data.datetime.date(0)
                    logger.info(f"记录清仓日期: {self.last_close_date}")
                
                # 更新当前持仓比例
                self.current_position_ratio = self.get_position_value_ratio()
                
                order.info = {'reason': self.trade_reason}  # 记录交易原因
                # 计算总资产和持仓市值 - 使用最新价格
                current_price = self.data.close[0]
                position_value = self.position.size * current_price if self.position else 0
                total_value = self.broker.getcash() + position_value
                
                order.info['total_value'] = total_value  # 记录总资产（含现金）
                order.info['position_value'] = position_value  # 记录持仓市值
                order.info['position_ratio'] = self.current_position_ratio  # 记录持仓比例
                order.info['avg_cost'] = last_avg_cost  # 记录卖出前的平均成本
                order.info['etf_code'] = self.etf_code  # 添加ETF代码
                order.info['execution_date'] = self.data.datetime.date(0)  # 添加执行日期
                self._orders.append(order)  # 添加到订单列表
                
                # 计算卖出收益
                profit = (order.executed.price - last_avg_cost) * order.executed.size if last_avg_cost and order.executed.price else 0
                profit_pct = ((order.executed.price / last_avg_cost) - 1.0) * 100 if last_avg_cost and order.executed.price else 0
                
                # 格式化价格和成本
                price_str = f"{order.executed.price:.2f}" if order.executed.price else "N/A"
                cost_str = f"{last_avg_cost:.4f}" if last_avg_cost else "N/A"
                
                logger.info(f'卖出执行 - 价格: {price_str}, 数量: {order.executed.size}, '
                          f'仓位比例: {self.current_position_ratio:.2%}, 平均成本: {cost_str}, '
                          f'收益: {profit:.2f}, 收益率: {profit_pct:.2f}%, 原因: {self.trade_reason}')
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            logger.warning(f'订单失败 - 状态: {order.getstatusname()}')
            
        self.order = None  # 重置订单状态

    def stop(self):
        """策略结束时的汇总信息"""
        portfolio_value = self.broker.getvalue()
        returns = (portfolio_value / self.broker.startingcash) - 1.0
        logger.info(f"策略结束 - 最终资金: {portfolio_value:.2f}, 收益率: {returns:.2%}") 

    def get_position_value_ratio(self):
        """计算当前持仓市值占总资产的比例"""
        if not self.position:
            return 0.0
        
        position_value = self.position.size * self.data.close[0]
        total_value = self.broker.getvalue()
        return position_value / total_value 