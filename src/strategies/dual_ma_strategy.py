import backtrader as bt
from loguru import logger
from src.indicators.trailing_stop import TrailingStop
import math

class DualMAStrategy(bt.Strategy):
    params = (
        ('fast_period', 5),      # 快速移动平均线周期
        ('slow_period', 30),      # 慢速移动平均线周期
        ('trail_percent', 2.0),   # 追踪止损百分比
        ('risk_ratio', 0.02),     # 单次交易风险比率
        ('max_drawdown', 0.15),   # 最大回撤限制
        ('price_limit', 0.10),    # 涨跌停限制(10%)
    )

    def __init__(self):
        # 移动平均线指标
        self.fast_ma = bt.indicators.SMA(
            self.data.close, period=self.p.fast_period)
        self.slow_ma = bt.indicators.SMA(
            self.data.close, period=self.p.slow_period)
        self.crossover = bt.indicators.CrossOver(self.fast_ma, self.slow_ma)
        
        # 追踪止损指标
        self.trailing_stop = TrailingStop(self.data, trailing=self.p.trail_percent/100.0)
        self.trailing_stop._owner = self  # 显式设置所有者
        
        # 波动率指标（用于计算头寸）
        self.atr = bt.indicators.ATR(self.data)
        
        # 记录最高净值，用于计算回撤
        self.highest_value = self.broker.getvalue()
        
        # 用于跟踪订单和持仓
        self.order = None
        self.entry_price = None  # 记录入场价格
        
        # T+1交易限制
        self.buy_dates = set()  # 记录买入日期
        
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
                self.close()
                logger.info(f"触发最大回撤限制 - 当前回撤: {drawdown:.2%}, 限制: {self.p.max_drawdown:.2%}")
            return
            
        # 检查是否触及涨跌停
        if not self.check_price_limit(self.data.close[0]):
            return
            
        current_price = self.data.close[0]
        
        # 强制更新指标
        if self.position:
            self.trailing_stop.next()
        
        if not self.position:  # 没有持仓
            if self.crossover > 0:  # 金叉，买入信号
                shares = self.calculate_trade_size(current_price)
                
                if shares >= 100:  # 确保至少有100股
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
                
            # 获取当前止损价
            stop_price = self.trailing_stop[0]
            logger.info(f"持仓检查 - 今天日期: {current_date}, 当前价格: {current_price:.2f}, 止损价: {stop_price:.2f}, 最高价: {self.trailing_stop.max_price:.2f}")
            
            if self.crossover < 0:  # 死叉，卖出信号
                self.order = self.close()
                if self.order:
                    logger.info(f"卖出信号 - 价格: {current_price:.2f}")
            
            # 追踪止损检查
            elif current_price < stop_price:
                self.order = self.close()
                if self.order:
                    logger.info(f"追踪止损触发 - 当前价格: {current_price:.2f}, 止损价: {stop_price:.2f}, 最高价: {self.trailing_stop.max_price:.2f}")

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
            
        if order.status in [order.Completed]:
            if order.isbuy():
                # 买入订单执行后立即重置追踪止损，使用实际成交价
                self.trailing_stop.reset(price=order.executed.price)
                logger.info(
                    f'买入执行 - 价格: {order.executed.price:.2f}, '
                    f'数量: {order.executed.size}, '
                )
            else:
                # 重置入场价格并停止追踪
                self.entry_price = None
                self.trailing_stop.stop_tracking()
                logger.info(
                    f'卖出执行 - 价格: {order.executed.price:.2f}, '
                    f'数量: {order.executed.size}, '
                )
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            logger.warning(f'订单失败 - 状态: {order.getstatusname()}')
            
        self.order = None  # 重置订单状态

    def stop(self):
        """策略结束时的汇总信息"""
        portfolio_value = self.broker.getvalue()
        returns = (portfolio_value / self.broker.startingcash) - 1.0
        logger.info(f"策略结束 - 最终资金: {portfolio_value:.2f}, 收益率: {returns:.2%}") 