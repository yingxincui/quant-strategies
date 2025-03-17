import backtrader as bt
from loguru import logger
from src.indicators.trailing_stop import TrailingStop
from src.signals.sentiment_data import get_sentiment_data
import math

class MarketSentimentStrategy(bt.Strategy):
    params = (
        ('sentiment_buy_1', 25.0),   # 第一层情绪买入阈值
        ('sentiment_buy_2', 21.0),   # 第二层情绪买入阈值
        ('sentiment_sell', 45.0),    # 情绪卖出阈值
        ('position_layers', 8),       # 最大仓位层数
        ('trail_percent', 2.0),      # 追踪止损百分比
        ('risk_ratio', 0.02),        # 单次交易风险比率
        ('max_drawdown', 0.15),      # 最大回撤限制
        ('price_limit', 0.10),       # 涨跌停限制(10%)
    )

    def __init__(self):
        # 情绪指标
        self.sentiment_data = None
        
        # 追踪止损指标
        self.trailing_stop = TrailingStop(self.data, trailing=self.p.trail_percent/100.0)
        self.trailing_stop._owner = self
        
        # 波动率指标
        self.atr = bt.indicators.ATR(self.data)
        
        # 记录最高净值
        self.highest_value = self.broker.getvalue()
        
        # 订单和持仓跟踪
        self.order = None
        self.entry_price = None
        self.trade_reason = None  # 记录交易原因
        self._orders = []  # 记录所有订单
        
        # T+1交易限制
        self.buy_dates = set()
        
        logger.info(f"市场情绪策略初始化 - 买入阈值: {self.p.sentiment_buy_1}, {self.p.sentiment_buy_2}, 卖出阈值: {self.p.sentiment_sell}")

    def round_shares(self, shares):
        """将股数调整为100的整数倍"""
        return math.floor(shares / 100) * 100
        
    def check_price_limit(self, price):
        """检查是否触及涨跌停"""
        prev_close = self.data.close[-1]
        upper_limit = prev_close * (1 + self.p.price_limit)
        lower_limit = prev_close * (1 - self.p.price_limit)
        return lower_limit <= price <= upper_limit
        
    def calculate_trade_size(self, price, sentiment_value):
        """计算可交易的股数，根据情绪分数决定仓位大小"""
        cash = self.broker.getcash() * 0.95
        total_value = self.broker.getvalue()
        
        # 根据情绪分数确定目标仓位
        if sentiment_value <= self.p.sentiment_buy_2:
            target_position = total_value / self.p.position_layers  # 8层仓位
        elif sentiment_value <= self.p.sentiment_buy_1:
            target_position = total_value / (self.p.position_layers * 2)  # 4层仓位
        else:
            return 0
            
        # 计算当前持仓市值
        current_position_value = self.position.size * price if self.position else 0
        
        # 如果当前持仓已经超过目标仓位，不再加仓
        if current_position_value >= target_position:
            return 0
            
        # 计算需要买入的市值
        value_to_buy = target_position - current_position_value
        
        # 考虑风险控制
        risk_amount = total_value * self.p.risk_ratio
        current_atr = self.atr[0]
        risk_per_share = max(current_atr * 1.5, price * (self.p.trail_percent/100.0))
        
        risk_size = risk_amount / risk_per_share if risk_per_share > 0 else 0
        cash_size = min(cash, value_to_buy) / price
        
        shares = min(risk_size, cash_size)
        shares = self.round_shares(shares)
        
        if shares * price > cash:
            shares = self.round_shares(cash / price)
            
        return shares if shares >= 100 else 0

    def next(self):
        if self.order:
            return
            
        # 重置交易原因（在每个新的交易周期开始时）
        self.trade_reason = None
            
        # 第一次调用next时获取情绪数据
        if self.sentiment_data is None:
            try:
                self.sentiment_data = get_sentiment_data()
                if not self.sentiment_data:
                    logger.error("无法获取市场情绪数据")
                    return
            except Exception as e:
                logger.error(f"获取市场情绪数据失败: {str(e)}")
                return
            
        # 计算回撤
        current_value = self.broker.getvalue()
        self.highest_value = max(self.highest_value, current_value)
        drawdown = (self.highest_value - current_value) / self.highest_value
        
        if drawdown > self.p.max_drawdown:
            if self.position:
                self.trade_reason = f"触发最大回撤限制 ({drawdown:.2%})"
                self.close()
                logger.info(f"触发最大回撤限制 - 当前回撤: {drawdown:.2%}")
            return
            
        current_price = self.data.close[0]
        if not self.check_price_limit(current_price):
            return
            
        if self.position:
            self.trailing_stop.next()
            
        # 获取当前日期的情绪数据
        current_date = self.data.datetime.date()
        # 将情绪数据转换为字典,避免每次循环查找
        if not hasattr(self, 'sentiment_dict'):
            self.sentiment_dict = {data['date']: data['value'] for data in self.sentiment_data['sentiment']}
            
        sentiment_value = self.sentiment_dict.get(current_date.strftime('%Y-%m-%d'))
        if sentiment_value is not None:
            logger.info(f"找到情绪数据: {sentiment_value}, 日期: {current_date.strftime('%Y-%m-%d')}")
        else:
            logger.info(f"未找到情绪数据, 日期: {current_date.strftime('%Y-%m-%d')}")
            return
            
        # 买入或加仓逻辑
        if sentiment_value <= self.p.sentiment_buy_1:
            shares = self.calculate_trade_size(current_price, sentiment_value)
            if shares >= 100:
                if sentiment_value <= self.p.sentiment_buy_2:
                    self.trade_reason = f"市场情绪极度低迷 ({sentiment_value:.1f})"
                else:
                    self.trade_reason = f"市场情绪偏低 ({sentiment_value:.1f})"
                self.order = self.buy(size=shares)
                if self.order:
                    self.buy_dates.add(current_date)
                    self.entry_price = current_price
                    position_value = (self.position.size + shares) * current_price if self.position else shares * current_price
                    total_value = self.broker.getvalue()
                    position_ratio = position_value / total_value
                    logger.info(f"{'加仓' if self.position else '买入'}信号 - 市场情绪: {sentiment_value:.2f}, "
                              f"数量: {shares}, 目标仓位: {position_ratio:.2%}")
                        
        elif self.position:
            if current_date in self.buy_dates:
                return
                
            stop_price = self.trailing_stop[0]
            
            # 卖出条件：市场情绪高于阈值
            if sentiment_value > self.p.sentiment_sell:
                self.trade_reason = f"市场情绪过热 ({sentiment_value:.1f})"
                self.order = self.close()
                if self.order:
                    logger.info(f"卖出信号 - 市场情绪: {sentiment_value:.2f}")
            
            # 追踪止损
            elif current_price < stop_price:
                self.trade_reason = f"触发追踪止损 (止损价: {stop_price:.2f})"
                self.order = self.close()
                if self.order:
                    logger.info(f"追踪止损触发 - 当前价格: {current_price:.2f}, 止损价: {stop_price:.2f}")

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
            
        if order.status in [order.Completed]:
            if order.isbuy():
                self.trailing_stop.reset(price=order.executed.price)
                order.info = {'reason': self.trade_reason}  # 记录交易原因
                self._orders.append(order)  # 添加到订单列表
                logger.info(f'买入执行 - 价格: {order.executed.price:.2f}, 数量: {order.executed.size}, 原因: {self.trade_reason}')
            else:
                self.entry_price = None
                self.trailing_stop.stop_tracking()
                order.info = {'reason': self.trade_reason}  # 记录交易原因
                self._orders.append(order)  # 添加到订单列表
                logger.info(f'卖出执行 - 价格: {order.executed.price:.2f}, 数量: {order.executed.size}, 原因: {self.trade_reason}')
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            logger.warning(f'订单失败 - 状态: {order.getstatusname()}')
            
        self.order = None
        # 不在这里重置交易原因，移到下一个交易周期开始时重置
        # self.trade_reason = None  # 重置交易原因

    def stop(self):
        """策略结束时的汇总信息"""
        portfolio_value = self.broker.getvalue()
        returns = (portfolio_value / self.broker.startingcash) - 1.0
        logger.info(f"策略结束 - 最终资金: {portfolio_value:.2f}, 收益率: {returns:.2%}") 