import backtrader as bt
from loguru import logger
from src.indicators.trailing_stop import TrailingStop
from src.signals.sentiment_data import get_sentiment_data
import math
import numpy as np

class MarketSentimentStrategy(bt.Strategy):
    params = (
        ('sentiment_buy_1', 20.0),   # 第一层情绪买入阈值
        ('sentiment_buy_2', 10.0),   # 第二层情绪买入阈值
        ('sentiment_sell', 80.0),    # 情绪卖出阈值
        ('position_layers', 4),       # 默认仓位层数
        ('trail_percent', 2.0),      # 追踪止损百分比
        ('risk_ratio', 0.02),        # 单次交易风险比率
        ('max_drawdown', 0.15),      # 最大回撤限制
        ('price_limit', 0.10),       # 涨跌停限制(10%)
        ('use_trailing_stop', False), # 是否启用追踪止损
        ('atr_take_profit', 3.0),    # ATR止盈倍数
        ('use_atr_take_profit', True), # 是否启用ATR止盈
        ('ema_period', 5),           # EMA周期
        ('ema_up_days', 2),          # EMA连续向上天数
        ('bb_period', 20),           # 布林带周期
        ('bb_devfactor', 2.0),       # 布林带标准差倍数
    )

    def __init__(self):
        # 情绪指标
        self.sentiment_data = None
        
        # 追踪止损指标
        if self.p.use_trailing_stop:
            self.trailing_stop = TrailingStop(self.data, trailing=self.p.trail_percent/100.0)
            self.trailing_stop._owner = self
        else:
            self.trailing_stop = None
        
        # 波动率指标
        self.atr = bt.indicators.ATR(self.data)
        
        # 添加布林带指标用于动态调整止盈倍数
        self.bb = bt.indicators.BollingerBands(self.data, period=self.p.bb_period, devfactor=self.p.bb_devfactor)
        
        # 添加EMA指标
        self.ema = bt.indicators.EMA(self.data.close, period=self.p.ema_period)
        # 添加更长周期的EMA作为趋势确认
        self.ema_long = bt.indicators.EMA(self.data.close, period=self.p.ema_period * 2)
        
        # 记录最高净值
        self.highest_value = self.broker.getvalue()
        
        # 订单和持仓跟踪
        self.order = None
        self.entry_price = None
        self.trade_reason = None  # 记录交易原因
        self._orders = []  # 记录所有订单
        
        # 止盈相关变量
        self.take_profit_price = None
        self.entry_price_for_tp = None  # 用于计算止盈的入场价格
        
        # T+1交易限制
        self.buy_dates = set()
        
        # 记录EMA历史值，用于判断趋势
        self.ema_history = []
        self.close_history = []
        
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
            # 市场情绪极度低迷，首次开仓用40%资金，加仓用剩余资金的80%
            if not self.position:
                # 首次开仓，使用总资金的40%
                target_position = total_value * 0.4
            else:
                # 已有仓位，加仓使用剩余现金的80%
                target_position = self.position.size * price + cash * 0.8
        elif sentiment_value <= self.p.sentiment_buy_1:
            # 市场情绪偏低，使用较小仓位
            target_position = total_value / self.p.position_layers  # 4层仓位
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
                # 重置相关状态
                self.entry_price = None
                self.take_profit_price = None
                self.entry_price_for_tp = None
                if self.p.use_trailing_stop:
                    self.trailing_stop.stop_tracking()
                # 重置EMA历史
                self.ema_history = []
                self.close_history = []
                # 重置买入日期集合
                self.buy_dates = set()
                # 重置最高净值，这样策略可以重新开始
                self.highest_value = current_value
                logger.info(f"重置最高净值至当前值: {current_value:.2f}")
            return
            
        current_price = self.data.close[0]
        if not self.check_price_limit(current_price):
            return
            
        if self.position and self.p.use_trailing_stop:
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
            
        # 如果有持仓且启用了ATR止盈，每天动态更新止盈价格
        if self.position and self.p.use_atr_take_profit and self.entry_price_for_tp is not None:
            current_atr = self.atr[0]
            
            # 计算布林带波动率比率
            volatility_ratio = (self.bb.top[0] - self.bb.bot[0]) / self.bb.mid[0]
            
            # 根据市场情绪和波动率动态调整止盈倍数
            base_multiplier = self.p.atr_take_profit
            
            # 市场情绪调整
            sentiment_factor = 1.0
            if sentiment_value > 60:
                # 市场情绪高于60，适度放大止盈倍数
                sentiment_factor = 1.5
            elif sentiment_value < 30:
                # 市场情绪低于30，适度缩小止盈倍数
                sentiment_factor = 0.8
            
            # 结合布林带波动率和市场情绪计算最终倍数
            atr_multiplier = base_multiplier * (1 + volatility_ratio) * sentiment_factor
            
            # 限制倍数范围，避免极端值
            atr_multiplier = max(1.5, min(atr_multiplier, 6.0))
                
            # 更新止盈价格
            new_take_profit_price = self.entry_price_for_tp + (current_atr * atr_multiplier)
            
            # 如果止盈价格有显著变化，记录日志
            if self.take_profit_price is None or abs(new_take_profit_price - self.take_profit_price) > 0.01:
                logger.info(f"更新ATR止盈价格: {new_take_profit_price:.2f} (当前ATR: {current_atr:.2f}, "
                          f"布林带波动率: {volatility_ratio:.2f}, 情绪因子: {sentiment_factor:.1f}, "
                          f"最终倍数: {atr_multiplier:.2f}, 情绪值: {sentiment_value:.1f})")
                self.take_profit_price = new_take_profit_price
        
        # 更新EMA和收盘价历史值
        self.ema_history.append(self.ema[0])
        self.close_history.append(current_price)
        
        # 保持历史记录不超过20个值
        if len(self.ema_history) > 20:
            self.ema_history.pop(0)
        if len(self.close_history) > 20:
            self.close_history.pop(0)
            
        # 检查EMA是否连续向上
        ema_up = True
        # if len(self.ema_history) >= self.p.ema_up_days + 3:  # 需要更多历史数据来确认趋势
        #     # 1. 检查EMA是否连续上涨指定天数 - 向量化实现
        #     ema_array = np.array(self.ema_history[-(self.p.ema_up_days+1):])
        #     is_ema_up_trend = np.all(np.diff(ema_array) > 0)
            
        #     # 2. 检查价格是否高于EMA
        #     price_above_ema = True
        #     # current_price > self.ema[0]
            
        #     # 3. 检查短期EMA是否高于长期EMA (趋势确认)
        #     ema_trend_aligned = self.ema[0] > self.ema_long[0]
            
        #     # 4. 检查收盘价是否也连续上涨 - 向量化实现
        #     is_price_up_trend = True
        #     # if len(self.close_history) >= 3:
        #     #     close_array = np.array(self.close_history[-3:])
        #     #     is_price_up_trend = np.all(np.diff(close_array) > 0)
                    
        #     # 5. 检查EMA斜率 (计算最近几天的平均变化率) - 向量化实现
        #     ema_slope = 0
        #     if len(self.ema_history) >= 5:
        #         ema_recent = np.array(self.ema_history[-5:])
        #         ema_changes = np.diff(ema_recent) / ema_recent[:-1]
        #         ema_slope = np.mean(ema_changes)
            
        #     positive_slope = ema_slope > 0.0001  # 要求EMA有正斜率
            
        #     # 同时满足所有条件
        #     ema_up = is_ema_up_trend and price_above_ema and ema_trend_aligned and is_price_up_trend and positive_slope
            
        #     logger.info(f"EMA趋势检查 - EMA连续上涨: {is_ema_up_trend}, 价格高于EMA: {price_above_ema}, "
        #               f"EMA趋势一致: {ema_trend_aligned}, 价格连续上涨: {is_price_up_trend}, "
        #               f"EMA斜率: {ema_slope:.6f}, 斜率足够: {positive_slope}, "
        #               f"当前EMA: {self.ema[0]:.4f}, 长期EMA: {self.ema_long[0]:.4f}")
        
        # 买入或加仓逻辑
        if sentiment_value <= self.p.sentiment_buy_1:
            # 检查当天情绪是否比前一天低
            prev_date = (current_date - bt.datetime.timedelta(days=1)).strftime('%Y-%m-%d')
            prev_sentiment = self.sentiment_dict.get(prev_date)
            
            # 如果前一天的情绪数据存在，且当天情绪比前一天低，则不开仓
            if prev_sentiment is not None and sentiment_value < prev_sentiment:
                logger.info(f"当天情绪({sentiment_value:.1f})低于前一天({prev_sentiment:.1f})，放弃买入")
                return
            
            # 增加EMA向上的条件
            if not ema_up:
                logger.info(f"EMA条件不满足，放弃买入 - 当前EMA: {self.ema[0]:.4f}, 当前价格: {current_price:.4f}")
                return
                
            shares = self.calculate_trade_size(current_price, sentiment_value)
            if shares >= 100:
                if sentiment_value <= self.p.sentiment_buy_2:
                    self.trade_reason = f"市场情绪极度低迷 ({sentiment_value:.1f}) 且EMA向上"
                else:
                    self.trade_reason = f"市场情绪偏低 ({sentiment_value:.1f}) 且EMA向上"
                self.order = self.buy(size=shares)
                if self.order:
                    self.buy_dates.add(current_date)
                    self.entry_price = current_price
                    position_value = (self.position.size + shares) * current_price if self.position else shares * current_price
                    total_value = self.broker.getvalue()
                    position_ratio = position_value / total_value
                    logger.info(f"{'加仓' if self.position else '买入'}信号 - 市场情绪: {sentiment_value:.2f}, EMA向上, "
                              f"数量: {shares}, 目标仓位: {position_ratio:.2%}")
                        
        elif self.position:
            if current_date in self.buy_dates:
                return
                
            # 只在启用追踪止损时检查止损条件
            if self.p.use_trailing_stop:
                stop_price = self.trailing_stop[0]
                if current_price < stop_price:
                    self.trade_reason = f"触发追踪止损 (止损价: {stop_price:.2f})"
                    self.order = self.close()
                    if self.order:
                        logger.info(f"追踪止损触发 - 当前价格: {current_price:.2f}, 止损价: {stop_price:.2f}")
                    return
            
            # 检查ATR止盈条件
            if self.p.use_atr_take_profit and self.take_profit_price is not None:
                if current_price >= self.take_profit_price:
                    self.trade_reason = f"触发ATR止盈 (止盈价: {self.take_profit_price:.2f})"
                    self.order = self.close()
                    if self.order:
                        logger.info(f"ATR止盈触发 - 当前价格: {current_price:.2f}, 止盈价: {self.take_profit_price:.2f}")
                    return
            
            # 卖出条件：市场情绪高于阈值
            if sentiment_value > self.p.sentiment_sell:
                self.trade_reason = f"市场情绪过热 ({sentiment_value:.1f})"
                self.order = self.close()
                if self.order:
                    logger.info(f"卖出信号 - 市场情绪: {sentiment_value:.2f}")

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
            
        if order.status in [order.Completed]:
            if order.isbuy():
                if self.p.use_trailing_stop:
                    self.trailing_stop.reset(price=order.executed.price)
                
                # 记录入场价格用于计算止盈
                if self.p.use_atr_take_profit:
                    # 如果是首次建仓或全部平仓后再次建仓，更新入场价格
                    if self.entry_price_for_tp is None:
                        self.entry_price_for_tp = order.executed.price
                    
                    # 初始设置止盈价格
                    current_atr = self.atr[0]
                    
                    # 获取当前日期的情绪数据
                    current_date = self.data.datetime.date()
                    sentiment_value = self.sentiment_dict.get(current_date.strftime('%Y-%m-%d'), 0)
                    
                    # 计算布林带波动率比率
                    volatility_ratio = (self.bb.top[0] - self.bb.bot[0]) / self.bb.mid[0]
                    
                    # 根据市场情绪和波动率动态调整止盈倍数
                    base_multiplier = self.p.atr_take_profit
                    
                    # 市场情绪调整
                    sentiment_factor = 1.0
                    if sentiment_value > 60:
                        # 市场情绪高于60，适度放大止盈倍数
                        sentiment_factor = 1.5
                        logger.info(f"市场情绪较高({sentiment_value:.1f})，调整情绪因子至: {sentiment_factor:.1f}")
                    elif sentiment_value < 30:
                        # 市场情绪低于30，适度缩小止盈倍数
                        sentiment_factor = 0.8
                        logger.info(f"市场情绪较低({sentiment_value:.1f})，调整情绪因子至: {sentiment_factor:.1f}")
                    
                    # 结合布林带波动率和市场情绪计算最终倍数
                    atr_multiplier = base_multiplier * (1 + volatility_ratio) * sentiment_factor
                    
                    # 限制倍数范围，避免极端值
                    atr_multiplier = max(1.5, min(atr_multiplier, 6.0))
                    
                    self.take_profit_price = self.entry_price_for_tp + (current_atr * atr_multiplier)
                    logger.info(f"初始设置ATR止盈价格: {self.take_profit_price:.2f} (入场价: {self.entry_price_for_tp:.2f}, "
                              f"ATR: {current_atr:.2f}, 布林带波动率: {volatility_ratio:.2f}, "
                              f"情绪因子: {sentiment_factor:.1f}, 最终倍数: {atr_multiplier:.2f})")
                
                order.info = {'reason': self.trade_reason}  # 记录交易原因
                order.info['total_value'] = self.broker.getvalue()  # 记录总资产（含现金）
                order.info['position_value'] = self.position.size * order.executed.price if self.position else 0  # 记录持仓市值
                self._orders.append(order)  # 添加到订单列表
                logger.info(f'买入执行 - 价格: {order.executed.price:.2f}, 数量: {order.executed.size}, 原因: {self.trade_reason}')
            else:
                self.entry_price = None
                self.take_profit_price = None  # 重置止盈价格
                self.entry_price_for_tp = None  # 重置用于计算止盈的入场价格
                if self.p.use_trailing_stop:
                    self.trailing_stop.stop_tracking()
                order.info = {'reason': self.trade_reason}  # 记录交易原因
                order.info['total_value'] = self.broker.getvalue()  # 记录总资产（含现金）
                order.info['position_value'] = self.position.size * order.executed.price if self.position else 0  # 记录持仓市值
                self._orders.append(order)  # 添加到订单列表
                logger.info(f'卖出执行 - 价格: {order.executed.price:.2f}, 数量: {order.executed.size}, 原因: {self.trade_reason}')
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            logger.warning(f'订单失败 - 状态: {order.getstatusname()}')
            
        self.order = None

    def stop(self):
        """策略结束时的汇总信息"""
        portfolio_value = self.broker.getvalue()
        returns = (portfolio_value / self.broker.startingcash) - 1.0
        logger.info(f"策略结束 - 最终资金: {portfolio_value:.2f}, 收益率: {returns:.2%}") 