import json
import os
import backtrader as bt
from loguru import logger
from src.indicators.trailing_stop import TrailingStop
from src.strategies.market_sentiment.sentiment_data import get_sentiment_data
import math
import numpy as np
import pandas as pd
from src.strategies.market_sentiment.utils import TrendStateDetector, PositionManager
from src.strategies.market_sentiment.etf_dividend_handler import ETFDividendHandler
from datetime import datetime

class MarketSentimentStrategy(bt.Strategy):
    params = (
        ('sentiment_core', 5.0),     # 核心信号情绪阈值
        ('sentiment_secondary', 15.0), # 次级信号情绪阈值
        ('sentiment_warning', 9.0),   # 预警信号情绪阈值
        ('sentiment_sell_1', 40.0),    # 第一阶段止盈阈值
        ('sentiment_sell_2', 55.0),    # 第二阶段止盈阈值
        ('sentiment_sell_3', 70.0),   # 第三阶段止盈阈值
        ('position_core', 0.5),       # 核心信号仓位比例
        ('position_grid_step', 0.1),  # 网格加仓步长
        ('position_warning', 0.05),   # 预警信号初始仓位
        ('position_bb_signal', 0.1),  # 布林带突破信号仓位
        ('min_momentum', -5.0),       # 最小动量要求(10日涨跌幅)
        ('vol_threshold', 20.0),      # 高波动率阈值
        ('quick_profit_days', 5),     # 短期获利天数
        ('quick_profit_pct', 15.0),   # 短期获利比例阈值
        ('high_vol_profit_pct', 20.0),# 高波动市场获利阈值
        ('trail_percent', 2.0),       # 追踪止损百分比
        ('risk_ratio', 0.02),         # 单次交易风险比率
        ('max_drawdown', 0.15),       # 最大回撤限制
        ('price_limit', 0.10),        # 涨跌停限制(10%)
        ('use_trailing_stop', False),  # 是否启用追踪止损
        ('bb_period', 20),            # 布林带周期
        ('bb_devfactor', 2.0),        # 布林带标准差倍数
        ('rsi_period', 14),           # RSI周期
        ('volume_ma_period', 20),     # 成交量均线周期
        ('garch_vol_threshold', 25.0),# GARCH波动率阈值
        ('handle_dividend', True),    # 是否处理分红
        ('min_shares', 100),          # 最小交易股数
        ('min_position', 3000),       # 最小持仓股数
        ('cash_buffer', 0.95),        # 现金缓冲比例
        ('core_min_value', 500000),   # 核心信号最小买入金额
        ('secondary_min_value', 200000), # 次级信号最小买入金额
        ('core_min_shares', 35000),    # 核心信号最大买入股数
        ('secondary_min_shares', 30000), # 次级信号最大买入股数
        ('min_profit_pct', 1.0),      # 最小盈利要求
        ('price_drop_threshold', 0.99), # 加仓价格条件阈值
        ('atr_multiplier', 1),      # ATR倍数
        ('history_length', 60),       # 历史数据长度
        ('atr_tp_multiplier_min', 1),   # ATR止盈最小倍数
        ('atr_tp_multiplier_max', 3),   # ATR止盈最大倍数
        ('atr_tp_min_profit', 0.5),   # ATR止盈最小盈利要求
        ('tp_min_profit', 2.0),       # 普通止盈最小盈利要求
        ('vol_atr_ratio', 0.8),       # 波动率与ATR倍数的相关系数
        ('sentiment_entry', 25.0),    # 情绪开仓阈值
        ('max_add_positions', 5),     # 最大补仓次数
        ('max_atr_tp_times', 2),      # 最大ATR止盈次数
    )

    def __init__(self):
        """初始化策略"""
        super().__init__()
        
        # 获取数据源
        self.data = self.datas[0]
        
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
            
        # 设置数据源的名称
        self.data._name = self.etf_code
        
        # 获取回测区间
        self.start_date = None
        self.end_date = None
        try:
            if hasattr(self.data, 'params'):
                if hasattr(self.data.params, 'fromdate'):
                    self.start_date = self.data.params.fromdate
                    if isinstance(self.start_date, datetime):
                        self.start_date = self.start_date.date()
                if hasattr(self.data.params, 'todate'):
                    self.end_date = self.data.params.todate
                    if isinstance(self.end_date, datetime):
                        self.end_date = self.end_date.date()
        except Exception as e:
            logger.warning(f"获取回测区间时出错: {str(e)}")
            
        logger.info(f"初始化分红处理器 - 股票代码: {self.etf_code}, 回测区间: {self.start_date} 至 {self.end_date}")
        
        # 初始化分红处理器
        self.dividend_handler = None
        if self.p.handle_dividend:
            if self.etf_code:
                self.dividend_handler = ETFDividendHandler(ts_code=self.etf_code)
                self.dividend_handler.update_dividend_data(start_date=self.start_date, end_date=self.end_date)
            else:
                logger.warning("无法获取股票代码，分红处理功能将不可用")
        
        # 累计分红金额
        self.total_dividend = 0.0
        
        # 趋势识别和仓位管理
        self.trend_detector = TrendStateDetector()
        self.position_manager = PositionManager()
        
        # 记录上一次市场状态
        self.last_market_regime = None
        
        # 存储价格和成交量数据用于趋势分析
        self.price_history = []
        self.volume_history = []
        
        # 情绪指标
        self.sentiment_data = None
        self.sentiment_details = {}  # 存储详细情绪数据
        
        # 追踪止损指标
        if self.p.use_trailing_stop:
            self.trailing_stop = TrailingStop(self.data, trailing=self.p.trail_percent/100.0)
            self.trailing_stop._owner = self
        else:
            self.trailing_stop = None
        
        # 基础指标
        self.atr = bt.indicators.ATR(self.data)
        self.rsi = bt.indicators.RSI(self.data, period=self.p.rsi_period)
        self.bb = bt.indicators.BollingerBands(self.data, period=self.p.bb_period, devfactor=self.p.bb_devfactor)
        self.ema55 = bt.indicators.EMA(self.data.close, period=55)  # EMA55指标
        self.ema5 = bt.indicators.EMA(self.data.close, period=5)    # 添加EMA5指标
        
        # 量价指标
        self.volume_ma = bt.indicators.SMA(self.data.volume, period=self.p.volume_ma_period)
        self.returns = bt.indicators.PercentChange(self.data.close, period=1)  # 日收益率
        
        # 动量指标 - 计算10日和5日收益率
        self.pct_change_10d = bt.indicators.PercentChange(self.data.close, period=10)
        self.pct_change_5d = bt.indicators.PercentChange(self.data.close, period=5)
                
        # 记录最高净值
        self.highest_value = self.broker.getvalue()
        
        # 订单和持仓跟踪
        self.order = None
        self.entry_price = None
        self.avg_cost = None  # 平均持仓成本
        self.trade_reason = None  # 记录交易原因
        self._orders = []  # 记录所有订单
        
        # 持仓管理
        self.current_position_ratio = 0.0  # 当前仓位比例
        
        # 止盈相关变量
        self.take_profit_price = None
        self.entry_price_for_tp = None  # 用于计算止盈的入场价格
        
        # T+1交易限制
        self.buy_dates = set()
        
        # 记录是否是核心信号开仓
        self.is_core_signal_position = False
        
        # 记录上次清仓日期
        self.last_close_date = None
        
        # 跟踪情绪得分
        self.last_sentiment = None
        self.current_sentiment = None
        self.is_continuous_low = False  # 是否连续低于阈值
        
        # 跟踪补仓和止盈次数
        self.add_position_count = 0  # 补仓次数
        self.atr_tp_count = 0       # ATR止盈次数
        
        # 记录连续低于EMA55的天数
        self.below_ema55_days = 0
        
        # 记录最后一次加仓日期
        self.last_add_position_date = None
        # 记录5日前的价格
        self.price_5days_ago = None
        # 价格历史记录
        self.price_history_5d = []
        # 记录市场状态历史
        self.market_regime_history = []
        # 记录是否已经触发过阶梯止盈
        self.ladder_tp_triggered = False
        
        logger.info(f"多维度市场情绪策略初始化 - 核心信号: {self.p.sentiment_core}, "
                   f"次级信号: {self.p.sentiment_secondary}, 预警信号: {self.p.sentiment_warning}")

    def round_shares(self, shares):
        """将股数调整为100的整数倍"""
        return math.floor(shares / self.p.min_shares) * self.p.min_shares
        
    def check_price_limit(self, price):
        """检查是否触及涨跌停"""
        prev_close = self.data.close[-1]
        upper_limit = prev_close * (1 + self.p.price_limit)
        lower_limit = prev_close * (1 - self.p.price_limit)
        return lower_limit <= price <= upper_limit
    
    def get_position_value_ratio(self):
        """计算当前持仓市值占总资产的比例"""
        if not self.position:
            return 0.0
        
        position_value = self.position.size * self.data.close[0]
        total_value = self.broker.getvalue()
        return position_value / total_value
        
    def calculate_trade_size(self, price, target_ratio):
        """计算可交易的股数，根据目标仓位比例决定"""
        # 使用当前总资产而不是初始本金
        total_value = self.broker.getvalue()
        cash = total_value * self.p.cash_buffer  # 留5%作为缓冲
        
        # 计算当前持仓市值及其占比
        current_position_value = self.position.size * price if self.position else 0
        current_ratio = current_position_value / total_value if total_value > 0 else 0
        
        # 如果当前持仓比例已经超过目标仓位，不再加仓
        if current_ratio >= target_ratio:
            logger.info(f"当前持仓比例已经超过目标仓位，不再加仓 - 当前比例: {current_ratio:.2f}, 目标比例: {target_ratio:.2f}")
            return 0
        
        # 加仓价格条件：如果已有持仓，只有当前价格比持仓均价低5%以上才允许加仓
        if self.position and self.position.size > 0:
            avg_cost = self.get_avg_cost()
            if price > avg_cost * self.p.price_drop_threshold:  # 当前价格比均价的降幅不足5%
                return 0  # 不满足加仓价格条件，不加仓
            
        # 计算需要买入的目标市值
        target_position_value = total_value * target_ratio
        value_to_buy = target_position_value - current_position_value
        
        # 考虑风险控制
        risk_amount = total_value * self.p.risk_ratio
        current_atr = self.atr[0]
        risk_per_share = max(current_atr * self.p.atr_multiplier, price * (self.p.trail_percent/100.0))
        
        risk_size = risk_amount / risk_per_share if risk_per_share > 0 else 0
        cash_size = min(cash, value_to_buy) / price
        
        shares = min(risk_size, cash_size)
        shares = self.round_shares(shares)
        
        if shares * price > cash:
            shares = self.round_shares(cash / price)
            
        return shares if shares >= self.p.min_shares else 0
        
    def calculate_core_signal_size(self, price, target_ratio):
        """为核心信号计算交易规模，确保最小买入数量"""
        # 先获取通常的交易规模计算结果
        normal_size = self.calculate_trade_size(price, target_ratio)
        
        # 如果正常计算结果超过5000股，则返回该结果
        if normal_size >= self.p.core_min_shares:
            return normal_size
        
        # 对于核心信号，即使已有持仓，也应该检查价格条件
        if self.position and self.position.size > 0:
            avg_cost = self.get_avg_cost()
            # 如果价格高于均价的95%，仍然要根据情绪计算最小买入量
            if price > avg_cost * self.p.price_drop_threshold:
                # 情绪极低时（低于2.5分）执行首次建仓策略，不受价格限制
                if self.sentiment_dict.get(self.data.datetime.date().strftime('%Y-%m-%d'), 99) < self.p.sentiment_core:
                    # 继续执行计算最小买入量的逻辑
                    pass
                else:
                    return 0  # 不是核心信号或价格不满足条件，不加仓
            
        # 否则，计算基于资金的最小买入数量（至少占总资产的10%）
        total_value = self.broker.getvalue()
        cash = total_value * self.p.cash_buffer  # 留5%作为缓冲
        
        # 计算最小买入市值 - 至少是总资产的10%或500,000元中的较小值
        min_value_to_buy = min(total_value * 0.1, self.p.core_min_value)
        
        # 确保不超过可用现金
        min_value_to_buy = min(min_value_to_buy, cash)
        
        # 计算最小股数
        min_shares = self.round_shares(min_value_to_buy / price)
        
        # 取正常计算和最小买入量中的较大值
        result_shares = max(normal_size, min_shares)
        
        # 确保不超过可用现金
        if result_shares * price > cash:
            result_shares = self.round_shares(cash / price)
            
        # 确保最小为100股
        return result_shares if result_shares >= self.p.min_shares else 0

    def calculate_secondary_signal_size(self, price, target_ratio):
        """为次级信号计算交易规模，确保最小买入数量"""
        # 先获取通常的交易规模计算结果
        normal_size = self.calculate_trade_size(price, target_ratio)

        logger.info(f"次级信号计算交易规模 - 当前价格: {price:.2f}, 目标仓位比例: {target_ratio:.2f}, 正常计算结果: {normal_size:.2f}")
        
        # 如果已有持仓，需要满足价格条件：当前价格比持仓均价低5%以上才允许加仓
        if self.position and self.position.size > 0:
            avg_cost = self.get_avg_cost()
            if price > avg_cost * self.p.price_drop_threshold:  # 当前价格比均价的降幅不足5%
                return 0  # 不满足加仓价格条件，不加仓
        
        # 如果正常计算结果超过最小股数要求，则返回该结果
        if normal_size >= self.p.min_shares:
            return normal_size
        
        # 否则加仓target_ratio的数量
        total_value = self.broker.getvalue()
        cash = self.broker.getcash() * self.p.cash_buffer  # 留5%作为缓冲
        
        # 计算基于目标比例的买入数量
        target_value = total_value * target_ratio
        target_shares = self.round_shares(target_value / price)
        
        # 确保不超过可用现金
        if target_shares * price > cash:
            target_shares = self.round_shares(cash / price)
        
        # 确保最小为100股
        return target_shares if target_shares >= self.p.min_shares else 0

    def get_sentiment_details(self, date_str):
        """获取指定日期的详细情绪数据"""
        if date_str in self.sentiment_details:
            return self.sentiment_details[date_str]
        
        # 从sentiment_data中获取details
        for sentiment_item in self.sentiment_data['sentiment']:
            if sentiment_item['date'] == date_str and 'details' in sentiment_item:
                details = sentiment_item['details']
                # 存储计算结果
                self.sentiment_details[date_str] = details
                return details
            
        # 如果没有找到数据,返回默认值
        return {
            'change': 0,
            'change_5d': 0,
            'conditional_vol': 0,
            'rsi': 50,
            'bb_position': 0,
            'volume_ratio': 1.0
        }

    def get_avg_cost(self):
        """获取当前持仓的平均成本"""
        if not self.position or self.position.size == 0:
            return 0.0
            
        # 如果已计算过平均成本，直接返回
        if self.avg_cost is not None:
            return self.avg_cost
            
        # 如果没有计算过，则使用entry_price作为近似值
        return self.entry_price or self.data.close[0]

    def next(self):
        if self.order:
            return
            
        # 重置交易原因
        self.trade_reason = None
            
        # 检查是否有分红
        current_date = self.data.datetime.date()
        current_date_str = current_date.strftime('%Y-%m-%d')
        
        # 处理ETF分红
        if self.p.handle_dividend and self.dividend_handler and self.position and self.position.size > 0:
            dividend_amount = self.dividend_handler.process_dividend(
                current_date_str, 
                self.position.size,
                self.data.close[0]
            )
            
            if dividend_amount > 0:
                # 分红金额直接计入现金
                self.broker.add_cash(dividend_amount)
                self.total_dividend += dividend_amount
                
                # 调整平均成本 - 考虑分红后的成本降低
                if self.avg_cost is not None and self.avg_cost > 0:
                    # 每股分红
                    div_per_share = dividend_amount / self.position.size
                    # 调整平均成本
                    self.avg_cost = max(0.001, self.avg_cost - div_per_share)  # 确保成本不会变为负数或0
                    logger.info(f"ETF分红后调整平均成本 - 原成本: {self.avg_cost + div_per_share:.4f}, 新成本: {self.avg_cost:.4f}, 每股分红: {div_per_share:.4f}, 总分红: {dividend_amount:.2f}")
            
        # 更新历史数据
        self.price_history.append(self.data.close[0])
        self.volume_history.append(self.data.volume[0])
        
        # 保持历史数据长度为60个周期
        if len(self.price_history) > self.p.history_length:
            self.price_history = self.price_history[-self.p.history_length:]
            self.volume_history = self.volume_history[-self.p.history_length:]
            
        # 第一次调用next时获取情绪数据
        if self.sentiment_data is None:
            try:
                self.sentiment_data = get_sentiment_data(start_date=self.start_date, end_date=self.end_date)
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
                self._reset_position_state()
            return
            
        current_price = self.data.close[0]
        if not self.check_price_limit(current_price):
            return
            
        if self.position and self.p.use_trailing_stop:
            self.trailing_stop.next()
            
        # 获取当前日期的情绪数据
        current_date = self.data.datetime.date()
        current_date_str = current_date.strftime('%Y-%m-%d')
        
        # 将情绪数据转换为字典
        if not hasattr(self, 'sentiment_dict'):
            self.sentiment_dict = {data['date']: data['value'] for data in self.sentiment_data['sentiment']}
            
        sentiment_value = self.sentiment_dict.get(current_date_str)
        if sentiment_value is None:
            return
            
        # 获取详细情绪数据
        details = self.get_sentiment_details(current_date_str)
        
        # 更新当前持仓比例
        self.current_position_ratio = self.get_position_value_ratio()
        
        # 检测市场状态
        if len(self.price_history) >= self.p.history_length:  # 确保有足够的历史数据
            market_regime = self.trend_detector.detect(
                np.array(self.price_history),
                np.array(self.volume_history),
                current_date
            )
            
            # 更新市场状态历史
            self.market_regime_history.append(market_regime)
            if len(self.market_regime_history) > 7:
                self.market_regime_history.pop(0)
            
            # 更新情绪值
            self.last_sentiment = self.current_sentiment
            self.current_sentiment = sentiment_value

            # 检查是否满足清仓后等待期要求
            if self.last_close_date:
                days_since_close = (current_date - self.last_close_date).days
                if days_since_close < 5:
                    logger.info(f"清仓后等待期不足5个交易日，跳过开仓 - 已等待: {days_since_close}天")
                    return
            
            # 检查是否符合开仓条件
            if len(self.market_regime_history) >= 7:
                regime_changes = sum(1 for i in range(len(self.market_regime_history)-1) 
                                    if self.market_regime_history[i] != self.market_regime_history[i+1])
                if regime_changes >= 2:  # 如果7天内状态变化超过3次，认为是震荡市
                    logger.info(f"最近7个交易日市场震荡（状态变化{regime_changes}次），跳过开仓")
                    return
                
            if (market_regime == 'uptrend' or market_regime == 'potential_uptrend') and self.last_sentiment is not None and self.current_sentiment is not None:
                if self.current_sentiment < self.p.sentiment_entry and self.current_sentiment > self.p.sentiment_secondary:
                    # 没有持仓时开仓
                    # if not self.position:
                        # 检查最近7个交易日是否存在震荡市
                        if self.current_sentiment > self.last_sentiment:  # 情绪值开始回升时开仓
                            # 计算开仓数量
                            shares_to_buy = self.calculate_core_signal_size(current_price, self.p.position_core)
                            if shares_to_buy >= self.p.min_shares:
                                self.trade_reason = f"情绪回升开仓 - 前值({self.last_sentiment:.2f}) < 当前值({self.current_sentiment:.2f})"
                                self.order = self.buy(size=shares_to_buy)
                                return

            if self.position and self.position.size > 0:
                # 如果价格在EMA55上方，检查加仓条件
                if self.data.close[0] > self.ema55[0]:
                    # 如果已经触发过阶梯止盈，不再加仓
                    if self.ladder_tp_triggered:
                        logger.info("已触发阶梯止盈，不再加仓")
                        return
                        
                    # 检查是否距离上次加仓超过5个交易日
                    if self.last_add_position_date:
                        days_since_last_add = (current_date - self.last_add_position_date).days
                        if days_since_last_add <= 5:
                            return  # 不满足加仓间隔要求
                    
                    # 检查是否可以加仓
                    if self.add_position_count < self.p.max_add_positions:
                        shares_to_buy = self.calculate_secondary_signal_size(self.data.close[0], self.p.position_grid_step)
                        if shares_to_buy >= self.p.min_shares:
                            self.trade_reason = f"EMA55上方加仓({self.add_position_count + 1}/{self.p.max_add_positions})"
                            self.order = self.buy(size=shares_to_buy)
                            if self.order:
                                self.last_add_position_date = current_date  # 更新最后加仓日期
                                self.add_position_count += 1
                            return
                
            # 更新上一次市场状态
            self.last_market_regime = market_regime
            
            # 每周记录一次当前市场状态
            # logger.info(f"当前日期: {current_date_str}, 当前市场状态: {market_regime}, 情绪: {sentiment_value:.2f}, 波动率: {details['conditional_vol']:.2f}%")
            
            # 生成交易信号
            # signals = generate_signals(sentiment_value, market_regime, details['conditional_vol'])
            
            # 如果是下跌或震荡趋势，检查是否需要减仓
            if (market_regime == 'downtrend' or market_regime == 'potential_downtrend') and self.position and self.position.size > 0:
                # 如果是核心信号开仓，则不执行减仓操作
                # if not self.is_core_signal_position:
                # 判断是否需要减仓
                avg_cost = self.get_avg_cost()
                profit_pct = ((current_price / avg_cost) - 1.0) * 100 if avg_cost > 0 else 0
                
                if profit_pct < -3.0:  # 如果没有盈利，且跌幅超过3%，止损清仓
                    self.trade_reason = f"下跌趋势止损清仓 - 亏损({profit_pct:.2f}%)"
                    self.order = self.close()
                    if self.order:
                        logger.info(f"下跌趋势止损清仓 - 情绪: {sentiment_value:.2f}, 亏损: {profit_pct:.2f}%, 平仓数量: {self.position.size}")
                        self._reset_position_state()
                    return

        # =============== 执行动态止盈策略 ===============
        if self.position:
            if current_date in self.buy_dates:
                return  # T+1交易限制
                
            # 计算当前盈亏比例 - 使用平均成本
            avg_cost = self.get_avg_cost()
            profit_pct = ((current_price / avg_cost) - 1.0) * 100 if avg_cost > 0 else 0
            min_profit_pct = self.p.min_profit_pct  # 最小盈利要求：1%
            
            # ATR止盈检查
            if self.position and self.position.size > 0:
                # 获取当前情绪值
                current_date_str = self.data.datetime.date().strftime('%Y-%m-%d')
                sentiment_value = self.sentiment_dict.get(current_date_str)
                
                # 如果情绪值大于第二阶段止盈阈值，跳过ATR止盈
                if sentiment_value and sentiment_value >= self.p.sentiment_sell_2:
                    logger.info(f"情绪值大于第二阶段止盈阈值，跳过ATR止盈 - 情绪: {sentiment_value:.1f}")
                else:
                    atr = self.atr[0]
                    # 根据波动率动态调整ATR倍数
                    vol_ratio = details['conditional_vol'] / self.p.vol_threshold
                    dynamic_multiplier = self.p.atr_tp_multiplier_min + (self.p.atr_tp_multiplier_max - self.p.atr_tp_multiplier_min) * min(vol_ratio * self.p.vol_atr_ratio, 1.0)
                    atr_tp_price = avg_cost + (atr * dynamic_multiplier)

                    logger.info(f"ATR倍数: {dynamic_multiplier:.2f}, ATR止盈价格: {atr_tp_price:.2f}, 盈利: {profit_pct:.2f}%")
                    
                    # 检查是否满足ATR止盈条件，并且价格不在EMA5上方
                    if ((current_price >= atr_tp_price and profit_pct >= self.p.atr_tp_min_profit) or (profit_pct >= self.p.tp_min_profit)) and current_price <= self.ema5[0]:
                        # 根据波动率调整卖出比例
                        sell_ratio = 0.5
                        if vol_ratio > 0.05:  # 高波动市场卖出更多
                            sell_ratio = 0.8
                        
                        shares_to_sell = self.round_shares(self.position.size * sell_ratio)
                        if self.position.size - shares_to_sell < self.p.min_position:
                            self.trade_reason = f"ATR止盈转清仓 - 价格突破ATR止盈点({atr_tp_price:.2f}), 盈利: {profit_pct:.2f}%, ATR倍数: {dynamic_multiplier:.2f}"
                            self.order = self.close()
                            if self.order:
                                logger.info(f"ATR止盈转清仓 - ATR: {atr:.2f}, 止盈价: {atr_tp_price:.2f}, 盈利: {profit_pct:.2f}%, ATR倍数: {dynamic_multiplier:.2f}, 平仓比例: 100%")
                                self._reset_position_state()
                            return
                        
                        if shares_to_sell >= self.p.min_shares:
                            self.trade_reason = f"ATR止盈 - 价格突破ATR止盈点({atr_tp_price:.2f}), 盈利: {profit_pct:.2f}%, ATR倍数: {dynamic_multiplier:.2f}"
                            self.order = self.sell(size=shares_to_sell)
                            if self.order:
                                logger.info(f"ATR止盈触发 - ATR: {atr:.2f}, 止盈价: {atr_tp_price:.2f}, 盈利: {profit_pct:.2f}%, ATR倍数: {dynamic_multiplier:.2f}, 平仓数量: {shares_to_sell}")
                            return
                    elif (current_price >= atr_tp_price and profit_pct >= self.p.atr_tp_min_profit) and current_price > self.ema5[0] and self.add_position_count > 0:
                        logger.info(f"满足ATR止盈条件但价格在EMA5上方，且补过仓，止盈清仓 - 当前价格: {current_price:.2f}, EMA5: {self.ema5[0]:.2f}")
                        self.trade_reason = f"ATR止盈 - 价格突破ATR止盈点({atr_tp_price:.2f}), 盈利: {profit_pct:.2f}%, ATR倍数: {dynamic_multiplier:.2f}"
                        self.order = self.close()
                        if self.order:
                            logger.info(f"ATR止盈触发清仓 - ATR: {atr:.2f}, 止盈价: {atr_tp_price:.2f}, 盈利: {profit_pct:.2f}%, ATR倍数: {dynamic_multiplier:.2f}")
                            self._reset_position_state()
                        return
            
            # 阶梯式止盈机制
            if sentiment_value >= self.p.sentiment_sell_3 and self.current_position_ratio > 0 and profit_pct >= min_profit_pct:
                self.trade_reason = f"阶梯止盈3 - 情绪达到顶部({sentiment_value:.1f}), 盈利: {profit_pct:.2f}%"
                self.order = self.close()
                if self.order:
                    logger.info(f"阶梯止盈3触发 - 情绪: {sentiment_value:.1f}, 盈利: {profit_pct:.2f}%, 平仓比例: 100%")
                    self._reset_position_state()
                return
                
            elif sentiment_value >= self.p.sentiment_sell_2 and self.current_position_ratio > 0 and profit_pct >= min_profit_pct:
                shares_to_sell = self.round_shares(self.position.size * 0.3)
                if self.position.size - shares_to_sell < self.p.min_position:
                    self.trade_reason = f"阶梯止盈2转清仓 - 情绪较高({sentiment_value:.1f}), 卖出后剩余股数不足{self.p.min_position}"
                    self.order = self.close()
                    if self.order:
                        logger.info(f"阶梯止盈2转清仓 - 情绪: {sentiment_value:.1f}, 盈利: {profit_pct:.2f}%, 平仓比例: 100%")
                        self._reset_position_state()
                    return
                
                if shares_to_sell >= self.p.min_shares:
                    self.trade_reason = f"阶梯止盈2 - 情绪较高({sentiment_value:.1f}), 盈利: {profit_pct:.2f}%"
                    self.order = self.sell(size=shares_to_sell)
                    if self.order:
                        logger.info(f"阶梯止盈2触发 - 情绪: {sentiment_value:.1f}, 盈利: {profit_pct:.2f}%, 平仓数量: {shares_to_sell}")
                        self.ladder_tp_triggered = True  # 标记已触发阶梯止盈
                    return
                    
            elif sentiment_value >= self.p.sentiment_sell_1 and self.current_position_ratio > 0 and profit_pct >= min_profit_pct:
                shares_to_sell = self.round_shares(self.position.size * 0.5)
                if self.position.size - shares_to_sell < self.p.min_position:
                    self.trade_reason = f"阶梯止盈1转清仓 - 情绪回升({sentiment_value:.1f}), 卖出后剩余股数不足{self.p.min_position}"
                    self.order = self.close()
                    if self.order:
                        logger.info(f"阶梯止盈1转清仓 - 情绪: {sentiment_value:.1f}, 盈利: {profit_pct:.2f}%, 平仓比例: 100%")
                        self._reset_position_state()
                    return
                
                if shares_to_sell >= 100:
                    self.trade_reason = f"阶梯止盈1 - 情绪回升({sentiment_value:.1f}), 盈利: {profit_pct:.2f}%"
                    self.order = self.sell(size=shares_to_sell)
                    if self.order:
                        logger.info(f"阶梯止盈1触发 - 情绪: {sentiment_value:.1f}, 盈利: {profit_pct:.2f}%, 平仓数量: {shares_to_sell}")
                        self.ladder_tp_triggered = True  # 标记已触发阶梯止盈
                    return
            
            # 追踪止损检查
            if self.p.use_trailing_stop:
                stop_price = self.trailing_stop[0]
                if current_price < stop_price:
                    self.trade_reason = f"触发追踪止损 (止损价: {stop_price:.2f})"
                    self.order = self.close()
                    if self.order:
                        logger.info(f"追踪止损触发 - 当前价格: {current_price:.2f}, 止损价: {stop_price:.2f}")
                    return

    def _reset_position_state(self):
        """重置持仓相关的状态变量"""
        self.entry_price = None
        self.take_profit_price = None
        self.entry_price_for_tp = None
        self.current_position_ratio = 0.0
        self.is_core_signal_position = False  # 重置核心信号开仓标记
        if self.p.use_trailing_stop:
            self.trailing_stop.stop_tracking()
        self.buy_dates = set()
        self.highest_value = self.broker.getvalue()
        self.add_position_count = 0
        self.atr_tp_count = 0
        self.is_continuous_low = False
        self.below_ema55_days = 0  # 重置EMA55计数
        self.last_add_position_date = None  # 重置最后加仓日期
        self.price_history_5d = []  # 重置5日价格历史
        self.market_regime_history = []  # 重置市场状态历史
        self.ladder_tp_triggered = False  # 重置阶梯止盈标记

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
            
        if order.status in [order.Completed]:
            if order.isbuy():
                if self.p.use_trailing_stop:
                    self.trailing_stop.reset(price=order.executed.price)
                
                # 更新平均成本 - 买入时更新
                if self.avg_cost is None:
                    # 首次买入
                    self.avg_cost = order.executed.price
                else:
                    # 计算新的平均成本
                    # 当前持仓量是 position.size - order.executed.size (因为position.size已经包含了新买入的数量)
                    current_position = self.position.size - order.executed.size
                    current_value = current_position * self.avg_cost
                    new_value = order.executed.size * order.executed.price
                    total_position = current_position + order.executed.size
                    self.avg_cost = (current_value + new_value) / total_position
                    logger.info(f"当前持仓量: {current_position}, 当前持仓市值: {current_value}, 新买入市值: {new_value}, 总持仓量: {total_position}")


                logger.info(f"买入平均成本: {self.avg_cost:.4f}")
                # 记录入场价格用于计算止盈
                self.entry_price = order.executed.price  # 保留最后一次买入价格
                
                if self.p.use_trailing_stop:
                    # 如果是首次建仓或全部平仓后再次建仓，更新入场价格
                    if self.entry_price_for_tp is None:
                        self.entry_price_for_tp = order.executed.price
                
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
                    self.take_profit_price = None
                    self.entry_price_for_tp = None
                    if self.p.use_trailing_stop:
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
            
        self.order = None

    def stop(self):
        """策略结束时的汇总信息"""
        portfolio_value = self.broker.getvalue()
        returns = (portfolio_value / self.broker.startingcash) - 1.0
        logger.info(f"策略结束 - 最终资金: {portfolio_value:.2f}, 收益率: {returns:.2%}")
        
        # 记录分红收益信息
        if self.p.handle_dividend:
            dividend_return = (self.total_dividend / self.broker.startingcash) * 100
            total_return = returns * 100
            price_return = total_return - dividend_return
            
            logger.info(f"分红统计 - 总分红: {self.total_dividend:.2f}, 分红收益率: {dividend_return:.2f}%")
            logger.info(f"收益分解 - 总收益率: {total_return:.2f}%, 价格收益率: {price_return:.2f}%, 分红收益率: {dividend_return:.2f}%") 
