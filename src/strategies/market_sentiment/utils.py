import numpy as np
import pandas as pd
import talib
from loguru import logger

class TrendStateDetector:
    def __init__(self, window=60):
        self.window = window
        self.last_regime = None
        self.regime_change_date = None
        # 缓存历史计算结果
        self.cached_ema5 = None
        self.cached_ema13 = None
        self.cached_ema55 = None
        self.cached_momentum = None
        self.crossover_threshold = 0.001  # 0.1%的最小上穿幅度要求
        
    def detect(self, price_series, volume_series, current_date=None):
        """趋势检测逻辑，基于EMA交叉和价格位置"""
        price_series_pd = pd.Series(price_series)
        
        # 计算EMA指标
        ema5 = talib.EMA(price_series_pd, timeperiod=5)
        ema13 = talib.EMA(price_series_pd, timeperiod=10)
        ema55 = talib.EMA(price_series_pd, timeperiod=55)
        
        # 动量计算
        momentum = talib.MOM(price_series_pd, timeperiod=5)
        sma_momentum = talib.SMA(momentum, timeperiod=3)
        
        # 成交量指标
        volume_series_pd = pd.Series(volume_series)
        obv = talib.OBV(price_series_pd, volume_series_pd)
        obv_ema = talib.EMA(obv, timeperiod=5)
        
        current_regime = 'normal'
        
        if len(price_series) >= 55:
            # 获取当前和前一日的值
            ema5_current = ema5.iloc[-1]
            ema5_prev = ema5.iloc[-2]
            ema13_current = ema13.iloc[-1]
            ema13_prev = ema13.iloc[-2]
            ema55_current = ema55.iloc[-1]
            ema55_prev = ema55.iloc[-2]
            price_current = price_series_pd.iloc[-1]
            price_prev = price_series_pd.iloc[-2]
            
            # 成交量确认
            volume_confirmed = obv.iloc[-1] > obv_ema.iloc[-1]
            
            # 1. EMA5上穿EMA13时设置为potential_uptrend
            crossover_pct = (ema5_current - ema13_current) / ema13_current
            if (ema5_current > ema13_current and ema5_prev <= ema13_prev and 
                crossover_pct > self.crossover_threshold):
                current_regime = 'potential_uptrend'
            
            # 2. EMA5上穿EMA13后，收盘价格连续3个交易日站上EMA55，设置为uptrend
            if (ema5_current > ema13_current and 
                price_current > ema55_current and 
                price_series_pd.iloc[-2] > ema55.iloc[-2] and 
                price_series_pd.iloc[-3] > ema55.iloc[-3] and
                volume_confirmed):
                current_regime = 'uptrend'
            
            # 3. EMA5下穿EMA13，但收盘价格在EMA55之上，设置为potential_downtrend
            if (ema5_current < ema13_current and 
                ema5_prev >= ema13_prev and 
                price_current > ema55_current):
                current_regime = 'potential_downtrend'
            
            # 4. EMA5下穿EMA55，设置为downtrend
            if ema5_current < ema55_current and ema5_prev >= ema55_prev:
                current_regime = 'downtrend'
        
        # 趋势变化检测
        if current_regime != self.last_regime:
            self.regime_change_date = current_date
            if current_date is not None:
                logger.info(f"趋势变化: {self.last_regime} -> {current_regime}, 日期: {current_date}")
            self.last_regime = current_regime
        
        return current_regime

class PositionManager:
    def __init__(self, max_risk=0.45):
        self.max_risk = max_risk
        
    def adjust_position(self, target_ratio, volatility):
        """简化的仓位管理，只根据波动率调整仓位"""
        # 波动率调整系数 - 这里的2是百分比形式的波动率基准值(2%)
        # 实际波动率通常在1%~3%之间
        vol_adj = np.clip(volatility / 2, 0.5, 1.5)
        
        # 根据波动率调整目标仓位
        adjusted_ratio = target_ratio * vol_adj
        
        # 设置风险上限
        if adjusted_ratio > self.max_risk * 2:  # 最大允许仓位90%
            adjusted_ratio = self.max_risk * 2
        
        logger.info(f"波动率: {volatility:.2f}, 波动率调整系数: {vol_adj:.2f}, 目标仓位: {target_ratio:.2f}, 调整后仓位: {adjusted_ratio:.2f}")
            
        return adjusted_ratio

# # 信号生成参数
# SENTIMENT_THRESHOLDS = {
#     'core': 2.5,      # 核心信号阈值
#     'secondary': 9.0,  # 次级信号阈值
#     'light': 9.0     # 轻仓信号阈值
# }

# POSITION_WEIGHTS = {
#     'core': 0.95,      # 核心信号仓位
#     'secondary': 0.9,  # 次级信号仓位
#     'light': 0.8      # 轻仓信号仓位
# }

# def generate_signals(sentiment_score, regime, volatility):
#     """简化的信号生成，只区分下跌趋势和正常趋势"""
#     signals = []
    
#     # 下跌趋势不开仓
#     if regime == 'downtrend':
#         if sentiment_score < SENTIMENT_THRESHOLDS['core']:
#             signals.append({'type': 'buy', 'weight': POSITION_WEIGHTS['core']})  # 核心信号
#             return signals
#         else:
#             return []
    
#     # 其他情况根据情绪分数决定
#     if sentiment_score < SENTIMENT_THRESHOLDS['core']:
#         signals.append({'type': 'buy', 'weight': POSITION_WEIGHTS['core']})  # 核心信号
#     elif SENTIMENT_THRESHOLDS['core'] <= sentiment_score < SENTIMENT_THRESHOLDS['secondary']:
#         signals.append({'type': 'buy', 'weight': POSITION_WEIGHTS['secondary']})  # 次级信号
#     elif SENTIMENT_THRESHOLDS['secondary'] <= sentiment_score < SENTIMENT_THRESHOLDS['light']:
#         signals.append({'type': 'buy', 'weight': POSITION_WEIGHTS['light']})  # 轻仓信号
    
#     return signals