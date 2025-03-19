import numpy as np
import pandas as pd
from loguru import logger

class TrendStateDetector:
    def __init__(self, window=60):
        self.window = window
        self.last_regime = None  # 记录上一次的市场状态
        self.regime_change_date = None  # 记录状态变化的日期
        
    def detect(self, price_series, volume_series, rsi_series, current_date=None):
        """简化的市场状态检测，重点识别下跌趋势"""
        # 价格通道突破（3个月窗口）
        ma60 = pd.Series(price_series).rolling(60).mean()
        price_ratio = (price_series - ma60) / ma60
        
        # 成交量异动检测
        vol_mean = pd.Series(volume_series).rolling(20).mean()
        vol_std = pd.Series(volume_series).rolling(20).std()
        vol_zscore = (volume_series - vol_mean) / vol_std
        
        # 周线趋势检测 - 使用多重时间框架确认
        current_regime = 'normal'  # 默认为正常市场
        
        # 检查1: 短期均线方向 (20日 ~ 一个月交易日)
        if len(price_series) >= 20:
            ma20 = pd.Series(price_series).rolling(20).mean()
            if len(ma20) >= 10 and not pd.isna(ma20.iloc[-10]) and not pd.isna(ma20.iloc[-1]):
                # 如果最近10天的移动平均线向下，判定为下跌趋势的一个条件
                if ma20.iloc[-1] < ma20.iloc[-10]:
                    downtrend_signals = 1
                else:
                    downtrend_signals = 0
                    
                # 检查2: 价格低于长期均线
                if not pd.isna(ma60.iloc[-1]) and price_series[-1] < ma60.iloc[-1]:
                    downtrend_signals += 1
                
                # 检查3: 长期均线向下
                if len(ma60) >= 10 and not pd.isna(ma60.iloc[-10]) and not pd.isna(ma60.iloc[-1]):
                    if ma60.iloc[-1] < ma60.iloc[-10]:
                        downtrend_signals += 1
                
                # 如果至少满足2个条件，判定为下跌趋势
                if downtrend_signals >= 2:
                    current_regime = 'downtrend'
        
        # 检测趋势变化并记录
        if self.last_regime != current_regime:
            self.regime_change_date = current_date
            # 记录趋势变化
            if current_date is not None and self.last_regime is not None:
                logger.info(f"市场趋势变化: {self.last_regime} -> {current_regime}, 日期: {current_date}")
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

# 信号生成参数
SENTIMENT_THRESHOLDS = {
    'core': 2.5,      # 核心信号阈值
    'secondary': 20.0,  # 次级信号阈值
    'light': 22.0     # 轻仓信号阈值
}

POSITION_WEIGHTS = {
    'core': 0.95,      # 核心信号仓位
    'secondary': 0.9,  # 次级信号仓位
    'light': 0.8      # 轻仓信号仓位
}

def generate_signals(sentiment_score, regime, volatility):
    """简化的信号生成，只区分下跌趋势和正常趋势"""
    signals = []
    
    # 下跌趋势不开仓
    if regime == 'downtrend':
        if sentiment_score < SENTIMENT_THRESHOLDS['core']:
            signals.append({'type': 'buy', 'weight': POSITION_WEIGHTS['core']})  # 核心信号
            return signals
        else:
            return []
    
    # 其他情况根据情绪分数决定
    if sentiment_score < SENTIMENT_THRESHOLDS['core']:
        signals.append({'type': 'buy', 'weight': POSITION_WEIGHTS['core']})  # 核心信号
    elif SENTIMENT_THRESHOLDS['core'] <= sentiment_score < SENTIMENT_THRESHOLDS['secondary']:
        signals.append({'type': 'buy', 'weight': POSITION_WEIGHTS['secondary']})  # 次级信号
    elif SENTIMENT_THRESHOLDS['secondary'] <= sentiment_score < SENTIMENT_THRESHOLDS['light']:
        signals.append({'type': 'buy', 'weight': POSITION_WEIGHTS['light']})  # 轻仓信号
    
    return signals