import pandas as pd
import numpy as np
import talib
from typing import Dict, List, Tuple, Union, Optional

class FeatureEngineer:
    """
    金融特征工程类
    
    用于计算和添加金融技术指标和特征
    """
    
    def __init__(self, 
                 use_technical_indicators: bool = True,
                 use_vix: bool = False,
                 use_turbulence: bool = False,
                 use_sentiment: bool = False):
        """
        初始化特征工程器
        
        参数:
            use_technical_indicators: 是否使用技术指标
            use_vix: 是否使用波动率指数
            use_turbulence: 是否使用市场波动指标
            use_sentiment: 是否使用情绪指标
        """
        self.use_technical_indicators = use_technical_indicators
        self.use_vix = use_vix
        self.use_turbulence = use_turbulence
        self.use_sentiment = use_sentiment
    
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        预处理数据并添加特征
        
        参数:
            df: 原始数据DataFrame
            
        返回:
            处理后的DataFrame
        """
        data = df.copy()
        
        # 添加技术指标
        if self.use_technical_indicators:
            data = self.add_technical_indicators(data)
        
        # 添加波动率指数
        if self.use_vix:
            data = self.add_vix(data)
        
        # 添加市场波动指标
        if self.use_turbulence:
            data = self.add_turbulence(data)
        
        # 添加情绪指标
        if self.use_sentiment:
            data = self.add_sentiment(data)
        
        # 填充NaN值
        data = self.fill_missing_values(data)
        
        return data
    
    def add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        添加技术指标
        
        参数:
            data: 价格数据DataFrame
            
        返回:
            添加技术指标后的DataFrame
        """
        df = data.copy()
        
        # 确保列名标准化
        price_col = 'close' if 'close' in df.columns else 'Close'
        high_col = 'high' if 'high' in df.columns else 'High'
        low_col = 'low' if 'low' in df.columns else 'Low'
        volume_col = 'volume' if 'volume' in df.columns else 'Volume'
        
        # 使用TA-Lib计算技术指标 (如果可用)
        try:
            # MACD
            macd, macd_signal, macd_hist = talib.MACD(df[price_col])
            df['macd'] = macd
            df['macd_signal'] = macd_signal
            df['macd_hist'] = macd_hist
            
            # RSI
            df['rsi_6'] = talib.RSI(df[price_col], timeperiod=6)
            df['rsi_14'] = talib.RSI(df[price_col], timeperiod=14)
            df['rsi_30'] = talib.RSI(df[price_col], timeperiod=30)
            
            # CCI
            df['cci'] = talib.CCI(df[high_col], df[low_col], df[price_col], timeperiod=14)
            
            # ADX
            df['adx'] = talib.ADX(df[high_col], df[low_col], df[price_col], timeperiod=14)
            
            # 布林带
            df['boll_upper'], df['boll_middle'], df['boll_lower'] = talib.BBANDS(
                df[price_col], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
            
            # ATR
            df['atr'] = talib.ATR(df[high_col], df[low_col], df[price_col], timeperiod=14)
            
            # 移动平均线
            df['sma_5'] = talib.SMA(df[price_col], timeperiod=5)
            df['sma_10'] = talib.SMA(df[price_col], timeperiod=10)
            df['sma_20'] = talib.SMA(df[price_col], timeperiod=20)
            df['sma_60'] = talib.SMA(df[price_col], timeperiod=60)
            
            # WILLR
            df['willr'] = talib.WILLR(df[high_col], df[low_col], df[price_col], timeperiod=14)
            
            # ROC
            df['roc'] = talib.ROC(df[price_col], timeperiod=10)
            
            # OBV
            df['obv'] = talib.OBV(df[price_col], df[volume_col])
        
        except (ImportError, AttributeError):
            # 如果TA-Lib不可用，使用Pandas计算
            # MACD
            df['ema12'] = df[price_col].ewm(span=12, adjust=False).mean()
            df['ema26'] = df[price_col].ewm(span=26, adjust=False).mean()
            df['macd'] = df['ema12'] - df['ema26']
            df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
            df['macd_hist'] = df['macd'] - df['macd_signal']
            
            # RSI
            delta = df[price_col].diff()
            gain = delta.copy()
            loss = delta.copy()
            gain[gain < 0] = 0
            loss[loss > 0] = 0
            loss = -loss
            
            # RSI 14
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / avg_loss
            df['rsi_14'] = 100 - (100 / (1 + rs))
            
            # CCI
            df['tp'] = (df[high_col] + df[low_col] + df[price_col]) / 3
            df['tp_ma'] = df['tp'].rolling(window=20).mean()
            mean_dev = df['tp'].rolling(window=20).apply(lambda x: abs(x - x.mean()).mean())
            df['cci'] = (df['tp'] - df['tp_ma']) / (0.015 * mean_dev)
            
            # 布林带
            df['sma_20'] = df[price_col].rolling(window=20).mean()
            df['boll_upper'] = df['sma_20'] + 2 * df[price_col].rolling(window=20).std()
            df['boll_lower'] = df['sma_20'] - 2 * df[price_col].rolling(window=20).std()
            
            # ATR
            df['tr1'] = df[high_col] - df[low_col]
            df['tr2'] = abs(df[high_col] - df[price_col].shift())
            df['tr3'] = abs(df[low_col] - df[price_col].shift())
            df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
            df['atr'] = df['tr'].rolling(window=14).mean()
            
            # 移动平均线
            df['sma_5'] = df[price_col].rolling(window=5).mean()
            df['sma_10'] = df[price_col].rolling(window=10).mean()
            df['sma_60'] = df[price_col].rolling(window=60).mean()
            
            # 清理临时列
            df = df.drop(['tr1', 'tr2', 'tr3', 'tr', 'tp', 'tp_ma', 'ema12', 'ema26'], 
                         axis=1, errors='ignore')
        
        # 涨跌幅
        df['daily_return'] = df[price_col].pct_change()
        
        # 波动率
        df['volatility'] = df['daily_return'].rolling(window=20).std() * np.sqrt(252)
        
        return df
    
    def add_vix(self, data: pd.DataFrame, vix_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        添加VIX波动率指数
        
        参数:
            data: 原始数据DataFrame
            vix_data: VIX数据DataFrame（可选）
            
        返回:
            添加VIX后的DataFrame
        """
        df = data.copy()
        
        if vix_data is not None:
            # 如果提供了VIX数据，合并到主数据中
            vix_data = vix_data.rename(columns={'close': 'vix'})
            df = pd.merge(df, vix_data[['vix']], 
                          left_index=True, right_index=True, 
                          how='left')
        else:
            # 如果没有VIX数据，使用收益率的滚动波动率作为VIX代理
            price_col = 'close' if 'close' in df.columns else 'Close'
            returns = df[price_col].pct_change()
            df['vix'] = returns.rolling(window=20).std() * np.sqrt(252) * 100
        
        return df
    
    def add_turbulence(self, data: pd.DataFrame, window: int = 252) -> pd.DataFrame:
        """
        添加市场波动指标
        
        参数:
            data: 原始数据DataFrame
            window: 计算窗口
            
        返回:
            添加波动指标后的DataFrame
        """
        df = data.copy()
        
        # 计算收益率
        price_col = 'close' if 'close' in df.columns else 'Close'
        df['return'] = df[price_col].pct_change()
        
        # 计算波动指标
        df['turbulence'] = df['return'].rolling(window=window).apply(
            lambda x: np.sum(np.square(x - x.mean())) / len(x)
        )
        
        return df
    
    def add_sentiment(self, data: pd.DataFrame, sentiment_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        添加市场情绪指标
        
        参数:
            data: 原始数据DataFrame
            sentiment_data: 情绪数据DataFrame（可选）
            
        返回:
            添加情绪指标后的DataFrame
        """
        df = data.copy()
        
        if sentiment_data is not None:
            # 如果提供了情绪数据，合并到主数据中
            df = pd.merge(df, sentiment_data, 
                          left_index=True, right_index=True, 
                          how='left')
        else:
            # 如果没有情绪数据，简单地添加一个基于技术指标的情绪代理
            if 'rsi_14' in df.columns:
                # RSI的简单情绪指标: RSI高表示乐观，RSI低表示悲观
                df['sentiment'] = (df['rsi_14'] - 50) / 50  # 归一化到[-1, 1]
            elif 'macd' in df.columns and 'macd_signal' in df.columns:
                # 基于MACD信号的情绪: MACD > 信号线表示乐观
                df['sentiment'] = np.where(df['macd'] > df['macd_signal'], 1, -1)
            else:
                # 如果没有其他指标，使用收益率的动量作为情绪代理
                price_col = 'close' if 'close' in df.columns else 'Close'
                returns = df[price_col].pct_change()
                df['sentiment'] = returns.rolling(window=5).mean() / returns.rolling(window=5).std()
        
        return df
    
    def fill_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        填充缺失值
        
        参数:
            data: 包含缺失值的DataFrame
            
        返回:
            填充缺失值后的DataFrame
        """
        df = data.copy()
        
        # 前向填充（填充交易日中间缺失的数据）
        df = df.fillna(method='ffill')
        
        # 后向填充（处理最早的数据）
        df = df.fillna(method='bfill')
        
        # 对于仍然缺失的值，用0填充
        df = df.fillna(0)
        
        return df 