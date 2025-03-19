"""
数据处理器模块

这个模块提供了处理金融市场数据的各种处理器，支持不同的数据源:
- TushareProcessor: 处理来自Tushare API的数据
- AKShareProcessor: 处理来自AKShare的数据
- DataProcessor: 数据处理器的基类

主要功能:
- 下载和预处理ETF价格数据
- 添加技术指标
- 准备用于强化学习环境的数据

使用示例:
```python
from src.strategies.rl_model_finrl.meta.data_processors import DataProcessor, TushareProcessor

# 使用Tushare处理器
processor = TushareProcessor(token="your_tushare_token")
data = processor.download_data(
    start_date="2018-01-01",
    end_date="2021-12-31",
    ticker_list=["510050"]
)

# 添加技术指标
data_with_indicators = processor.add_technical_indicators(data)

# 准备训练数据
train_data = processor.data_split(data_with_indicators, start_date="2018-01-01", end_date="2020-12-31")
```
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Union, Optional

class DataProcessor(ABC):
    """
    数据处理器基类
    
    定义了数据处理的通用接口，所有特定数据源的处理器都应该继承这个类
    """
    
    @abstractmethod
    def download_data(self, **kwargs) -> pd.DataFrame:
        """
        从数据源下载数据
        
        参数:
            **kwargs: 下载参数，如起止日期、股票代码等
            
        返回:
            下载的数据DataFrame
        """
        pass
    
    @abstractmethod
    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        清洗数据
        
        参数:
            data: 原始数据DataFrame
            
        返回:
            清洗后的数据DataFrame
        """
        pass
    
    def add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        添加技术指标
        
        参数:
            data: 原始价格数据DataFrame
            
        返回:
            添加技术指标后的数据DataFrame
        """
        df = data.copy()
        
        # 确保列名标准化
        price_col = 'close' if 'close' in df.columns else 'Close'
        high_col = 'high' if 'high' in df.columns else 'High'
        low_col = 'low' if 'low' in df.columns else 'Low'
        volume_col = 'volume' if 'volume' in df.columns else 'Volume'
        
        # 计算MACD
        df['ema12'] = df[price_col].ewm(span=12, adjust=False).mean()
        df['ema26'] = df[price_col].ewm(span=26, adjust=False).mean()
        df['macd'] = df['ema12'] - df['ema26']
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # 计算RSI (相对强弱指标)
        delta = df[price_col].diff()
        gain = delta.copy()
        loss = delta.copy()
        gain[gain < 0] = 0
        loss[loss > 0] = 0
        loss = -loss
        
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        
        rs = avg_gain / avg_loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # 计算CCI (商品通道指标)
        df['tp'] = (df[high_col] + df[low_col] + df[price_col]) / 3
        df['tp_ma'] = df['tp'].rolling(window=20).mean()
        mean_dev = df['tp'].rolling(window=20).apply(lambda x: abs(x - x.mean()).mean())
        df['cci'] = (df['tp'] - df['tp_ma']) / (0.015 * mean_dev)
        
        # 计算布林带
        df['sma20'] = df[price_col].rolling(window=20).mean()
        df['bollinger_upper'] = df['sma20'] + 2 * df[price_col].rolling(window=20).std()
        df['bollinger_lower'] = df['sma20'] - 2 * df[price_col].rolling(window=20).std()
        
        # 计算ATR (真实波幅均值)
        df['tr1'] = df[high_col] - df[low_col]
        df['tr2'] = abs(df[high_col] - df[price_col].shift())
        df['tr3'] = abs(df[low_col] - df[price_col].shift())
        df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
        df['atr'] = df['tr'].rolling(window=14).mean()
        
        # 移除临时列
        df = df.drop(['tr1', 'tr2', 'tr3', 'tr', 'tp', 'tp_ma', 'ema12', 'ema26'], axis=1, errors='ignore')
        
        # 使用涨跌幅代替价格
        df['daily_return'] = df[price_col].pct_change()
        
        return df
    
    def data_split(
        self, 
        df: pd.DataFrame, 
        start_date: str, 
        end_date: str
    ) -> pd.DataFrame:
        """
        按日期范围分割数据
        
        参数:
            df: 原始数据DataFrame
            start_date: 开始日期
            end_date: 结束日期
            
        返回:
            指定日期范围的数据
        """
        data = df.copy()
        
        # 确保索引是日期类型
        if not isinstance(data.index, pd.DatetimeIndex):
            if 'date' in data.columns:
                data['date'] = pd.to_datetime(data['date'])
                data = data.set_index('date')
            else:
                raise ValueError("数据必须有日期列或日期索引")
        
        # 转换日期字符串为日期类型
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        
        # 筛选日期范围内的数据
        data = data.loc[start_date:end_date]
        
        return data
    
    def prepare_data_for_training(self, **kwargs) -> Tuple[Dict[str, pd.DataFrame], Optional[pd.DataFrame]]:
        """
        准备用于训练的数据
        
        参数:
            **kwargs: 准备数据的参数
            
        返回:
            (ETF数据字典, 市场指数数据)元组
        """
        pass

from src.strategies.rl_model_finrl.meta.data_processors.tushare_processor import TushareProcessor
from src.strategies.rl_model_finrl.meta.data_processors.akshare_processor import AKShareProcessor

__all__ = [
    'DataProcessor',
    'TushareProcessor',
    'AKShareProcessor'
] 