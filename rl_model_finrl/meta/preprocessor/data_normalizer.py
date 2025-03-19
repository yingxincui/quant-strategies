import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Union, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

class DataNormalizer:
    """
    数据归一化处理器
    
    用于金融数据的归一化和标准化，以便更好地用于机器学习模型
    """
    
    def __init__(self, 
                method: str = 'standard',  # 'standard', 'minmax', 'robust'
                feature_range: Tuple[float, float] = (0, 1),
                ):
        """
        初始化归一化处理器
        
        参数:
            method: 归一化方法: 'standard'(标准化), 'minmax'(最小最大归一化), 'robust'(稳健归一化)
            feature_range: 特征范围 (针对minmax方法)
        """
        self.method = method
        self.feature_range = feature_range
        self.scalers = {}
        self.feature_columns = None
    
    def fit(self, df: pd.DataFrame, feature_columns: Optional[List[str]] = None) -> 'DataNormalizer':
        """
        拟合归一化器
        
        参数:
            df: 数据DataFrame
            feature_columns: 要归一化的特征列名列表，如果为None，将使用所有数值型列
            
        返回:
            归一化处理器实例
        """
        data = df.copy()
        
        # 如果未指定特征列，使用所有数值列
        if feature_columns is None:
            self.feature_columns = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
        else:
            self.feature_columns = feature_columns
        
        # 拟合每个特征的归一化器
        for col in self.feature_columns:
            if col in data.columns:
                # 创建适当的缩放器
                if self.method == 'standard':
                    scaler = StandardScaler()
                elif self.method == 'minmax':
                    scaler = MinMaxScaler(feature_range=self.feature_range)
                elif self.method == 'robust':
                    scaler = RobustScaler()
                else:
                    raise ValueError(f"不支持的归一化方法: {self.method}")
                
                # 拟合缩放器
                values = data[col].values.reshape(-1, 1)
                scaler.fit(values)
                self.scalers[col] = scaler
        
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        转换数据
        
        参数:
            df: 要转换的数据DataFrame
            
        返回:
            归一化后的DataFrame
        """
        if not self.scalers:
            raise ValueError("请先调用fit方法")
        
        data = df.copy()
        
        # 转换每个特征
        for col, scaler in self.scalers.items():
            if col in data.columns:
                values = data[col].values.reshape(-1, 1)
                data[col] = scaler.transform(values)
        
        return data
    
    def fit_transform(self, df: pd.DataFrame, feature_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        拟合并转换数据
        
        参数:
            df: 数据DataFrame
            feature_columns: 要归一化的特征列名列表
            
        返回:
            归一化后的DataFrame
        """
        self.fit(df, feature_columns)
        return self.transform(df)
    
    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        反向转换数据(恢复原始比例)
        
        参数:
            df: 归一化后的数据DataFrame
            
        返回:
            原始比例的DataFrame
        """
        if not self.scalers:
            raise ValueError("请先调用fit方法")
        
        data = df.copy()
        
        # 反向转换每个特征
        for col, scaler in self.scalers.items():
            if col in data.columns:
                values = data[col].values.reshape(-1, 1)
                data[col] = scaler.inverse_transform(values)
        
        return data
    
    def save(self, path: str) -> None:
        """
        保存归一化器状态
        
        参数:
            path: 保存路径
        """
        import joblib
        
        state = {
            'method': self.method,
            'feature_range': self.feature_range,
            'scalers': self.scalers,
            'feature_columns': self.feature_columns
        }
        
        joblib.dump(state, path)
    
    @classmethod
    def load(cls, path: str) -> 'DataNormalizer':
        """
        从文件加载归一化器
        
        参数:
            path: 文件路径
            
        返回:
            加载的归一化器实例
        """
        import joblib
        
        state = joblib.load(path)
        
        normalizer = cls(
            method=state['method'],
            feature_range=state['feature_range']
        )
        
        normalizer.scalers = state['scalers']
        normalizer.feature_columns = state['feature_columns']
        
        return normalizer
    
    def normalize_price_data(self, df: pd.DataFrame, price_columns: List[str]) -> pd.DataFrame:
        """
        归一化价格数据
        
        参数:
            df: 价格数据DataFrame
            price_columns: 价格列名列表
            
        返回:
            归一化后的DataFrame
        """
        data = df.copy()
        
        # 当价格数据用于机器学习时，通常更好的做法是转换为收益率或相对变化
        for col in price_columns:
            if col in data.columns:
                # 计算收益率
                data[f'{col}_return'] = data[col].pct_change()
                
                # 计算相对于初始价格的变化
                data[f'{col}_rel'] = data[col] / data[col].iloc[0] - 1
                
                # 对原始价格列进行归一化
                if self.method == 'standard':
                    scaler = StandardScaler()
                elif self.method == 'minmax':
                    scaler = MinMaxScaler(feature_range=self.feature_range)
                elif self.method == 'robust':
                    scaler = RobustScaler()
                
                values = data[col].values.reshape(-1, 1)
                data[f'{col}_norm'] = scaler.fit_transform(values)
                
                # 保存缩放器以便将来反归一化
                self.scalers[col] = scaler
        
        return data 