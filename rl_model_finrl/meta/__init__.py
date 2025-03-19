"""
FinRL-Meta模块

此模块包含FinRL框架的基础组件，包括：
- data_processors: 数据处理器，用于获取和处理市场数据
- preprocessor: 数据预处理，用于生成技术指标和特征

FinRL-Meta提供了金融强化学习的基础设施，支持：
- 多种数据源的集成和处理
- 标准化的交易环境接口
- 丰富的市场特征和状态表示
"""

from src.strategies.rl_model_finrl.meta.data_processors import TushareProcessor, AKShareProcessor
from src.strategies.rl_model_finrl.meta.preprocessor import FeatureEngineer, DataNormalizer

__all__ = [
    'TushareProcessor',
    'AKShareProcessor',
    'FeatureEngineer',
    'DataNormalizer'
] 