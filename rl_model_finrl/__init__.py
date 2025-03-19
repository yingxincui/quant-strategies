"""
RL模型基于FinRL框架优化的ETF交易模块

此模块包含使用强化学习进行ETF交易的完整框架，基于FinRL架构设计。
模块使用了三层架构：
1. 数据层（meta/data_processors）: 负责从Tushare和AKShare获取A股ETF数据
2. 环境层（meta/env_stock_trading）: 提供了ETF交易的交互环境
3. 智能体层（agents）: 实现了各种RL算法，包括DQN等

主要功能：
- 使用Tushare和AKShare获取A股ETF数据
- 构建多ETF交易环境
- 实现DQN等强化学习算法
- 提供训练和回测功能
"""

# 版本信息
__version__ = "0.1.0"

# 核心组件
from src.strategies.rl_model_finrl.applications.stock_trading.etf_env import ETFTradingEnv
from src.strategies.rl_model_finrl.agents.stablebaseline3.dqn_agent import DQNAgent

# 导出接口
__all__ = [
    'ETFTradingEnv',
    'DQNAgent'
] 