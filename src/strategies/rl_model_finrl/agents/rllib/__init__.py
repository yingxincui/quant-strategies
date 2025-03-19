"""
RLlib智能体模块

这个模块提供了基于Ray RLlib框架的强化学习智能体实现，支持:
- PPO (Proximal Policy Optimization)

主要组件:
- RLlibPPOAgent: 基于RLlib的PPO算法实现

使用示例:
```python
from src.strategies.rl_model_finrl.agents.rllib.ppo_agent import RLlibPPOAgent
from src.strategies.rl_model_finrl.meta.env_stock_trading.etf_trading_env import ETFTradingEnv

# 创建环境
env = ETFTradingEnv(...)

# 初始化PPO智能体
agent = RLlibPPOAgent(env=env)

# 训练模型
agent.learn(total_timesteps=100000)

# 保存模型
agent.save("models/ppo_rllib")

# 加载模型
agent.load("models/ppo_rllib")

# 测试模型
asset_memory, action_memory = agent.test(test_env)
```
"""

from src.strategies.rl_model_finrl.agents.rllib.ppo_agent import RLlibPPOAgent

__all__ = [
    'RLlibPPOAgent'
] 