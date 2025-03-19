"""
基于ElegantRL的强化学习智能体

此模块包含使用ElegantRL库实现的强化学习智能体：
- PPOAgent: 基于PPO算法的交易智能体

ElegantRL提供了高效、轻量级的深度强化学习实现，尤其适合金融应用场景。
其优势包括：
- 更高效的训练速度和采样效率
- 灵活的网络架构设计
- 对多进程训练的良好支持
"""

from src.strategies.rl_model_finrl.agents.elegantrl.ppo_agent import PPOAgent

__all__ = [
    'PPOAgent',
] 