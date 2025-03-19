import os
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Tuple, Type, Union
import torch
import torch.nn as nn
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.logger import configure
import logging
import matplotlib.pyplot as plt

from src.strategies.rl_model_finrl.config import (
    GAMMA,
    LEARNING_RATE,
    BATCH_SIZE,
    REPLAY_BUFFER_SIZE,
    NUM_EPISODES,
    TENSORBOARD_PATH
)

class TensorboardCallback(BaseCallback):
    """用于记录训练过程中的指标的回调函数"""
    
    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)
        self.training_env = None
        
    def _on_training_start(self) -> None:
        """训练开始时设置训练环境引用"""
        self.training_env = self.model.get_env()
        self._log_freq = 1  # 记录频率 = 1个回合
        
    def _on_step(self) -> bool:
        """每步执行，并记录指标"""
        if self.n_calls % self._log_freq == 0:
            # 从训练环境中获取信息
            info = self.training_env.buf_infos[0]
            portfolio_value = info.get('portfolio_value', 0)
            step_return = info.get('step_return', 0)
            
            # 记录到TensorBoard
            self.logger.record('portfolio_value', portfolio_value)
            self.logger.record('step_return', step_return)
        
        return True

class DQNAgent:
    """使用 stable-baselines3 的DQN智能体"""
    
    def __init__(
        self,
        env,
        model_name: str = "dqn_etf_trading",
        policy_type: str = "MlpPolicy",
        learning_rate: float = LEARNING_RATE,
        gamma: float = GAMMA,
        batch_size: int = BATCH_SIZE,
        buffer_size: int = REPLAY_BUFFER_SIZE,
        exploration_fraction: float = 0.2,
        exploration_initial_eps: float = 1.0,
        exploration_final_eps: float = 0.1,
        verbose: int = 1,
        tensorboard_log: str = TENSORBOARD_PATH,
        device: str = "auto"
    ):
        """
        初始化DQN智能体
        
        参数:
            env: 交易环境
            model_name: 模型名称
            policy_type: 策略类型，默认MlpPolicy
            learning_rate: 学习率
            gamma: 折扣因子
            batch_size: 批处理大小
            buffer_size: 经验回放缓冲区大小
            exploration_fraction: 探索阶段占总训练步数的比例
            exploration_initial_eps: 初始探索率
            exploration_final_eps: 最终探索率
            verbose: 日志详细程度
            tensorboard_log: TensorBoard日志目录
            device: 运行设备，"auto"会自动选择GPU或CPU
        """
        self.env = env
        self.model_name = model_name
        
        # 设置日志
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # 检查环境是否被向量化
        if not isinstance(env, DummyVecEnv):
            self.env = DummyVecEnv([lambda: env])
        
        # 创建日志目录
        if not os.path.exists(tensorboard_log):
            os.makedirs(tensorboard_log)
        
        # 设置TensorBoard日志
        log_dir = os.path.join(tensorboard_log, model_name)
        self.logger.info(f"TensorBoard日志目录: {log_dir}")
        
        # 创建DQN模型
        self.model = DQN(
            policy=policy_type,
            env=self.env,
            learning_rate=learning_rate,
            gamma=gamma,
            batch_size=batch_size,
            buffer_size=buffer_size,
            exploration_fraction=exploration_fraction,
            exploration_initial_eps=exploration_initial_eps,
            exploration_final_eps=exploration_final_eps,
            tensorboard_log=log_dir,
            verbose=verbose,
            device=device
        )
        
        self.logger.info(f"初始化DQN智能体，模型名称: {model_name}")
    
    def train(
        self,
        total_timesteps: int = NUM_EPISODES * 100,  # 假设每个回合约100步
        tb_log_name: str = "dqn_training",
        eval_freq: int = 10000,
        n_eval_episodes: int = 5,
        eval_env = None
    ):
        """
        训练DQN智能体
        
        参数:
            total_timesteps: 总训练步数
            tb_log_name: TensorBoard日志名称
            eval_freq: 评估频率
            n_eval_episodes: 评估回合数
            eval_env: 评估环境，如果None则使用训练环境
            
        返回:
            训练后的模型
        """
        # 设置评估环境
        if eval_env is None:
            eval_env = self.env
        
        # 创建TensorBoard回调
        callback = TensorboardCallback()
        
        # 开始训练
        self.logger.info(f"开始训练DQN模型，总步数: {total_timesteps}")
        self.model.learn(
            total_timesteps=total_timesteps,
            tb_log_name=tb_log_name,
            callback=callback
        )
        
        # 保存模型
        model_path = os.path.join("models", f"{self.model_name}.zip")
        self.model.save(model_path)
        self.logger.info(f"模型已保存到 {model_path}")
        
        return self.model
    
    def predict(self, observation, state=None, deterministic=True):
        """
        使用模型进行预测
        
        参数:
            observation: 观察状态
            state: 隐藏状态（如适用）
            deterministic: 是否确定性预测
            
        返回:
            预测的动作
        """
        return self.model.predict(observation, state, deterministic)
    
    def load(self, path):
        """
        加载训练好的模型
        
        参数:
            path: 模型路径
            
        返回:
            加载的模型
        """
        self.model = DQN.load(path, env=self.env)
        self.logger.info(f"模型已加载: {path}")
        return self.model
    
    def save(self, path):
        """
        保存当前模型
        
        参数:
            path: 保存路径
        """
        self.model.save(path)
        self.logger.info(f"模型已保存: {path}")
    
    def test(self, test_env, num_episodes=1, render=False):
        """
        在测试环境中评估模型
        
        参数:
            test_env: 测试环境
            num_episodes: 测试回合数
            render: 是否渲染环境
            
        返回:
            资产记忆和交易记忆
        """
        self.logger.info(f"开始测试模型，回合数: {num_episodes}")
        
        # 如果测试环境不是向量化的，则向量化
        if not isinstance(test_env, DummyVecEnv):
            test_env = DummyVecEnv([lambda: test_env])
        
        # 初始化统计信息
        total_rewards = []
        portfolio_values = []
        all_trades = []
        
        # 多次测试
        for episode in range(num_episodes):
            # 初始化环境
            obs = test_env.reset()
            done = False
            total_reward = 0
            step = 0
            
            while not done:
                # 使用模型选择动作
                action, _states = self.model.predict(obs, deterministic=True)
                
                # 执行动作
                obs, reward, done, info = test_env.step(action)
                
                # 更新统计信息
                total_reward += reward[0]
                step += 1
                
                # 收集交易信息
                if 'trades' in info[0]:
                    all_trades.extend(info[0]['trades'])
                
                # 收集投资组合价值
                if 'portfolio_value' in info[0]:
                    portfolio_values.append(info[0]['portfolio_value'])
                
                # 如果需要，渲染环境
                if render:
                    test_env.render()
            
            # 记录回合奖励
            total_rewards.append(total_reward)
            self.logger.info(f"回合 {episode+1}/{num_episodes}, 总奖励: {total_reward:.4f}")
        
        # 获取资产记忆
        asset_memory = test_env.envs[0].save_asset_memory()
        
        # 获取交易记忆
        action_memory = test_env.envs[0].save_action_memory()
        
        # 绘制测试结果
        if render:
            self._plot_test_results(asset_memory, action_memory)
        
        # 获取最终统计信息
        stats = test_env.envs[0].get_final_stats()
        self.logger.info(f"测试结果统计: {stats}")
        
        return asset_memory, action_memory
    
    def _plot_test_results(self, asset_memory, action_memory):
        """
        绘制测试结果
        
        参数:
            asset_memory: 资产记忆DataFrame
            action_memory: 交易记忆DataFrame
        """
        # 绘制投资组合价值
        plt.figure(figsize=(12, 6))
        plt.plot(asset_memory.index, asset_memory['portfolio_value'], label='投资组合价值')
        
        # 添加买入卖出标记
        if not action_memory.empty:
            for idx, row in action_memory.iterrows():
                date = row['date']
                action = row['action']
                if date in asset_memory.index:
                    portfolio_value = asset_memory.loc[date, 'portfolio_value']
                    if action == 'buy':
                        plt.scatter(date, portfolio_value, marker='^', color='green', s=100)
                    elif action == 'sell':
                        plt.scatter(date, portfolio_value, marker='v', color='red', s=100)
        
        plt.title('ETF交易投资组合价值')
        plt.xlabel('日期')
        plt.ylabel('价值')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"results/{self.model_name}_portfolio_value.png")
        plt.show()
        
        # 绘制收益率
        returns = asset_memory['portfolio_value'].pct_change().dropna()
        plt.figure(figsize=(12, 6))
        plt.plot(returns.index, returns, label='每日收益率')
        plt.title('ETF交易每日收益率')
        plt.xlabel('日期')
        plt.ylabel('收益率')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"results/{self.model_name}_daily_returns.png")
        plt.show() 