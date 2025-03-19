import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict, List, Tuple, Any, Optional
import time
import logging
import matplotlib.pyplot as plt
from stable_baselines3.common.vec_env import DummyVecEnv

from src.strategies.rl_model_finrl.config import (
    GAMMA,
    LEARNING_RATE,
    BATCH_SIZE,
    NUM_EPISODES,
    TENSORBOARD_PATH
)

class ActorNetwork(nn.Module):
    """Actor网络，用于生成策略"""
    
    def __init__(
        self, 
        state_dim: int, 
        action_dim: int,
        hidden_dim: int = 128
    ):
        """
        初始化Actor网络
        
        参数:
            state_dim: 状态空间维度
            action_dim: 动作空间维度
            hidden_dim: 隐藏层神经元数量
        """
        super(ActorNetwork, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, action_dim)
        self.fc_std = nn.Linear(hidden_dim, action_dim)
        
        # 初始化最后一层权重，使得初始策略方差较小
        nn.init.xavier_uniform_(self.fc_mean.weight)
        nn.init.constant_(self.fc_mean.bias, 0.0)
        nn.init.xavier_uniform_(self.fc_std.weight)
        nn.init.constant_(self.fc_std.bias, -0.5)  # 初始方差小一些
        
        self.action_dim = action_dim
        self.action_distribution = None
    
    def forward(self, state):
        """
        前向传播计算策略分布
        
        参数:
            state: 输入状态
            
        返回:
            动作概率分布
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        
        # 计算离散动作的概率
        action_probs = F.softmax(self.fc_mean(x), dim=-1)
        
        return action_probs
    
    def get_action(self, state, deterministic=False):
        """
        根据状态选择动作
        
        参数:
            state: 输入状态
            deterministic: 是否使用确定性策略
            
        返回:
            选择的动作和对应的对数概率
        """
        action_probs = self.forward(state)
        
        if deterministic:
            # 确定性策略：选择概率最高的动作
            action = torch.argmax(action_probs, dim=-1)
        else:
            # 随机采样
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample()
            
        log_prob = torch.log(action_probs + 1e-10).gather(1, action.unsqueeze(-1)).squeeze(-1)
        
        return action, log_prob
    
    def evaluate(self, state, action):
        """
        评估已选择动作的对数概率和熵
        
        参数:
            state: 输入状态
            action: 已选择的动作
            
        返回:
            (log_probs, entropy) 元组
        """
        action_probs = self.forward(state)
        action_dist = torch.distributions.Categorical(action_probs)
        
        # 动作的对数概率
        log_probs = action_dist.log_prob(action)
        
        # 策略熵
        entropy = action_dist.entropy()
        
        return log_probs, entropy


class CriticNetwork(nn.Module):
    """Critic网络，用于估计状态价值"""
    
    def __init__(
        self, 
        state_dim: int,
        hidden_dim: int = 128
    ):
        """
        初始化Critic网络
        
        参数:
            state_dim: 状态空间维度
            hidden_dim: 隐藏层神经元数量
        """
        super(CriticNetwork, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_value = nn.Linear(hidden_dim, 1)
        
        # 初始化最后一层权重
        nn.init.xavier_uniform_(self.fc_value.weight)
        nn.init.constant_(self.fc_value.bias, 0.0)
    
    def forward(self, state):
        """
        前向传播计算状态价值
        
        参数:
            state: 输入状态
            
        返回:
            状态价值
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        value = self.fc_value(x)
        
        return value


class PPOMemory:
    """PPO算法的经验回放缓冲区"""
    
    def __init__(self, batch_size: int = 64):
        """
        初始化回放缓冲区
        
        参数:
            batch_size: 批处理大小
        """
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []
        
        self.batch_size = batch_size
    
    def store(self, state, action, probs, vals, reward, done):
        """
        存储经验元组
        
        参数:
            state: 状态
            action: 动作
            probs: 动作的对数概率
            vals: 状态价值
            reward: 奖励
            done: 是否结束
        """
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)
    
    def clear(self):
        """清空缓冲区"""
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []
    
    def generate_batches(self):
        """
        生成批次数据
        
        返回:
            包含索引和相应数据的字典
        """
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]
        
        return {
            'states': np.array(self.states),
            'actions': np.array(self.actions),
            'probs': np.array(self.probs),
            'vals': np.array(self.vals),
            'rewards': np.array(self.rewards),
            'dones': np.array(self.dones),
            'batches': batches
        }


class PPOAgent:
    """基于ElegantRL的PPO (Proximal Policy Optimization) 强化学习智能体"""
    
    def __init__(
        self,
        env,
        model_name: str = "ppo_etf_trading",
        state_dim: int = None,
        action_dim: int = None,
        learning_rate: float = LEARNING_RATE,
        gamma: float = GAMMA,
        gae_lambda: float = 0.95,
        policy_clip: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        batch_size: int = BATCH_SIZE,
        n_epochs: int = 10,
        hidden_dim: int = 128,
        device: str = "auto",
        tensorboard_log: str = TENSORBOARD_PATH
    ):
        """
        初始化PPO智能体
        
        参数:
            env: 交易环境
            model_name: 模型名称
            state_dim: 状态空间维度，如果为None则从环境自动获取
            action_dim: 动作空间维度，如果为None则从环境自动获取
            learning_rate: 学习率
            gamma: 折扣因子
            gae_lambda: GAE (Generalized Advantage Estimation) lambda参数
            policy_clip: 策略裁剪参数
            value_coef: 价值损失系数
            entropy_coef: 熵正则化系数
            batch_size: 批处理大小
            n_epochs: 每次更新的训练轮数
            hidden_dim: 网络隐藏层维度
            device: 计算设备
            tensorboard_log: TensorBoard日志目录
        """
        # 设置日志
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # 存储参数
        self.model_name = model_name
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.policy_clip = policy_clip
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        
        # 确保环境被包装
        if not isinstance(env, DummyVecEnv):
            self.env = DummyVecEnv([lambda: env])
        else:
            self.env = env
        
        # 设置设备
        if device == "auto":
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        self.logger.info(f"使用设备: {self.device}")
        
        # 获取环境维度
        if state_dim is None:
            self.state_dim = env.observation_space.shape[0]
        else:
            self.state_dim = state_dim
            
        if action_dim is None:
            if isinstance(env.action_space.n, int):
                self.action_dim = env.action_space.n
            else:
                self.action_dim = env.action_space.shape[0]
        else:
            self.action_dim = action_dim
        
        # 创建actor和critic网络
        self.actor = ActorNetwork(self.state_dim, self.action_dim, hidden_dim).to(self.device)
        self.critic = CriticNetwork(self.state_dim, hidden_dim).to(self.device)
        
        # 创建优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)
        
        # 创建经验回放缓冲区
        self.memory = PPOMemory(batch_size)
        
        # 创建TensorBoard日志目录
        if not os.path.exists(tensorboard_log):
            os.makedirs(tensorboard_log)
            
        self.tensorboard_log = os.path.join(tensorboard_log, model_name)
        
        # 训练统计信息
        self.training_step = 0
        self.best_reward = float('-inf')
        
        self.logger.info(f"初始化PPO智能体，模型名称: {model_name}")
    
    def train(
        self, 
        total_timesteps: int = NUM_EPISODES * 100,
        eval_freq: int = 10000,
        n_eval_episodes: int = 5,
        log_interval: int = 1000,
    ):
        """
        训练PPO智能体
        
        参数:
            total_timesteps: 总训练步数
            eval_freq: 评估频率
            n_eval_episodes: 每次评估的回合数
            log_interval: 日志输出间隔
            
        返回:
            训练后的模型
        """
        self.logger.info(f"开始训练PPO模型，总步数: {total_timesteps}")
        
        # 当前时间步
        time_step = 0
        
        # 训练循环
        while time_step < total_timesteps:
            # 重置环境
            state = self.env.reset()
            done = False
            score = 0
            
            # 每个回合的循环
            while not done and time_step < total_timesteps:
                # 将状态转换为张量
                state_tensor = torch.FloatTensor(state).to(self.device)
                
                # 获取动作、动作概率和状态价值
                with torch.no_grad():
                    action, prob = self.actor.get_action(state_tensor)
                    value = self.critic(state_tensor)
                
                # 执行动作
                action_cpu = action.cpu().numpy()[0]
                next_state, reward, done_array, info = self.env.step([action_cpu])
                done = done_array[0]
                reward = reward[0]
                
                # 将数据存入缓冲区
                self.memory.store(
                    state[0], 
                    action_cpu, 
                    prob.cpu().numpy()[0], 
                    value.cpu().numpy()[0], 
                    reward, 
                    done
                )
                
                # 记录得分
                score += reward
                
                # 更新状态
                state = next_state
                time_step += 1
                
                # 判断是否需要更新策略
                if time_step % self.batch_size == 0:
                    self._update_policy()
                
                # 日志输出
                if time_step % log_interval == 0:
                    self.logger.info(f"Timestep: {time_step}/{total_timesteps}, Score: {score:.2f}")
                
                # 评估模型
                if time_step % eval_freq == 0:
                    mean_reward = self._evaluate(n_eval_episodes)
                    self.logger.info(f"Evaluation at timestep {time_step}: Mean Reward: {mean_reward:.2f}")
                    
                    # 如果性能更好，保存模型
                    if mean_reward > self.best_reward:
                        self.best_reward = mean_reward
                        self.save(os.path.join("models", f"{self.model_name}_best.pt"))
                        self.logger.info(f"保存最佳模型，平均奖励: {mean_reward:.2f}")
            
            # 回合结束，记录得分
            self.logger.info(f"Episode completed, Score: {score:.2f}")
        
        # 训练完成，保存最终模型
        self.save(os.path.join("models", f"{self.model_name}.pt"))
        self.logger.info(f"训练完成，总步数: {time_step}")
        
        return self
    
    def _update_policy(self):
        """更新策略网络和价值网络"""
        # 计算状态价值估计
        values = np.array(self.memory.vals)
        rewards = np.array(self.memory.rewards)
        dones = np.array(self.memory.dones)
        states = np.array(self.memory.states)
        actions = np.array(self.memory.actions)
        old_probs = np.array(self.memory.probs)
        
        # 计算优势估计（使用GAE）
        advantages = np.zeros(len(rewards), dtype=np.float32)
        gae = 0
        
        # 从后向前计算GAE
        for t in reversed(range(len(rewards) - 1)):
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
        
        # 计算回报
        returns = advantages + values
        
        # 转换为张量
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        old_probs = torch.FloatTensor(old_probs).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        
        # 进行多个epoch的训练
        for _ in range(self.n_epochs):
            # 每个batch的训练
            for batch in range(0, len(states), self.batch_size):
                end = min(batch + self.batch_size, len(states))
                
                # 获取当前batch
                batch_states = states[batch:end]
                batch_actions = actions[batch:end]
                batch_old_probs = old_probs[batch:end]
                batch_returns = returns[batch:end]
                batch_advantages = advantages[batch:end]
                
                # 获取新的动作概率和熵
                action_probs = self.actor(batch_states)
                action_dist = torch.distributions.Categorical(action_probs)
                new_probs = action_dist.log_prob(batch_actions)
                entropy = action_dist.entropy().mean()
                
                # 计算新旧概率比率
                prob_ratio = torch.exp(new_probs - batch_old_probs)
                
                # 计算策略目标
                weighted_advantages = batch_advantages * prob_ratio
                clipped_advantages = batch_advantages * torch.clamp(
                    prob_ratio, 1 - self.policy_clip, 1 + self.policy_clip
                )
                actor_loss = -torch.min(weighted_advantages, clipped_advantages).mean()
                
                # 计算价值损失
                critic_value = self.critic(batch_states).squeeze()
                critic_loss = F.mse_loss(critic_value, batch_returns)
                
                # 总损失
                total_loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy
                
                # 更新网络
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                total_loss.backward()
                self.actor_optimizer.step()
                self.critic_optimizer.step()
        
        # 清空缓冲区
        self.memory.clear()
    
    def _evaluate(self, n_episodes: int = 5) -> float:
        """
        评估当前策略
        
        参数:
            n_episodes: 评估回合数
            
        返回:
            平均奖励
        """
        rewards = []
        
        for _ in range(n_episodes):
            state = self.env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                # 确定性选择动作
                action = self.predict(state, deterministic=True)
                
                # 执行动作
                next_state, reward, done_array, _ = self.env.step([action])
                done = done_array[0]
                
                # 累积奖励
                episode_reward += reward[0]
                
                # 更新状态
                state = next_state
            
            rewards.append(episode_reward)
        
        # 计算平均奖励
        mean_reward = np.mean(rewards)
        
        return mean_reward
    
    def predict(
        self, 
        observation,
        state=None,
        deterministic=True
    ):
        """
        使用当前策略预测动作
        
        参数:
            observation: 观测状态
            state: 隐藏状态（如适用）
            deterministic: 是否使用确定性策略
            
        返回:
            预测的动作
        """
        # 转换为张量
        observation_tensor = torch.FloatTensor(observation).to(self.device)
        
        # 使用Actor网络预测动作
        with torch.no_grad():
            action, _ = self.actor.get_action(observation_tensor, deterministic)
        
        # 返回预测的动作
        return action.cpu().numpy()[0]
    
    def save(self, path: str):
        """
        保存模型
        
        参数:
            path: 保存路径
        """
        # 确保目录存在
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # 保存模型和优化器状态
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'model_info': {
                'state_dim': self.state_dim,
                'action_dim': self.action_dim,
                'gamma': self.gamma,
                'best_reward': self.best_reward
            }
        }, path)
        
        self.logger.info(f"模型已保存至: {path}")
    
    def load(self, path: str):
        """
        加载模型
        
        参数:
            path: 模型路径
            
        返回:
            加载的模型
        """
        if not os.path.exists(path):
            self.logger.error(f"模型文件不存在: {path}")
            raise FileNotFoundError(f"模型文件不存在: {path}")
        
        # 加载模型
        checkpoint = torch.load(path, map_location=self.device)
        
        # 更新模型信息
        model_info = checkpoint.get('model_info', {})
        self.state_dim = model_info.get('state_dim', self.state_dim)
        self.action_dim = model_info.get('action_dim', self.action_dim)
        self.gamma = model_info.get('gamma', self.gamma)
        self.best_reward = model_info.get('best_reward', self.best_reward)
        
        # 如果网络尺寸不匹配，重新创建网络
        if hasattr(self, 'actor') and (self.actor.fc1.in_features != self.state_dim or self.actor.fc_mean.out_features != self.action_dim):
            self.actor = ActorNetwork(self.state_dim, self.action_dim).to(self.device)
            self.critic = CriticNetwork(self.state_dim).to(self.device)
            self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.learning_rate)
            self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.learning_rate)
        
        # 加载网络权重
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        
        # 加载优化器状态
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        
        self.logger.info(f"已加载模型: {path}")
        
        return self
    
    def test(self, test_env, num_episodes=1, render=False):
        """
        在测试环境中测试模型
        
        参数:
            test_env: 测试环境
            num_episodes: 测试回合数
            render: 是否渲染环境
            
        返回:
            (asset_memory, action_memory)元组
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
            state = test_env.reset()
            done = False
            total_reward = 0
            step = 0
            
            while not done:
                # 使用模型选择动作
                action = self.predict(state, deterministic=True)
                
                # 执行动作
                next_state, reward, done_array, info = test_env.step([action])
                done = done_array[0]
                
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
                
                # 更新状态
                state = next_state
            
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
        
        plt.title(f'PPO ETF交易策略 - 投资组合价值')
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
        plt.title(f'PPO ETF交易策略 - 每日收益率')
        plt.xlabel('日期')
        plt.ylabel('收益率')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"results/{self.model_name}_daily_returns.png")
        plt.show() 