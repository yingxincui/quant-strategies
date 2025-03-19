import os
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Any, Optional
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.tune.registry import register_env

logger = logging.getLogger(__name__)

class RLlibPPOAgent:
    """
    RLlib PPO 智能体实现
    
    这个类封装了Ray RLlib的PPO算法，用于ETF交易环境
    """
    
    def __init__(
        self,
        env,
        model_name: str = "ppo_rllib",
        tensorboard_log: Optional[str] = None,
        seed: Optional[int] = None,
        verbose: int = 1,
        device: str = "auto",
        **kwargs
    ):
        """
        初始化RLlib PPO智能体
        
        参数:
            env: 训练环境
            model_name: 模型名称
            tensorboard_log: TensorBoard日志目录
            seed: 随机种子
            verbose: 详细程度
            device: 设备选择 ('cpu', 'cuda', 'auto')
            **kwargs: 传递给PPO构造函数的其他参数
        """
        self.env = env
        self.model_name = model_name
        self.tensorboard_log = tensorboard_log
        self.seed = seed
        self.verbose = verbose
        self.device = device
        self.ppo_config = {
            "framework": "torch",
            "num_gpus": 0 if device == "cpu" else 1,
            "seed": seed,
            # PPO特定参数
            "lambda": 0.95,
            "kl_coeff": 0.5,
            "clip_param": 0.2,
            "vf_clip_param": 10.0,
            "entropy_coeff": 0.01,
            "train_batch_size": 5000,
            "sgd_minibatch_size": 500,
            "num_sgd_iter": 10,
            # 通用参数
            "gamma": 0.99,
            "lr": 3e-4,
            "log_level": "WARN",
        }
        
        # 更新配置
        self.ppo_config.update(kwargs)
        
        # 注册环境
        self._register_env()
        
        # 初始化Ray（如果尚未初始化）
        if not ray.is_initialized():
            if self.verbose > 0:
                logger.info("初始化Ray...")
            ray.init(ignore_reinit_error=True, logging_level=logging.ERROR)
        
        # 创建PPO算法实例
        self.model = PPO(
            config=self.ppo_config,
            env=self.env.__class__.__name__,
        )
        
        if self.verbose > 0:
            logger.info(f"RLlib PPO智能体初始化完成: {model_name}")
    
    def _register_env(self):
        """注册环境到Ray"""
        env_name = self.env.__class__.__name__
        
        # 注册环境创建函数
        def env_creator(env_config):
            return self.env
        
        # 注册环境
        register_env(env_name, env_creator)
    
    def learn(
        self,
        total_timesteps: int = 100000,
        callback: Any = None,
        log_interval: int = 4,
        eval_env = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "PPO",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ) -> "RLlibPPOAgent":
        """
        训练模型
        
        参数:
            total_timesteps: 总训练步数
            callback: 回调函数
            log_interval: 日志记录间隔
            eval_env: 评估环境
            eval_freq: 评估频率
            n_eval_episodes: 评估轮数
            tb_log_name: TensorBoard日志名称
            eval_log_path: 评估日志路径
            reset_num_timesteps: 是否重置时间步计数
            
        返回:
            训练后的智能体
        """
        if self.verbose > 0:
            logger.info(f"开始训练模型，总步数: {total_timesteps}")
        
        iterations = total_timesteps // self.ppo_config["train_batch_size"]
        
        for i in range(iterations):
            if self.verbose > 0 and i % log_interval == 0:
                logger.info(f"训练迭代 {i+1}/{iterations}")
            
            # 执行训练
            result = self.model.train()
            
            # 打印训练结果
            if self.verbose > 0 and i % log_interval == 0:
                logger.info(f"  训练回报: {result['episode_reward_mean']:.2f}")
                logger.info(f"  训练长度: {result['episode_len_mean']:.2f}")
            
            # 评估（如果需要）
            if eval_env is not None and eval_freq > 0 and i % eval_freq == 0:
                self.evaluate(eval_env, n_eval_episodes)
        
        return self
    
    def predict(
        self,
        observation: np.ndarray,
        state: Optional[Tuple] = None,
        deterministic: bool = True,
    ) -> Tuple[np.ndarray, Optional[Tuple]]:
        """
        预测动作
        
        参数:
            observation: 当前观察
            state: RNN状态（如果适用）
            deterministic: 是否确定性预测
            
        返回:
            (动作, 状态)元组
        """
        action = self.model.compute_single_action(
            observation,
            explore=not deterministic
        )
        return action, state
    
    def save(self, path: str) -> None:
        """
        保存模型
        
        参数:
            path: 保存路径
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        checkpoint_path = self.model.save(path)
        if self.verbose > 0:
            logger.info(f"模型已保存到: {checkpoint_path}")
    
    def load(self, path: str) -> None:
        """
        加载模型
        
        参数:
            path: 模型路径
        """
        self.model.restore(path)
        if self.verbose > 0:
            logger.info(f"已从{path}加载模型")
    
    def evaluate(
        self,
        eval_env,
        n_eval_episodes: int = 10,
        deterministic: bool = True,
        render: bool = False,
    ) -> Tuple[float, float]:
        """
        评估模型
        
        参数:
            eval_env: 评估环境
            n_eval_episodes: 评估轮数
            deterministic: 是否确定性预测
            render: 是否渲染环境
            
        返回:
            (平均奖励, 标准差)元组
        """
        if self.verbose > 0:
            logger.info(f"开始评估模型，轮数: {n_eval_episodes}")
        
        episode_rewards = []
        episode_lengths = []
        
        for i in range(n_eval_episodes):
            obs = eval_env.reset()
            done = False
            episode_reward = 0.0
            episode_length = 0
            
            while not done:
                action, _ = self.predict(obs, deterministic=deterministic)
                obs, reward, done, info = eval_env.step(action)
                
                episode_reward += reward
                episode_length += 1
                
                if render:
                    eval_env.render()
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
        
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        
        if self.verbose > 0:
            logger.info(f"评估结果 - 平均奖励: {mean_reward:.2f} +/- {std_reward:.2f}")
        
        return mean_reward, std_reward
    
    def test(
        self,
        test_env,
        num_episodes: int = 1,
        render: bool = False
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        测试模型并返回资产和动作记录
        
        参数:
            test_env: 测试环境
            num_episodes: 测试轮数
            render: 是否渲染环境
            
        返回:
            (资产记录, 动作记录)元组
        """
        if self.verbose > 0:
            logger.info(f"开始测试模型，轮数: {num_episodes}")
        
        # 重置环境
        state = test_env.reset()
        
        # 初始化记录
        episode_rewards = []
        
        # 决策步骤
        done = False
        episode_reward = 0.0
        
        while not done:
            action, _ = self.predict(state, deterministic=True)
            next_state, reward, done, info = test_env.step(action)
            
            state = next_state
            episode_reward += reward
            
            if render:
                test_env.render()
        
        if self.verbose > 0:
            logger.info(f"测试完成，总奖励: {episode_reward:.2f}")
        
        # 获取回测数据
        asset_memory = test_env.save_asset_memory()
        action_memory = test_env.save_action_memory()
        
        return asset_memory, action_memory 