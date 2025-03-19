import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from typing import List, Dict, Any, Optional, Union, Tuple
import logging

from stable_baselines3 import PPO, A2C, DDPG, TD3, SAC, DQN
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback

from src.strategies.rl_model_finrl.meta.data_processors import DataProcessor
from src.strategies.rl_model_finrl.applications.stock_trading.etf_env import ETFTradingEnv
from src.strategies.rl_model_finrl.agents.stablebaseline3 import DQNAgent
from src.strategies.rl_model_finrl.agents.elegantrl import PPOAgent

from src.strategies.rl_model_finrl.config import (
    INITIAL_AMOUNT,
    TRANSACTION_COST_PCT,
    TURBULENCE_THRESHOLD,
    TRAIN_START_DATE,
    TRAIN_END_DATE,
    TEST_START_DATE,
    TEST_END_DATE,
    TECHNICAL_INDICATORS_LIST,
    TICKER_LIST,
    MODEL_SAVE_PATH,
    NUM_EPISODES,
    TENSORBOARD_LOG_PATH
)


class TensorboardCallback(BaseCallback):
    """
    自定义回调，记录训练过程中的奖励和组合价值
    """
    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)
        self.rewards = []
        
    def _on_step(self) -> bool:
        # 获取最近的奖励
        if len(self.model.env.envs[0].rewards_memory) > 0:
            latest_reward = self.model.env.envs[0].rewards_memory[-1]
            self.rewards.append(latest_reward)
            # 记录到tensorboard
            self.logger.record('train/reward', latest_reward)
            self.logger.record('train/portfolio_value', self.model.env.envs[0].total_asset)
            
        return True


def prepare_etf_data(
    processor: DataProcessor, 
    ticker_list: List[str], 
    start_date: str, 
    end_date: str,
    data_source: str = "tushare"
) -> pd.DataFrame:
    """
    准备ETF交易数据
    
    参数:
        processor: 数据处理器
        ticker_list: ETF代码列表
        start_date: 开始日期
        end_date: 结束日期
        data_source: 数据源
        
    返回:
        处理后的DataFrame
    """
    # 如果没有提供ETF列表，使用默认列表
    if not ticker_list:
        ticker_list = TICKER_LIST
    
    # 获取原始数据
    df = processor.download_data(
        ticker_list=ticker_list,
        start_date=start_date,
        end_date=end_date,
        data_source=data_source
    )
    
    # 处理数据
    df = processor.clean_data(df)
    
    # 添加技术指标
    df = processor.add_technical_indicators(df, TECHNICAL_INDICATORS_LIST)
    
    # 添加波动性指标
    df = processor.add_turbulence(df)
    
    # 填充缺失值
    df = df.fillna(method='ffill').fillna(method='bfill')
    
    # 确保日期是正确的格式
    df.index = pd.to_datetime(df.index)
    
    return df


def run_etf_strategy(
    start_date: str = TRAIN_START_DATE,
    end_date: str = TRAIN_END_DATE,
    ticker_list: List[str] = None,
    data_source: str = "tushare",
    time_interval: str = "1d",
    technical_indicator_list: List[str] = TECHNICAL_INDICATORS_LIST,
    initial_amount: float = INITIAL_AMOUNT,
    transaction_cost_pct: float = TRANSACTION_COST_PCT,
    agent: str = "ppo",
    model_name: str = None,
    turbulence_threshold: float = TURBULENCE_THRESHOLD,
    if_store_model: bool = True,
    num_episodes: int = NUM_EPISODES,
    **kwargs
) -> Any:
    """
    运行ETF交易策略
    
    参数:
        start_date: 训练开始日期
        end_date: 训练结束日期
        ticker_list: ETF代码列表
        data_source: 数据源
        time_interval: 时间间隔
        technical_indicator_list: 技术指标列表
        initial_amount: 初始资金
        transaction_cost_pct: 交易成本百分比
        agent: 智能体类型 (ppo, dqn, a2c等)
        model_name: 模型名称
        turbulence_threshold: 市场波动阈值
        if_store_model: 是否存储模型
        num_episodes: 训练回合数
        **kwargs: 传递给agent的其他参数
        
    返回:
        训练好的模型
    """
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # 如果未指定模型名称，则自动生成
    if model_name is None:
        model_name = f"{agent}_etf_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    logger.info(f"开始运行ETF交易策略, 智能体: {agent}, 模型名称: {model_name}")
    
    # 初始化数据处理器
    processor = DataProcessor(data_source=data_source, time_interval=time_interval)
    
    # 准备数据
    logger.info(f"准备ETF数据: {start_date} 到 {end_date}")
    df = prepare_etf_data(
        processor=processor,
        ticker_list=ticker_list or TICKER_LIST,
        start_date=start_date,
        end_date=end_date,
        data_source=data_source
    )
    
    # 创建ETF交易环境
    stock_dimension = len(df['tic'].unique())
    env = ETFTradingEnv(
        df=df,
        stock_dim=stock_dimension,
        hmax=100,
        initial_amount=initial_amount,
        transaction_cost_pct=transaction_cost_pct,
        reward_scaling=1.0,
        tech_indicator_list=technical_indicator_list,
        turbulence_threshold=turbulence_threshold,
        day_trade=True,
        reward_type='sharpe',
        cash_penalty_proportion=0.1,
    )
    
    # 创建向量化环境
    env_vec = env.get_sb_env()
    
    # 创建智能体
    if agent.lower() == "ppo_elegant":
        # ElegantRL PPO实现
        model = PPOAgent(
            env=env_vec,
            model_name=model_name,
            learning_rate=kwargs.get('learning_rate', 0.0003),
            gamma=kwargs.get('gamma', 0.99),
            tensorboard_log=TENSORBOARD_LOG_PATH
        )
    elif agent.lower() == "dqn":
        # Stable-Baselines3 DQN实现
        model = DQNAgent(
            env=env_vec,
            model_name=model_name,
            learning_rate=kwargs.get('learning_rate', 0.0001),
            gamma=kwargs.get('gamma', 0.99),
            tensorboard_log=TENSORBOARD_LOG_PATH
        )
    elif agent.lower() == "ppo":
        # Stable-Baselines3 PPO实现
        model = PPO(
            "MlpPolicy",
            env_vec,
            verbose=1,
            learning_rate=kwargs.get('learning_rate', 0.0003),
            gamma=kwargs.get('gamma', 0.99),
            tensorboard_log=TENSORBOARD_LOG_PATH
        )
    elif agent.lower() == "a2c":
        # Stable-Baselines3 A2C实现
        model = A2C(
            "MlpPolicy",
            env_vec,
            verbose=1,
            learning_rate=kwargs.get('learning_rate', 0.0007),
            gamma=kwargs.get('gamma', 0.99),
            tensorboard_log=TENSORBOARD_LOG_PATH
        )
    elif agent.lower() == "ddpg":
        # Stable-Baselines3 DDPG实现
        model = DDPG(
            "MlpPolicy",
            env_vec,
            verbose=1,
            learning_rate=kwargs.get('learning_rate', 0.0001),
            gamma=kwargs.get('gamma', 0.99),
            tensorboard_log=TENSORBOARD_LOG_PATH
        )
    elif agent.lower() == "sac":
        # Stable-Baselines3 SAC实现
        model = SAC(
            "MlpPolicy",
            env_vec,
            verbose=1,
            learning_rate=kwargs.get('learning_rate', 0.0003),
            gamma=kwargs.get('gamma', 0.99),
            tensorboard_log=TENSORBOARD_LOG_PATH
        )
    else:
        raise ValueError(f"不支持的智能体类型: {agent}")
    
    # 创建回调
    callback = TensorboardCallback()
    
    # 创建模型保存路径
    if not os.path.exists(MODEL_SAVE_PATH):
        os.makedirs(MODEL_SAVE_PATH)
    
    # 训练模型
    logger.info(f"开始训练模型: {model_name}")
    
    # 根据不同模型类型采用不同的训练方法
    if agent.lower() in ["ppo_elegant"]:
        # ElegantRL训练方法
        model.train(
            total_timesteps=num_episodes * 100,
            eval_freq=1000,
            n_eval_episodes=5,
            log_interval=100
        )
    else:
        # Stable-Baselines3训练方法
        model.learn(
            total_timesteps=num_episodes * 100,
            callback=callback,
            tb_log_name=model_name
        )
    
    # 保存模型
    if if_store_model:
        if agent.lower() in ["ppo_elegant"]:
            # ElegantRL模型保存
            model_path = os.path.join(MODEL_SAVE_PATH, f"{model_name}.pt")
            model.save(model_path)
        else:
            # Stable-Baselines3模型保存
            model_path = os.path.join(MODEL_SAVE_PATH, f"{model_name}.zip")
            model.save(model_path)
        
        logger.info(f"模型已保存至: {model_path}")
    
    # 绘制训练曲线
    if len(callback.rewards) > 0:
        plt.figure(figsize=(12, 6))
        plt.plot(callback.rewards)
        plt.title(f'{agent.upper()} ETF交易策略 - 训练奖励')
        plt.xlabel('步数')
        plt.ylabel('奖励')
        plt.grid(True)
        plt.savefig(f"results/{model_name}_train_rewards.png")
        plt.close()
    
    # 返回训练好的模型
    return model


def load_etf_model(
    model_name: str,
    agent: str = "ppo",
    env = None
) -> Any:
    """
    加载已训练的ETF交易模型
    
    参数:
        model_name: 模型名称
        agent: 智能体类型
        env: 环境实例(可选)
        
    返回:
        加载的模型
    """
    if agent.lower() == "ppo_elegant":
        # ElegantRL模型加载
        model_path = os.path.join(MODEL_SAVE_PATH, f"{model_name}.pt")
        model = PPOAgent(env=env)
        model.load(model_path)
    elif agent.lower() == "dqn":
        # Stable-Baselines3 DQN模型加载
        model_path = os.path.join(MODEL_SAVE_PATH, f"{model_name}.zip")
        model = DQN.load(model_path, env=env)
    elif agent.lower() == "ppo":
        # Stable-Baselines3 PPO模型加载
        model_path = os.path.join(MODEL_SAVE_PATH, f"{model_name}.zip")
        model = PPO.load(model_path, env=env)
    elif agent.lower() == "a2c":
        # Stable-Baselines3 A2C模型加载
        model_path = os.path.join(MODEL_SAVE_PATH, f"{model_name}.zip")
        model = A2C.load(model_path, env=env)
    elif agent.lower() == "ddpg":
        # Stable-Baselines3 DDPG模型加载
        model_path = os.path.join(MODEL_SAVE_PATH, f"{model_name}.zip")
        model = DDPG.load(model_path, env=env)
    elif agent.lower() == "sac":
        # Stable-Baselines3 SAC模型加载
        model_path = os.path.join(MODEL_SAVE_PATH, f"{model_name}.zip")
        model = SAC.load(model_path, env=env)
    else:
        raise ValueError(f"不支持的智能体类型: {agent}")
    
    return model


if __name__ == "__main__":
    # 示例用法
    model = run_etf_strategy(
        start_date=TRAIN_START_DATE,
        end_date=TRAIN_END_DATE,
        ticker_list=TICKER_LIST,
        agent="ppo",
        model_name="ppo_etf_demo",
        num_episodes=NUM_EPISODES
    ) 