import os
import numpy as np
import torch

# 设备配置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# A股ETF数据配置
TICKER_LIST = [
    "159915.SZ",  # 易方达创业板ETF
    "510300.SH",  # 华泰柏瑞沪深300ETF
    "510500.SH",  # 南方中证500ETF
    "512100.SH",  # 南方中证1000ETF
    "510050.SH",  # 华夏上证50ETF
    "512880.SH",  # 国泰中证军工ETF
    "512690.SH",  # 鹏华中证医药卫生ETF
    "512980.SH",  # 广发中证传媒ETF
]  # A股ETF代码列表

TECHNICAL_INDICATORS_LIST = [
    "macd",
    "boll_ub",
    "boll_lb",
    "rsi_30",
    "cci_30",
    "dx_30",
    "close_30_sma", 
    "close_60_sma",
    "volatility_30",
    "momentum_30",
]  # 技术指标列表
INDICATORS_NORMALIZE = True  # 是否标准化指标
TURBULENCE_THRESHOLD = 0.01  # 波动性阈值

# 交易环境配置
INITIAL_AMOUNT = 1000000.0  # 初始资金
TRANSACTION_COST_PCT = 0.0003  # 交易成本百分比，ETF一般费率更低
MAX_POSITION_PCT = 0.3  # 单个ETF最大仓位
REWARD_SCALING = 1e-3  # 奖励缩放系数
STATE_SPACE_DIM = len(TECHNICAL_INDICATORS_LIST) + 4  # 状态空间维度 (技术指标 + 持仓量 + 现金比例 + 大盘指标 + 情绪指标)
ACTION_SPACE_DIM = 3  # 动作空间维度（买入、卖出、持有）

# 训练配置
TRAIN_START_DATE = "2018-01-01"
TRAIN_END_DATE = "2021-12-31"
TEST_START_DATE = "2022-01-01"
TEST_END_DATE = "2023-12-31"
TIME_INTERVAL = "1d"  # 时间间隔（日频）

# 数据源配置
TUSHARE_TOKEN = ""  # 需要填入您的tushare token
USE_TUSHARE = True
USE_AKSHARE = True

# 智能体配置
REPLAY_BUFFER_SIZE = 100000  # 回放缓冲区大小
GAMMA = 0.99  # 折扣因子
LEARNING_RATE = 1e-4  # 学习率
BATCH_SIZE = 256  # 批处理大小
TARGET_UPDATE_FREQ = 100  # 目标网络更新频率
NUM_EPISODES = 1000  # 训练回合数
EPSILON_START = 0.9  # 探索率初始值
EPSILON_END = 0.05  # 探索率终值
EPSILON_DECAY = 1000  # 探索率衰减系数

# 路径配置
DATA_SAVE_PATH = "data"
MODEL_SAVE_PATH = "models"
RESULTS_PATH = "results"
TENSORBOARD_PATH = "runs"
TENSORBOARD_LOG_PATH = "runs/tensorboard"

# 创建必要的目录
for path in [DATA_SAVE_PATH, MODEL_SAVE_PATH, RESULTS_PATH, TENSORBOARD_PATH]:
    os.makedirs(path, exist_ok=True) 