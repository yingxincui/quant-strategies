"""
ETF交易应用模块

本模块提供了基于强化学习的ETF交易应用案例，用于展示如何使用FinRL框架构建ETF交易策略。
该应用在模拟和实际市场数据上进行回测，以评估策略的性能。

主要组件:
1. ETF交易环境: 一个特定为ETF交易定制的强化学习环境
2. 交易策略示例: 基于RL模型的ETF交易策略
3. 回测工具: 用于评估模型性能的回测工具
4. 性能分析: 交易策略的收益率、风险和其他相关指标分析

用法示例:
```python
from src.strategies.rl_model_finrl.applications.stock_trading import (
    run_etf_strategy,
    backtest_etf_strategy,
    ETFTradingEnv
)

# 训练ETF交易策略
trained_model = run_etf_strategy(
    start_date='2010-01-01',
    end_date='2020-12-31',
    ticker_list=['SPY', 'QQQ', 'IWM', 'EEM'],
    agent='ppo'
)

# 回测策略
performance = backtest_etf_strategy(
    model=trained_model,
    test_start='2021-01-01',
    test_end='2021-12-31'
)
```
"""

from src.strategies.rl_model_finrl.applications.stock_trading.etf_env import ETFTradingEnv
from src.strategies.rl_model_finrl.applications.stock_trading.run_strategy import run_etf_strategy
from src.strategies.rl_model_finrl.applications.stock_trading.backtest import backtest_etf_strategy
from src.strategies.rl_model_finrl.applications.stock_trading.analysis import ETFStrategyAnalyzer

__all__ = [
    'ETFTradingEnv',
    'run_etf_strategy',
    'backtest_etf_strategy',
    'ETFStrategyAnalyzer'
] 