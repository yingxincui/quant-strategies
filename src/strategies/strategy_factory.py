from typing import Type, Dict
import backtrader as bt
from .dual_ma_hedging_strategy import DualMAHedgingStrategy
from .dual_ma_strategy import DualMAStrategy
from .market_sentiment_strategy import MarketSentimentStrategy
from .etf_rotation_strategy import ETFRotationStrategy

class StrategyFactory:
    _strategies: Dict[str, Type[bt.Strategy]] = {
        "双均线策略": DualMAStrategy,
        "市场情绪策略": MarketSentimentStrategy,
        "ETF轮动策略": ETFRotationStrategy,
        "双均线对冲策略": DualMAHedgingStrategy,
    }

    @classmethod
    def register_strategy(cls, name: str, strategy_class: Type[bt.Strategy]):
        """注册新的策略"""
        cls._strategies[name] = strategy_class

    @classmethod
    def get_strategy(cls, name: str) -> Type[bt.Strategy]:
        """获取策略类"""
        return cls._strategies.get(name)

    @classmethod
    def get_all_strategies(cls) -> Dict[str, Type[bt.Strategy]]:
        """获取所有已注册的策略"""
        return cls._strategies.copy()

    @classmethod
    def get_strategy_names(cls) -> list:
        """获取所有策略名称"""
        return list(cls._strategies.keys()) 