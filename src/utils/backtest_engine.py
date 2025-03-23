import backtrader as bt
import pandas as pd
from datetime import datetime
from loguru import logger
from .analysis import Analysis
from .plot import Plot
import sys

class BacktestEngine:
    def __init__(self, strategy_class, data_feed, cash=100000.0, commission=0.00025, strategy_params=None):
        """初始化回测引擎
        Args:
            strategy_class: 策略类
            data_feed: 数据源或数据源列表
            cash: 初始资金
            commission: 股票交易手续费率
            strategy_params: 策略参数
        """
        self.cerebro = bt.Cerebro()
        self.cerebro.broker.setcash(cash)
        
        # 设置手续费
        if isinstance(data_feed, list):
            # 默认设置股票/ETF手续费率
            self.cerebro.broker.setcommission(commission=commission)
            
            # 创建多数据源的特定手续费处理
            for i, feed in enumerate(data_feed):
                if i == 1:  # 期货数据源使用固定手续费
                    # 为期货设置固定手续费
                    self.cerebro.broker.addcommissioninfo(
                        bt.CommissionInfo(
                            commission=1.51/100000,  # 固定手续费转换为相对值
                            margin=0.10,             # 保证金比例
                            mult=10,                 # 合约乘数
                            commtype=0               # 固定手续费类型(0=固定手续费)
                        )
                    )
        else:
            self.cerebro.broker.setcommission(commission=commission)
        
        # 添加数据源
        try:
            if isinstance(data_feed, list):
                # 如果是数据源列表，添加所有数据源
                for feed in data_feed:
                    self.cerebro.adddata(feed)
            else:
                # 如果是单个数据源，直接添加
                self.cerebro.adddata(data_feed)
        except Exception as e:
            logger.warning(f"添加数据源时出错: {str(e)}")
            # 如果出错，尝试不带ts_code参数添加
            if hasattr(data_feed, 'params'):
                data_feed.params.pop('ts_code', None)
            self.cerebro.adddata(data_feed)
            
        # 添加策略和参数
        if strategy_params:
            self.cerebro.addstrategy(strategy_class, **strategy_params)
        else:
            self.cerebro.addstrategy(strategy_class)
            
        # 添加分析器
        self.cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', riskfreerate=0.02)
        self.cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        self.cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
        self.cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
        self.cerebro.addanalyzer(bt.analyzers.VWR, _name='vwr')  # 波动率加权收益
        self.cerebro.addanalyzer(bt.analyzers.SQN, _name='sqn')  # 系统质量指数
        
        # 为每个数据源添加单独的交易记录分析器
        if isinstance(data_feed, list):
            for feed in data_feed:
                self.cerebro.addanalyzer(bt.analyzers.Transactions, _name=f'txn_{feed._name}')
        else:
            self.cerebro.addanalyzer(bt.analyzers.Transactions, _name='txn')
        
        self.trades = []  # 存储交易记录
        
    def run(self):
        """运行回测"""
        results = self.cerebro.run()
        
        self.strategy = results[0]
                
        analysis = self._get_analysis(self.strategy)
        
        logger.info("=== 回测统计 ===")
        logger.info(f"总收益率: {analysis['total_return']:.2%}")
        logger.info(f"年化收益率: {analysis['annualized_return']:.2%}")
        logger.info(f"夏普比率: {analysis['sharpe_ratio']:.2f}")
        logger.info(f"最大回撤: {analysis['max_drawdown']:.2%}")
        logger.info(f"胜率: {analysis['win_rate']:.2%}")
        logger.info(f"盈亏比: {analysis['profit_factor']:.2f}")
        logger.info(f"系统质量指数(SQN): {analysis['sqn']:.2f}")
        
        return analysis
    
    def plot(self, **kwargs):
        """使用Plotly绘制交互式回测结果"""
        fig = Plot(self.strategy).plot()
        return fig
        
    def _get_analysis(self, strategy):
        """获取回测分析结果"""
        analysis = Analysis()._get_analysis(self, strategy)
        return analysis 