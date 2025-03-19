import backtrader as bt
import pandas as pd
from datetime import datetime
from loguru import logger
from .analysis import Analysis
from .plot import Plot
import sys

class AShareCommissionInfo(bt.CommInfoBase):
    """A股交易费用"""
    params = (
        ('commission', 0.00025),     # 佣金费率 0.025%
        ('stocklike', True),        # 股票模式
        ('commtype', bt.CommInfoBase.COMM_PERC),  # 按百分比收取
    )

    def _getcommission(self, size, price, pseudoexec):
        """计算交易费用"""
        value = abs(size) * price
        
        # 计算佣金（买入和卖出都收取）
        commission = value * self.p.commission
        
        return commission

class BacktestEngine:
    def __init__(self, strategy_class, data_feed, cash=100000.0, commission=0.00025, strategy_params=None):
        """初始化回测引擎"""
        self.cerebro = bt.Cerebro()
        self.cerebro.broker.setcash(cash)
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
        logger.info(analysis)
        return analysis 