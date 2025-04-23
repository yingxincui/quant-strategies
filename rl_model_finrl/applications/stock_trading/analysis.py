import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import logging
import json
import empyrical

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/analysis.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ETFStrategyAnalyzer:
    """量化策略分析器类，提供全面的策略绩效分析功能"""
    
    def __init__(
        self,
        results_dir: str,
        benchmark_ticker: str = '000300',  # 使用沪深300作为默认基准
        risk_free_rate: float = 0.03,      # 年化无风险利率（默认3%）
    ):
        """
        初始化量化策略分析器
        
        参数:
            results_dir: 结果数据目录路径
            benchmark_ticker: 基准指数代码
            risk_free_rate: 年化无风险利率
        """
        self.results_dir = results_dir
        self.benchmark_ticker = benchmark_ticker
        self.risk_free_rate = risk_free_rate / 252  # 转换为日度无风险利率
        self.strategy_data = None
        self.benchmark_data = None
        
        # 创建结果目录（如果不存在）
        os.makedirs(self.results_dir, exist_ok=True)
    
    def load_strategy_data(self, strategy_file: str) -> pd.DataFrame:
        """
        加载策略回测数据
        
        参数:
            strategy_file: 策略回测数据文件路径
            
        返回:
            策略回测数据DataFrame
        """
        logger.info(f"加载策略回测数据: {strategy_file}")
        
        try:
            data = pd.read_csv(strategy_file, index_col=0)
            data.index = pd.to_datetime(data.index)
            self.strategy_data = data
            return data
        except Exception as e:
            logger.error(f"加载策略数据失败: {str(e)}")
            raise
    
    def load_benchmark_data(self, benchmark_file: str = None) -> pd.DataFrame:
        """
        加载基准数据
        
        参数:
            benchmark_file: 基准数据文件路径，如果为None则尝试从结果目录加载
            
        返回:
            基准数据DataFrame
        """
        if benchmark_file is None:
            benchmark_file = os.path.join(self.results_dir, f"benchmark_{self.benchmark_ticker}.csv")
        
        logger.info(f"加载基准数据: {benchmark_file}")
        
        try:
            data = pd.read_csv(benchmark_file, index_col=0)
            data.index = pd.to_datetime(data.index)
            self.benchmark_data = data
            return data
        except Exception as e:
            logger.error(f"加载基准数据失败: {str(e)}")
            raise
    
    def calculate_returns(self) -> Tuple[pd.Series, pd.Series]:
        """
        计算策略和基准的收益率序列
        
        返回:
            (策略日收益率序列, 基准日收益率序列)元组
        """
        if self.strategy_data is None or self.benchmark_data is None:
            logger.error("计算收益率前需要先加载策略和基准数据")
            raise ValueError("计算收益率前需要先加载策略和基准数据")
        
        # 计算策略日收益率
        strategy_returns = self.strategy_data['portfolio_value'].pct_change().dropna()
        
        # 计算基准日收益率
        benchmark_returns = self.benchmark_data['close'].pct_change().dropna()
        
        # 确保两个序列有相同的日期索引
        common_index = strategy_returns.index.intersection(benchmark_returns.index)
        strategy_returns = strategy_returns.loc[common_index]
        benchmark_returns = benchmark_returns.loc[common_index]
        
        return strategy_returns, benchmark_returns
    
    def calculate_performance_metrics(self) -> Dict:
        """
        计算策略绩效指标
        
        返回:
            包含各项绩效指标的字典
        """
        strategy_returns, benchmark_returns = self.calculate_returns()
        
        # 使用empyrical库计算各项指标
        metrics = {}
        
        # 年化收益率
        metrics['annual_return'] = empyrical.annual_return(strategy_returns)
        metrics['benchmark_annual_return'] = empyrical.annual_return(benchmark_returns)
        
        # 最大回撤
        metrics['max_drawdown'] = empyrical.max_drawdown(strategy_returns)
        metrics['benchmark_max_drawdown'] = empyrical.max_drawdown(benchmark_returns)
        
        # 夏普比率
        metrics['sharpe_ratio'] = empyrical.sharpe_ratio(
            strategy_returns, risk_free=self.risk_free_rate
        )
        metrics['benchmark_sharpe_ratio'] = empyrical.sharpe_ratio(
            benchmark_returns, risk_free=self.risk_free_rate
        )
        
        # 索提诺比率
        metrics['sortino_ratio'] = empyrical.sortino_ratio(
            strategy_returns, required_return=self.risk_free_rate
        )
        metrics['benchmark_sortino_ratio'] = empyrical.sortino_ratio(
            benchmark_returns, required_return=self.risk_free_rate
        )
        
        # 卡尔玛比率
        metrics['calmar_ratio'] = empyrical.calmar_ratio(strategy_returns)
        metrics['benchmark_calmar_ratio'] = empyrical.calmar_ratio(benchmark_returns)
        
        # 波动率
        metrics['volatility'] = empyrical.annual_volatility(strategy_returns)
        metrics['benchmark_volatility'] = empyrical.annual_volatility(benchmark_returns)
        
        # α、β系数
        metrics['alpha'] = empyrical.alpha(
            strategy_returns, benchmark_returns, risk_free=self.risk_free_rate
        )
        metrics['beta'] = empyrical.beta(strategy_returns, benchmark_returns)
        
        # 信息比率
        metrics['information_ratio'] = empyrical.excess_sharpe(
            strategy_returns, benchmark_returns
        )
        
        # 胜率
        metrics['win_rate'] = len(strategy_returns[strategy_returns > 0]) / len(strategy_returns)
        
        # 年化超额收益
        metrics['excess_return'] = metrics['annual_return'] - metrics['benchmark_annual_return']
        
        # 累计收益
        metrics['cumulative_return'] = empyrical.cum_returns_final(strategy_returns)
        metrics['benchmark_cumulative_return'] = empyrical.cum_returns_final(benchmark_returns)
        
        return metrics
    
    def plot_equity_curve(self, save_path: str = None) -> None:
        """
        绘制策略与基准的权益曲线对比
        
        参数:
            save_path: 图表保存路径，默认保存到结果目录
        """
        if save_path is None:
            save_path = os.path.join(self.results_dir, "equity_curve.png")
        
        strategy_returns, benchmark_returns = self.calculate_returns()
        
        # 计算累计收益曲线
        strategy_cum_returns = (1 + strategy_returns).cumprod()
        benchmark_cum_returns = (1 + benchmark_returns).cumprod()
        
        # 绘制累计收益曲线
        plt.figure(figsize=(14, 8))
        plt.plot(strategy_cum_returns.index, strategy_cum_returns, 
                 label='ETF Strategy', linewidth=2, color='blue')
        plt.plot(benchmark_cum_returns.index, benchmark_cum_returns, 
                 label='Benchmark', linewidth=2, color='red', alpha=0.7)
        
        # 添加图表元素
        plt.title('Strategy vs Benchmark Cumulative Returns', fontsize=15)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Cumulative Return', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        
        # 设置日期格式
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.xticks(rotation=45)
        
        # 添加收益率数据
        metrics = self.calculate_performance_metrics()
        plt.figtext(0.15, 0.01, 
                    f"Strategy Return: {metrics['annual_return']:.2%} | "
                    f"Benchmark Return: {metrics['benchmark_annual_return']:.2%} | "
                    f"Alpha: {metrics['alpha']:.2%} | "
                    f"Beta: {metrics['beta']:.2f} | "
                    f"Sharpe: {metrics['sharpe_ratio']:.2f}",
                    fontsize=12)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        logger.info(f"权益曲线已保存到: {save_path}")
    
    def plot_drawdown(self, save_path: str = None) -> None:
        """
        绘制策略与基准的回撤对比图
        
        参数:
            save_path: 图表保存路径，默认保存到结果目录
        """
        if save_path is None:
            save_path = os.path.join(self.results_dir, "drawdown.png")
        
        strategy_returns, benchmark_returns = self.calculate_returns()
        
        # 计算回撤序列
        strategy_dd = empyrical.roll_max_drawdown(strategy_returns, window=252)
        benchmark_dd = empyrical.roll_max_drawdown(benchmark_returns, window=252)
        
        # 绘制回撤图
        plt.figure(figsize=(14, 8))
        plt.plot(strategy_dd.index, strategy_dd * 100, 
                 label='ETF Strategy Drawdown', linewidth=2, color='blue')
        plt.plot(benchmark_dd.index, benchmark_dd * 100, 
                 label='Benchmark Drawdown', linewidth=2, color='red', alpha=0.7)
        plt.fill_between(strategy_dd.index, 0, strategy_dd * 100, color='blue', alpha=0.1)
        plt.fill_between(benchmark_dd.index, 0, benchmark_dd * 100, color='red', alpha=0.1)
        
        # 添加图表元素
        plt.title('Strategy vs Benchmark Drawdown', fontsize=15)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Drawdown (%)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        
        # 设置日期格式
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.xticks(rotation=45)
        
        # 添加最大回撤信息
        metrics = self.calculate_performance_metrics()
        plt.figtext(0.15, 0.01, 
                    f"Strategy Max Drawdown: {metrics['max_drawdown']:.2%} | "
                    f"Benchmark Max Drawdown: {metrics['benchmark_max_drawdown']:.2%}",
                    fontsize=12)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        logger.info(f"回撤图已保存到: {save_path}")
    
    def plot_monthly_returns(self, save_path: str = None) -> None:
        """
        绘制策略的月度收益热力图
        
        参数:
            save_path: 图表保存路径，默认保存到结果目录
        """
        if save_path is None:
            save_path = os.path.join(self.results_dir, "monthly_returns.png")
        
        strategy_returns, _ = self.calculate_returns()
        
        # 计算月度收益
        monthly_returns = strategy_returns.resample('M').apply(
            lambda x: (1 + x).prod() - 1
        )
        
        # 创建月度收益的透视表
        monthly_returns_table = pd.DataFrame(monthly_returns)
        monthly_returns_table['year'] = monthly_returns_table.index.year
        monthly_returns_table['month'] = monthly_returns_table.index.month
        monthly_returns_pivot = monthly_returns_table.pivot_table(
            index='year', 
            columns='month', 
            values=0
        )
        
        # 绘制热力图
        plt.figure(figsize=(14, 8))
        cmap = sns.diverging_palette(10, 220, as_cmap=True)
        sns.heatmap(
            monthly_returns_pivot * 100, 
            annot=True, 
            fmt=".2f", 
            cmap=cmap,
            center=0, 
            linewidths=1, 
            cbar_kws={"shrink": .75}
        )
        
        # 设置月份标签
        month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        plt.xticks(np.arange(12) + 0.5, month_labels)
        
        plt.title('Monthly Returns (%)', fontsize=15)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        logger.info(f"月度收益热力图已保存到: {save_path}")
    
    def plot_rolling_statistics(self, window: int = 60, save_path: str = None) -> None:
        """
        绘制滚动统计指标图（滚动收益率、波动率、夏普比率）
        
        参数:
            window: 滚动窗口天数
            save_path: 图表保存路径，默认保存到结果目录
        """
        if save_path is None:
            save_path = os.path.join(self.results_dir, f"rolling_stats_{window}d.png")
        
        strategy_returns, benchmark_returns = self.calculate_returns()
        
        # 计算滚动指标
        rolling_return_strategy = empyrical.roll_annual_return(
            strategy_returns, window=window
        )
        rolling_return_benchmark = empyrical.roll_annual_return(
            benchmark_returns, window=window
        )
        
        rolling_vol_strategy = empyrical.roll_annual_volatility(
            strategy_returns, window=window
        )
        rolling_vol_benchmark = empyrical.roll_annual_volatility(
            benchmark_returns, window=window
        )
        
        rolling_sharpe_strategy = empyrical.roll_sharpe_ratio(
            strategy_returns, risk_free=self.risk_free_rate, window=window
        )
        rolling_sharpe_benchmark = empyrical.roll_sharpe_ratio(
            benchmark_returns, risk_free=self.risk_free_rate, window=window
        )
        
        # 绘制滚动收益率
        plt.figure(figsize=(14, 18))
        
        plt.subplot(3, 1, 1)
        plt.plot(rolling_return_strategy.index, rolling_return_strategy * 100, 
                label=f'Strategy {window}d Rolling Return', linewidth=2, color='blue')
        plt.plot(rolling_return_benchmark.index, rolling_return_benchmark * 100, 
                label=f'Benchmark {window}d Rolling Return', linewidth=2, color='red', alpha=0.7)
        plt.title(f'{window}-Day Rolling Annualized Return (%)', fontsize=14)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Return (%)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='upper left', fontsize=12)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.xticks(rotation=45)
        
        # 绘制滚动波动率
        plt.subplot(3, 1, 2)
        plt.plot(rolling_vol_strategy.index, rolling_vol_strategy * 100, 
                label=f'Strategy {window}d Rolling Vol', linewidth=2, color='blue')
        plt.plot(rolling_vol_benchmark.index, rolling_vol_benchmark * 100, 
                label=f'Benchmark {window}d Rolling Vol', linewidth=2, color='red', alpha=0.7)
        plt.title(f'{window}-Day Rolling Annualized Volatility (%)', fontsize=14)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Volatility (%)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='upper left', fontsize=12)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.xticks(rotation=45)
        
        # 绘制滚动夏普比率
        plt.subplot(3, 1, 3)
        plt.plot(rolling_sharpe_strategy.index, rolling_sharpe_strategy, 
                label=f'Strategy {window}d Rolling Sharpe', linewidth=2, color='blue')
        plt.plot(rolling_sharpe_benchmark.index, rolling_sharpe_benchmark, 
                label=f'Benchmark {window}d Rolling Sharpe', linewidth=2, color='red', alpha=0.7)
        plt.title(f'{window}-Day Rolling Sharpe Ratio', fontsize=14)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Sharpe Ratio', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='upper left', fontsize=12)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        logger.info(f"滚动统计指标图已保存到: {save_path}")
    
    def generate_performance_report(self, report_path: str = None) -> Dict:
        """
        生成完整的绩效报告
        
        参数:
            report_path: 报告保存路径，默认保存到结果目录
            
        返回:
            绩效指标字典
        """
        if report_path is None:
            report_path = os.path.join(self.results_dir, "performance_report.json")
        
        # 计算绩效指标
        metrics = self.calculate_performance_metrics()
        
        # 生成绩效图表
        self.plot_equity_curve()
        self.plot_drawdown()
        self.plot_monthly_returns()
        self.plot_rolling_statistics(window=60)
        
        # 保存绩效指标到JSON文件
        with open(report_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        logger.info(f"绩效报告已保存到: {report_path}")
        
        return metrics


def analyze_etf_strategy(
    strategy_file: str,
    benchmark_file: str = None,
    results_dir: str = None,
    benchmark_ticker: str = '000300',
    risk_free_rate: float = 0.03
) -> Dict:
    """
    分析ETF交易策略绩效的主函数
    
    参数:
        strategy_file: 策略回测结果文件路径
        benchmark_file: 基准数据文件路径
        results_dir: 结果保存目录
        benchmark_ticker: 基准指数代码
        risk_free_rate: 年化无风险利率
        
    返回:
        绩效指标字典
    """
    if results_dir is None:
        results_dir = os.path.dirname(strategy_file)
    
    logger.info(f"开始分析量化策略: {os.path.basename(strategy_file)}")
    
    # 创建分析器实例
    analyzer = ETFStrategyAnalyzer(
        results_dir=results_dir,
        benchmark_ticker=benchmark_ticker,
        risk_free_rate=risk_free_rate
    )
    
    # 加载数据
    analyzer.load_strategy_data(strategy_file)
    analyzer.load_benchmark_data(benchmark_file)
    
    # 生成绩效报告
    metrics = analyzer.generate_performance_report()
    
    # 打印主要绩效指标
    logger.info("量化策略绩效摘要:")
    logger.info(f"年化收益率: {metrics['annual_return']:.2%} (基准: {metrics['benchmark_annual_return']:.2%})")
    logger.info(f"最大回撤: {metrics['max_drawdown']:.2%} (基准: {metrics['benchmark_max_drawdown']:.2%})")
    logger.info(f"夏普比率: {metrics['sharpe_ratio']:.2f} (基准: {metrics['benchmark_sharpe_ratio']:.2f})")
    logger.info(f"Alpha: {metrics['alpha']:.2%}, Beta: {metrics['beta']:.2f}")
    logger.info(f"信息比率: {metrics['information_ratio']:.2f}")
    
    return metrics


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='量化策略绩效分析工具')
    parser.add_argument('--strategy_file', type=str, required=True,
                      help='策略回测结果文件路径')
    parser.add_argument('--benchmark_file', type=str, default=None,
                      help='基准数据文件路径')
    parser.add_argument('--results_dir', type=str, default=None,
                      help='结果保存目录')
    parser.add_argument('--benchmark_ticker', type=str, default='000300',
                      help='基准指数代码')
    parser.add_argument('--risk_free_rate', type=float, default=0.03,
                      help='年化无风险利率')
    
    args = parser.parse_args()
    
    analyze_etf_strategy(
        strategy_file=args.strategy_file,
        benchmark_file=args.benchmark_file,
        results_dir=args.results_dir,
        benchmark_ticker=args.benchmark_ticker,
        risk_free_rate=args.risk_free_rate
    ) 