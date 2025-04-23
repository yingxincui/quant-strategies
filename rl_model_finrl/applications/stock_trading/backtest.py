import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import PercentFormatter
import seaborn as sns
import os
import logging
from typing import Any, List, Dict, Optional, Union, Tuple
import datetime

from src.strategies.rl_model_finrl.meta.data_processors import DataProcessor
from src.strategies.rl_model_finrl.applications.stock_trading.etf_env import ETFTradingEnv
from src.strategies.rl_model_finrl.applications.stock_trading.run_strategy import prepare_etf_data

from src.strategies.rl_model_finrl.config import (
    TEST_START_DATE,
    TEST_END_DATE,
    TECHNICAL_INDICATORS_LIST,
    TICKER_LIST,
    INITIAL_AMOUNT,
    TRANSACTION_COST_PCT,
    MODEL_SAVE_PATH,
    RESULTS_PATH
)


def backtest_etf_strategy(
    model: Any,
    test_start: str = TEST_START_DATE,
    test_end: str = TEST_END_DATE,
    ticker_list: List[str] = None,
    data_source: str = "tushare",
    time_interval: str = "1d",
    technical_indicator_list: List[str] = TECHNICAL_INDICATORS_LIST,
    initial_amount: float = INITIAL_AMOUNT,
    transaction_cost_pct: float = TRANSACTION_COST_PCT,
    risk_free_rate: float = 0.0,
    market_benchmark: str = 'CSI300',
    render: bool = True,
    save_result: bool = True,
    result_filename: str = None
) -> Dict[str, Any]:
    """
    使用训练好的RL模型回测ETF交易策略
    
    参数:
        model: 已训练的强化学习模型
        test_start: 测试开始日期
        test_end: 测试结束日期
        ticker_list: ETF代码列表
        data_source: 数据源
        time_interval: 时间间隔
        technical_indicator_list: 技术指标列表
        initial_amount: 初始资金
        transaction_cost_pct: 交易成本百分比
        risk_free_rate: 无风险利率
        market_benchmark: 市场基准标的
        render: 是否渲染结果图表
        save_result: 是否保存结果
        result_filename: 结果文件名
        
    返回:
        回测结果统计字典
    """
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info(f"开始回测ETF交易策略: {test_start} 到 {test_end}")
    
    # 如果未提供ETF列表，使用默认列表
    if ticker_list is None:
        ticker_list = TICKER_LIST
    
    # 确保结果目录存在
    if not os.path.exists(RESULTS_PATH):
        os.makedirs(RESULTS_PATH)
    
    # 初始化数据处理器
    processor = DataProcessor(data_source=data_source, time_interval=time_interval)
    
    # 准备回测数据
    df = prepare_etf_data(
        processor=processor,
        ticker_list=ticker_list,
        start_date=test_start,
        end_date=test_end,
        data_source=data_source
    )
    
    # 获取基准数据
    if market_benchmark:
        benchmark_df = None
        try:
            benchmark_df = processor.download_data(
                ticker_list=[market_benchmark],
                start_date=test_start,
                end_date=test_end,
                data_source=data_source
            )
            benchmark_df = processor.clean_data(benchmark_df)
        except Exception as e:
            logger.warning(f"无法获取基准数据: {e}")
            market_benchmark = None
    
    # 创建回测环境
    stock_dimension = len(df['tic'].unique())
    env_test = ETFTradingEnv(
        df=df,
        stock_dim=stock_dimension,
        hmax=100,
        initial_amount=initial_amount,
        transaction_cost_pct=transaction_cost_pct,
        reward_scaling=1.0,
        tech_indicator_list=technical_indicator_list,
        turbulence_threshold=None,  # 在回测中不使用波动阈值
        risk_free_rate=risk_free_rate,
        cash_penalty_proportion=0.0  # 回测中不惩罚现金
    )
    
    # 进行回测（运行一个episode）
    logger.info("执行回测...")
    vec_env_test = env_test.get_sb_env()
    
    # 使用确定性策略进行预测
    obs = vec_env_test.reset()
    done = False
    
    # 回测统计变量
    rewards = []
    actions = []
    portfolio_values = []
    
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env_test.step(action)
        rewards.append(reward[0])
        actions.append(action)
        portfolio_values.append(info[0]['portfolio_value'])
    
    # 获取回测记录
    df_actions = env_test.save_action_memory()
    df_asset = env_test.save_asset_memory()
    
    # 计算账户价值日变化率
    df_asset['daily_return'] = df_asset['portfolio_value'].pct_change()
    
    # 获取基准回报
    if market_benchmark and benchmark_df is not None:
        benchmark_daily_returns = benchmark_df[benchmark_df['tic'] == market_benchmark]['close'].pct_change().dropna()
        # 确保两个时间序列有相同的日期索引
        common_dates = set(df_asset.index).intersection(set(benchmark_daily_returns.index))
        df_asset = df_asset.loc[common_dates]
        benchmark_daily_returns = benchmark_daily_returns.loc[common_dates]
        df_asset['benchmark_return'] = benchmark_daily_returns.values
    
    # 计算累积回报
    df_asset['cumulative_return'] = (1 + df_asset['daily_return']).cumprod() - 1
    if 'benchmark_return' in df_asset.columns:
        df_asset['benchmark_cumulative_return'] = (1 + df_asset['benchmark_return']).cumprod() - 1
    
    # 获取最终统计结果
    final_stats = env_test.get_final_stats()
    
    # 如果有基准，计算超额收益
    if 'benchmark_return' in df_asset.columns:
        # 计算每日超额收益
        df_asset['excess_return'] = df_asset['daily_return'] - df_asset['benchmark_return']
        
        # 计算累积超额收益
        df_asset['cumulative_excess_return'] = (1 + df_asset['excess_return']).cumprod() - 1
        
        # 计算信息比率
        excess_returns = df_asset['excess_return'].values
        if len(excess_returns) > 0:
            tracking_error = np.std(excess_returns) * np.sqrt(252)  # 年化跟踪误差
            information_ratio = (final_stats['annual_return'] - benchmark_df[benchmark_df['tic'] == market_benchmark]['close'].pct_change().mean() * 252) / tracking_error if tracking_error > 0 else np.nan
            final_stats['information_ratio'] = information_ratio
            final_stats['tracking_error'] = tracking_error
    
    # 绘制回测结果
    if render:
        # 创建结果目录
        if not os.path.exists(os.path.join(RESULTS_PATH, 'plots')):
            os.makedirs(os.path.join(RESULTS_PATH, 'plots'))
        
        # 绘制投资组合价值图
        plt.figure(figsize=(15, 6))
        plt.plot(df_asset.index, df_asset['portfolio_value'], label='量化策略')
        plt.title('量化策略 - 投资组合价值')
        plt.xlabel('日期')
        plt.ylabel('价值($)')
        plt.grid(True)
        plt.legend()
        
        # 格式化日期轴
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_PATH, 'plots', 'portfolio_value.png'))
        
        # 绘制累积收益对比图
        plt.figure(figsize=(15, 6))
        plt.plot(df_asset.index, df_asset['cumulative_return'] * 100, label='量化策略')
        if 'benchmark_cumulative_return' in df_asset.columns:
            plt.plot(df_asset.index, df_asset['benchmark_cumulative_return'] * 100, label=market_benchmark, linestyle='--')
        plt.title('量化策略 vs 基准 - 累积收益率(%)')
        plt.xlabel('日期')
        plt.ylabel('累积收益率(%)')
        plt.grid(True)
        plt.legend()
        
        # 格式化y轴为百分比
        plt.gca().yaxis.set_major_formatter(PercentFormatter())
        
        # 格式化日期轴
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_PATH, 'plots', 'cumulative_returns.png'))
        
        # 绘制每日收益率分布图
        plt.figure(figsize=(15, 6))
        sns.histplot(df_asset['daily_return'].dropna() * 100, kde=True, bins=50)
        plt.title('量化策略 - 每日收益率分布(%)')
        plt.xlabel('每日收益率(%)')
        plt.ylabel('频率')
        plt.grid(True)
        plt.axvline(x=0, color='r', linestyle='--')
        
        # 格式化x轴为百分比
        plt.gca().xaxis.set_major_formatter(PercentFormatter())
        
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_PATH, 'plots', 'daily_returns_distribution.png'))
        
        # 绘制月度收益热力图
        if len(df_asset) > 30:  # 至少需要一个月的数据
            # 计算月度收益
            df_asset['year'] = df_asset.index.year
            df_asset['month'] = df_asset.index.month
            monthly_returns = df_asset.groupby(['year', 'month'])['daily_return'].apply(lambda x: (1 + x).prod() - 1) * 100
            monthly_returns = monthly_returns.unstack()
            
            # 绘制热力图
            plt.figure(figsize=(15, 8))
            sns.heatmap(monthly_returns, annot=True, fmt=".2f", cmap="RdYlGn", cbar=True, linewidths=.5)
            plt.title('量化策略 - 月度收益率热力图(%)')
            plt.xlabel('月份')
            plt.ylabel('年份')
            
            plt.tight_layout()
            plt.savefig(os.path.join(RESULTS_PATH, 'plots', 'monthly_returns_heatmap.png'))
    
    # 保存回测结果
    if save_result:
        if result_filename is None:
            result_filename = f"backtest_result_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # 保存资产记录
        df_asset.to_csv(os.path.join(RESULTS_PATH, f"{result_filename}_asset.csv"))
        
        # 保存交易记录
        if not df_actions.empty:
            df_actions.to_csv(os.path.join(RESULTS_PATH, f"{result_filename}_trades.csv"))
        
        # 保存统计摘要
        with open(os.path.join(RESULTS_PATH, f"{result_filename}_summary.txt"), 'w') as f:
            f.write("ETF交易策略回测摘要\n")
            f.write(f"回测期间: {test_start} 到 {test_end}\n")
            f.write(f"初始资金: ${initial_amount:.2f}\n")
            f.write(f"最终资产: ${final_stats['final_value']:.2f}\n")
            f.write(f"总收益率: {final_stats['total_return']*100:.2f}%\n")
            f.write(f"年化收益率: {final_stats['annual_return']*100:.2f}%\n")
            f.write(f"年化波动率: {final_stats['volatility']*100:.2f}%\n")
            f.write(f"夏普比率: {final_stats['sharpe']:.4f}\n")
            f.write(f"索提诺比率: {final_stats['sortino']:.4f}\n")
            f.write(f"最大回撤: {final_stats['max_drawdown']*100:.2f}%\n")
            f.write(f"胜率: {final_stats['win_rate']*100:.2f}%\n")
            f.write(f"交易次数: {final_stats['total_trades']}\n")
            f.write(f"分红总额: ${final_stats['total_dividend']:.2f}\n")
            
            if 'information_ratio' in final_stats:
                f.write(f"信息比率: {final_stats['information_ratio']:.4f}\n")
                f.write(f"跟踪误差: {final_stats['tracking_error']*100:.2f}%\n")
    
    logger.info(f"回测完成，结果已保存至 {RESULTS_PATH}")
    
    return final_stats


def evaluation_comparison(
    models: Dict[str, Any],
    test_start: str = TEST_START_DATE,
    test_end: str = TEST_END_DATE,
    ticker_list: List[str] = None,
    data_source: str = "tushare",
    initial_amount: float = INITIAL_AMOUNT,
    transaction_cost_pct: float = TRANSACTION_COST_PCT,
    market_benchmark: str = 'CSI300',
    save_result: bool = True
):
    """
    比较多个模型的回测性能
    
    参数:
        models: 模型字典，键为模型名称，值为模型实例
        test_start: 测试开始日期
        test_end: 测试结束日期
        ticker_list: ETF代码列表
        data_source: 数据源
        initial_amount: 初始资金
        transaction_cost_pct: 交易成本百分比
        market_benchmark: 市场基准标的
        save_result: 是否保存结果
    """
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info(f"开始比较{len(models)}个模型的回测性能")
    
    # 保存每个模型的回测结果
    results = {}
    
    # 对每个模型进行回测
    for model_name, model in models.items():
        logger.info(f"开始回测模型: {model_name}")
        
        result_filename = f"compare_{model_name}_{datetime.datetime.now().strftime('%Y%m%d')}"
        
        # 执行回测
        stats = backtest_etf_strategy(
            model=model,
            test_start=test_start,
            test_end=test_end,
            ticker_list=ticker_list,
            data_source=data_source,
            initial_amount=initial_amount,
            transaction_cost_pct=transaction_cost_pct,
            market_benchmark=market_benchmark,
            render=False,  # 单独模型不需要渲染
            save_result=True,
            result_filename=result_filename
        )
        
        # 保存该模型的结果
        results[model_name] = stats
    
    # 比较模型并绘制比较图表
    if results:
        # 创建比较DataFrame
        compare_df = pd.DataFrame({
            'Model': list(results.keys()),
            'Total Return (%)': [results[model]['total_return'] * 100 for model in results],
            'Annual Return (%)': [results[model]['annual_return'] * 100 for model in results],
            'Volatility (%)': [results[model]['volatility'] * 100 for model in results],
            'Sharpe Ratio': [results[model]['sharpe'] for model in results],
            'Max Drawdown (%)': [results[model]['max_drawdown'] * 100 for model in results],
            'Win Rate (%)': [results[model]['win_rate'] * 100 for model in results],
            'Total Trades': [results[model]['total_trades'] for model in results]
        })
        
        # 按总回报排序
        compare_df = compare_df.sort_values('Total Return (%)', ascending=False)
        
        # 绘制性能对比图
        plt.figure(figsize=(15, 10))
        
        # 绘制总回报对比
        plt.subplot(2, 2, 1)
        sns.barplot(x='Model', y='Total Return (%)', data=compare_df)
        plt.title('总回报对比')
        plt.xticks(rotation=45)
        plt.grid(True, axis='y')
        
        # 绘制夏普比率对比
        plt.subplot(2, 2, 2)
        sns.barplot(x='Model', y='Sharpe Ratio', data=compare_df)
        plt.title('夏普比率对比')
        plt.xticks(rotation=45)
        plt.grid(True, axis='y')
        
        # 绘制最大回撤对比
        plt.subplot(2, 2, 3)
        sns.barplot(x='Model', y='Max Drawdown (%)', data=compare_df)
        plt.title('最大回撤对比')
        plt.xticks(rotation=45)
        plt.grid(True, axis='y')
        
        # 绘制胜率对比
        plt.subplot(2, 2, 4)
        sns.barplot(x='Model', y='Win Rate (%)', data=compare_df)
        plt.title('胜率对比')
        plt.xticks(rotation=45)
        plt.grid(True, axis='y')
        
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_PATH, 'model_comparison.png'))
        
        # 保存比较结果
        if save_result:
            compare_df.to_csv(os.path.join(RESULTS_PATH, 'model_comparison.csv'), index=False)
        
        logger.info(f"模型比较完成，结果已保存至 {RESULTS_PATH}")
        
        return compare_df
    else:
        logger.warning("没有有效的模型结果进行比较")
        return None


if __name__ == "__main__":
    # 示例用法 - 需要先训练模型获取model实例
    from src.strategies.rl_model_finrl.applications.stock_trading.run_strategy import load_etf_model
    
    # 加载已训练的模型
    model = load_etf_model("ppo_etf_demo", agent="ppo")
    
    # 执行回测
    result = backtest_etf_strategy(
        model=model,
        test_start=TEST_START_DATE,
        test_end=TEST_END_DATE,
        ticker_list=TICKER_LIST,
        render=True,
        save_result=True
    ) 