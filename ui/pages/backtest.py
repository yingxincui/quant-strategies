import streamlit as st
import pandas as pd
from src.strategies.strategy_factory import StrategyFactory
from src.data.data_loader import DataLoader
from src.utils.backtest_engine import BacktestEngine
from src.utils.logger import setup_logger
import os

logger = setup_logger()

def render_backtest(params):
    """渲染回测界面"""
    if st.button("开始回测", type="primary"):
        # 检查市场情绪策略的token
        if params['strategy_name'] == "市场情绪策略" and not params['tushare_token']:
            st.error("市场情绪策略必须提供Tushare Token")
            return
            
        with st.spinner("正在进行回测..."):
            try:
                # 下载数据
                logger.info(f"开始下载数据 - 股票代码: {params['selected_etfs'] if params['strategy_name'] == 'ETF轮动策略' else params['symbol']}")
                data_loader = DataLoader(tushare_token=params['tushare_token'])
                
                # 根据策略类型下载数据
                if params['strategy_name'] == "ETF轮动策略":
                    data = data_loader.download_data(params['selected_etfs'], params['start_date'], params['end_date'])
                    # 打印每个数据源的ETF代码
                    if isinstance(data, list):
                        for d in data:
                            etf_code = d.params.ts_code if hasattr(d, 'params') and hasattr(d.params, 'ts_code') else '未知'
                            logger.info(f"加载ETF数据源: {etf_code}")
                else:
                    data = data_loader.download_data(params['symbol'], params['start_date'], params['end_date'])
                
                if data is None:
                    st.error("未获取到数据，请检查股票代码和日期范围")
                    return
                    
                # 获取策略类
                strategy_class = StrategyFactory.get_strategy(params['strategy_name'])
                if not strategy_class:
                    st.error(f"未找到策略: {params['strategy_name']}")
                    return
                
                # 设置策略参数
                strategy_params = {
                    'trail_percent': params['trail_percent'],
                    'risk_ratio': params['risk_ratio'] / 100,
                    'max_drawdown': params['max_drawdown'] / 100,
                }
                
                # 如果是双均线策略，添加特定参数
                if params['strategy_name'] == "双均线策略":
                    strategy_params.update({
                        'fast_period': params['fast_period'],
                        'slow_period': params['slow_period'],
                    })
                
                # 如果是ETF轮动策略，添加特定参数
                if params['strategy_name'] == "ETF轮动策略":
                    strategy_params.update({
                        'momentum_period': params['momentum_period'],
                        'rebalance_interval': params['rebalance_interval'],
                        'num_positions': params['num_positions'],
                    })
                
                # 如果是市场情绪策略，添加tushare token
                if params['strategy_name'] == "市场情绪策略":
                    os.environ['TUSHARE_TOKEN'] = params['tushare_token']
                
                # 创建回测引擎
                engine = BacktestEngine(
                    strategy_class,
                    data,
                    cash=params['initial_cash'],
                    commission=params['commission'],
                    strategy_params=strategy_params
                )
                
                # 运行回测
                results = engine.run()
                
                # 显示回测结果
                st.header("回测结果")
                
                # 显示交易统计
                trades = results.get('trades', pd.DataFrame())  # 获取交易记录DataFrame

                # 打印交易记录
                logger.info(f"交易记录: {trades}")
                
                # 获取交易统计
                total_pnl = results.get('total_pnl', 0)  # 使用引擎计算的总盈亏
                total_trades = len(trades) if not trades.empty else 0
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("总盈亏", f"{total_pnl:.2f}")
                with col2:
                    st.metric("总交易次数", total_trades)
                with col3:
                    # 使用引擎计算的胜率，与后台日志保持一致
                    st.metric("胜率", f"{results['win_rate']:.2%}" if total_trades > 0 else "0%")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("总收益率", f"{results['total_return']:.2%}")
                with col2:
                    st.metric("夏普比率", f"{results['sharpe_ratio']:.2f}")
                with col3:
                    st.metric("最大回撤", f"{results['max_drawdown']:.2%}")
                
                # 显示交易记录
                st.subheader("交易记录")
                if total_trades > 0:
                    try:
                        # 显示交易记录表格
                        st.dataframe(
                            trades,
                            use_container_width=True,
                            hide_index=True
                        )
                        
                    except Exception as e:
                        logger.error(f"格式化交易记录时出错: {str(e)}")
                        st.error("显示交易记录时出错，请检查数据格式")
                else:
                    st.info("回测期间没有产生交易")
                    
                # 费用统计
                st.subheader("费用统计")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("分红总额", f"{results['total_dividend']:.2f}")
                with col2:
                    st.metric("佣金", f"{results['total_commission']:.2f}")
                
                # 绘制回测图表
                st.subheader("回测结果图表")
                
                # 使用新的Plotly可视化
                fig = engine.plot()
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                logger.error(f"回测过程中出现错误: {str(e)}")
                import traceback
                traceback.print_exc()
                st.error(f"回测失败: {str(e)}") 