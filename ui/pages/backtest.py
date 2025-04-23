import streamlit as st
import pandas as pd
from src.strategies.strategy_factory import StrategyFactory
from src.data.data_loader import DataLoader
from src.data.future_data_loader import FutureDataLoader
from src.utils.backtest_engine import BacktestEngine
from src.utils.logger import setup_logger
import os
import plotly.graph_objects as go
import tushare as ts
from datetime import datetime, timedelta
import json

logger = setup_logger()

def screen_index_stocks(params, index_code, index_name):
    """根据指数筛选成分股
    
    Args:
        params: 参数字典
        index_code: 指数代码
        index_name: 指数名称
    """
    os.environ['TUSHARE_TOKEN'] = params['tushare_token']

    with st.spinner("正在进行回测筛选..."):
        try:
            # 初始化Tushare
            ts.set_token(params['tushare_token'])
            pro = ts.pro_api()
            
            # 获取成分股列表
            today = datetime.now().strftime('%Y%m%d')
            csi_index = pro.index_weight(index_code=index_code, trade_date=today)
            if csi_index.empty:
                # 如果当天数据不可用，尝试获取最近的数据
                dates = pro.trade_cal(exchange='', start_date=(datetime.now() - timedelta(days=30)).strftime('%Y%m%d'), 
                                    end_date=today, is_open='1')
                for date in sorted(dates['cal_date'].tolist(), reverse=True):
                    csi_index = pro.index_weight(index_code=index_code, trade_date=date)
                    if not csi_index.empty:
                        break
            
            if csi_index.empty:
                st.error(f"未获取到{index_name}成分股列表")
                return
                
            # 获取成分股的基本信息
            stocks = pro.stock_basic(exchange='', list_status='L')
            csi_index_stocks = stocks[stocks['ts_code'].isin(csi_index['con_code'])]
            
            if csi_index_stocks.empty:
                st.error(f"未获取到{index_name}成分股信息")
                return
                
            # 创建进度条
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # 存储筛选结果
            positive_stocks = []
            total_stocks = len(csi_index_stocks)
            
            # 遍历每只股票进行回测
            for i, (idx, row) in enumerate(csi_index_stocks.iterrows()):
                # 更新进度（确保不超过1.0）
                progress = min(1.0, (i + 1) / total_stocks)
                progress_bar.progress(progress)
                status_text.text(f"正在回测 ({i + 1}/{total_stocks}): {row['name']} ({row['ts_code']})")
                
                try:
                    # 下载数据
                    data_loader = DataLoader(tushare_token=params['tushare_token'])
                    data = data_loader.download_data(row['ts_code'], params['start_date'], params['end_date'])
                    
                    if data is None:
                        continue
                        
                    # 获取策略类
                    strategy_class = StrategyFactory.get_strategy(params['strategy_name'])
                    if not strategy_class:
                        continue
                        
                    # 设置策略参数
                    strategy_params = {
                        'trail_percent': params['trail_percent'],
                        'risk_ratio': params['risk_ratio'] / 100,
                        'max_drawdown': params['max_drawdown'] / 100,
                    }
                    
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
                    
                    # 如果总盈亏为正，添加到结果列表
                    if results.get('total_pnl', 0) > 0:
                        # 获取该股票在指数中的权重
                        weight = csi_index[csi_index['con_code'] == row['ts_code']]['weight'].iloc[0]
                        positive_stocks.append({
                            '股票代码': row['ts_code'],
                            '股票名称': row['name'],
                            '总盈亏': results['total_pnl'],
                            '总收益率': results['total_return'],
                            '年化收益率': results['annualized_return'],
                            '夏普比率': results['sharpe_ratio'],
                            '最大回撤': results['max_drawdown'],
                            '胜率': results['win_rate'],
                            '指数权重(%)': weight
                        })
                        
                except Exception as e:
                    logger.error(f"回测股票 {row['ts_code']} 时出错: {str(e)}")
                    continue
            
            # 完成后设置进度条为100%
            progress_bar.progress(1.0)
            status_text.text(f"{index_name}成分股回测筛选完成！")
            
            # 显示筛选结果
            if positive_stocks:
                st.subheader(f"{index_name}成分股回测筛选结果")
                # 将结果转换为DataFrame并排序
                results_df = pd.DataFrame(positive_stocks)
                results_df = results_df.sort_values('总收益率', ascending=False)
                
                # 格式化百分比列
                for col in ['总收益率', '年化收益率', '最大回撤', '胜率']:
                    results_df[col] = results_df[col].apply(lambda x: f"{x:.2%}")
                
                # 格式化其他数值列
                results_df['总盈亏'] = results_df['总盈亏'].apply(lambda x: f"{x:,.2f}")
                results_df['夏普比率'] = results_df['夏普比率'].apply(lambda x: f"{x:.2f}")
                results_df['指数权重(%)'] = results_df['指数权重(%)'].apply(lambda x: f"{x:.3f}")
                
                # 显示结果表格
                st.dataframe(
                    results_df,
                    use_container_width=True,
                    hide_index=True
                )
                
                # 显示统计信息
                st.subheader("统计信息")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("符合条件的股票数量", len(positive_stocks))
                with col2:
                    st.metric("筛选比例", f"{(len(positive_stocks) / total_stocks * 100):.2f}%")
                with col3:
                    total_weight = sum(float(s['指数权重(%)']) for s in positive_stocks)
                    st.metric("总指数权重", f"{total_weight:.2f}%")
            else:
                st.info("未找到符合条件的股票")
                
        except Exception as e:
            logger.error(f"{index_name}成分股回测筛选过程中出现错误: {str(e)}")
            st.error(f"{index_name}成分股回测筛选失败: {str(e)}")
            import traceback
            traceback.print_exc()

def render_backtest(params):
    """渲染回测界面"""
    # 如果是市场情绪策略，显示回测筛选按钮
    if params['strategy_name'] == "市场情绪策略":
        # 添加指数筛选按钮
        index_options = {
            "上证50": "000016.SH",
            "沪深300": "000300.SH",
            "中证500": "000905.SH"
        }
        selected_index = st.selectbox("选择指数进行成分股批量回测", options=list(index_options.keys()))
        
        if st.button(f"{selected_index}成分股回测", type="secondary"):
            if not params['tushare_token']:
                st.error(f"{selected_index}成分股回测需要提供Tushare Token")
                return
            
            screen_index_stocks(params, index_options[selected_index], selected_index)
    
    if st.button("开始回测", type="primary"):
        # 检查市场情绪策略的token
        if params['strategy_name'] == "市场情绪策略" and not params['tushare_token']:
            st.error("市场情绪策略必须提供Tushare Token")
            return
        
        # 检查双均线对冲策略的token
        if params['strategy_name'] == "双均线对冲策略" and not params['tushare_token']:
            st.error("双均线对冲策略必须提供Tushare Token")
            return
            
        with st.spinner("正在进行回测..."):
            try:
                # 下载数据
                logger.info(f"开始下载数据 - 股票代码: {params['selected_etfs'] if params['strategy_name'] == 'ETF轮动策略' else params['symbol']}, 开始日期: {params['start_date']}, 结束日期: {params['end_date']}")
                data_loader = DataLoader(tushare_token=params['tushare_token'])
                
                # 根据策略类型下载数据
                if params['strategy_name'] == "ETF轮动策略":
                    data = data_loader.download_data(params['selected_etfs'], params['start_date'], params['end_date'])
                    # 打印每个数据源的ETF代码
                    if isinstance(data, list):
                        for d in data:
                            etf_code = d.params.ts_code if hasattr(d, 'params') and hasattr(d.params, 'ts_code') else '未知'
                            logger.info(f"加载ETF数据源: {etf_code}")
                elif params['strategy_name'] == "双均线对冲策略":
                    # 加载ETF数据
                    etf_data = data_loader.download_data(params['symbol'], params['start_date'], params['end_date'])
                    
                    # 加载豆粕期货数据
                    future_loader = FutureDataLoader(
                        start_date=params['start_date'], 
                        end_date=params['end_date'],
                        token=params['tushare_token']
                    )
                    future_data = future_loader.load()
                    
                    # 将两个数据源组合
                    data = [etf_data, future_data]
                    logger.info(f"加载ETF数据源: {etf_data._name} 和期货数据源: {future_data._name}")
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
                if params['strategy_name'] == "双均线策略" or params['strategy_name'] == "双均线对冲策略":
                    strategy_params.update({
                        'fast_period': params['fast_period'],
                        'slow_period': params['slow_period'],
                        'atr_loss_multiplier': params['atr_loss_multiplier'],
                        'atr_profit_multiplier': params['atr_profit_multiplier'],
                        'enable_trailing_stop': params['enable_trailing_stop'],
                        'enable_death_cross': params['enable_death_cross'],
                    })
                
                # 如果是双均线对冲策略，添加对冲相关参数
                if params['strategy_name'] == "双均线对冲策略":
                    strategy_params.update({
                        'hedge_contract_size': params['hedge_contract_size'],
                        'hedge_profit_multiplier': params['hedge_profit_multiplier'],
                        'enable_hedging': params['enable_hedging'],
                        'enable_ma_cross_hedge': params['enable_ma_cross_hedge'],
                        'enable_macd_hedge': params['enable_macd_hedge'],
                        'enable_sync_long_hedge': params['enable_sync_long_hedge'],
                    })
                
                # 如果是ETF轮动策略，添加特定参数
                if params['strategy_name'] == "ETF轮动策略":
                    strategy_params.update({
                        'momentum_short': params['momentum_short'],
                        'momentum_long': params['momentum_long'],
                        'rebalance_interval': params['rebalance_interval'],
                        'num_positions': params['num_positions'],
                        'profit_target1': params['profit_target1'],
                        'profit_target2': params['profit_target2'],
                        'market_trend_threshold': params['market_trend_threshold'],
                        'vix_threshold': params['vix_threshold'],
                        'momentum_decay': params['momentum_decay'],
                        'atr_multiplier': params['atr_multiplier'],
                    })
                
                # 如果是市场情绪策略或双均线对冲策略，添加tushare token
                if params['strategy_name'] in ["市场情绪策略", "双均线对冲策略"]:
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
                
                # 检查是否是双账户策略
                if 'is_dual_account' in results and results['is_dual_account']:
                    st.header("双账户回测结果")
                    
                    # ETF账户信息
                    st.subheader("ETF账户")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("ETF账户最终资金", f"{results['etf_value']:.2f}")
                    with col2:
                        st.metric("ETF账户收益率", f"{results['etf_return']:.2%}")
                    with col3:
                        st.metric("ETF初始资金", "100,000.00")
                        
                    # 期货账户信息
                    st.subheader("期货账户")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("期货账户最终资金", f"{results['future_value']:.2f}")
                    with col2:
                        st.metric("期货账户收益率", f"{results['future_return']:.2%}")
                    with col3:
                        st.metric("期货初始资金", "100,000.00")
                        
                    # 总体表现
                    st.subheader("总体表现")
                else:
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
                    st.metric("年化收益率", f"{results['annualized_return']:.2%}")
                with col3:
                    st.metric("夏普比率", f"{results['sharpe_ratio']:.2f}")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("最大回撤", f"{results['max_drawdown']:.2%}")
                with col2:
                    st.metric("胜率", f"{results['win_rate']:.2%}")
                with col3:
                    st.metric("盈亏比", f"{results['profit_factor']:.2f}")
                
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

                # 添加情绪指标图表
                try:
                    with open('cache/sentiment_data.json', 'r', encoding='utf-8') as f:
                        sentiment_data = json.load(f)
                    
                    # 从sentiment数组创建DataFrame
                    sentiment_df = pd.DataFrame(sentiment_data['sentiment'])
                    
                    # 将value列重命名为sentiment
                    sentiment_df = sentiment_df.rename(columns={'value': 'sentiment'})
                    
                    # 转换日期格式
                    sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
                    
                    # 按日期排序
                    sentiment_df = sentiment_df.sort_values('date')
                    
                    fig_sentiment = go.Figure()
                    fig_sentiment.add_trace(go.Scatter(
                        x=sentiment_df['date'],
                        y=sentiment_df['sentiment'],
                        mode='lines',
                        name='市场情绪',
                        line=dict(color='purple')
                    ))
                    
                    fig_sentiment.update_layout(
                        title='市场情绪指标',
                        xaxis_title='日期',
                        yaxis_title='情绪值',
                        hovermode='x unified',
                        xaxis=dict(tickformat='%Y年%m月%d日'),
                        # 添加网格线使图表更清晰
                        xaxis_gridcolor='lightgrey',
                        yaxis_gridcolor='lightgrey',
                        plot_bgcolor='white'
                    )
                    
                    st.plotly_chart(fig_sentiment, use_container_width=True)
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    if params['strategy_name'] == "市场情绪策略":
                        logger.error(f"加载情绪指标数据失败: {str(e)}")
                        logger.error(f"数据内容: {sentiment_data if 'sentiment_data' in locals() else '未加载'}")
                        st.warning("无法加载情绪指标数据，请检查数据格式")

                # 添加总资产变化图表
                if not trades.empty:                    
                    # 从交易记录中获取总资产数据
                    trades['交易时间'] = pd.to_datetime(trades['交易时间'])
                    trades['总资产'] = pd.to_numeric(trades['总资产'], errors='coerce')
                    
                    # 创建图表
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=trades['交易时间'],
                        y=trades['总资产'],
                        mode='lines',
                        name='总资产',
                        line=dict(color='blue')
                    ))
                    
                    # 更新布局
                    fig.update_layout(
                        title='总资产变化',
                        xaxis_title='日期',
                        yaxis_title='总资产',
                        hovermode='x unified',
                        # 设置y轴格式为完整数字
                        yaxis=dict(
                            tickformat='.0f'
                        ),
                        # 设置x轴格式为中文日期
                        xaxis=dict(
                            tickformat='%Y年%m月%d日'
                        )
                    )
                    
                    # 显示图表
                    st.plotly_chart(fig, use_container_width=True)
                
                # 使用新的Plotly可视化
                fig = engine.plot()
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                logger.error(f"回测过程中出现错误: {str(e)}")
                import traceback
                traceback.print_exc()
                st.error(f"回测失败: {str(e)}") 