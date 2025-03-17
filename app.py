import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import os
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from bokeh.plotting import figure
from bokeh.layouts import column

from src.strategies.strategy_factory import StrategyFactory
from src.data.data_loader import DataLoader, PandasData
from src.utils.backtest_engine import BacktestEngine
from src.utils.logger import setup_logger

# 设置日志
logger = setup_logger()

# 设置页面
st.set_page_config(
    page_title="ETF策略回测系统",
    layout="wide",
    initial_sidebar_state="expanded"
)

def plot_equity_curve(df):
    """绘制资金曲线"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['equity_curve'],
        mode='lines',
        name='资金曲线',
        line=dict(color='#00b894')
    ))
    fig.update_layout(
        title='策略资金曲线',
        xaxis_title='日期',
        yaxis_title='资金',
        template='plotly_white'
    )
    return fig

def main():
    st.title("ETF策略回测系统")
    
    # 添加系统介绍
    with st.expander("ETF策略回测系统", expanded=True):
        st.markdown("""
        ### 🎯 系统功能
        这是一个专业的 ETF 量化交易策略回测系统，支持多种交易策略的回测和分析。系统具有以下特点：
        
        - 📊 **实时数据**：支持通过 Tushare（专业版）或 AKShare（免费）获取实时行情数据
        - 🚀 **多策略支持**：采用工厂模式设计，支持多种交易策略，便于扩展
        - 📈 **可视化分析**：使用 Plotly 提供交互式图表，包括 K 线、均线、交易点位等
        - ⚠️ **风险控制**：内置追踪止损、最大回撤限制等风险控制机制
        - 💰 **费用模拟**：精确计算交易费用，包括佣金等
        - 📝 **详细日志**：记录每笔交易的详细信息，便于分析和优化
       
        ### ⚠️ 风险提示
        本系统仅供学习和研究使用，不构成任何投资建议。使用本系统进行实盘交易需要自行承担风险。
        """)
    
    # 侧边栏参数设置
    with st.sidebar:
        st.header("策略参数设置")
        
        # 策略选择
        st.subheader("策略选择")
        strategy_name = st.selectbox(
            "选择策略",
            options=StrategyFactory.get_strategy_names(),
            index=0
        )
        
        # 数据源设置
        st.subheader("数据源配置")
        if strategy_name == "市场情绪策略":
            tushare_token = st.text_input("Tushare Token（必填）", type="password", help="市场情绪策略需要使用Tushare数据源")
            if not tushare_token:
                st.error("市场情绪策略必须提供Tushare Token")
        else:
            tushare_token = st.text_input("Tushare Token（可选，如不填则使用akshare）", type="password")
        symbol = st.text_input("ETF代码", value="510050.SH", help="支持：A股(000001.SZ)、ETF(510300.SH)、港股(00700.HK)")
        
        # 移动平均线参数（仅在选择双均线策略时显示）
        if strategy_name == "双均线策略":
            st.subheader("均线参数")
            col1, col2 = st.columns(2)
            with col1:
                fast_period = st.number_input("快线周期", value=5, min_value=1)
            with col2:
                slow_period = st.number_input("慢线周期", value=30, min_value=1)
        
        # 风险控制参数
        st.subheader("风险控制")
        trail_percent = st.slider("追踪止损比例(%)", 0.5, 5.0, 2.0, 0.1)
        risk_ratio = st.slider("单次交易风险比例(%)", 0.5, 5.0, 2.0, 0.1)
        max_drawdown = st.slider("最大回撤限制(%)", 5.0, 30.0, 15.0, 1.0)
            
        # 回测区间
        st.subheader("回测区间")
        start_date = st.date_input(
            "开始日期",
            datetime.now() - timedelta(days=365)
        )
        end_date = st.date_input("结束日期", datetime.now())
        
        # 资金设置
        st.subheader("资金设置")
        initial_cash = st.number_input("初始资金", value=100000.0, min_value=1000.0)
        commission = st.number_input("佣金费率（双向收取，默认万分之2.5）", value=0.00025, min_value=0.0, max_value=0.01, format="%.5f",
                                   help="双向收取，例如：0.00025表示万分之2.5")
        
    # 主界面
    if st.button("开始回测", type="primary"):
        # 检查市场情绪策略的token
        if strategy_name == "市场情绪策略" and not tushare_token:
            st.error("市场情绪策略必须提供Tushare Token")
            return
            
        with st.spinner("正在进行回测..."):
            try:
                # 下载数据
                logger.info(f"开始下载数据 - 股票代码: {symbol}")
                data_loader = DataLoader(tushare_token=tushare_token)
                df = data_loader.download_data(symbol, start_date, end_date)
                
                if df.empty:
                    st.error("未获取到数据，请检查股票代码和日期范围")
                    return
                    
                # 创建数据源
                data = PandasData(dataname=df)
                
                # 获取策略类
                strategy_class = StrategyFactory.get_strategy(strategy_name)
                if not strategy_class:
                    st.error(f"未找到策略: {strategy_name}")
                    return
                
                # 设置策略参数
                strategy_params = {
                    'trail_percent': trail_percent,
                    'risk_ratio': risk_ratio / 100,
                    'max_drawdown': max_drawdown / 100,
                }
                
                # 如果是双均线策略，添加特定参数
                if strategy_name == "双均线策略":
                    strategy_params.update({
                        'fast_period': fast_period,
                        'slow_period': slow_period,
                    })
                
                # 如果是市场情绪策略，添加tushare token
                if strategy_name == "市场情绪策略":
                    os.environ['TUSHARE_TOKEN'] = tushare_token
                
                # 创建回测引擎
                engine = BacktestEngine(
                    strategy_class,
                    data,
                    cash=initial_cash,
                    commission=commission,
                    strategy_params=strategy_params
                )
                
                # 运行回测
                results = engine.run()
                
                # 显示回测结果
                st.header("回测结果")
                
                # 显示交易统计
                trades = results.get('trades', [])
                trades_df = pd.DataFrame(trades)
                
                # 计算交易统计
                if not trades_df.empty:
                    total_pnl = sum(float(trade.get('pnl', 0)) for trade in trades)
                    # 不再自己计算胜率，而是使用引擎返回的结果
                    total_trades = len(trades)
                else:
                    total_pnl = 0
                    total_trades = 0
                
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
                        # 格式化数据
                        trades_df['time'] = pd.to_datetime(trades_df['time']).dt.strftime('%Y-%m-%d')
                        trades_df['price'] = trades_df['price'].map('{:.3f}'.format)
                        trades_df['avg_price'] = trades_df['avg_price'].map('{:.4f}'.format)
                        trades_df['return'] = trades_df['return'].map('{:.2%}'.format)
                        trades_df['pnl'] = trades_df['pnl'].map('{:.2f}'.format)
                        trades_df['size'] = trades_df['size'].astype(int)
                        
                        # 转换方向
                        trades_df['direction'] = trades_df['direction'].map({'Long': '买入', 'Short': '卖出'})
                        
                        # 重命名列并选择要显示的列
                        display_df = trades_df[['time', 'direction', 'price', 'avg_price', 'size', 'pnl', 'return', 'reason']]
                        display_df = display_df.rename(columns={
                            'time': '交易时间',
                            'direction': '方向',
                            'price': '成交价',
                            'avg_price': '持仓均价',
                            'size': '数量',
                            'pnl': '盈亏',
                            'return': '收益率',
                            'reason': '交易原因'
                        })
                        
                        # 显示交易记录表格
                        st.dataframe(
                            display_df,
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
                    st.metric("总费用", f"{results['total_cost']:.2f}")
                with col2:
                    st.metric("佣金", f"{results['total_commission']:.2f}")
                
                # 绘制回测图表
                st.subheader("回测结果图表")
                
                # 使用新的Plotly可视化
                fig = engine.plot()
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                logger.error(f"回测过程中出现错误: {str(e)}")
                st.error(f"回测失败: {str(e)}")

if __name__ == "__main__":
    main() 