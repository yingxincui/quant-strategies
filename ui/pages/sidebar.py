import streamlit as st
from datetime import datetime, timedelta
from src.strategies.strategy_factory import StrategyFactory

def render_sidebar():
    """渲染侧边栏"""
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
        if strategy_name in ["市场情绪策略", "双均线对冲策略"]:
            tushare_token = st.text_input("Tushare Token（必填）", value="", type="password", help="市场情绪策略需要使用Tushare数据源")
            if not tushare_token:
                st.error(f"{strategy_name}必须提供Tushare Token")
        else:
            tushare_token = st.text_input("Tushare Token（可选，如不填则使用akshare）", type="password")
            
        # ETF轮动策略的ETF选择
        if strategy_name == "ETF轮动策略":
            etf_list = [
                '510050.SH',  # 上证50ETF
                '510300.SH',  # 沪深300ETF
                '510500.SH',  # 中证500ETF
                '159915.SZ',  # 创业板ETF
                '512880.SH',  # 证券ETF
                '512690.SH',  # 酒ETF
                '512660.SH',  # 军工ETF
                '512010.SH',  # 医药ETF
                '512800.SH',  # 银行ETF
                '512170.SH',  # 医疗ETF
                '512760.SH',  # 芯片ETF
                '159928.SZ',  # 消费ETF
                '512480.SH',  # 半导体ETF
                '512980.SH',  # 科技ETF
                '512580.SH',  # 环保ETF
                '512400.SH',  # 有色金属ETF
                '512200.SH',  # 地产ETF
                '516160.SH',  # 新能源车ETF
                '159939.SZ',  # 信息技术ETF
                '512600.SH',  # 主要消费ETF
                '512070.SH',  # 证券保险ETF
                '159869.SZ',  # 新基建ETF
                '515030.SH',  # 新能源ETF
                '515790.SH',  # 光伏ETF
                '513050.SH',  # 中概互联ETF
            ]
            selected_etfs = st.multiselect(
                "选择ETF",
                options=etf_list,
                default=etf_list[:5],  # 默认选择前5个ETF
                help="选择要轮动的ETF，建议选择3-5个相关性较低的ETF"
            )
            if not selected_etfs:
                st.error("请至少选择一个ETF")
                return None
        else:
            if strategy_name == "双均线对冲策略":
                symbol = st.text_input("ETF代码", value="159985.SZ", help="使用M豆粕主力合约作为被对冲标的")
            else:
                symbol = st.text_input("ETF代码", value="510050.SH", help="支持：A股(000001.SZ)、ETF(510300.SH)、港股(00700.HK)")
        
        # 移动平均线参数（对双均线策略和双均线对冲策略显示）
        if strategy_name in ["双均线策略", "双均线对冲策略"]:
            st.subheader("均线参数")
            col1, col2 = st.columns(2)
            with col1:
                fast_period = st.number_input("快线周期", value=5, min_value=1)
            with col2:
                slow_period = st.number_input("慢线周期", value=13, min_value=1)
                
            # 止损止盈参数
            st.subheader("止损止盈参数")
            col1, col2 = st.columns(2)
            with col1:
                atr_multiplier = st.number_input("ATR倍数", value=1.0, min_value=0.5, max_value=5.0, step=0.1, 
                                               help="ATR倍数用于计算止损距离，值越大止损距离越远")
            with col2:
                atr_period = st.number_input("ATR周期", value=14, min_value=5, max_value=30, step=1)
                
            # 其他技术参数
            col1, col2 = st.columns(2)
            with col1:
                enable_trailing_stop = st.checkbox("启用追踪止损", value=False, 
                                                help="追踪止损会在价格创新高后设置止损价，可以锁定更多利润")
            with col2:
                enable_death_cross = st.checkbox("启用死叉卖出", value=False,
                                              help="启用后，当快线下穿慢线时将卖出持仓")
            
        # 双均线对冲策略特有参数
        if strategy_name == "双均线对冲策略":
            st.subheader("对冲参数")
            col1, col2 = st.columns(2)
            with col1:
                hedge_contract_size = st.number_input("对冲合约手数", value=10, min_value=1, max_value=100, step=1,
                                                   help="豆粕期货对冲合约手数，建议10-20手")
            with col2:
                hedge_profit_multiplier = st.number_input("对冲盈利倍数", value=1.0, min_value=0.5, max_value=5.0, step=0.1,
                                                       help="对冲目标盈利 = 原始损失 × (1 + 盈利倍数)")
        
        # ETF轮动策略参数
        if strategy_name == "ETF轮动策略":
            st.subheader("轮动参数")
            col1, col2 = st.columns(2)
            with col1:
                momentum_short = st.number_input("短期动量周期", value=10, min_value=1)
            with col2:
                momentum_long = st.number_input("长期动量周期", value=60, min_value=1)
            col1, col2 = st.columns(2)
            with col1:
                rebalance_interval = st.number_input("调仓间隔(天)", value=20, min_value=1)
            with col2:
                num_positions = st.number_input("持仓数量", value=3, min_value=1, max_value=10)
            
            # 止盈止损参数
            st.subheader("止盈止损参数")
            col1, col2 = st.columns(2)
            with col1:
                profit_target1 = st.number_input("第一止盈目标(%)", value=5.0, min_value=1.0, max_value=100.0, step=1.0)
            with col2:
                profit_target2 = st.number_input("第二止盈目标(%)", value=10.0, min_value=1.0, max_value=100.0, step=1.0)
            
            # 市场状态参数
            st.subheader("市场状态参数")
            col1, col2 = st.columns(2)
            with col1:
                market_trend_threshold = st.number_input("市场趋势阈值(%)", value=-5.0, min_value=-20.0, max_value=0.0, step=1.0)
            with col2:
                vix_threshold = st.number_input("波动率阈值(%)", value=3.0, min_value=1.0, max_value=10.0, step=0.5)
            
            # 动量衰减参数
            momentum_decay = st.slider("动量衰减阈值(%)", 10.0, 50.0, 30.0, 1.0)
            atr_multiplier = st.slider("ATR倍数", 1.0, 3.0, 2.0, 0.1)
        
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
        
        # 返回所有参数
        params = {
            'strategy_name': strategy_name,
            'tushare_token': tushare_token,
            'selected_etfs': selected_etfs if strategy_name == "ETF轮动策略" else None,
            'symbol': symbol if strategy_name != "ETF轮动策略" else None,
            'fast_period': fast_period if strategy_name in ["双均线策略", "双均线对冲策略"] else None,
            'slow_period': slow_period if strategy_name in ["双均线策略", "双均线对冲策略"] else None,
            'atr_multiplier': atr_multiplier if strategy_name in ["双均线策略", "双均线对冲策略", "ETF轮动策略"] else None,
            'atr_period': atr_period if strategy_name in ["双均线策略", "双均线对冲策略"] else None,
            'enable_trailing_stop': enable_trailing_stop if strategy_name in ["双均线策略", "双均线对冲策略"] else None,
            'enable_death_cross': enable_death_cross if strategy_name in ["双均线策略", "双均线对冲策略"] else None,
            'hedge_contract_size': hedge_contract_size if strategy_name == "双均线对冲策略" else None,
            'hedge_profit_multiplier': hedge_profit_multiplier if strategy_name == "双均线对冲策略" else None,
            'momentum_short': momentum_short if strategy_name == "ETF轮动策略" else None,
            'momentum_long': momentum_long if strategy_name == "ETF轮动策略" else None,
            'rebalance_interval': rebalance_interval if strategy_name == "ETF轮动策略" else None,
            'num_positions': num_positions if strategy_name == "ETF轮动策略" else None,
            'profit_target1': profit_target1 / 100 if strategy_name == "ETF轮动策略" else None,
            'profit_target2': profit_target2 / 100 if strategy_name == "ETF轮动策略" else None,
            'market_trend_threshold': market_trend_threshold / 100 if strategy_name == "ETF轮动策略" else None,
            'vix_threshold': vix_threshold / 100 if strategy_name == "ETF轮动策略" else None,
            'momentum_decay': momentum_decay / 100 if strategy_name == "ETF轮动策略" else None,
            'trail_percent': trail_percent,
            'risk_ratio': risk_ratio,
            'max_drawdown': max_drawdown,
            'start_date': start_date,
            'end_date': end_date,
            'initial_cash': initial_cash,
            'commission': commission
        }
        
        return params 