import streamlit as st
import pandas as pd
import os
from datetime import datetime, timedelta
from src.utils.logger import setup_logger
from src.trading.market_executor import MarketExecutor
import threading
import tushare as ts

logger = setup_logger()

def start_trading(params):
    """启动实盘交易"""
    try:
        # 从session state获取tushare token
        if not params['tushare_token']:
            st.error("请先在侧边栏设置Tushare Token")
            return
            
        # 设置tushare token
        ts.set_token(params['tushare_token'])
        pro = ts.pro_api()
            
        # 获取上证50成分股
        today = datetime.now().strftime('%Y%m%d')
        sz50 = pro.index_weight(index_code='000016.SH', trade_date=today)
        if sz50.empty:
            # 如果当天数据不可用，尝试获取最近的数据
            dates = pro.trade_cal(exchange='', start_date=(datetime.now() - timedelta(days=30)).strftime('%Y%m%d'), 
                                end_date=today, is_open='1')
            for date in sorted(dates['cal_date'].tolist(), reverse=True):
                sz50 = pro.index_weight(index_code='000016.SH', trade_date=date)
                if not sz50.empty:
                    break
        
        if sz50.empty:
            st.error("未获取到上证50成分股列表")
            return
        symbols = sz50['con_code'].tolist()
        logger.info(f"上证50成分股: {symbols}")
        
        # 创建并启动MarketExecutor
        executor = MarketExecutor(symbols, tushare_token=params['tushare_token'])
        thread = threading.Thread(target=executor.run_continuously)
        thread.daemon = True
        thread.start()
        
        st.session_state.trading_thread = thread
        st.session_state.is_trading = True
        st.success("实盘交易已启动")
        
    except Exception as e:
        st.error(f"启动实盘交易失败: {str(e)}")
        logger.error(f"启动实盘交易失败: {str(e)}")
        import traceback
        traceback.print_exc()

def stop_trading():
    """停止实盘交易"""
    try:
        if hasattr(st.session_state, 'trading_thread'):
            # 这里需要实现一个优雅的停止机制
            st.session_state.is_trading = False
            st.session_state.trading_thread.join(timeout=5)
            st.success("实盘交易已停止")
    except Exception as e:
        st.error(f"停止实盘交易失败: {str(e)}")
        logger.error(f"停止实盘交易失败: {str(e)}")

def render_market(params):
    st.header("实盘交易记录")
    
    # 创建交易记录目录
    records_dir = "data/trading_records"
    os.makedirs(records_dir, exist_ok=True)
    
    # 初始化session state
    if 'is_trading' not in st.session_state:
        st.session_state.is_trading = False
    
    # 添加启动/停止按钮
    col1, col2 = st.columns(2)
    with col1:
        if not st.session_state.is_trading:
            if st.button("启动实盘", type="primary"):
                start_trading(params)
    with col2:
        if st.session_state.is_trading:
            if st.button("停止实盘", type="secondary"):
                stop_trading()
    
    # 显示当前状态
    status = "运行中" if st.session_state.is_trading else "已停止"
    st.info(f"实盘交易状态: {status}")
    
    # 获取最新的交易记录文件
    record_files = [f for f in os.listdir(records_dir) if f.endswith('.csv')]
    if not record_files:
        st.info("暂无交易记录")
        return
        
    latest_file = max(record_files, key=lambda x: os.path.getctime(os.path.join(records_dir, x)))
    
    # 读取并显示交易记录
    try:
        df = pd.read_csv(os.path.join(records_dir, latest_file))
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp', ascending=False)
        
        # 显示交易统计
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("总交易次数", len(df))
        with col2:
            st.metric("买入次数", len(df[df['action'] == 'buy']))
        with col3:
            st.metric("卖出次数", len(df[df['action'] == 'sell']))
            
        # 显示交易记录表格
        st.dataframe(df)
        
    except Exception as e:
        st.error(f"读取交易记录时出错: {str(e)}")
        logger.error(f"读取交易记录时出错: {str(e)}")
