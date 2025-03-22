import json
from loguru import logger
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import os
import akshare as ak
from src.data.data_loader import DataLoader
from arch import arch_model
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# 优化1：趋势检测函数
def detect_trend(series, window=14):
    """复合趋势检测（新增ADX指标）"""
    clean_series = series.ffill()
    
    # 计算ADX指标
    high = clean_series.rolling(window).max()
    low = clean_series.rolling(window).min()
    tr = pd.DataFrame({'high': high, 'low': low, 'close': clean_series}).apply(
        lambda x: max(x.high - x.low, 
                     abs(x.high - x.close), 
                     abs(x.low - x.close)), axis=1)
    
    plus_dm = (clean_series.diff().apply(lambda x: x if x > 0 else 0)).rolling(window).mean()
    minus_dm = (clean_series.diff().apply(lambda x: -x if x < 0 else 0)).rolling(window).mean()
    
    dx = 100 * (plus_dm - minus_dm).abs() / (plus_dm + minus_dm).replace(0, 0.001)
    adx = dx.rolling(window).mean()
    
    # 双均线交叉增强（增加发散程度判断）
    fast_ma = clean_series.ewm(span=5, adjust=False).mean()
    slow_ma = clean_series.ewm(span=20, adjust=False).mean()
    ma_spread = (fast_ma - slow_ma) / slow_ma * 100
    
    # 动量指标改进（加入速度变化率）
    momentum = (clean_series / clean_series.shift(12) - 1) * 100
    momentum_acc = momentum.diff(3)  # 3日动量加速度
    
    # 综合信号（增加ADX强度判断）
    trend = np.select(
        [
            (fast_ma > slow_ma) & (adx > 25) & (ma_spread > 1.5),
            (fast_ma < slow_ma) & (adx > 25) & (ma_spread < -1.5)
        ],
        [3, -3],  # 增强趋势信号强度
        default=np.select(
            [
                (fast_ma > slow_ma) & (momentum_acc > 0),
                (fast_ma < slow_ma) & (momentum_acc < 0)
            ],
            [2, -2],
            default=0
        )
    )
    return pd.Series(trend, index=series.index).fillna(0)

# 优化2：混合归一化改进
def hybrid_normalize(series, window=30, decay_factor=0.9):
    """动态衰减历史极值归一化"""
    clean_series = series.ffill().bfill()
    
    # 自适应极值衰减机制
    adjusted_high = clean_series.expanding().max() * decay_factor
    adjusted_low = clean_series.expanding().min() * (2 - decay_factor)
    
    # 结合滚动窗口和长期极值
    roll_high = clean_series.rolling(window, min_periods=1).max()
    roll_low = clean_series.rolling(window, min_periods=1).min()
    
    q_high = np.maximum(roll_high, adjusted_high)
    q_low = np.minimum(roll_low, adjusted_low)
    
    # 动态调整极值范围
    range_adjust = (q_high - q_low) * 0.1
    q_high += range_adjust
    q_low -= range_adjust
    
    diff = q_high - q_low
    diff = np.where(diff < 1e-6, 1.0, diff)
    
    norm = 100 * (clean_series - q_low) / diff
    return pd.Series(np.clip(norm, 0, 100), index=series.index)

# 优化3：钝化函数调优
def smooth_plateau(raw_score, trend_strength):
    """趋势强度自适应钝化"""
    raw_score = np.clip(raw_score, 0, 100)
    
    # 趋势强度影响因子（0-1之间）
    trend_factor = np.clip(abs(trend_strength) / 3, 0, 1)
    
    # 动态调整压缩区间
    if raw_score <= 60 + 20*trend_factor:
        return raw_score * (1 + 0.2*trend_factor)  # 增强趋势期的低分段
    elif raw_score <= 90 + 5*trend_factor:
        return 60 + (30 + 5*trend_factor)*(raw_score-60)/(30 + 20*(1-trend_factor))
    else:
        # 强趋势下保持线性增长
        if trend_factor > 0.7:
            return 90 + (raw_score-90)*(1 + 0.5*trend_factor)
        else:
            return 90 + 15*(1 - np.exp(-0.1*(raw_score-90)))

# 优化4：GARCH模型计算波动率
def calculate_garch_vol(returns, window=60):
    """使用GARCH模型估计条件波动率"""
    vols = []
    for i in range(len(returns)):
        if i < window:
            vols.append(0.01)  # 使用1%的初始波动率而非NA
            continue
        
        try:
            # 使用最近的window个数据拟合GARCH(1,1)模型
            r = returns.iloc[i-window:i].dropna().values
            if len(r) < window/2:  # 数据太少，使用传统方法
                std = returns.iloc[max(0, i-window):i].std()
                vols.append(std if not pd.isna(std) else 0.01)
                continue
                
            # 处理极端值
            r = np.clip(r, -0.1, 0.1)  # 限制收益率范围
            
            model = arch_model(r, vol='Garch', p=1, q=1, rescale=False)
            res = model.fit(disp='off', show_warning=False, options={'maxiter': 100})
            forecast = res.forecast(horizon=1)
            vol = np.sqrt(forecast.variance.values[-1,0])
            vols.append(vol if not np.isnan(vol) and vol > 0 else 0.01)
        except Exception as e:
            # 如果GARCH拟合失败，回退到传统方法
            logger.warning(f"GARCH fitting failed, fallback to traditional method: {str(e)[:100]}")
            std = returns.iloc[max(0, i-window):i].std()
            vols.append(std if not pd.isna(std) else 0.01)
    
    # 确保没有极端值
    vols = np.clip(vols, 0.001, 0.5)  # 限制波动率范围
    return pd.Series(vols, index=returns.index)

# 优化5：RSI权重机制改进
def rsi_smooth_weight(rsi, price_trend):
    """改进的顶背离检测"""
    rsi_clean = rsi.fillna(50)
    price_clean = price_trend.ffill()
    
    # 使用3日平滑价格和RSI
    price_smooth = price_clean.ewm(span=3).mean()
    rsi_smooth = rsi_clean.ewm(span=3).mean()
    
    # 动态窗口检测（趋势强度越大，窗口越长）
    trend_strength = price_clean.diff(5).abs().rolling(10).mean()
    # 处理NA和inf值
    trend_strength = trend_strength.replace([np.inf, -np.inf], np.nan)
    trend_strength = trend_strength.fillna(trend_strength.median())
    
    # 安全地计算动态窗口
    median_strength = trend_strength.median()
    if median_strength == 0:
        median_strength = 1  # 避免除零
    
    dynamic_window = np.clip((trend_strength / median_strength).astype(int)*10, 20, 60)
    
    divergence = pd.Series(False, index=rsi.index)
    for i in range(len(rsi)):
        if i < 30: continue
        window_size = int(dynamic_window.iloc[i])
        lookback = max(0, i - window_size)
        
        price_window = price_smooth.iloc[lookback:i+1]
        rsi_window = rsi_smooth.iloc[lookback:i+1]
        
        price_current = price_smooth.iloc[i]
        rsi_current = rsi_smooth.iloc[i]
        
        # 价格新高条件放宽（允许1%的波动）
        price_condition = (price_current >= price_window.max() * 0.99)
        # RSI条件收紧（必须低于前高的95%）
        rsi_condition = (rsi_current < rsi_window.max() * 0.95)
        
        divergence.iloc[i] = price_condition & rsi_condition
    
    base_weight = 1 / (1 + np.exp(-0.15*(rsi_clean - 65)))
    return pd.Series(np.where(divergence, base_weight*0.7, base_weight), index=rsi.index)

def get_sentiment_data(start_date = None, end_date = None):
    """获取市场情绪指标"""
    try:
        # 从环境变量获取token并初始化DataLoader
        import os
        from src.data.data_loader import DataLoader
        
        tushare_token = os.getenv('TUSHARE_TOKEN')
        if not tushare_token:
            raise ValueError("未设置TUSHARE_TOKEN环境变量")
            
        data_loader = DataLoader(tushare_token=tushare_token)
        
        result = {
            'sentiment': []
        }
        # 获取时间范围
        if start_date is None or end_date is None:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365 * 10)
        
        logger.info(f"获取市场情绪数据 - 开始日期: {start_date}, 结束日期: {end_date}")

        # 检查缓存目录是否存在
        if not os.path.exists('cache'):
            os.makedirs('cache')
        
        cache_file = 'cache/sentiment_data.json'
        
        # 尝试从缓存读取
        cache_data_valid = False
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                result = json.loads(f.read())
                
            # 检查缓存数据的时间范围是否覆盖当前回测区间
            if result and 'sentiment' in result and isinstance(result['sentiment'], list):
                # 获取交易日历
                try:
                    from src.data.data_loader import DataLoader
                    tushare_token = os.getenv('TUSHARE_TOKEN')
                    if not tushare_token:
                        raise ValueError("未设置TUSHARE_TOKEN环境变量")
                    data_loader = DataLoader(tushare_token=tushare_token)
                    start_date_str = start_date.strftime('%Y%m%d')
                    end_date_str = end_date.strftime('%Y%m%d')
                    trade_cal = data_loader.pro.trade_cal(exchange='SSE', 
                                        start_date=start_date_str,
                                        end_date=end_date_str,
                                        is_open=1)
                    
                    if not trade_cal.empty:
                        actual_start_date = trade_cal['cal_date'].min()
                        actual_end_date = trade_cal['cal_date'].max()
                        actual_start_date = f"{actual_start_date[:4]}-{actual_start_date[4:6]}-{actual_start_date[6:]}"
                        actual_end_date = f"{actual_end_date[:4]}-{actual_end_date[4:6]}-{actual_end_date[6:]}"
                        
                        # 从sentiment数组中提取日期
                        sentiment_dates = [item['date'] for item in result['sentiment'] if isinstance(item, dict) and 'date' in item]
                        
                        if sentiment_dates:
                            min_date = min(sentiment_dates)
                            max_date = max(sentiment_dates)
                            if min_date <= actual_start_date and max_date >= actual_end_date:
                                cache_data_valid = True
                                logger.info(f"使用缓存的市场情绪数据 - 缓存范围: {min_date} 到 {max_date}")
                            else:
                                logger.info(f"缓存数据时间范围不匹配 - 缓存范围: {min_date} 到 {max_date}, 需要范围: {actual_start_date} 到 {actual_end_date}")
                    else:
                        logger.warning("无法获取交易日历，将使用原始日期范围检查")
                        sentiment_dates = [item['date'] for item in result['sentiment'] if isinstance(item, dict) and 'date' in item]
                        
                        if sentiment_dates:
                            min_date = min(sentiment_dates)
                            max_date = max(sentiment_dates)
                            if min_date <= start_date and max_date >= end_date:
                                cache_data_valid = True
                                logger.info(f"使用缓存的市场情绪数据 - 缓存范围: {min_date} 到 {max_date}")
                            else:
                                logger.info(f"缓存数据时间范围不匹配 - 缓存范围: {min_date} 到 {max_date}, 需要范围: {start_date} 到 {end_date}")
                except Exception as e:
                    logger.error(f"获取交易日历失败: {str(e)}，将使用原始日期范围检查")
                    import traceback
                    traceback.print_exc()
                    sentiment_dates = [item['date'] for item in result['sentiment'] if isinstance(item, dict) and 'date' in item]
                    
                    if sentiment_dates:
                        min_date = min(sentiment_dates)
                        max_date = max(sentiment_dates)
                        if min_date <= start_date and max_date >= end_date:
                            cache_data_valid = True
                            logger.info(f"使用缓存的市场情绪数据 - 缓存范围: {min_date} 到 {max_date}")
                        else:
                            logger.info(f"缓存数据时间范围不匹配 - 缓存范围: {min_date} 到 {max_date}, 需要范围: {start_date} 到 {end_date}")
                return result
            else:
                logger.warning("缓存数据格式不正确或为空")
        
        # 如果缓存不存在、为空或时间范围不匹配，重新获取数据
        if not result or not cache_data_valid:
            logger.info(f"开始获取市场情绪数据 - 开始日期: {start_date}, 结束日期: {end_date}")
        
        try:
            # 获取市场情绪指标
            # 使用tushare获取多个指数日线数据，并设置权重
            index_weights = {
                '000001.SH': 0.2,  # 上证指数
                '000300.SH': 0.5,  # 沪深300指数
                '000016.SH': 0.2,  # 上证50指数
                '399240.SZ': 0.1,  # 金融指数
            }
            index_codes = list(index_weights.keys())
                        
            index_data_list = []
            for ts_code in index_codes:
                try:
                    df = data_loader.pro.index_daily(
                        ts_code=ts_code,
                        start_date=start_date.strftime('%Y%m%d'),
                        end_date=end_date.strftime('%Y%m%d')
                    )
                    
                    # 检查是否为当前交易日且需要添加实时数据
                    today = datetime.now()
                    if end_date.strftime('%Y%m%d') == today.strftime('%Y%m%d') and df is not None and not df.empty:
                        # 检查是否有当天数据
                        today_str = today.strftime('%Y%m%d')
                        if today_str not in df['trade_date'].astype(str).values:
                            try:
                                # 获取实时行情数据
                                # 转换为雪球指数代码格式
                                if ts_code == '000001.SH':
                                    market_code = 'SH000001'
                                elif ts_code == '000300.SH':
                                    market_code = 'SH000300'
                                elif ts_code == '000016.SH':
                                    market_code = 'SH000016'
                                elif ts_code == '399240.SZ':
                                    market_code = 'SZ399240'
                                else:
                                    market_code = ts_code.replace('.SH', 'SH').replace('.SZ', 'SZ')
                                    
                                realtime_data = ak.stock_individual_spot_xq(symbol=market_code, token="d679467b716fd5b0a0af195f7e8143774d271a40")
                                if realtime_data is not None and not realtime_data.empty:
                                    # 将数据转换为以item为索引的格式
                                    realtime_data = realtime_data.set_index('item')
                                    # 检查df中是否已有相同日期的数据
                                    existing_today = df[df['trade_date'].astype(str) == today_str]
                                    if existing_today.empty:
                                        # 创建当天数据行
                                        today_row = {
                                            'ts_code': ts_code,
                                            'trade_date': today_str,
                                            'open': float(realtime_data.loc['今开', 'value']),
                                            'high': float(realtime_data.loc['最高', 'value']),
                                            'low': float(realtime_data.loc['最低', 'value']),
                                            'close': float(realtime_data.loc['现价', 'value']),
                                            'pre_close': float(realtime_data.loc['昨收', 'value']),
                                            'change': float(realtime_data.loc['涨跌', 'value']),
                                            'pct_chg': float(realtime_data.loc['涨幅', 'value']),
                                            'vol': float(realtime_data.loc['成交量', 'value']),
                                            'amount': float(realtime_data.loc['成交额', 'value'])
                                        }
                                        # 添加到数据框
                                        df = pd.concat([df, pd.DataFrame([today_row])], ignore_index=True)
                                        logger.info(f"已添加实时数据到 {ts_code}")
                            except Exception as e:
                                logger.error(f"获取实时数据失败 {ts_code}: {e}")
                                import traceback
                                traceback.print_exc()
                    
                    if df is not None and not df.empty:
                        df['index_code'] = ts_code
                        df['weight'] = index_weights[ts_code]  # 添加权重
                        index_data_list.append(df)
                except Exception as e:
                    logger.error(f"Error fetching data for {ts_code}: {e}")
                    continue
            
            if index_data_list:
                # 合并所有指数数据
                index_data = pd.concat(index_data_list, ignore_index=True)
                # 将日期列转换为datetime
                index_data['trade_date'] = pd.to_datetime(index_data['trade_date'])
                
                # 按指数分组计算技术指标
                sentiment_results = []
                for ts_code in index_codes:
                    single_index = index_data[index_data['index_code'] == ts_code].copy()
                    if single_index.empty:
                        continue
                        
                    # 确保数据按日期排序
                    single_index = single_index.sort_values('trade_date')
                    
                    # 预处理：确保数值列没有NA
                    for col in ['close', 'vol', 'pct_chg']:
                        if col in single_index.columns:
                            single_index[col] = single_index[col].ffill().bfill()
                            # 检测无穷值
                            single_index[col] = single_index[col].replace([np.inf, -np.inf], np.nan).fillna(
                                single_index[col].mean() if not single_index[col].empty else 0
                            )
                    
                    # 使用优化1：复合趋势检测
                    single_index['trend'] = detect_trend(single_index['close'])
                    
                    # 计算技术指标
                    # 使用优化4：GARCH模型波动率
                    returns = single_index['pct_chg'] / 100
                    
                    # 尝试使用GARCH模型，如果失败则回退到原方法
                    try:
                        # 计算条件波动率
                        single_index['conditional_vol'] = calculate_garch_vol(returns) * np.sqrt(252) * 100
                        
                        # 分别计算上涨和下跌的条件波动率
                        positive_cond = returns > 0
                        single_index['positive_volatility'] = single_index['conditional_vol'] * positive_cond
                        single_index['negative_volatility'] = single_index['conditional_vol'] * (~positive_cond)
                    except Exception as e:
                        logger.warning(f"Error in GARCH calculation, fallback to traditional method: {e}")
                        # 回退到原方法计算波动率
                        positive_returns = returns.where(returns > 0, 0)
                        negative_returns = returns.where(returns < 0, 0)
                        single_index['positive_volatility'] = positive_returns.ewm(span=10, adjust=False).std() * np.sqrt(252) * 100
                        single_index['negative_volatility'] = (-negative_returns).ewm(span=10, adjust=False).std() * np.sqrt(252) * 100
                    
                    # 确保波动率没有NA或Inf
                    for col in ['positive_volatility', 'negative_volatility', 'conditional_vol']:
                        if col in single_index.columns:
                            single_index[col] = single_index[col].fillna(0).replace([np.inf, -np.inf], 0)
                    
                    # 2. RSI - 使用双重EMA平滑
                    delta = single_index['close'].diff()
                    gain = delta.where(delta > 0, 0.0)
                    loss = -delta.where(delta < 0, 0.0)
                    # 使用更长的EMA窗口进行平滑
                    avg_gain = gain.ewm(alpha=1/21, adjust=False).mean()  # 21日EMA
                    avg_loss = loss.ewm(alpha=1/21, adjust=False).mean()
                    # 避免除零错误
                    rs = avg_gain / avg_loss.replace(0, 0.000001)
                    single_index['rsi'] = 100 - (100 / (1 + rs))
                    single_index['rsi'] = single_index['rsi'].clip(0, 100)  # 限制RSI在0-100范围内
                    
                    # 使用优化5：改进的RSI权重计算
                    single_index['rsi_weight'] = rsi_smooth_weight(single_index['rsi'], single_index['close'])
                    
                    # 3. 布林带 - 使用EMA平滑
                    sma = single_index['close'].ewm(span=20, adjust=False).mean()
                    std = single_index['close'].ewm(span=20, adjust=False).std()
                    single_index['bb_position'] = (single_index['close'] - sma) / (2 * std.replace(0, 0.000001))
                    single_index['bb_position'] = single_index['bb_position'].clip(-3, 3)  # 限制布林带位置
                    
                    # 4. 成交量变化（使用EMA平滑）
                    single_index['volume_ma'] = single_index['vol'].ewm(span=20, adjust=False).mean()
                    # 避免除零错误
                    single_index['volume_ratio'] = single_index['vol'] / single_index['volume_ma'].replace(0, 0.000001)
                    single_index['volume_ratio'] = single_index['volume_ratio'].clip(0, 5)  # 限制成交量比例
                    
                    # 计算综合情绪分数（考虑指标方向性和趋势）
                    # 新增趋势持续因子（基于价格新高和成交量）
                    price_high_series = single_index['close'].rolling(30, min_periods=1).max()
                    volume_high_series = single_index['vol'].rolling(30, min_periods=1).max()

                    trend_persistence = (
                        (single_index['close'] >= price_high_series*0.99).astype(int) * 
                        (single_index['vol'] >= volume_high_series*0.8).astype(int)
                    ).rolling(5).sum() / 5

                    # 使用优化2：改进的混合归一化函数
                    volatility_score = (
                        hybrid_normalize(single_index['positive_volatility']) * 0.2 +  # 上涨波动率正向贡献
                        (100 - hybrid_normalize(single_index['negative_volatility'])) * 0.2  # 下跌波动率反向处理
                    )
                    
                    # RSI非线性加权（使用混合归一化）
                    rsi_score = hybrid_normalize(single_index['rsi']) * single_index['rsi_weight']
                    
                    # 在原始分数计算中加入趋势持续因子
                    raw_sentiment_score = (
                        volatility_score * 0.35 + 
                        rsi_score * 0.35 +
                        hybrid_normalize(single_index['bb_position']) * 0.1 +
                        hybrid_normalize(single_index['volume_ratio']) * 0.1 +
                        trend_persistence * 20  # 新增20%的持续因子影响
                    ) * (1 + 0.3 * single_index['trend']/3)  # 增强趋势影响
                    
                    # 使用优化3：改进的钝化函数
                    single_index['sentiment_score'] = [
                        smooth_plateau(score, trend) 
                        for score, trend in zip(raw_sentiment_score, single_index['trend'])
                    ]
                    
                    # 最终情绪分二次平滑（5日EMA）
                    single_index['sentiment_score'] = single_index['sentiment_score'].ewm(span=5, adjust=False).mean()
                    
                    # 确保结果中没有NA或inf
                    single_index = single_index.ffill().bfill()
                    # 处理所有可能的inf值
                    for col in single_index.columns:
                        if single_index[col].dtype == 'float64' or single_index[col].dtype == 'int64':
                            single_index[col] = single_index[col].replace([np.inf, -np.inf], np.nan).fillna(
                                single_index[col].mean() if not pd.isna(single_index[col]).all() else 0
                            )
                    
                    # 添加到结果列表
                    for _, row in single_index.iterrows():
                        # 仅当sentiment_score存在且有效时添加
                        if 'sentiment_score' in row and not pd.isna(row['sentiment_score']):
                            sentiment_results.append({
                                'date': row['trade_date'].strftime('%Y-%m-%d'),
                                'value': float(row['sentiment_score']),
                                'weight': float(row['weight']),
                                'details': {
                                    'index_code': ts_code,
                                    'positive_volatility': float(row['positive_volatility']),
                                    'negative_volatility': float(row['negative_volatility']),
                                    'rsi': float(row['rsi']),
                                    'rsi_weight': float(row['rsi_weight']),
                                    'bb_position': float(row['bb_position']),
                                    'volume_ratio': float(row['volume_ratio']),
                                    'trend': int(row['trend']),
                                    'close': float(row['close']),
                                    'change': float(row['pct_chg']),
                                    'conditional_vol': float(row['conditional_vol']) if 'conditional_vol' in row else 0
                                }
                            })
                
                if sentiment_results:
                    # 按日期分组计算加权平均情绪分数
                    sentiment_df = pd.DataFrame(sentiment_results)
                    try:
                        result['sentiment'] = sentiment_df.groupby('date').apply(
                            lambda group: {
                                'date': group['date'].iloc[0],
                                'value': float((group['value'] * group['weight']).sum() / group['weight'].sum()),
                                'details': {
                                    'positive_volatility': float(group['details'].apply(lambda x: x['positive_volatility']).mean()),
                                    'negative_volatility': float(group['details'].apply(lambda x: x['negative_volatility']).mean()),
                                    'rsi': float(group['details'].apply(lambda x: x['rsi']).mean()),
                                    'rsi_weight': float(group['details'].apply(lambda x: x['rsi_weight']).mean()),
                                    'bb_position': float(group['details'].apply(lambda x: x['bb_position']).mean()),
                                    'volume_ratio': float(group['details'].apply(lambda x: x['volume_ratio']).mean()),
                                    'trend': int(round(group['details'].apply(lambda x: x['trend']).mean())),
                                    'close': float(group['details'].apply(lambda x: x['close']).mean()),
                                    'change': float(group['details'].apply(lambda x: x['change']).mean()),
                                    'conditional_vol': float(group['details'].apply(lambda x: x.get('conditional_vol', 0)).mean()),
                                    'indices': [
                                        {
                                            'code': detail['index_code'],
                                            'close': float(detail['close']),
                                            'change': float(detail['change']),
                                            'positive_volatility': float(detail['positive_volatility']),
                                            'negative_volatility': float(detail['negative_volatility']),
                                            'rsi': float(detail['rsi']),
                                            'rsi_weight': float(detail['rsi_weight']),
                                            'bb_position': float(detail['bb_position']),
                                            'volume_ratio': float(detail['volume_ratio']),
                                            'trend': int(detail['trend']),
                                            'conditional_vol': float(detail.get('conditional_vol', 0))
                                        }
                                        for detail in group['details']
                                    ]
                                }
                            }
                        ).tolist()
                    except Exception as e:
                        logger.error(f"Error in groupby aggregation: {e}")
                        # 回退方案：简单返回不分组的结果
                        result['sentiment'] = [
                            {
                                'date': item['date'],
                                'value': float(item['value']),
                                'details': item['details']
                            }
                            for item in sentiment_results
                        ]
                
                # 按日期排序
                result['sentiment'].sort(key=lambda x: x['date'])
                
            else:
                result['sentiment'] = []
                logger.warning("No index data available for sentiment calculation")
        except Exception as e:
            logger.error(f"Error calculating sentiment indicators: {e}")
            import traceback
            logger.error(traceback.format_exc())  # 打印完整堆栈跟踪
            result['sentiment'] = []
        
        # 对所有数据按日期排序
        for key in result:
            result[key].sort(key=lambda x: x['date'])

        if result and 'sentiment' in result and isinstance(result['sentiment'], list):
            # 写入缓存
            with open(cache_file, 'w') as f:
                json.dump(result, f)
            logger.info("成功更新市场情绪数据缓存")
        else:
            return None

        return result

    except Exception as e:
        logger.error(f"Error in get_sentiment_data: {e}")
        import traceback
        logger.error(traceback.format_exc())  # 打印完整堆栈跟踪
        return None