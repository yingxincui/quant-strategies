from loguru import logger
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import os
from src.data.data_loader import DataLoader
# 添加arch库导入支持GARCH模型
from arch import arch_model

def get_sentiment_data(start_date = None, end_date = None):
    """获取市场情绪指标"""
    try:
        # 从环境变量获取token并初始化DataLoader
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
            
            # 优化2：混合归一化改进
            def hybrid_normalize(series, short_window=30, long_window=180):
                """改进的动态窗口归一化"""
                # 自适应窗口调整（根据近期波动率）
                volatility = series.diff().abs().rolling(10).std()
                # 修复：确保动态窗口始终有有效整数值，处理NA和inf
                volatility = volatility.fillna(0)  # 填充NA为0
                volatility = np.clip(volatility, 0, 0.1)  # 限制波动率范围，避免极端值
                
                # 计算动态窗口并确保为整数
                dynamic_short = np.maximum(10, np.minimum(60, (short_window * (1 + volatility/0.02))))
                dynamic_short = dynamic_short.astype(int)  # 确保转换为整数
                
                # 滚动极值计算（前向窗口）
                def forward_max(s, window_series):
                    """使用变长窗口计算前向最大值"""
                    result = pd.Series(index=s.index)
                    for i in range(len(s)):
                        if pd.isna(s.iloc[i]):
                            result.iloc[i] = np.nan
                            continue
                        
                        # 确保窗口大小为有效整数
                        window = max(1, int(window_series.iloc[i]))
                        start_idx = max(0, i - window + 1)
                        window_data = s.iloc[start_idx:i+1]
                        result.iloc[i] = window_data.max() if not window_data.empty else s.iloc[i]
                    return result
                
                def forward_min(s, window_series):
                    """使用变长窗口计算前向最小值"""
                    result = pd.Series(index=s.index)
                    for i in range(len(s)):
                        if pd.isna(s.iloc[i]):
                            result.iloc[i] = np.nan
                            continue
                        
                        # 确保窗口大小为有效整数
                        window = max(1, int(window_series.iloc[i]))
                        start_idx = max(0, i - window + 1)
                        window_data = s.iloc[start_idx:i+1]
                        result.iloc[i] = window_data.min() if not window_data.empty else s.iloc[i]
                    return result
                
                # 处理series中的NA值
                clean_series = series.copy()
                clean_series = clean_series.fillna(method='ffill').fillna(method='bfill')
                
                # 动态极值计算
                max_vals = forward_max(clean_series, dynamic_short)
                min_vals = forward_min(clean_series, dynamic_short)
                
                # 安全归一化
                diff = max_vals - min_vals
                # 处理零或接近零的差值
                diff = np.where(diff < 1e-6, 1.0, diff)  # 避免除以零或接近零的值
                norm = np.where(diff < 1e-6, 50, 100 * (clean_series - min_vals) / diff)
                
                return pd.Series(norm, index=series.index)
            
            # 优化5：RSI权重机制改进
            def rsi_smooth_weight(rsi, price_trend):
                """动态RSI权重，考虑价格趋势"""
                # 确保输入数据没有NA值
                rsi_clean = rsi.fillna(50)  # RSI默认值50
                price_clean = price_trend.fillna(method='ffill').fillna(method='bfill')
                
                base_weight = 1 / (1 + np.exp(-0.15*(rsi_clean - 65)))
                
                # 检测顶背离：价格新高但RSI未新高
                # 确保rolling操作不产生NA值
                price_high = price_clean.rolling(30, min_periods=1).max()
                rsi_high = rsi_clean.rolling(30, min_periods=1).max()
                
                divergence = (price_clean >= price_high * 0.99) & (rsi_clean < rsi_high * 0.95)
                
                # 出现顶背离时降低权重
                result = np.where(divergence, base_weight*0.5, base_weight)
                return pd.Series(result, index=rsi.index)
            
            # 优化3：钝化函数调优
            def smooth_plateau(raw_score):
                """改进的S型钝化曲线"""
                # 确保输入值有效
                if pd.isna(raw_score):
                    return 50.0  # 默认中性值
                
                # 避免极端值
                raw_score = np.clip(raw_score, 0, 100)
                
                # 保持80-100分的线性区间，避免过度抑制
                if raw_score <= 80:
                    return 0.8 * raw_score  # 0-80分线性缩放
                else:
                    exp_portion = 100 * (1 - np.exp(-0.02*(raw_score-80)))
                    return 80 + exp_portion*(20/16.5)  # 80-100分渐进曲线
            
            # 优化1：趋势检测函数
            def detect_trend(series):
                """复合趋势检测"""
                # 处理NA值
                clean_series = series.fillna(method='ffill').fillna(method='bfill')
                
                # 双均线交叉
                fast_ma = clean_series.ewm(span=5, adjust=False).mean()
                slow_ma = clean_series.ewm(span=20, adjust=False).mean()
                cross_over = (fast_ma > slow_ma) & (fast_ma.shift(1).fillna(0) <= slow_ma.shift(1).fillna(0))
                cross_under = (fast_ma < slow_ma) & (fast_ma.shift(1).fillna(0) >= slow_ma.shift(1).fillna(0))
                
                # 价格通道突破
                high_20 = clean_series.rolling(20, min_periods=1).max()
                low_20 = clean_series.rolling(20, min_periods=1).min()
                break_up = clean_series > high_20.shift(1).fillna(0)
                break_down = clean_series < low_20.shift(1).fillna(float('inf'))  # 使用无穷大确保初始值不会触发
                
                # 综合判断
                trend = np.select(
                    [cross_over | break_up, cross_under | break_down],
                    [1, -1],
                    default=0  # 使用0作为默认值而不是np.nan
                )
                
                # 使用前向填充处理剩余的可能缺失值
                result = pd.Series(trend, index=series.index).fillna(0)
                return result
            
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
                                realtime_data = data_loader.pro.realtime_quote(ts_code=ts_code)
                                if realtime_data is not None and not realtime_data.empty:
                                    # 检查df中是否已有相同日期的数据
                                    existing_today = df[df['trade_date'].astype(str) == today_str]
                                    if existing_today.empty:
                                        # 创建当天数据行
                                        today_row = {
                                            'ts_code': ts_code,
                                            'trade_date': today_str,
                                            'open': float(realtime_data['open'].iloc[0]),
                                            'high': float(realtime_data['high'].iloc[0]),
                                            'low': float(realtime_data['low'].iloc[0]),
                                            'close': float(realtime_data['price'].iloc[0]),
                                            'pre_close': float(realtime_data['pre_close'].iloc[0]),
                                            'change': float(realtime_data['price'].iloc[0]) - float(realtime_data['pre_close'].iloc[0]),
                                            'pct_chg': (float(realtime_data['price'].iloc[0]) / float(realtime_data['pre_close'].iloc[0]) - 1) * 100,
                                            'vol': float(realtime_data['vol'].iloc[0]),
                                            'amount': float(realtime_data['amount'].iloc[0])
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
                            single_index[col] = single_index[col].fillna(method='ffill').fillna(method='bfill')
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
                    # 使用更平缓的趋势因子
                    trend_factor = np.where(single_index['trend'] == 1, 1.1, 0.9)  # 降低趋势影响
                    
                    # 波动率处理：上涨波动率正向贡献，下跌波动率反向处理
                    # 使用优化2：改进的混合归一化函数
                    volatility_score = (
                        hybrid_normalize(single_index['positive_volatility']) * 0.15 +  # 上涨波动率正向贡献
                        (100 - hybrid_normalize(single_index['negative_volatility'])) * 0.15  # 下跌波动率反向处理
                    )
                    
                    # RSI非线性加权（使用混合归一化）
                    rsi_score = hybrid_normalize(single_index['rsi']) * single_index['rsi_weight']
                    
                    # 计算原始情绪分数
                    raw_sentiment_score = (
                        volatility_score * trend_factor +  # 波动率得分与趋势协同
                        rsi_score * 0.3 +                 # RSI非线性加权
                        hybrid_normalize(single_index['bb_position']) * 0.2 +  # 布林带位置保留方向
                        hybrid_normalize(single_index['volume_ratio']) * 0.2 * trend_factor  # 成交量与趋势协同
                    )
                    
                    # 波动率过滤机制
                    volatility_threshold = 20  # 年化波动率>20%时认为是高波动期
                    volatility_mask = ((single_index['positive_volatility'] + single_index['negative_volatility']) > volatility_threshold).astype(float)
                    raw_sentiment_score = raw_sentiment_score * (0.2 + 0.8 * volatility_mask)
                    
                    # 使用优化3：改进的钝化函数
                    single_index['sentiment_score'] = raw_sentiment_score.apply(lambda x: smooth_plateau(x))
                    
                    # 最终情绪分二次平滑（5日EMA）
                    single_index['sentiment_score'] = single_index['sentiment_score'].ewm(span=5, adjust=False).mean()
                    
                    # 确保结果中没有NA或inf
                    single_index = single_index.fillna(method='ffill').fillna(method='bfill')
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
        
        logger.info(f"获取市场情绪数据 - 结果: {result}")

        # 对所有数据按日期排序
        for key in result:
            result[key].sort(key=lambda x: x['date'])

        return result

    except Exception as e:
        logger.error(f"Error in get_sentiment_data: {e}")
        import traceback
        logger.error(traceback.format_exc())  # 打印完整堆栈跟踪
        return None