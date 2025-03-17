from loguru import logger
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import os
from src.data.data_loader import DataLoader

def get_sentiment_data():
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
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * 10)
        
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
            
            def hybrid_normalize(series, short_window=60, long_window=252):
                """动态混合归一化函数
                短期窗口提供敏感性，长期窗口提供稳定性
                根据近期波动率动态调整两个窗口的权重
                """
                # 处理零除情况
                def safe_normalize(s, window):
                    min_val = s.rolling(window, min_periods=1).min()
                    max_val = s.rolling(window, min_periods=1).max()
                    diff = max_val - min_val
                    # 当最大最小值相同时返回50
                    return np.where(diff == 0, 50, 100 * (s - min_val) / diff)
                
                # 短期和长期归一化
                short_norm = safe_normalize(series, short_window)
                long_norm = safe_normalize(series, long_window)
                
                # 计算动态混合权重
                volatility = series.diff().abs().rolling(10).mean()
                blend_ratio = np.clip(volatility / 0.05, 0, 1)  # 波动率>5%时完全使用短期
                
                # 混合结果
                return pd.Series(short_norm * blend_ratio + long_norm * (1 - blend_ratio), index=series.index)
            
            def rsi_smooth_weight(rsi):
                """RSI平滑权重函数，使用sigmoid实现渐变过渡"""
                return 1 / (1 + np.exp(-0.2 * (rsi - 70)))  # RSI=70时权重0.5，80时0.88，90时0.98
            
            def smooth_plateau(raw_score):
                """渐进式高位钝化函数"""
                return 100 * (1 - np.exp(-0.03 * raw_score))  # 分数越高增速越慢，避免陡峭顶部
            
            index_data_list = []
            for ts_code in index_codes:
                try:
                    df = data_loader.pro.index_daily(
                        ts_code=ts_code,
                        start_date=start_date.strftime('%Y%m%d'),
                        end_date=end_date.strftime('%Y%m%d')
                    )
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
                    
                    # 计算趋势方向（使用EMA平滑）
                    single_index['ma20'] = single_index['close'].ewm(span=20, adjust=False).mean()
                    single_index['trend'] = np.where(single_index['ma20'].diff() > 0, 1, -1)
                    
                    # 计算技术指标
                    # 1. 分别计算上涨和下跌波动率（使用EMA平滑）
                    returns = single_index['pct_chg'] / 100
                    # 计算正向和负向波动率
                    positive_returns = returns.where(returns > 0, 0)
                    negative_returns = returns.where(returns < 0, 0)
                    single_index['positive_volatility'] = positive_returns.ewm(span=10, adjust=False).std() * np.sqrt(252) * 100
                    single_index['negative_volatility'] = (-negative_returns).ewm(span=10, adjust=False).std() * np.sqrt(252) * 100
                    
                    # 2. RSI - 使用双重EMA平滑
                    delta = single_index['close'].diff()
                    gain = delta.where(delta > 0, 0.0)
                    loss = -delta.where(delta < 0, 0.0)
                    # 使用更长的EMA窗口进行平滑
                    avg_gain = gain.ewm(alpha=1/21, adjust=False).mean()  # 21日EMA
                    avg_loss = loss.ewm(alpha=1/21, adjust=False).mean()
                    rs = avg_gain / avg_loss
                    single_index['rsi'] = 100 - (100 / (1 + rs))
                    # 应用平滑RSI权重
                    single_index['rsi_weight'] = single_index['rsi'].apply(rsi_smooth_weight)
                    
                    # 3. 布林带 - 使用EMA平滑
                    sma = single_index['close'].ewm(span=20, adjust=False).mean()
                    std = single_index['close'].ewm(span=20, adjust=False).std()
                    single_index['bb_position'] = (single_index['close'] - sma) / (2 * std)
                    
                    # 4. 成交量变化（使用EMA平滑）
                    single_index['volume_ma'] = single_index['vol'].ewm(span=20, adjust=False).mean()
                    single_index['volume_ratio'] = single_index['vol'] / single_index['volume_ma']
                    
                    # 计算综合情绪分数（考虑指标方向性和趋势）
                    # 使用更平缓的趋势因子
                    trend_factor = np.where(single_index['trend'] == 1, 1.1, 0.9)  # 降低趋势影响
                    
                    # 波动率处理：上涨波动率正向贡献，下跌波动率反向处理
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
                    
                    # 应用渐进式高位钝化
                    single_index['sentiment_score'] = raw_sentiment_score.apply(smooth_plateau)
                    
                    # 最终情绪分二次平滑（5日EMA）
                    single_index['sentiment_score'] = single_index['sentiment_score'].ewm(span=5, adjust=False).mean()
                    
                    # 仅使用前向填充
                    single_index = single_index.fillna(method='ffill')
                    
                    # 添加到结果列表
                    sentiment_results.extend([
                        {
                            'date': row['trade_date'].strftime('%Y-%m-%d'),
                            'value': float(row['sentiment_score']) if not pd.isna(row['sentiment_score']) else 50,
                            'weight': float(row['weight']),
                            'details': {
                                'index_code': ts_code,
                                'positive_volatility': float(row['positive_volatility']) if not pd.isna(row['positive_volatility']) else 0,
                                'negative_volatility': float(row['negative_volatility']) if not pd.isna(row['negative_volatility']) else 0,
                                'rsi': float(row['rsi']) if not pd.isna(row['rsi']) else 50,
                                'rsi_weight': float(row['rsi_weight']) if not pd.isna(row['rsi_weight']) else 0.5,
                                'bb_position': float(row['bb_position']) if not pd.isna(row['bb_position']) else 0,
                                'volume_ratio': float(row['volume_ratio']) if not pd.isna(row['volume_ratio']) else 1,
                                'trend': int(row['trend']) if not pd.isna(row['trend']) else 0,
                                'close': float(row['close']),
                                'change': float(row['pct_chg'])
                            }
                        }
                        for _, row in single_index.iterrows()
                        if not pd.isna(row['sentiment_score'])
                    ])
                
                # 按日期分组计算加权平均情绪分数
                sentiment_df = pd.DataFrame(sentiment_results)
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
                            'trend': int(group['details'].apply(lambda x: x['trend']).mean()),
                            'close': float(group['details'].apply(lambda x: x['close']).mean()),
                            'change': float(group['details'].apply(lambda x: x['change']).mean()),
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
                                    'trend': int(detail['trend'])
                                }
                                for detail in group['details']
                            ]
                        }
                    }
                ).tolist()
                
                # 按日期排序
                result['sentiment'].sort(key=lambda x: x['date'])
                
            else:
                result['sentiment'] = []
                logger.warning("No index data available for sentiment calculation")
        except Exception as e:
            logger.error(f"Error calculating sentiment indicators: {e}")
            result['sentiment'] = []

        # 对所有数据按日期排序
        for key in result:
            result[key].sort(key=lambda x: x['date'])

        return result

    except Exception as e:
        logger.error(f"Error in get_sentiment_data: {e}")
        return None