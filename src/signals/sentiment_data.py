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
            # 使用tushare获取多个指数日线数据
            index_codes = [
                '000001.SH',  # 上证指数
                '000300.SH',  # 沪深300指数
                '000016.SH',  # 上证50指数
                '399240.SZ',  # 金融指数
            ]
            
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
                        
                    # 计算技术指标
                    # 1. 波动率 - 20日年化波动率
                    single_index['returns'] = single_index['pct_chg'] / 100  # 使用涨跌幅计算收益率
                    single_index['volatility'] = single_index['returns'].rolling(window=20, min_periods=5).std() * np.sqrt(252) * 100
                    
                    # 2. RSI - 14日相对强弱指数
                    delta = single_index['close'].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=3).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=3).mean()
                    rs = gain / loss
                    single_index['rsi'] = 100 - (100 / (1 + rs))
                    
                    # 3. 布林带 - 计算布林带位置
                    sma = single_index['close'].rolling(window=20, min_periods=5).mean()
                    std = single_index['close'].rolling(window=20, min_periods=5).std()
                    single_index['bb_position'] = (single_index['close'] - sma) / (2 * std)
                    
                    # 4. 成交量变化
                    single_index['volume_ma'] = single_index['vol'].rolling(window=20, min_periods=5).mean()
                    single_index['volume_ratio'] = single_index['vol'] / single_index['volume_ma']
                    
                    # 5. 改进的归一化函数
                    def normalize(series):
                        # 使用全局最大最小值而不是滚动窗口
                        min_val = series.min()
                        max_val = series.max()
                        if max_val == min_val:
                            return pd.Series(50, index=series.index)  # 如果所有值相同，返回中性值
                        return 100 * (series - min_val) / (max_val - min_val)
                    
                    # 计算综合情绪分数
                    single_index['sentiment_score'] = (
                        normalize(single_index['volatility']) * 0.3 +  # 波动率权重
                        normalize(single_index['rsi']) * 0.3 +         # RSI权重
                        normalize(abs(single_index['bb_position'])) * 0.2 +  # 布林带偏离度权重
                        normalize(single_index['volume_ratio']) * 0.2   # 成交量比率权重
                    )
                    
                    # 填充可能的NaN值
                    single_index = single_index.fillna(method='ffill').fillna(method='bfill')
                    
                    # 添加到结果列表
                    sentiment_results.extend([
                        {
                            'date': row['trade_date'].strftime('%Y-%m-%d'),
                            'value': float(row['sentiment_score']) if not pd.isna(row['sentiment_score']) else 50,
                            'details': {
                                'index_code': ts_code,
                                'volatility': float(row['volatility']) if not pd.isna(row['volatility']) else 0,
                                'rsi': float(row['rsi']) if not pd.isna(row['rsi']) else 50,
                                'bb_position': float(row['bb_position']) if not pd.isna(row['bb_position']) else 0,
                                'volume_ratio': float(row['volume_ratio']) if not pd.isna(row['volume_ratio']) else 1,
                                'close': float(row['close']),
                                'change': float(row['pct_chg'])
                            }
                        }
                        for _, row in single_index.iterrows()
                        if not pd.isna(row['sentiment_score'])
                    ])
                
                # 按日期分组计算平均情绪分数
                sentiment_df = pd.DataFrame(sentiment_results)
                result['sentiment'] = sentiment_df.groupby('date').apply(
                    lambda group: {
                        'date': group['date'].iloc[0],
                        'value': float(group['value'].mean()),
                        'details': {
                            'volatility': float(group['details'].apply(lambda x: x['volatility']).mean()),
                            'rsi': float(group['details'].apply(lambda x: x['rsi']).mean()),
                            'bb_position': float(group['details'].apply(lambda x: x['bb_position']).mean()),
                            'volume_ratio': float(group['details'].apply(lambda x: x['volume_ratio']).mean()),
                            'close': float(group['details'].apply(lambda x: x['close']).mean()),
                            'change': float(group['details'].apply(lambda x: x['change']).mean()),
                            'indices': [
                                {
                                    'code': detail['index_code'],
                                    'close': float(detail['close']),
                                    'change': float(detail['change']),
                                    'volatility': float(detail['volatility']),
                                    'rsi': float(detail['rsi']),
                                    'bb_position': float(detail['bb_position']),
                                    'volume_ratio': float(detail['volume_ratio'])
                                }
                                for detail in group['details']
                            ]
                        }
                    }
                ).tolist()
                
                # 按日期排序
                result['sentiment'].sort(key=lambda x: x['date'])
                # logger.info(f"情绪指标: {result['sentiment']}")
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