import tushare as ts
import pandas as pd
from src.data.data_loader import DataLoader
from datetime import datetime, timedelta
from loguru import logger
import os
import time
import json
import numpy as np

class ETFDividendHandler:
    """处理ETF分红的类"""
    def __init__(self, ts_code=None):
        self.ts_code = ts_code  # ETF代码
        self.dividend_data = None  # 分红数据DataFrame
        self.last_api_call = 0  # 上次API调用时间
        self.min_interval = 1.0  # 最小调用间隔（秒）
        self.cache_file = f'cache/dividend_{ts_code}.json'  # 缓存文件路径
        
        # 确保缓存目录存在
        os.makedirs('cache', exist_ok=True)
        
        # 初始化Tushare
        tushare_token = os.getenv('TUSHARE_TOKEN')
        if not tushare_token:
            raise ValueError("未设置TUSHARE_TOKEN环境变量")
        ts.set_token(tushare_token)
        self.pro = ts.pro_api()
        
    def _wait_for_rate_limit(self):
        """等待以满足API调用频率限制"""
        current_time = time.time()
        elapsed = current_time - self.last_api_call
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self.last_api_call = time.time()
        
    def _load_from_cache(self):
        """从缓存加载分红数据"""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r') as f:
                    data = json.load(f)
                    df = pd.DataFrame(data)
                    df['date'] = pd.to_datetime(df['date'])
                    return df
        except Exception as e:
            logger.warning(f"从缓存加载分红数据失败: {str(e)}")
        return None
        
    def _save_to_cache(self, df):
        """保存分红数据到缓存"""
        try:
            # 将DataFrame转换为可序列化的格式
            df_copy = df.copy()
            df_copy['date'] = df_copy['date'].dt.strftime('%Y-%m-%d')
            data = df_copy.to_dict('records')
            
            with open(self.cache_file, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            logger.warning(f"保存分红数据到缓存失败: {str(e)}")
            
    def _clean_dividend_data(self, df):
        """清理分红数据，过滤掉非数值的分红记录"""
        try:
            # 将分红列转换为数值类型，非数值将变为NaN
            df['dividend'] = pd.to_numeric(df['dividend'], errors='coerce')
            
            # 删除分红为NaN的记录
            df = df.dropna(subset=['dividend'])
            
            # 删除分红为0的记录
            df = df[df['dividend'] > 0]
            
            return df
        except Exception as e:
            logger.error(f"清理分红数据时出错: {str(e)}")
            return df
            
    def update_dividend_data(self, start_date=None, end_date=None):
        """更新分红数据"""
        try:
            # 先尝试从缓存加载
            cached_data = self._load_from_cache()
            if cached_data is not None:
                # 检查缓存数据是否覆盖了所需的日期范围
                if start_date and end_date:
                    cached_start = cached_data['date'].min().date()
                    cached_end = cached_data['date'].max().date()
                    if cached_start <= start_date and cached_end >= end_date:
                        self.dividend_data = cached_data
                        logger.info(f"从缓存加载ETF分红数据成功 - {self.ts_code}, 数据长度: {len(cached_data)}")
                        return True
            
            # 如果缓存不存在或数据不完整，从API获取
            self._wait_for_rate_limit()
            
            # 获取分红数据
            df = self.pro.fund_div(
                ts_code=self.ts_code,
                start_date=start_date.strftime('%Y%m%d') if start_date else None,
                end_date=end_date.strftime('%Y%m%d') if end_date else None
            )
            
            if df is not None and not df.empty:
                # 重命名列
                df = df.rename(columns={
                    'ann_date': 'date',
                    'div_cash': 'dividend'
                })
                
                # 转换日期格式
                df['date'] = pd.to_datetime(df['date'])
                
                # 清理分红数据
                df = self._clean_dividend_data(df)
                
                # 按日期排序
                df = df.sort_values('date')
                
                # 保存到缓存
                self._save_to_cache(df)
                
                self.dividend_data = df
                logger.info(f"成功获取ETF分红数据 - {self.ts_code}, 数据长度: {len(df)}")
                return True
            else:
                logger.warning(f"未获取到ETF分红数据 - {self.ts_code}")
                return False
                
        except Exception as e:
            logger.error(f"获取ETF分红数据出错: {str(e)}")
            return False
        
    def process_dividend(self, date_str, position_size, current_price):
        """处理指定日期的分红"""
        if self.dividend_data is None:
            return 0.0
            
        try:
            # 将日期字符串转换为datetime对象
            date = pd.to_datetime(date_str)
            
            # 查找当天的分红记录
            dividend_record = self.dividend_data[self.dividend_data['date'].dt.date == date.date()]
            
            if not dividend_record.empty:
                # 获取每股分红金额
                dividend_per_share = float(dividend_record['dividend'].iloc[0])
                
                # 计算总分红金额
                total_dividend = dividend_per_share * position_size
                
                logger.info(f"处理ETF分红 - 日期: {date_str}, 每股分红: {dividend_per_share:.4f}, "
                          f"持仓数量: {position_size}, 总分红: {total_dividend:.2f}")
                
                return total_dividend
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"处理ETF分红时出错: {str(e)}")
            return 0.0