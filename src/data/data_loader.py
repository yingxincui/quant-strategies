import tushare as ts
import akshare as ak
import pandas as pd
import backtrader as bt
from datetime import datetime
import os
from loguru import logger

class DataLoader:
    def __init__(self, tushare_token=None):
        """
        初始化数据加载器
        :param tushare_token: Tushare的API token
        """
        self.tushare_token = tushare_token
        if tushare_token:
            ts.set_token(tushare_token)
            self.pro = ts.pro_api()
            logger.info("Tushare API初始化成功")

    def download_data(self, symbol, start_date, end_date):
        """
        下载数据，支持A股、ETF和港股
        :param symbol: 股票代码（格式：000001.SZ, 510300.SH, 00700.HK等）
        :param start_date: 开始日期
        :param end_date: 结束日期
        :return: DataFrame
        """
        try:
            # 转换日期格式
            start_str = start_date.strftime("%Y%m%d")
            end_str = end_date.strftime("%Y%m%d")
            
            # 判断市场类型
            if symbol.endswith(('.SH', '.SZ')):  # A股或ETF
                if symbol.startswith('51') or symbol.startswith('159'):  # ETF
                    logger.info(f"使用AKShare下载ETF数据: {symbol}")
                    df = self._download_etf_data(symbol, start_date, end_date)
                else:  # A股
                    logger.info(f"使用Tushare下载A股数据: {symbol}")
                    df = self._download_stock_data(symbol, start_str, end_str)
            elif symbol.endswith('.HK'):  # 港股
                logger.info(f"使用AKShare下载港股数据: {symbol}")
                df = self._download_hk_data(symbol, start_date, end_date)
            else:
                raise ValueError(f"不支持的市场类型: {symbol}")
                
            logger.info(f"下载数据成功: {symbol}，数据长度: {len(df)}")

            if df.empty:
                return None
                
            # 创建PandasData对象并设置股票代码
            data = PandasData(dataname=df, ts_code=symbol)
            return data
                
        except Exception as e:
            logger.error(f"下载数据失败: {str(e)}")
            raise

    def _download_stock_data(self, symbol, start_date, end_date):
        """下载A股数据"""
        if not self.tushare_token:
            raise ValueError("需要设置Tushare token才能下载A股数据")
            
        df = self.pro.daily(ts_code=symbol, start_date=start_date, end_date=end_date)
        if df.empty:
            return pd.DataFrame()
            
        # 重命名列以匹配backtrader要求
        df = df.rename(columns={
            'trade_date': 'date',
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'vol': 'volume'
        })
        
        # 转换日期格式并设置为索引
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        df = df.sort_index()  # 确保按日期升序
        
        return df[['open', 'high', 'low', 'close', 'volume']]

    def _download_etf_data(self, symbol, start_date, end_date):
        """下载ETF数据"""
        symbol_code = symbol.split('.')[0]  # 去掉市场后缀
        
        try:
            df = ak.fund_etf_hist_em(symbol=symbol_code, period="daily")
            
            # 重命名列以匹配backtrader要求
            df = df.rename(columns={
                '日期': 'date',
                '开盘': 'open',
                '最高': 'high',
                '最低': 'low',
                '收盘': 'close',
                '成交量': 'volume'
            })
            
            # 转换日期格式并设置为索引
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
            
            # 过滤日期范围
            df = df.loc[start_date:end_date]
            
            return df[['open', 'high', 'low', 'close', 'volume']]
            
        except Exception as e:
            logger.error(f"下载ETF数据失败: {str(e)}")
            return pd.DataFrame()

    def _download_hk_data(self, symbol, start_date, end_date):
        """下载港股数据"""
        symbol_code = symbol.split('.')[0]  # 去掉市场后缀
        
        try:
            df = ak.stock_hk_daily(symbol=symbol_code)
            
            # 重命名列以匹配backtrader要求
            df = df.rename(columns={
                '日期': 'date',
                '开盘': 'open',
                '最高': 'high',
                '最低': 'low',
                '收盘': 'close',
                '成交量': 'volume'
            })
            
            # 转换日期格式并设置为索引
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
            
            # 过滤日期范围
            df = df.loc[start_date:end_date]
            
            return df[['open', 'high', 'low', 'close', 'volume']]
            
        except Exception as e:
            logger.error(f"下载港股数据失败: {str(e)}")
            return pd.DataFrame()

class PandasData(bt.feeds.PandasData):
    """自定义PandasData类，用于加载数据"""
    params = (
        ('datetime', None),  # 使用索引作为日期
        ('open', 'open'),
        ('high', 'high'),
        ('low', 'low'),
        ('close', 'close'),
        ('volume', 'volume'),
        ('openinterest', None),
        ('ts_code', None),  # 股票代码
    )
    
    def __init__(self, **kwargs):
        """初始化数据源"""
        # 从kwargs中获取ts_code
        self.ts_code = kwargs.pop('ts_code', None)
        super().__init__(**kwargs) 