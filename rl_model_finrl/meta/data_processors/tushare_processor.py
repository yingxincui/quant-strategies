import pandas as pd
import numpy as np
import tushare as ts
from datetime import datetime
import os
from typing import List, Tuple, Dict, Optional
import logging

from src.strategies.rl_model_finrl.config import (
    TUSHARE_TOKEN,
    DATA_SAVE_PATH,
    TECHNICAL_INDICATORS_LIST,
    TRAIN_START_DATE,
    TRAIN_END_DATE,
    TEST_START_DATE,
    TEST_END_DATE,
    TICKER_LIST,
)

class TushareProcessor:
    """
    TushareProcessor 类负责从Tushare获取A股ETF数据
    并进行数据预处理以供强化学习使用
    """
    
    def __init__(self, token: str = TUSHARE_TOKEN):
        """
        初始化Tushare数据处理器
        
        参数:
            token: Tushare API令牌
        """
        self.tushare_token = token
        if self.tushare_token:
            ts.set_token(self.tushare_token)
            self.pro = ts.pro_api()
        else:
            raise ValueError("请提供有效的Tushare令牌")
        
        # 确保数据保存目录存在
        self.data_path = os.path.join(DATA_SAVE_PATH, "tushare_data")
        os.makedirs(self.data_path, exist_ok=True)
        
        # 设置日志
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def download_etf_data(
        self, 
        ticker_list: List[str], 
        start_date: str, 
        end_date: str,
        adjust: str = "qfq"  # 前复权
    ) -> Dict[str, pd.DataFrame]:
        """
        下载ETF日线数据
        
        参数:
            ticker_list: ETF代码列表
            start_date: 开始日期，格式'YYYY-MM-DD'
            end_date: 结束日期，格式'YYYY-MM-DD'
            adjust: 复权类型，默认前复权'qfq'
            
        返回:
            字典，包含每个ETF代码及其对应的数据框
        """
        self.logger.info(f"开始下载 {len(ticker_list)} 只ETF的数据...")
        etf_data_dict = {}
        
        # 转换日期格式，将横线替换为空
        start_date_ts = start_date.replace("-", "")
        end_date_ts = end_date.replace("-", "")
        
        for ticker in ticker_list:
            try:
                # 使用Tushare获取ETF日线数据
                self.logger.info(f"下载 {ticker} 的数据...")
                
                # 分离市场和代码
                if ".SH" in ticker:
                    market = "SH"
                    raw_ticker = ticker.replace(".SH", "")
                elif ".SZ" in ticker:
                    market = "SZ"
                    raw_ticker = ticker.replace(".SZ", "")
                else:
                    self.logger.warning(f"不支持的代码格式: {ticker}")
                    continue
                
                # 下载数据
                df = ts.pro_bar(
                    ts_code=ticker,
                    start_date=start_date_ts,
                    end_date=end_date_ts,
                    asset="E",  # ETF
                    adj=adjust,
                )
                
                if df is None or df.empty:
                    self.logger.warning(f"无法获取 {ticker} 的数据")
                    continue
                
                # 重命名列以便后续处理
                df = df.rename(columns={
                    'trade_date': 'date',
                    'open': 'open',
                    'high': 'high',
                    'low': 'low',
                    'close': 'close',
                    'vol': 'volume',
                    'amount': 'amount',
                    'pct_chg': 'change_pct'
                })
                
                # 转换日期格式并设置为索引
                df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
                df = df.sort_values('date')
                df = df.set_index('date')
                
                # 添加代码列
                df['tic'] = ticker
                
                # 保存到字典
                etf_data_dict[ticker] = df
                self.logger.info(f"成功下载 {ticker} 的数据，共 {len(df)} 行")
                
                # 保存到CSV
                file_name = f"{ticker}_daily_{start_date_ts}_{end_date_ts}.csv"
                file_path = os.path.join(self.data_path, file_name)
                df.to_csv(file_path)
                self.logger.info(f"数据已保存到 {file_path}")
                
            except Exception as e:
                self.logger.error(f"下载 {ticker} 时出错: {str(e)}")
        
        return etf_data_dict
    
    def download_index_data(
        self,
        index_list: List[str] = ["000001.SH", "399001.SZ", "399006.SZ"],  # 默认上证指数、深证成指、创业板指
        start_date: str = None,
        end_date: str = None,
    ) -> pd.DataFrame:
        """
        下载指数数据作为市场状态的一部分
        
        参数:
            index_list: 指数代码列表
            start_date: 开始日期，格式'YYYY-MM-DD'
            end_date: 结束日期，格式'YYYY-MM-DD'
            
        返回:
            包含指数数据的DataFrame
        """
        # 如果未提供日期，则使用默认日期
        if start_date is None:
            start_date = TRAIN_START_DATE
        if end_date is None:
            end_date = TEST_END_DATE
            
        start_date_ts = start_date.replace("-", "")
        end_date_ts = end_date.replace("-", "")
        
        index_dfs = []
        
        for index_code in index_list:
            try:
                self.logger.info(f"下载指数 {index_code} 的数据...")
                
                # 获取指数日线数据
                df = self.pro.index_daily(
                    ts_code=index_code,
                    start_date=start_date_ts,
                    end_date=end_date_ts
                )
                
                if df is None or df.empty:
                    self.logger.warning(f"无法获取指数 {index_code} 的数据")
                    continue
                
                # 重命名列并处理日期
                df = df.rename(columns={
                    'trade_date': 'date',
                    'pct_chg': 'change_pct'
                })
                
                df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
                df = df.sort_values('date')
                df = df.set_index('date')
                
                # 添加指数代码
                df['index_code'] = index_code
                
                index_dfs.append(df)
                
                # 保存到CSV
                file_name = f"{index_code}_daily_{start_date_ts}_{end_date_ts}.csv"
                file_path = os.path.join(self.data_path, file_name)
                df.to_csv(file_path)
                
            except Exception as e:
                self.logger.error(f"下载指数 {index_code} 时出错: {str(e)}")
        
        # 合并所有指数数据
        if index_dfs:
            combined_index_df = pd.concat(index_dfs)
            return combined_index_df
        else:
            self.logger.warning("没有获取到任何指数数据")
            return pd.DataFrame()
    
    def add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        添加技术指标
        
        参数:
            data: 输入价格数据，包含open, high, low, close, volume等列
            
        返回:
            添加了技术指标的DataFrame
        """
        df = data.copy()
        
        # 确保DataFrame按日期排序
        if not df.index.is_monotonic_increasing:
            df = df.sort_index()
        
        # MACD
        if 'macd' in TECHNICAL_INDICATORS_LIST:
            exp12 = df['close'].ewm(span=12, adjust=False).mean()
            exp26 = df['close'].ewm(span=26, adjust=False).mean()
            macd = exp12 - exp26
            signal = macd.ewm(span=9, adjust=False).mean()
            df['macd'] = macd - signal
        
        # Bollinger Bands
        if any(x in TECHNICAL_INDICATORS_LIST for x in ['boll_ub', 'boll_lb']):
            ma20 = df['close'].rolling(window=20).mean()
            std20 = df['close'].rolling(window=20).std()
            if 'boll_ub' in TECHNICAL_INDICATORS_LIST:
                df['boll_ub'] = ma20 + (std20 * 2)
            if 'boll_lb' in TECHNICAL_INDICATORS_LIST:
                df['boll_lb'] = ma20 - (std20 * 2)
        
        # RSI
        if 'rsi_30' in TECHNICAL_INDICATORS_LIST:
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=30).mean()
            avg_loss = loss.rolling(window=30).mean()
            rs = avg_gain / avg_loss
            df['rsi_30'] = 100 - (100 / (1 + rs))
        
        # CCI (Commodity Channel Index)
        if 'cci_30' in TECHNICAL_INDICATORS_LIST:
            tp = (df['high'] + df['low'] + df['close']) / 3
            ma_tp = tp.rolling(window=30).mean()
            md_tp = (tp - ma_tp).abs().rolling(window=30).mean()
            df['cci_30'] = (tp - ma_tp) / (0.015 * md_tp)
        
        # DX (Directional Movement Index)
        if 'dx_30' in TECHNICAL_INDICATORS_LIST:
            up_move = df['high'].diff()
            down_move = df['low'].diff().abs()
            pos_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
            neg_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
            
            tr1 = df['high'] - df['low']
            tr2 = (df['high'] - df['close'].shift(1)).abs()
            tr3 = (df['low'] - df['close'].shift(1)).abs()
            tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
            
            atr = tr.rolling(window=30).mean()
            pos_di = 100 * (pd.Series(pos_dm).rolling(window=30).mean() / atr)
            neg_di = 100 * (pd.Series(neg_dm).rolling(window=30).mean() / atr)
            
            df['dx_30'] = 100 * ((pos_di - neg_di).abs() / (pos_di + neg_di))
        
        # SMA
        if 'close_30_sma' in TECHNICAL_INDICATORS_LIST:
            df['close_30_sma'] = df['close'].rolling(window=30).mean()
        if 'close_60_sma' in TECHNICAL_INDICATORS_LIST:
            df['close_60_sma'] = df['close'].rolling(window=60).mean()
        
        # Volatility
        if 'volatility_30' in TECHNICAL_INDICATORS_LIST:
            df['volatility_30'] = df['close'].pct_change().rolling(window=30).std()
        
        # Momentum
        if 'momentum_30' in TECHNICAL_INDICATORS_LIST:
            df['momentum_30'] = df['close'] - df['close'].shift(30)
        
        # 删除NaN值
        df = df.dropna()
        
        return df
    
    def get_trading_days(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        获取交易日历
        
        参数:
            start_date: 开始日期，格式'YYYY-MM-DD'
            end_date: 结束日期，格式'YYYY-MM-DD'
            
        返回:
            交易日历DataFrame
        """
        start_date_ts = start_date.replace("-", "")
        end_date_ts = end_date.replace("-", "")
        
        try:
            trading_cal = self.pro.trade_cal(
                exchange='SSE',
                start_date=start_date_ts,
                end_date=end_date_ts,
                is_open=1  # 只获取交易日
            )
            return trading_cal
        except Exception as e:
            self.logger.error(f"获取交易日历时出错: {str(e)}")
            return pd.DataFrame()
    
    def prepare_data_for_training(self) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
        """
        下载并准备训练和测试数据
        
        返回:
            (etf_data_dict, index_data)元组，包含ETF数据字典和指数数据
        """
        # 下载ETF历史数据，包括训练和测试期间
        etf_data_dict = self.download_etf_data(
            ticker_list=TICKER_LIST,
            start_date=TRAIN_START_DATE,
            end_date=TEST_END_DATE
        )
        
        # 下载指数数据
        index_data = self.download_index_data(
            start_date=TRAIN_START_DATE,
            end_date=TEST_END_DATE
        )
        
        # 为每个ETF添加技术指标
        for ticker, df in etf_data_dict.items():
            etf_data_dict[ticker] = self.add_technical_indicators(df)
            
        return etf_data_dict, index_data
    
    def get_news_sentiment(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        从Tushare获取新闻情绪数据（需要高级数据权限）
        
        参数:
            start_date: 开始日期，格式'YYYY-MM-DD'
            end_date: 结束日期，格式'YYYY-MM-DD'
            
        返回:
            新闻情绪DataFrame
        """
        # 注意：该API需要Tushare高级会员
        try:
            start_date_ts = start_date.replace("-", "")
            end_date_ts = end_date.replace("-", "")
            
            self.logger.info("尝试获取新闻情绪数据（需要高级会员）...")
            news_data = self.pro.news(
                start_date=start_date_ts,
                end_date=end_date_ts,
                src='sina'  # 新浪财经
            )
            
            # 如果获取成功，进行简单的情感处理（实际使用中应当使用更复杂的情感分析模型）
            if news_data is not None and not news_data.empty:
                # 对新闻进行简单处理，每天的新闻聚合
                news_data['date'] = pd.to_datetime(news_data['pub_time']).dt.date
                news_data['date'] = pd.to_datetime(news_data['date'])
                
                # 按日期分组
                daily_news = news_data.groupby('date').size().reset_index(name='news_count')
                daily_news = daily_news.set_index('date')
                
                return daily_news
            else:
                self.logger.warning("没有获取到新闻数据或无权限访问")
                return pd.DataFrame()
        except Exception as e:
            self.logger.error(f"获取新闻情绪数据时出错: {str(e)}")
            return pd.DataFrame() 