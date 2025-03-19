import pandas as pd
import numpy as np
import akshare as ak
import os
from typing import List, Dict, Optional, Tuple
import logging
from datetime import datetime

from src.strategies.rl_model_finrl.config import (
    DATA_SAVE_PATH,
    TRAIN_START_DATE,
    TRAIN_END_DATE,
    TEST_START_DATE,
    TEST_END_DATE,
    TICKER_LIST,
)

class AKShareProcessor:
    """
    AKShareProcessor 类负责从AKShare获取A股ETF相关数据
    作为补充数据源使用
    """
    
    def __init__(self):
        """初始化AKShare数据处理器"""
        # 确保数据保存目录存在
        self.data_path = os.path.join(DATA_SAVE_PATH, "akshare_data")
        os.makedirs(self.data_path, exist_ok=True)
        
        # 设置日志
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def download_etf_fund_info(self, ticker_list: List[str]) -> Dict[str, pd.DataFrame]:
        """
        下载ETF基金基本信息
        
        参数:
            ticker_list: ETF代码列表
            
        返回:
            包含ETF基本信息的字典
        """
        etf_info_dict = {}
        
        for ticker in ticker_list:
            try:
                # 处理代码格式
                ticker_code = ticker.split('.')[0]
                
                self.logger.info(f"获取ETF {ticker} 的基本信息...")
                
                # 获取ETF基本信息
                etf_info = ak.fund_etf_basic_info(symbol=ticker_code)
                
                if etf_info is not None and not etf_info.empty:
                    # 保存到字典
                    etf_info_dict[ticker] = etf_info
                    
                    # 保存到CSV
                    file_name = f"{ticker}_info.csv"
                    file_path = os.path.join(self.data_path, file_name)
                    etf_info.to_csv(file_path, index=False)
                    self.logger.info(f"ETF基本信息已保存到 {file_path}")
                else:
                    self.logger.warning(f"无法获取ETF {ticker} 的基本信息")
            
            except Exception as e:
                self.logger.error(f"获取ETF {ticker} 的基本信息时出错: {str(e)}")
        
        return etf_info_dict
    
    def download_etf_daily_data(
        self,
        ticker_list: List[str],
        start_date: str,
        end_date: str
    ) -> Dict[str, pd.DataFrame]:
        """
        从AKShare下载ETF日线数据
        
        参数:
            ticker_list: ETF代码列表
            start_date: 开始日期，格式'YYYY-MM-DD'
            end_date: 结束日期，格式'YYYY-MM-DD'
            
        返回:
            字典，包含每个ETF代码及其对应的数据框
        """
        etf_data_dict = {}
        
        for ticker in ticker_list:
            try:
                # 处理代码格式
                ticker_code = ticker.split('.')[0]
                
                self.logger.info(f"从AKShare获取ETF {ticker} 的日线数据...")
                
                # 获取ETF历史数据
                etf_data = ak.fund_etf_hist_em(
                    symbol=ticker_code,
                    period="daily",
                    start_date=start_date,
                    end_date=end_date,
                    adjust="qfq"
                )
                
                if etf_data is not None and not etf_data.empty:
                    # 重命名列以匹配tushare格式
                    etf_data = etf_data.rename(columns={
                        '日期': 'date',
                        '开盘': 'open',
                        '收盘': 'close',
                        '最高': 'high',
                        '最低': 'low',
                        '成交量': 'volume',
                        '成交额': 'amount',
                        '振幅': 'amplitude',
                        '涨跌幅': 'change_pct',
                        '涨跌额': 'change',
                        '换手率': 'turnover'
                    })
                    
                    # 转换日期格式并设置为索引
                    etf_data['date'] = pd.to_datetime(etf_data['date'])
                    etf_data = etf_data.sort_values('date')
                    etf_data = etf_data.set_index('date')
                    
                    # 添加代码列
                    etf_data['tic'] = ticker
                    
                    # 保存到字典
                    etf_data_dict[ticker] = etf_data
                    
                    # 保存到CSV
                    file_name = f"{ticker}_daily_akshare_{start_date.replace('-', '')}_{end_date.replace('-', '')}.csv"
                    file_path = os.path.join(self.data_path, file_name)
                    etf_data.to_csv(file_path)
                    self.logger.info(f"ETF日线数据已保存到 {file_path}")
                else:
                    self.logger.warning(f"无法获取ETF {ticker} 的日线数据")
            
            except Exception as e:
                self.logger.error(f"获取ETF {ticker} 的日线数据时出错: {str(e)}")
        
        return etf_data_dict
    
    def download_etf_fund_flow(
        self,
        ticker_list: List[str],
        start_date: str,
        end_date: str
    ) -> Dict[str, pd.DataFrame]:
        """
        获取ETF资金流向数据
        
        参数:
            ticker_list: ETF代码列表
            start_date: 开始日期，格式'YYYY-MM-DD'
            end_date: 结束日期，格式'YYYY-MM-DD'
            
        返回:
            包含资金流向的字典
        """
        fund_flow_dict = {}
        
        for ticker in ticker_list:
            try:
                # 处理代码格式
                ticker_code = ticker.split('.')[0]
                
                self.logger.info(f"获取ETF {ticker} 的资金流向数据...")
                
                # 获取ETF资金流向
                try:
                    # 尝试获取资金流向数据，这个可能需要调整，因为AKShare的API可能有变化
                    fund_flow = ak.fund_etf_fund_flow_rank()
                    
                    # 筛选特定ETF的数据
                    if ticker_code in fund_flow['代码'].values:
                        fund_flow_single = fund_flow[fund_flow['代码'] == ticker_code]
                        fund_flow_dict[ticker] = fund_flow_single
                        
                        # 保存到CSV
                        file_name = f"{ticker}_fund_flow.csv"
                        file_path = os.path.join(self.data_path, file_name)
                        fund_flow_single.to_csv(file_path, index=False)
                        self.logger.info(f"ETF资金流向数据已保存到 {file_path}")
                    else:
                        self.logger.warning(f"未找到ETF {ticker} 的资金流向数据")
                except Exception as inner_e:
                    self.logger.error(f"获取ETF资金流向时出现内部错误: {str(inner_e)}")
            
            except Exception as e:
                self.logger.error(f"获取ETF {ticker} 的资金流向数据时出错: {str(e)}")
        
        return fund_flow_dict
    
    def download_etf_holdings(self, ticker_list: List[str]) -> Dict[str, pd.DataFrame]:
        """
        获取ETF持仓数据
        
        参数:
            ticker_list: ETF代码列表
            
        返回:
            包含ETF持仓的字典
        """
        holdings_dict = {}
        
        for ticker in ticker_list:
            try:
                # 处理代码格式
                ticker_code = ticker.split('.')[0]
                
                self.logger.info(f"获取ETF {ticker} 的持仓数据...")
                
                # 获取ETF持仓
                try:
                    # 尝试获取ETF持仓数据，这个API需要根据AKShare的最新文档调整
                    holdings = ak.fund_etf_spot_deal_em()
                    
                    # 筛选特定ETF的数据
                    if ticker_code in holdings['代码'].values:
                        holdings_single = holdings[holdings['代码'] == ticker_code]
                        holdings_dict[ticker] = holdings_single
                        
                        # 保存到CSV
                        file_name = f"{ticker}_holdings.csv"
                        file_path = os.path.join(self.data_path, file_name)
                        holdings_single.to_csv(file_path, index=False)
                        self.logger.info(f"ETF持仓数据已保存到 {file_path}")
                    else:
                        self.logger.warning(f"未找到ETF {ticker} 的持仓数据")
                except Exception as inner_e:
                    self.logger.error(f"获取ETF持仓时出现内部错误: {str(inner_e)}")
            
            except Exception as e:
                self.logger.error(f"获取ETF {ticker} 的持仓数据时出错: {str(e)}")
        
        return holdings_dict
    
    def download_market_sentiment(self) -> pd.DataFrame:
        """
        获取市场情绪数据（如恐慌指数等）
        
        返回:
            包含市场情绪的DataFrame
        """
        try:
            self.logger.info("获取A股市场情绪数据...")
            
            # 获取A股情绪指标
            # 注意：需要根据AKShare的最新文档确认获取情绪数据的API
            # 这里用股市情绪指标作为示例
            sentiment_data = ak.stock_market_emotion_baidu()
            
            if sentiment_data is not None and not sentiment_data.empty:
                # 保存到CSV
                file_name = f"market_sentiment_{datetime.now().strftime('%Y%m%d')}.csv"
                file_path = os.path.join(self.data_path, file_name)
                sentiment_data.to_csv(file_path, index=False)
                self.logger.info(f"市场情绪数据已保存到 {file_path}")
                
                return sentiment_data
            else:
                self.logger.warning("无法获取市场情绪数据")
                return pd.DataFrame()
        
        except Exception as e:
            self.logger.error(f"获取市场情绪数据时出错: {str(e)}")
            return pd.DataFrame()
    
    def prepare_supplementary_data(self) -> Dict[str, pd.DataFrame]:
        """
        准备所有补充数据
        
        返回:
            包含所有补充数据的字典
        """
        supplementary_data = {}
        
        # 1. 获取ETF基本信息
        etf_info = self.download_etf_fund_info(TICKER_LIST)
        supplementary_data['etf_info'] = etf_info
        
        # 2. 获取ETF资金流向
        fund_flow = self.download_etf_fund_flow(
            TICKER_LIST,
            TRAIN_START_DATE,
            TEST_END_DATE
        )
        supplementary_data['fund_flow'] = fund_flow
        
        # 3. 获取ETF持仓
        holdings = self.download_etf_holdings(TICKER_LIST)
        supplementary_data['holdings'] = holdings
        
        # 4. 获取市场情绪
        sentiment = self.download_market_sentiment()
        supplementary_data['market_sentiment'] = sentiment
        
        return supplementary_data 