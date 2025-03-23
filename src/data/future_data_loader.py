import pandas as pd
import backtrader as bt
from loguru import logger
import tushare as ts
import os

class FutureDataLoader:
    def __init__(self, ts_code=None, start_date=None, end_date=None, token=None):
        """初始化期货数据加载器
        Args:
            ts_code: 期货合约代码，如 'M2405.DCE' 表示大连商品交易所2405豆粕合约
                     如果为None，自动获取当前豆粕主力合约
            start_date: 开始日期，格式：'YYYY-MM-DD'
            end_date: 结束日期，格式：'YYYY-MM-DD'
            token: Tushare API token，如果不提供则从环境变量获取
        """
        self.ts_code = ts_code
        self.start_date = start_date
        self.end_date = end_date
        
        # 设置Tushare token
        self.token = token or os.environ.get('TUSHARE_TOKEN')
        if not self.token:
            logger.warning("Tushare token未设置，请设置TUSHARE_TOKEN环境变量或直接提供token参数")
        else:
            ts.set_token(self.token)
            
    def get_dominant_contract(self, product='M'):
        """获取主力合约
        Args:
            product: 期货品种代码，默认为豆粕(M)
        Returns:
            str: 主力合约代码
        """
        try:
            pro = ts.pro_api()
            # 获取期货主力合约信息
            df = pro.fut_mapping(ts_code=f'{product}.DCE')
            # 获取最新的主力合约
            latest_dominant = df.iloc[0]['mapping_ts_code'] if not df.empty else None
            logger.info(f"获取到豆粕主力合约: {latest_dominant}")
            return latest_dominant
        except Exception as e:
            logger.error(f"获取主力合约失败: {str(e)}")
            # 如果获取失败，返回一个默认的豆粕合约
            return "M2405.DCE"
        
    def load(self):
        """加载期货数据
        Returns:
            backtrader.feeds.PandasData: 期货数据源
        """
        try:
            # 如果没有指定合约代码，获取豆粕主力合约
            if not self.ts_code:
                self.ts_code = self.get_dominant_contract()
                logger.info(f"使用豆粕主力合约: {self.ts_code}")
            
            # 使用Tushare API获取期货日线数据
            pro = ts.pro_api()
            df = pro.fut_daily(ts_code=self.ts_code, 
                               start_date=self.start_date.strftime("%Y%m%d") if self.start_date else None,
                               end_date=self.end_date.strftime("%Y%m%d") if self.end_date else None)
            
            if df.empty:
                logger.error(f"未获取到期货数据: {self.ts_code}")
                raise ValueError(f"未获取到期货数据: {self.ts_code}")
                
            # 调整数据格式以符合backtrader要求
            df['datetime'] = pd.to_datetime(df['trade_date'])
            df = df.sort_values('datetime')  # 确保按时间排序
            
            # 重命名列名以符合backtrader需求
            df = df.rename(columns={
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'vol': 'volume',
                'oi': 'openinterest'  # 持仓量
            })
            
            # 设置索引
            df.set_index('datetime', inplace=True)
            
            # 创建数据源
            data = bt.feeds.PandasData(
                dataname=df,
                datetime=None,  # 使用索引作为日期
                open='open',
                high='high',
                low='low',
                close='close',
                volume='volume',
                openinterest='openinterest',
                name=self.ts_code
            )
            
            # 添加期货代码属性
            data._name = self.ts_code
            data.ts_code = self.ts_code
            
            logger.info(f"成功加载期货数据: {self.ts_code}, 数据行数: {len(df)}")
            return data
            
        except Exception as e:
            logger.error(f"加载期货数据失败: {str(e)}")
            import traceback
            traceback.print_exc()
            raise 