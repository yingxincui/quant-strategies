import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from loguru import logger
import talib
import tushare as ts
import os
import backtrader as bt

class FutureDataLoader:
    def __init__(self, ts_code=None, start_date=None, end_date=None, token=None):
        """初始化期货数据加载器
        Args:
            ts_code: 期货合约代码，如 'M2405.DCE' 表示大连商品交易所2405豆粕合约
                     如果为None，自动获取当前豆粕主力合约
            start_date: 开始日期，datetime对象
            end_date: 结束日期，datetime对象
            token: Tushare API token，如果不提供则从环境变量获取
        """
        self.ts_code = ts_code
        self.start_date = start_date
        self.end_date = end_date
        self.future_code = 'M'  # 期货代码，如'M'代表豆粕
        self.contract_multiplier = 10  # 合约乘数，豆粕为10吨/手
        
        # 设置Tushare token
        self.token = token or os.environ.get('TUSHARE_TOKEN')
        if not self.token:
            logger.warning("Tushare token未设置，请设置TUSHARE_TOKEN环境变量或直接提供token参数")
        else:
            ts.set_token(self.token)
            self.pro = ts.pro_api()
            logger.info("期货Tushare API初始化成功")
            
    def get_dominant_contracts(self):
        """获取指定时间段内的主力合约列表
        Returns:
            list: 主力合约列表，每个元素为(合约代码, 开始日期, 结束日期)的元组
        """
        if not self.start_date or not self.end_date:
            logger.error("未提供开始日期或结束日期")
            return []
            
        try:
            # 确保开始和结束日期是 datetime64[ns] 类型
            start_date = pd.to_datetime(self.start_date)
            end_date = pd.to_datetime(self.end_date)
            
            # 根据结束月份确定是否需要获取下一个主力合约
            end_month = end_date.month
            next_month_map = {
                1: 3,   # 1月份需要获取3月合约
                3: 5,   # 3月份需要获取5月合约
                5: 7,   # 5月份需要获取7月合约
                7: 9,   # 7月份需要获取9月合约
                9: 11,  # 9月份需要获取11月合约
                11: 1   # 11月份需要获取次年1月合约
            }
            
            # 如果结束月份在映射中，延长结束日期到下个月中旬
            if end_month in next_month_map:
                next_month = next_month_map[end_month]
                if next_month == 1:  # 如果是次年1月合约
                    extended_end = pd.Timestamp(year=end_date.year + 1, month=1, day=15)
                else:
                    extended_end = pd.Timestamp(year=end_date.year, month=next_month, day=15)
                logger.info(f"结束月份为{end_month}月，延长数据获取至{extended_end.strftime('%Y-%m-%d')}以获取下一个主力合约")
            else:
                extended_end = end_date
            
            logger.info(f"开始获取主力合约列表 - 时间段: {start_date.date()} 至 {extended_end.date()}")
            
            # 获取所有可交易的合约
            logger.info(f"正在获取{self.future_code}期货合约列表...")
            contracts = self.pro.fut_basic(fut_code=f"{self.future_code}", 
                                    exchange='DCE', )
            
            if contracts is None or contracts.empty:
                logger.error(f"未获取到{self.future_code}期货合约数据")
                return []
                
            logger.info(f"获取到{len(contracts)}个合约")
            logger.debug(f"合约列表:\n{contracts}")
            
            # 将日期字符串转换为datetime对象
            contracts['last_ddate'] = pd.to_datetime(contracts['last_ddate'])
            
            # 按交割日期排序
            contracts = contracts.sort_values('last_ddate')
            logger.info(f"按交割日期排序后的合约列表:\n{contracts[['ts_code', 'last_ddate']]}")
            
            # 获取主力合约列表
            dominant_contracts = []
            current_date = start_date
            
            while current_date <= extended_end:  # 使用延长后的结束日期
                logger.debug(f"处理日期: {current_date.date()}")
                
                # 找到当前日期可用的合约
                available_contracts = contracts[
                    (contracts['last_ddate'] > current_date) & 
                    (contracts['last_ddate'] <= extended_end + pd.Timedelta(days=15))  # 确保合约在回测期间内
                ]
                
                if len(available_contracts) == 0:
                    logger.warning(f"在{current_date.date()}之后没有可用的合约")
                    break
                
                # 获取每个合约的成交量数据
                contract_volumes = []
                for _, contract in available_contracts.iterrows():
                    try:
                        # 获取合约的成交量数据
                        df = self.pro.fut_daily(
                            ts_code=contract['ts_code'],
                            start_date=current_date.strftime('%Y%m%d'),
                            end_date=(current_date + pd.Timedelta(days=5)).strftime('%Y%m%d'),
                            fields='ts_code,trade_date,vol'
                        )
                        
                        if df is not None and not df.empty:
                            avg_volume = df['vol'].mean()
                            contract_volumes.append({
                                'ts_code': contract['ts_code'],
                                'last_ddate': contract['last_ddate'],
                                'avg_volume': avg_volume
                            })
                    except Exception as e:
                        logger.warning(f"获取合约{contract['ts_code']}成交量数据失败: {str(e)}")
                        continue
                
                if not contract_volumes:
                    logger.warning(f"在{current_date.date()}没有可用的合约成交量数据")
                    break
                
                # 按成交量排序，选择成交量最大的合约
                contract_volumes.sort(key=lambda x: x['avg_volume'], reverse=True)
                dominant_contract = contract_volumes[0]
                
                logger.debug(f"选择的主力合约: {dominant_contract['ts_code']}, 交割日期: {dominant_contract['last_ddate'].date()}, 平均成交量: {dominant_contract['avg_volume']:.0f}")
                
                # 确定该合约的主力时间段
                contract_start = current_date
                # 提前30天切换主力合约
                contract_end = min(extended_end, dominant_contract['last_ddate'] - pd.Timedelta(days=30))
                
                # 确保结束日期大于开始日期
                if contract_end <= contract_start:
                    logger.warning(f"合约{dominant_contract['ts_code']}的结束日期{contract_end.date()}小于等于开始日期{contract_start.date()}，跳过")
                    break
                
                dominant_contracts.append({
                    'ts_code': dominant_contract['ts_code'],
                    'start_date': contract_start,
                    'end_date': contract_end
                })
                
                logger.debug(f"添加主力合约: {dominant_contract['ts_code']}, 时间段: {contract_start.date()} - {contract_end.date()}")
                
                # 更新当前日期为下一个合约的开始日期
                current_date = contract_end + pd.Timedelta(days=1)
                
                # 防止死循环
                if current_date > extended_end:
                    break
            
            logger.info(f"最终获取到{len(dominant_contracts)}个主力合约")
            for contract in dominant_contracts:
                logger.info(f"主力合约: {contract['ts_code']}, 时间段: {contract['start_date'].date()} - {contract['end_date'].date()}")
                
            return dominant_contracts
            
        except Exception as e:
            logger.error(f"获取主力合约列表失败: {str(e)}")
            import traceback
            logger.error(f"错误详情:\n{traceback.format_exc()}")
            return []
        
    def load(self):
        """加载期货数据
        Returns:
            backtrader.feeds.PandasData: 期货数据源
        """
        try:
            # 获取主力合约列表
            dominant_contracts = self.get_dominant_contracts()
            
            if not dominant_contracts:
                logger.error(f"未找到{self.future_code}期货的主力合约")
                raise ValueError(f"未找到{self.future_code}期货的主力合约")
                
            logger.info(f"获取到{self.future_code}期货主力合约列表:")
            for contract in dominant_contracts:
                logger.info(f"合约: {contract['ts_code']}, 时间段: {contract['start_date'].date()} - {contract['end_date'].date()}")
            
            # 获取每个合约的数据并拼接
            all_data = []
            
            for contract in dominant_contracts:
                try:
                    # 获取合约数据
                    df = self.pro.fut_daily(
                        ts_code=contract['ts_code'],
                        start_date=contract['start_date'].strftime('%Y%m%d'),
                        end_date=contract['end_date'].strftime('%Y%m%d'),
                        fields='ts_code,trade_date,open,high,low,close,vol,amount,oi'  # 添加oi字段
                    )
                    
                    if df is not None and not df.empty:
                        # 添加合约信息
                        df['contract'] = contract['ts_code']
                        all_data.append(df)
                        logger.info(f"成功加载合约{contract['ts_code']}的数据，行数: {len(df)}")
                    else:
                        logger.warning(f"合约{contract['ts_code']}没有数据")
                        
                except Exception as e:
                    logger.error(f"加载合约{contract['ts_code']}数据时出错: {str(e)}")
                    continue
            
            if not all_data:
                logger.error("没有成功加载任何合约数据")
                raise ValueError("没有成功加载任何合约数据")
                
            # 合并所有合约数据
            combined_data = pd.concat(all_data, ignore_index=True)
            
            # 按日期排序
            combined_data = combined_data.sort_values('trade_date')
            
            # 创建合约代码映射
            self.contract_mapping = pd.Series(combined_data['contract'].values, index=pd.to_datetime(combined_data['trade_date'])).to_dict()
            
            # 调整数据格式以符合backtrader要求
            combined_data['datetime'] = pd.to_datetime(combined_data['trade_date'])
            
            # 重命名列名以符合backtrader需求
            combined_data = combined_data.rename(columns={
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'vol': 'volume',
                'oi': 'openinterest'  # 持仓量
            })
            
            # 设置索引
            combined_data.set_index('datetime', inplace=True)
            
            # 创建数据源
            data = bt.feeds.PandasData(
                dataname=combined_data,
                datetime=None,  # 使用索引作为日期
                open='open',
                high='high',
                low='low',
                close='close',
                volume='volume',
                openinterest='openinterest',
                name=f"{self.future_code}_DOMINANT"
            )
            
            # 添加期货代码属性和合约映射
            data._name = f"{self.future_code}_DOMINANT"
            data.ts_code = f"{self.future_code}_DOMINANT"
            data.contract_mapping = self.contract_mapping  # 添加合约映射字典
            
            logger.info(f"成功合并{len(dominant_contracts)}个合约的数据，总行数: {len(combined_data)}")
            return data
            
        except Exception as e:
            logger.error(f"加载期货数据失败: {str(e)}")
            import traceback
            traceback.print_exc()
            raise 