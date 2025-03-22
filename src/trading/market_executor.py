import pandas as pd
import os
from datetime import datetime, timedelta
import time
import tushare as ts
import akshare as ak
import backtrader as bt
from src.utils.logger import setup_logger
from src.utils.notification import send_notification
from src.strategies.market_sentiment_strategy import MarketSentimentStrategy
from src.data.data_loader import DataLoader
from src.utils.analysis import Analysis

logger = setup_logger()

class MarketExecutor:
    def __init__(self, symbols: list, tushare_token: str):
        self.symbols = symbols
        self.records_dir = "data/trading_records"
        self.data_loader = DataLoader(tushare_token=tushare_token)
        self.analysis = Analysis()
        os.makedirs(self.records_dir, exist_ok=True)
        
    def is_trading_day(self):
        """判断当前是否为交易日"""
        today = datetime.now().date()
        # 获取交易日历
        trade_cal = self.data_loader.pro.trade_cal(exchange='SSE', 
                                 start_date=today.strftime('%Y%m%d'),
                                 end_date=today.strftime('%Y%m%d'))
        return trade_cal.iloc[0]['is_open'] == 1
        
    def get_realtime_data(self, ts_code):
        """获取实时行情数据"""
        try:
            # 获取实时行情
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
                
            realtime_data = ak.stock_individual_spot_xq(symbol=market_code)
            if realtime_data is not None and not realtime_data.empty:
                # 将数据转换为以item为索引的格式
                realtime_data = realtime_data.set_index('item')
                # 转换数据格式以匹配原有接口
                return pd.DataFrame({
                    'ts_code': [ts_code],
                    'open': [float(realtime_data.loc['今开', 'value'])],
                    'high': [float(realtime_data.loc['最高', 'value'])],
                    'low': [float(realtime_data.loc['最低', 'value'])],
                    'close': [float(realtime_data.loc['现价', 'value'])],
                    'pre_close': [float(realtime_data.loc['昨收', 'value'])],
                    'vol': [float(realtime_data.loc['成交量', 'value'])],
                    'amount': [float(realtime_data.loc['成交额', 'value'])]
                })
            return None
        except Exception as e:
            logger.error(f"获取实时数据失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
            
    def execute(self):
        """执行策略回测并记录交易信号"""
        try:
            # 检查是否为交易日
            if not self.is_trading_day():
                logger.info("当前不是交易日，跳过执行")
                return
                
            # 获取当前时间
            current_time = datetime.now()
                            
            # 遍历上证50成分股进行回测
            for symbol in self.symbols:
                try:
                    # 创建回测引擎
                    cerebro = bt.Cerebro()
                    
                    # 设置初始资金
                    cerebro.broker.setcash(1000000.0)
                    
                    # 设置交易手续费
                    cerebro.broker.setcommission(commission=0.0003)
                    
                    # 获取历史数据
                    start_date = (current_time - timedelta(days=365))
                    end_date = current_time
                    
                    # 获取实时数据
                    # realtime_data = self.get_realtime_data(symbol)
                    # if realtime_data is None:
                    #     logger.error("获取实时数据失败，跳过本次执行")
                    #     continue
                    
                    # 使用DataLoader加载数据
                    data = self.data_loader.download_data(
                        symbol=symbol,
                        start_date=start_date,
                        end_date=end_date
                    )

                    if data is None:
                        logger.error("获取数据失败，跳过本次执行")
                        continue
                    
                    # 添加数据到回测引擎
                    cerebro.adddata(data)
                    
                    # 添加策略
                    cerebro.addstrategy(MarketSentimentStrategy)
                    
                    # 添加分析器
                    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', riskfreerate=0.02)
                    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
                    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
                    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
                    cerebro.addanalyzer(bt.analyzers.VWR, _name='vwr')
                    cerebro.addanalyzer(bt.analyzers.SQN, _name='sqn')
                    cerebro.addanalyzer(bt.analyzers.Transactions, _name='txn')
                    
                    # 运行回测
                    results = cerebro.run()
                    strategy = results[0]
                    
                    # 初始化trades列表
                    cerebro.trades = []
                    
                    # 使用Analysis类获取回测结果
                    analysis = self.analysis._get_analysis(cerebro, strategy)

                    logger.info(f"回测结果: {analysis}")
                    
                    # 获取交易记录
                    if 'trades' in analysis and not analysis['trades'].empty:
                        # 获取当天的交易记录
                        # today_trades = analysis['trades'][
                        #     analysis['trades']['交易时间'] == current_time.strftime('%Y-%m-%d')
                        # ]
                        today_trades = analysis['trades']
                        logger.info(f"当天交易记录: {today_trades}, 所有交易记录: {analysis['trades']}")

                        # 记录当天的交易
                        for _, trade in today_trades.iterrows():
                            action = "buy" if trade['方向'] == '买入' else "sell"
                            signal = 1 if action == "buy" else -1
                            
                            self._record_trade(
                                symbol=symbol,
                                action=action,
                                timestamp=current_time,
                                signal=signal,
                                price=float(trade['成交价']),
                                size=int(trade['数量']),
                                reason=trade['交易原因']
                            )
                            
                            # 发送通知
                            message = f"交易提醒: {symbol} {action.upper()} 信号\n"
                            message += f"价格: {trade['成交价']}, 数量: {trade['数量']}\n"
                            message += f"原因: {trade['交易原因']}"
                            send_notification(message)
                            
                except Exception as e:
                    logger.error(f"处理股票 {symbol} 时出错: {str(e)}")
                    continue
                    
        except Exception as e:
            logger.error(f"执行策略回测时出错: {str(e)}")
            send_notification(f"策略回测错误: {str(e)}")
            
    def _record_trade(self, symbol: str, action: str, timestamp: datetime, signal: float, 
                     price: float, size: int, reason: str):
        """记录交易到CSV文件"""
        record_file = os.path.join(self.records_dir, f"trades_{timestamp.strftime('%Y%m%d')}.csv")
        
        record = {
            "timestamp": timestamp,
            "symbol": symbol,
            "action": action,
            "signal": signal,
            "price": price,
            "size": size,
            "reason": reason,
            "strategy": "market_sentiment"
        }
        
        df = pd.DataFrame([record])
        
        if os.path.exists(record_file):
            df.to_csv(record_file, mode='a', header=False, index=False)
        else:
            df.to_csv(record_file, index=False)
            
    def run_continuously(self, interval: int = 3600):
        """持续运行策略回测，每小时执行一次"""
        logger.info("启动实盘交易系统")
        while True:
            self.execute()
            time.sleep(interval) 