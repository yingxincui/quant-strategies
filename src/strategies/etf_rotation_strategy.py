import backtrader as bt
import datetime
from loguru import logger
import math

class ETFRotationStrategy(bt.Strategy):
    params = (
        ('momentum_period', 20),  # 动量计算周期
        ('rebalance_interval', 30),  # 调仓间隔天数
        ('num_positions', 1),     # 持有前N个ETF
        ('risk_ratio', 0.02),     # 单次交易风险比率
        ('max_drawdown', 0.15),   # 最大回撤限制
        ('trail_percent', 2.5),   # 追踪止损百分比，从1.5%提高到2.5%
        ('min_hold_days', 5),     # 最小持仓天数
        ('verbose', True),
    )


    def __init__(self):
        # 初始化指标字典
        self.inds = {}
        
        # 存储ETF代码映射
        self.etf_codes = {}
        
        # 为每个数据源设置名称
        for i, d in enumerate(self.datas):
            # 从数据源获取ETF代码
            etf_code = None
            try:
                # 首先尝试从数据源的params属性获取
                if hasattr(d, 'params') and hasattr(d.params, 'ts_code'):
                    etf_code = d.params.ts_code
                # 然后尝试从数据源的其他属性获取
                elif hasattr(d, 'ts_code'):
                    etf_code = d.ts_code
                # 最后尝试从数据源的_name属性获取
                elif hasattr(d, '_name'):
                    etf_code = d._name
            except Exception as e:
                logger.warning(f"获取ETF代码时出错: {str(e)}")
            
            # 如果没有找到ETF代码，使用默认名称
            if not etf_code:
                etf_code = f"ETF_{i+1}"
            
            # 设置数据源的名称
            d._name = etf_code
            self.etf_codes[d] = etf_code
            
            # 计算每个ETF的动量（过去N日收益率）
            self.inds[d] = bt.indicators.Momentum(d.close, 
                                period=self.p.momentum_period)
            
            # 添加ATR指标用于风险管理
            self.inds[d].atr = bt.indicators.ATR(d, period=14)
            
            # 打印每个ETF的代码信息
            logger.info(f"初始化ETF数据源: {etf_code}")
        
        # 设置下次调仓日期
        self.last_rebalance = None
        
        # 记录订单
        self.orders = {}
        
        # 记录最高净值，用于计算回撤
        self.highest_value = self.broker.getvalue()
        
        # 原因记录
        self.trade_reasons = {}
        
        # 记录买入的ETF
        self.bought_etfs = []
        
        # 添加订单列表用于记录所有交易
        self._orders = []
        
        # 打印初始化信息
        logger.info(f"ETF轮换策略初始化完成 - 参数: 动量周期={self.p.momentum_period}, 调仓间隔={self.p.rebalance_interval}天, "
                  f"持仓数量={self.p.num_positions}, 风险比例={self.p.risk_ratio:.2%}, 追踪止损={self.p.trail_percent}%")
        logger.info(f"加载的ETF列表: {[d._name for d in self.datas]}")
        
        self.entry_dates = {}  # 记录每个ETF的买入日期

    def log(self, txt, dt=None):
        """日志功能"""
        if self.p.verbose:
            dt = dt or self.data.datetime.date(0)
            logger.info(f'{dt.isoformat()} - {txt}')

    def round_shares(self, shares):
        """将股数调整为100的整数倍"""
        return math.floor(shares / 100) * 100

    def calculate_position_size(self, data, price):
        """计算头寸大小，考虑风险"""
        cash = self.broker.getcash()
        # 预留手续费缓冲
        cash = cash * 0.95
        
        # 计算风险金额
        total_value = self.broker.getvalue()
        risk_amount = total_value * self.p.risk_ratio / self.p.num_positions
        
        # 使用ATR计算每股风险
        current_atr = self.inds[data].atr[0]
        risk_per_share = current_atr * 1.5
        
        # 如果ATR过小，使用传统的百分比止损
        min_risk = price * (self.p.trail_percent/100.0)
        risk_per_share = max(risk_per_share, min_risk)
        
        # 根据风险计算的股数
        risk_size = risk_amount / risk_per_share if risk_per_share > 0 else 0
        
        # 根据可用资金和ETF数量计算的股数
        equal_allocation = cash / self.p.num_positions
        cash_size = equal_allocation / price
        
        # 取较小值并调整为100股整数倍
        shares = min(risk_size, cash_size)
        shares = self.round_shares(shares)
        
        # 再次验证金额是否超过可用资金
        if shares * price > equal_allocation:
            shares = self.round_shares(equal_allocation / price)
            
        self.log(f"计算{data._name}持仓 - ATR: {current_atr:.2f}, 每股风险: {risk_per_share:.2f}, 总风险金额: {risk_amount:.2f}, 计算股数: {shares}")
        
        return shares if shares >= 100 else 0

    def next(self):
        # 计算当前回撤
        current_value = self.broker.getvalue()
        self.highest_value = max(self.highest_value, current_value)
        drawdown = (self.highest_value - current_value) / self.highest_value
        
        # 如果回撤超过限制，清仓不开新仓
        if drawdown > self.p.max_drawdown:
            if any(self.getposition(d).size > 0 for d in self.datas):
                self.log(f"触发最大回撤限制 - 当前回撤: {drawdown:.2%}, 限制: {self.p.max_drawdown:.2%}")
                for d in self.datas:
                    if self.getposition(d).size > 0:
                        order = self.close(data=d)
                        order.data = d  # 确保订单的data属性指向正确的ETF
                        self.orders[d] = order
                        self.trade_reasons[d] = f"触发最大回撤限制 ({drawdown:.2%})"
            return
        
        # 检查是否到达调仓日
        if not self._time_to_rebalance():
            # 非调仓日检查止损
            self._check_trailing_stop()
            return

        # 计算各ETF动量并排序
        rankings = sorted(
            self.datas,
            key=lambda d: self.inds[d][0],
            reverse=True
        )
        
        # 选择前N个ETF
        top_etfs = rankings[:self.p.num_positions]
        
        # 先卖出不在top_etfs中的当前持仓
        for d in self.datas:
            if self.getposition(d).size > 0 and d not in top_etfs:
                order = self.close(data=d)
                order.data = d  # 确保订单的data属性指向正确的ETF
                self.orders[d] = order
                self.trade_reasons[d] = f"调仓期间退出 - 不再是动量最强的ETF"
                self.log(f'卖出 {d._name} - 不再是动量最强的ETF')
                # 从已购买ETF列表中移除
                if d in self.bought_etfs:
                    self.bought_etfs.remove(d)

        # 买入新标的
        for d in top_etfs:
            if self.getposition(d).size == 0:
                # 计算头寸大小
                price = d.close[0]
                size = self.calculate_position_size(d, price)
                
                if size >= 100:  # 确保至少有100股
                    # 创建买入订单并设置data属性
                    order = self.buy(data=d, size=size)
                    order.data = d  # 确保订单的data属性指向正确的ETF
                    self.orders[d] = order
                    self.trade_reasons[d] = f"动量排名第{top_etfs.index(d)+1}，信号强度: {self.inds[d][0]:.2f}"
                    self.log(f'买入 {d._name} - 动量排名第{top_etfs.index(d)+1}，信号强度: {self.inds[d][0]:.2f}, 数量: {size}')
                    self.bought_etfs.append(d)
                    # 记录买入日期
                    self.entry_dates[d] = self.data.datetime.date(0)

    def _check_trailing_stop(self):
        """检查追踪止损"""
        for d in self.datas:
            position = self.getposition(d)
            if position.size > 0:
                # 检查最小持仓时间
                if d in self.entry_dates:
                    hold_days = (self.data.datetime.date(0) - self.entry_dates[d]).days
                    if hold_days < self.p.min_hold_days:
                        continue
                
                # 获取当前价格和入场价格
                current_price = d.close[0]
                entry_price = position.price
                
                # 计算最大价格（用于追踪止损）
                if not hasattr(self, 'max_prices'):
                    self.max_prices = {}
                
                if d not in self.max_prices:
                    self.max_prices[d] = entry_price
                else:
                    self.max_prices[d] = max(self.max_prices[d], current_price)
                
                # 计算止损价
                stop_price = self.max_prices[d] * (1 - self.p.trail_percent / 100)
                
                # 如果价格低于止损价，卖出
                if current_price < stop_price:
                    # 创建卖出订单并设置data属性
                    order = self.close(data=d)
                    order.data = d  # 确保订单的data属性指向正确的ETF
                    self.orders[d] = order
                    self.trade_reasons[d] = f"触发追踪止损 (止损价: {stop_price:.2f})"
                    self.log(f'追踪止损触发 - 卖出 {d._name}, 当前价格: {current_price:.2f}, 止损价: {stop_price:.2f}, 最高价: {self.max_prices[d]:.2f}')
                    # 从已购买ETF列表中移除
                    if d in self.bought_etfs:
                        self.bought_etfs.remove(d)
                    # 清除买入日期记录
                    if d in self.entry_dates:
                        del self.entry_dates[d]

    def _time_to_rebalance(self):
        # 按时间间隔调仓
        if not self.last_rebalance:
            self.last_rebalance = self.data.datetime.date(0)
            return True
            
        days_since = (self.data.datetime.date(0) - self.last_rebalance).days
        if days_since >= self.p.rebalance_interval:
            self.last_rebalance = self.data.datetime.date(0)
            return True
        return False

    def notify_order(self, order):
        """订单状态通知"""
        if order.status in [order.Submitted, order.Accepted]:
            return
            
        # 获取订单对应的数据源
        d = order.data if hasattr(order, 'data') else None
        
        # 如果是self.orders中的订单，使用原有的处理逻辑
        if d and d in self.orders and self.orders[d] == order:
            if order.status == order.Completed:
                # 计算持仓市值
                position = self.getposition(d)
                position_value = position.size * order.executed.price if order.isbuy() else 0
                
                # 计算所有ETF的总持仓市值
                total_position_value = sum(
                    self.getposition(data).size * data.close[0]
                    for data in self.datas
                )
                
                # 获取ETF代码
                etf_code = self.etf_codes.get(d, d._name)
                
                # 获取订单执行时的实际日期
                order_date = self.data.datetime.date(0)  # 使用当前数据时间作为订单执行时间
                
                # 添加总资产和持仓市值信息
                order.info = {
                    'reason': self.trade_reasons.get(d, "未记录"),
                    'total_value': self.broker.getvalue(),
                    'position_value': total_position_value,
                    'etf_code': etf_code,
                    'execution_date': order_date  # 添加执行日期
                }
                
                # 将订单添加到订单列表中
                self._orders.append(order)
                
                if order.isbuy():
                    self.log(f'{etf_code} 买入完成 - 价格: {order.executed.price:.2f}, 数量: {order.executed.size}, '
                           f'佣金: {order.executed.comm:.2f}, 原因: {self.trade_reasons.get(d, "未记录")}, '
                           f'总资产: {self.broker.getvalue():.2f}, 持仓市值: {total_position_value:.2f}',
                           dt=order_date)  # 使用订单执行日期
                else:
                    self.log(f'{etf_code} 卖出完成 - 价格: {order.executed.price:.2f}, 数量: {order.executed.size}, '
                           f'佣金: {order.executed.comm:.2f}, 原因: {self.trade_reasons.get(d, "未记录")}, '
                           f'总资产: {self.broker.getvalue():.2f}, 持仓市值: {total_position_value:.2f}',
                           dt=order_date)  # 使用订单执行日期
                    
                    # 如果是卖出操作，从最高价记录中移除该ETF
                    if hasattr(self, 'max_prices') and d in self.max_prices:
                        del self.max_prices[d]
                        
            elif order.status in [order.Canceled, order.Margin, order.Rejected]:
                etf_code = self.etf_codes.get(d, d._name)
                self.log(f'{etf_code} 订单失败 - 状态: {order.getstatusname()}')
                
            # 无论成功失败都从订单字典中移除
            del self.orders[d]
        # 处理其他订单（如close()创建的订单）
        elif order.status == order.Completed:
            # 获取数据源
            d = order.data if hasattr(order, 'data') else None
            if d:
                # 获取ETF代码
                etf_code = self.etf_codes.get(d, d._name)
                
                # 计算持仓市值
                position = self.getposition(d)
                position_value = position.size * order.executed.price if order.isbuy() else 0
                
                # 计算所有ETF的总持仓市值
                total_position_value = sum(
                    self.getposition(data).size * data.close[0]
                    for data in self.datas
                )
                
                # 获取订单执行时的实际日期
                order_date = self.data.datetime.date(0)  # 使用当前数据时间作为订单执行时间
                
                # 添加总资产和持仓市值信息
                order.info = {
                    'reason': self.trade_reasons.get(d, "未记录"),
                    'total_value': self.broker.getvalue(),
                    'position_value': total_position_value,
                    'etf_code': etf_code,
                    'execution_date': order_date  # 添加执行日期
                }
                
                # 将订单添加到订单列表中
                self._orders.append(order)
                
                if order.isbuy():
                    self.log(f'{etf_code} 买入完成 - 价格: {order.executed.price:.2f}, 数量: {order.executed.size}, '
                           f'佣金: {order.executed.comm:.2f}, 原因: {self.trade_reasons.get(d, "未记录")}, '
                           f'总资产: {self.broker.getvalue():.2f}, 持仓市值: {total_position_value:.2f}',
                           dt=order_date)  # 使用订单执行日期
                else:
                    self.log(f'{etf_code} 卖出完成 - 价格: {order.executed.price:.2f}, 数量: {order.executed.size}, '
                           f'佣金: {order.executed.comm:.2f}, 原因: {self.trade_reasons.get(d, "未记录")}, '
                           f'总资产: {self.broker.getvalue():.2f}, 持仓市值: {total_position_value:.2f}',
                           dt=order_date)  # 使用订单执行日期
                    
                    # 如果是卖出操作，从最高价记录中移除该ETF
                    if hasattr(self, 'max_prices') and d in self.max_prices:
                        del self.max_prices[d]

    def stop(self):
        """策略结束时的汇总信息"""
        portfolio_value = self.broker.getvalue()
        returns = (portfolio_value / self.broker.startingcash) - 1.0
        
        logger.info(f"ETF轮换策略结束 - 最终资金: {portfolio_value:.2f}, 收益率: {returns:.2%}")
        if self.bought_etfs:
            logger.info(f"最终持仓ETF: {', '.join([etf._name for etf in self.bought_etfs])}")
