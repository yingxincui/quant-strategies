import backtrader as bt
import datetime
from loguru import logger
import math

class ETFRotationStrategy(bt.Strategy):
    params = (
        ('momentum_short', 10),    # 短期动量周期
        ('momentum_long', 60),     # 长期动量周期
        ('rebalance_interval', 30), # 调仓间隔天数
        ('num_positions', 1),      # 持有前N个ETF
        ('risk_ratio', 0.02),      # 单次交易风险比率
        ('max_drawdown', 0.15),    # 最大回撤限制
        ('trail_percent', 2.5),    # 追踪止损百分比
        ('min_hold_days', 5),      # 最小持仓天数
        ('profit_target1', 0.15),  # 第一止盈目标
        ('profit_target2', 0.30),  # 第二止盈目标
        ('atr_multiplier', 2.0),   # ATR倍数
        ('momentum_decay', 0.3),   # 动量衰减阈值
        ('market_trend_threshold', -0.05), # 市场趋势阈值
        ('vix_threshold', 0.03),   # 波动率阈值
        ('verbose', True),
    )

    def __init__(self):
        # 初始化指标字典
        self.inds = {}
        
        # 存储ETF代码映射
        self.etf_codes = {}
        
        # 市场状态指标
        self.market_state = MarketState(self.datas[0])
        
        # 为每个数据源设置名称和指标
        for i, d in enumerate(self.datas):
            # 从数据源获取ETF代码
            etf_code = None
            try:
                if hasattr(d, 'params') and hasattr(d.params, 'ts_code'):
                    etf_code = d.params.ts_code
                elif hasattr(d, 'ts_code'):
                    etf_code = d.ts_code
                elif hasattr(d, '_name'):
                    etf_code = d._name
            except Exception as e:
                logger.warning(f"获取ETF代码时出错: {str(e)}")
            
            if not etf_code:
                etf_code = f"ETF_{i+1}"
            
            d._name = etf_code
            self.etf_codes[d] = etf_code
            
            # 初始化该数据源的指标字典
            self.inds[d] = {}
            
            # 计算双动量指标
            mom_short = bt.indicators.Momentum(d.close, period=self.p.momentum_short)
            mom_long = bt.indicators.Momentum(d.close, period=self.p.momentum_long)
            # 添加一个小的偏移量避免除零
            self.inds[d]['momentum_score'] = mom_short / (mom_long + 1e-6)
            
            # 计算相对动量强度
            if i > 0:  # 假设第一个数据是基准指数
                rel_strength = d.close / (self.datas[0].close + 1e-6)
                self.inds[d]['rel_momentum'] = bt.indicators.ROC(rel_strength, period=20)
            
            # 添加ATR指标
            self.inds[d]['atr'] = bt.indicators.ATR(d, period=14)
            # 添加30天平均ATR指标
            self.inds[d]['atr_ma'] = bt.indicators.SMA(self.inds[d]['atr'], period=30)
            
            logger.info(f"初始化ETF数据源: {etf_code}")
        
        # 设置下次调仓日期
        self.last_rebalance = None
        
        # 记录订单和交易原因
        self.orders = {}
        self.trade_reasons = {}
        
        # 记录最高净值和买入日期
        self.highest_value = self.broker.getvalue()
        self.entry_dates = {}
        
        # 记录买入的ETF和最高价
        self.bought_etfs = []
        self.max_prices = {}
        
        # 记录止盈目标
        self.profit_targets = {}
        
        # 添加订单列表
        self._orders = []
        
        logger.info(f"ETF轮换策略初始化完成 - 参数: 短期动量={self.p.momentum_short}, 长期动量={self.p.momentum_long}, "
                  f"调仓间隔={self.p.rebalance_interval}天, 持仓数量={self.p.num_positions}, "
                  f"风险比例={self.p.risk_ratio:.2%}, 追踪止损={self.p.trail_percent}%")
        logger.info(f"加载的ETF列表: {[d._name for d in self.datas]}")

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
        current_atr = self.inds[data]['atr'][0]
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
        
        # 如果回撤超过限制，清仓并重置状态
        if drawdown > self.p.max_drawdown:
            if any(self.getposition(d).size > 0 for d in self.datas):
                self.log(f"触发最大回撤限制 - 当前回撤: {drawdown:.2%}, 限制: {self.p.max_drawdown:.2%}")
                for d in self.datas:
                    if self.getposition(d).size > 0:
                        order = self.close(data=d)
                        order.data = d
                        self.orders[d] = order
                        self.trade_reasons[d] = f"触发最大回撤限制 ({drawdown:.2%})"
                
                # 重置策略状态
                self.highest_value = current_value  # 重置最高净值
                self.bought_etfs = []  # 清空持仓ETF列表
                self.entry_dates = {}  # 清空买入日期记录
                self.max_prices = {}  # 清空最高价记录
                self.profit_targets = {}  # 清空止盈目标
                self.last_rebalance = None  # 重置调仓日期
                return  # 返回等待下一个交易日
        
        # 检查市场状态
        if self.market_state.trend[0] < self.p.market_trend_threshold:
            # 熊市清仓
            for d in self.datas:
                if self.getposition(d).size > 0:
                    order = self.close(data=d)
                    order.data = d
                    self.orders[d] = order
                    self.trade_reasons[d] = "熊市清仓"
            return
        
        # 计算市场波动率
        vix = sum(self.inds[d]['atr'][0]/d.close[0] for d in self.datas)/len(self.datas)
        
        # 高波动市场减仓
        target_positions = max(1, self.p.num_positions//2) if vix > self.p.vix_threshold else self.p.num_positions
        
        # 检查是否到达调仓日
        if not self._time_to_rebalance():
            # 非调仓日检查止损和动量衰减
            self._check_trailing_stop()
            self._check_momentum_decay()
            return

        # 计算各ETF动量并排序
        rankings = sorted(
            self.datas,
            key=lambda d: self.inds[d]['momentum_score'][0] * (1 + self.inds[d]['rel_momentum'][0] if hasattr(self.inds[d], 'rel_momentum') else 1),
            reverse=True
        )
        
        # 选择前N个ETF
        top_etfs = rankings[:target_positions]
        
        # 先卖出不在top_etfs中的当前持仓
        for d in self.datas:
            if self.getposition(d).size > 0 and d not in top_etfs:
                order = self.close(data=d)
                order.data = d
                self.orders[d] = order
                self.trade_reasons[d] = f"调仓期间退出 - 不再是动量最强的ETF"
                self.log(f'卖出 {d._name} - 不再是动量最强的ETF')
                if d in self.bought_etfs:
                    self.bought_etfs.remove(d)

        # 买入新标的
        for d in top_etfs:
            if self.getposition(d).size == 0:
                price = d.close[0]
                size = self.calculate_position_size(d, price)
                
                if size >= 100:
                    order = self.buy(data=d, size=size)
                    order.data = d
                    self.orders[d] = order
                    self.trade_reasons[d] = f"动量排名第{top_etfs.index(d)+1}，信号强度: {self.inds[d]['momentum_score'][0]:.2f}"
                    self.log(f'买入 {d._name} - 动量排名第{top_etfs.index(d)+1}，信号强度: {self.inds[d]["momentum_score"][0]:.2f}, 数量: {size}')
                    self.bought_etfs.append(d)
                    self.entry_dates[d] = self.data.datetime.date(0)
                    
                    # 设置止盈目标
                    self.profit_targets[d] = {
                        'level1': price * (1 + self.p.profit_target1),
                        'level2': price * (1 + self.p.profit_target2)
                    }

    def _time_to_rebalance(self):
        # 计算市场波动率（以ATR为代理）
        current_atr = sum(self.inds[d]['atr'][0] for d in self.datas) / len(self.datas)
        # 使用30天平均ATR
        avg_atr = sum(self.inds[d]['atr_ma'][0] for d in self.datas) / len(self.datas)
        
        # 波动率上升时缩短调仓周期
        if current_atr > avg_atr * 1.2:
            return (self.data.datetime.date(0) - self.last_rebalance).days >= 15
        else:
            if not self.last_rebalance:
                self.last_rebalance = self.data.datetime.date(0)
                return True
                
            days_since = (self.data.datetime.date(0) - self.last_rebalance).days
            if days_since >= self.p.rebalance_interval:
                self.last_rebalance = self.data.datetime.date(0)
                return True
            return False

    def _check_momentum_decay(self):
        """检查动量衰减"""
        for d in self.datas:
            if self.getposition(d).size > 0:
                # 计算动量衰减率
                mom_curr = self.inds[d]['momentum_score'][0]
                mom_prev = self.inds[d]['momentum_score'][-5]
                # 添加一个小的偏移量避免除零
                decay_rate = (mom_prev - mom_curr) / (abs(mom_prev) + 1e-6)
                
                if decay_rate > self.p.momentum_decay:
                    order = self.close(data=d)
                    order.data = d
                    self.orders[d] = order
                    self.trade_reasons[d] = f"动量快速衰减 ({decay_rate:.2%})"
                    self.log(f'动量衰减触发 - 卖出 {d._name}, 衰减率: {decay_rate:.2%}')
                    if d in self.bought_etfs:
                        self.bought_etfs.remove(d)
                    if d in self.entry_dates:
                        del self.entry_dates[d]

    def _check_trailing_stop(self):
        """检查复合止损"""
        for d in self.datas:
            pos = self.getposition(d)
            if pos.size == 0:
                continue
            
            price = d.close[0]
            entry = pos.price
            
            # 初始化hold_days变量
            hold_days = 0
            if d in self.entry_dates:
                hold_days = (self.data.datetime.date(0) - self.entry_dates[d]).days
            
            # 检查最小持仓时间
            if hold_days < self.p.min_hold_days:
                continue
            
            # 1. 动态追踪止损（基于ATR）
            atr_stop = price - self.p.atr_multiplier * self.inds[d]['atr'][0]
            
            # 2. 时间止损（持仓超30天强制退出）
            time_stop = hold_days > 30
            
            # 3. 盈利回撤止损（从最高点回落7%）
            max_price = max(self.max_prices.get(d, entry), price)
            self.max_prices[d] = max_price
            drawdown_stop = price < max_price * 0.93
            
            # 4. 分阶段止盈
            if d in self.profit_targets:
                targets = self.profit_targets[d]
                if price >= targets['level2']:
                    order = self.close(data=d)
                    order.data = d
                    self.orders[d] = order
                    self.trade_reasons[d] = "达到第二止盈目标"
                    self.log(f'第二止盈目标触发 - 卖出 {d._name}, 价格: {price:.2f}')
                    if d in self.bought_etfs:
                        self.bought_etfs.remove(d)
                    if d in self.entry_dates:
                        del self.entry_dates[d]
                    if d in self.profit_targets:
                        del self.profit_targets[d]
                elif price >= targets['level1']:
                    # 达到第一目标位平仓1/2
                    order = self.sell(data=d, size=pos.size//2)
                    order.data = d
                    self.orders[d] = order
                    self.trade_reasons[d] = "达到第一止盈目标"
                    self.log(f'第一止盈目标触发 - 卖出 {d._name} 一半仓位, 价格: {price:.2f}')
                    # 更新第一目标价格为当前价格
                    targets['level1'] = price
            
            # 触发任一止损条件
            if any([price < atr_stop, time_stop, drawdown_stop]):
                order = self.close(data=d)
                order.data = d
                self.orders[d] = order
                reason = []
                if price < atr_stop: reason.append("ATR止损")
                if time_stop: reason.append("时间止损")
                if drawdown_stop: reason.append("回撤止损")
                self.trade_reasons[d] = f"复合止损触发 ({', '.join(reason)})"
                self.log(f'复合止损触发 - 卖出 {d._name}, 原因: {", ".join(reason)}')
                if d in self.bought_etfs:
                    self.bought_etfs.remove(d)
                if d in self.entry_dates:
                    del self.entry_dates[d]
                if d in self.profit_targets:
                    del self.profit_targets[d]

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

class MarketState(bt.Indicator):
    lines = ('trend',)
    params = (('fast', 50), ('slow', 200))
    
    def __init__(self):
        ma_fast = bt.indicators.SMA(self.data, period=self.p.fast)
        ma_slow = bt.indicators.SMA(self.data, period=self.p.slow)
        self.lines.trend = ma_fast / ma_slow - 1  # 快线相对慢线的位置
