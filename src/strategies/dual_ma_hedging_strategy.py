import backtrader as bt
from loguru import logger
from src.indicators.trailing_stop import TrailingStop
import math

class DualMAHedgingStrategy(bt.Strategy):
    params = (
        ('fast_period', 5),      # 快速移动平均线周期
        ('slow_period', 13),      # 慢速移动平均线周期
        ('trail_percent', 2.0),   # 追踪止损百分比
        ('risk_ratio', 0.02),     # 单次交易风险比率
        ('max_drawdown', 0.15),   # 最大回撤限制
        ('price_limit', 0.10),    # 涨跌停限制(10%)
        ('enable_trailing_stop', False),  # 是否启用追踪止损
        ('atr_multiplier', 1.0),  # ATR倍数
        ('atr_period', 14),       # ATR周期
        ('enable_death_cross', False),  # 是否启用死叉卖出信号
        ('enable_hedging', True),  # 是否启用对冲功能
        ('hedge_contract_size', 10),  # 对冲合约手数
        ('hedge_fee', 1.51),          # 对冲合约手续费
        ('hedge_profit_multiplier', 1.0),  # 对冲盈利倍数
        ('future_contract_multiplier', 10),  # 期货合约乘数，豆粕为10吨/手
        ('verbose', False),          # 是否启用详细日志
        ('crossover_threshold', 0.003),  # 快线上穿慢线的最小幅度阈值(0.3%)
        ('volume_ratio_threshold', 0.99),  # 量能放大阈值(1.2倍)
        ('volume_surge_threshold', 2.1),  # 量能暴增阈值(2倍)
        ('margin_ratio', 0.95),     # 保证金比例(95%)
    )

    def __init__(self):
        # 获取ETF代码
        self.etf_code = None
        try:
            # 首先尝试从数据源的params属性获取
            if hasattr(self.data, 'params') and hasattr(self.data.params, 'ts_code'):
                self.etf_code = self.data.params.ts_code
            # 然后尝试从数据源的其他属性获取
            elif hasattr(self.data, 'ts_code'):
                self.etf_code = self.data.ts_code
            # 最后尝试从数据源的_name属性获取
            elif hasattr(self.data, '_name'):
                self.etf_code = self.data._name
        except Exception as e:
            logger.warning(f"获取ETF代码时出错: {str(e)}")
            
        # 如果没有找到ETF代码，使用默认名称
        if not self.etf_code:
            self.etf_code = "ETF_1"
            
        # 设置数据源的名称
        self.data._name = self.etf_code
        
        # 保存数据加载器引用
        self.future_loader = None
        if hasattr(self.data1, 'params') and hasattr(self.data1.params, 'loader'):
            self.future_loader = self.data1.params.loader
            logger.info("成功获取期货数据加载器")
        else:
            logger.warning("无法获取期货数据加载器")
        
        # 设置独立账户
        self.etf_cash = 100000.0  # ETF股票账户资金
        self.future_cash = 100000.0  # 期货账户资金
        
        # 记录每个账户的最高净值，用于计算回撤
        self.etf_highest_value = self.etf_cash
        self.future_highest_value = self.future_cash
        
        # 记录初始账户状态
        logger.info(f"账户初始化 - ETF账户: {self.etf_cash:.2f}, 期货账户: {self.future_cash:.2f}, "
                  f"总初始资金: {self.etf_cash + self.future_cash:.2f}")
        
        # 移动平均线指标
        self.fast_ma = bt.indicators.SMA(
            self.data.close, period=self.p.fast_period)
        self.slow_ma = bt.indicators.SMA(
            self.data.close, period=self.p.slow_period)
        self.crossover = bt.indicators.CrossOver(self.fast_ma, self.slow_ma)
        
        # 记录上一次的移动平均线值，用于计算上穿幅度
        self.last_fast_ma = None
        self.last_slow_ma = None
        
        # 追踪止损指标（根据参数决定是否启用）
        self.trailing_stop = None
        if self.p.enable_trailing_stop:
            self.trailing_stop = TrailingStop(self.data, trailing=self.p.trail_percent/100.0)
            self.trailing_stop._owner = self
        
        # ATR指标
        self.atr = bt.indicators.ATR(self.data, period=self.p.atr_period)
        
        # 用于跟踪订单和持仓
        self.order = None
        self.entry_price = None  # 记录入场价格
        self.trade_reason = None  # 记录交易原因
        self._orders = []  # 记录所有订单
        
        # 持仓管理
        self.current_position_ratio = 0.0  # 当前仓位比例
        self.avg_cost = None  # 平均持仓成本
        
        # T+1交易限制
        self.buy_dates = set()  # 记录买入日期
        
        # 记录上次清仓日期
        self.last_close_date = None
        
        # 对冲相关变量
        self.hedge_position = None        # 对冲持仓
        self.hedge_entry_price = None     # 对冲入场价格
        self.original_loss = None         # 原始止损损失
        self.hedge_target_profit = None   # 对冲目标盈利
        self.hedge_order = None           # 对冲订单
        self.hedge_contract_code = None   # 对冲合约代码
        
        logger.info(f"策略初始化完成 - 参数: 快线={self.p.fast_period}, 慢线={self.p.slow_period}, 追踪止损={self.p.trail_percent}%, 风险比例={self.p.risk_ratio:.2%}, 最大回撤={self.p.max_drawdown:.2%}")
        
    def round_shares(self, shares):
        """将股数调整为100的整数倍"""
        return math.floor(shares / 100) * 100
        
    def check_price_limit(self, price):
        """检查是否触及涨跌停"""
        prev_close = self.data.close[-1]
        upper_limit = prev_close * (1 + self.p.price_limit)
        lower_limit = prev_close * (1 - self.p.price_limit)
        return lower_limit <= price <= upper_limit
        
    def calculate_trade_size(self, price):
        """计算可交易的股数（考虑资金、手续费和100股整数倍）"""
        cash = self.etf_cash  # 使用ETF账户资金
        
        # 预留更多手续费和印花税缓冲
        cash = cash * self.p.margin_ratio  # 预留5%的资金作为手续费和保证金缓冲
        
        # 计算风险金额（使用ETF账户总资产的一定比例）
        etf_value = cash
        if self.position:
            etf_value += self.position.size * self.data.close[0]
        
        # 确保资产价值为正数
        etf_value = max(etf_value, 0)
        risk_amount = etf_value * self.p.risk_ratio
        
        # 确保风险金额不为负
        risk_amount = max(risk_amount, 0)
        
        # 使用ATR计算每股风险
        current_atr = self.atr[0]  # 当前ATR值
        # 使用1.5倍ATR作为止损距离，这个系数可以根据需要调整
        risk_per_share = current_atr * 1.5
        
        # 如果ATR过小，使用传统的百分比止损
        min_risk = price * (self.p.trail_percent/100.0)
        risk_per_share = max(risk_per_share, min_risk)
        
        # 根据风险计算的股数
        risk_size = risk_amount / risk_per_share if risk_per_share > 0 else 0
        
        # 根据可用资金计算的股数
        cash_size = cash / price if cash > 0 else 0
        
        # 取较小值并调整为100股整数倍
        shares = min(risk_size, cash_size)
        # 确保股数非负
        shares = max(0, shares)
        shares = self.round_shares(shares)
        
        # 再次验证金额是否超过可用资金
        if shares * price > cash:
            shares = self.round_shares(cash / price)
            
        logger.info(f"计算持仓 - ATR: {current_atr:.2f}, 每股风险: {risk_per_share:.2f}, 总风险金额: {risk_amount:.2f}, 计算股数: {shares}, ETF可用资金: {cash:.2f}")
        
        return shares if shares >= 100 else 0

    def hedge(self, loss_amount):
        """对冲函数
        Args:
            loss_amount: 原始止损损失金额
        """
        # 首先检查是否启用对冲功能
        if not self.p.enable_hedging:
            logger.info("对冲功能未启用，跳过对冲操作")
            return
            
        if self.hedge_position is not None or self.hedge_order is not None:
            logger.info("已有对冲仓位或对冲订单，不再开仓")
            return
            
        # 检查是否有豆粕期货数据
        if not hasattr(self, 'data1') or self.data1 is None:
            logger.error("未找到豆粕期货数据，无法进行对冲")
            return
            
        try:
            # 确保损失金额为正数
            self.original_loss = abs(loss_amount)
            self.hedge_target_profit = self.original_loss * (1 + self.p.hedge_profit_multiplier)
            
            logger.info(f"对冲参数计算 - 原始损失: {loss_amount:.2f}, 取绝对值: {self.original_loss:.2f}, "
                      f"目标盈利: {self.hedge_target_profit:.2f}, 盈利倍数: {self.p.hedge_profit_multiplier}")
            
            # 对冲合约规模
            hedge_size = self.p.hedge_contract_size
            
            # 检查期货账户资金是否足够
            future_price = self.data1.close[0]
            margin_requirement = future_price * hedge_size * self.p.future_contract_multiplier * 0.10  # 10%保证金
            
            if margin_requirement > self.future_cash:
                logger.warning(f"期货账户资金不足，需要{margin_requirement:.2f}，当前可用{self.future_cash:.2f}")
                # 根据可用资金调整手数
                adjusted_size = int(self.future_cash / (future_price * self.p.future_contract_multiplier * 0.10))
                if adjusted_size < 1:
                    logger.error("期货账户资金不足以开仓一手")
                    return
                hedge_size = adjusted_size
                logger.info(f"已调整对冲手数为: {hedge_size}")
            
            # 记录期货交易信息
            trade_date = self.data.datetime.date(0)  # 获取当前交易日期
            self.hedge_trade_date = trade_date
            
            # 实际对冲时使用的手数
            actual_hedge_size = hedge_size
            
            # 开空豆粕期货
            self.hedge_order = self.sell(data=self.data1, size=actual_hedge_size)
            
            # 记录入场价格用于后续计算盈亏
            self.hedge_entry_price = self.data1.close[0]
            
            # 计算实际占用的保证金
            actual_margin = self.hedge_entry_price * actual_hedge_size * self.p.future_contract_multiplier * 0.10
            
            # 扣除期货账户资金（保证金）
            pre_future_cash = self.future_cash
            self.future_cash -= actual_margin
            
            logger.info(f"期货资金变动 - 对冲前资金: {pre_future_cash:.2f}, 占用保证金: {actual_margin:.2f}, 对冲后资金: {self.future_cash:.2f}")
            
            logger.info(f"开启对冲 - 开空{actual_hedge_size}手豆粕期货, 入场价: {self.hedge_entry_price:.2f}, "
                      f"原始损失: {self.original_loss:.2f}, 目标盈利: {self.hedge_target_profit:.2f}, "
                      f"占用保证金: {actual_margin:.2f}, 剩余期货资金: {self.future_cash:.2f}")
            
        except Exception as e:
            logger.error(f"对冲开仓失败: {str(e)}")

    def next(self):
        # 重置交易原因（在每个新的交易周期开始时）
        self.trade_reason = None
        
        # 如果有未完成的订单，不执行新的交易
        if self.order:
            return
            
        # 检查对冲持仓盈亏情况
        if self.hedge_entry_price is not None and self.hedge_position is None and self.hedge_order is None:
            # 已经平仓的对冲仓位，重置对冲相关变量
            self.hedge_entry_price = None
            self.original_loss = None
            self.hedge_target_profit = None
        
        # 如果当前有对冲持仓，检查是否达到止盈条件
        if self.hedge_position is not None and self.hedge_target_profit is not None:
            # 计算当前对冲仓位盈亏
            current_hedge_price = self.data1.close[0]
            hedge_price_diff = self.hedge_entry_price - current_hedge_price  # 空仓盈利 = 开仓价 - 当前价
            hedge_profit = hedge_price_diff * self.hedge_position.size * self.p.future_contract_multiplier
            
            logger.info(f"对冲持仓检查 - 当前价格: {current_hedge_price:.2f}, 入场价: {self.hedge_entry_price:.2f}, "
                      f"价差: {hedge_price_diff:.2f}, 手数: {self.hedge_position.size}, 合约乘数: {self.p.future_contract_multiplier}, "
                      f"当前盈利: {hedge_profit:.2f}, 目标盈利: {self.hedge_target_profit:.2f}")
            
            # 如果达到止盈目标，平掉对冲仓位
            if hedge_profit >= self.hedge_target_profit:
                if self.hedge_order is None:  # 确保没有未完成订单
                    self.hedge_order = self.close(data=self.data1)
                    logger.info(f"对冲止盈下单 - 当前盈利: {hedge_profit:.2f}, 目标盈利: {self.hedge_target_profit:.2f}")
            
            # 对冲止损逻辑 - 当亏损等于原始亏损金额时止损
            elif hedge_profit <= -self.original_loss:
                if self.hedge_order is None:  # 确保没有未完成订单
                    self.hedge_order = self.close(data=self.data1)
                    logger.info(f"对冲止损下单 - 当前亏损: {-hedge_profit:.2f}, 原始亏损: {self.original_loss:.2f}")
        
        # 计算当前回撤
        etf_value = self.etf_cash
        if self.position:
            etf_value += self.position.size * self.data.close[0]
        
        # 确保etf_value不为负值
        etf_value = max(0, etf_value)
        
        self.etf_highest_value = max(self.etf_highest_value, etf_value)
        etf_drawdown = (self.etf_highest_value - etf_value) / self.etf_highest_value if self.etf_highest_value > 0 else 0
        
        # 更新日志内容以便跟踪资金情况
        if self.position or self.p.verbose:
            logger.info(f"账户状态 - ETF资金: {self.etf_cash:.2f}, 期货资金: {self.future_cash:.2f}, "
                      f"ETF持仓: {self.position.size if self.position else 0}, ETF价格: {self.data.close[0]:.2f}, "
                      f"ETF账户总值: {etf_value:.2f}, ETF回撤: {etf_drawdown:.2%}")
        
        # 检查是否触及涨跌停
        if not self.check_price_limit(self.data.close[0]):
            return
            
        current_price = self.data.close[0]
        
        # 强制更新指标
        if self.position:
            if self.p.enable_trailing_stop:
                self.trailing_stop.next()
        
        if not self.position:  # 没有持仓
            # 计算快线上穿慢线的幅度
            crossover_pct = 0
            if self.last_fast_ma is not None and self.last_slow_ma is not None:
                crossover_pct = (self.fast_ma[0] - self.slow_ma[0]) / self.slow_ma[0]
            
            # 更新上一次的移动平均线值
            self.last_fast_ma = self.fast_ma[0]
            self.last_slow_ma = self.slow_ma[0]
            
            if self.crossover > 0:  # 金叉，买入信号
                # 检查上穿幅度是否满足阈值要求
                if crossover_pct > self.p.crossover_threshold:
                    # 检查量能是否大于前一日
                    current_volume = self.data.volume[0]
                    prev_volume = self.data.volume[-1]
                    
                    if current_volume > self.p.volume_ratio_threshold * prev_volume:
                        shares = self.calculate_trade_size(current_price)
                        
                        if shares >= 100:  # 确保至少有100股
                            # 计算实际所需资金（包括手续费）
                            required_cash = shares * current_price * 1.0003  # 包含0.03%手续费
                            
                            # 确保有足够资金
                            if required_cash > self.etf_cash * self.p.margin_ratio:
                                # 根据实际可用资金调整交易规模
                                adjusted_shares = int(self.etf_cash * self.p.margin_ratio / current_price / 1.0003 / 100) * 100
                                if adjusted_shares >= 100:
                                    shares = adjusted_shares
                                    logger.warning(f"资金不足，已调整交易规模 - 原始: {shares}, 调整后: {adjusted_shares}")
                                else:
                                    logger.warning(f"资金不足以买入最低交易单位(100股) - 可用: {self.etf_cash:.2f}, 需要: {required_cash:.2f}")
                                    return
                        
                            self.trade_reason = f"快线上穿慢线 ({self.p.fast_period}日均线上穿{self.p.slow_period}日均线), 上穿幅度: {crossover_pct:.4%}, 量能放大: {current_volume/prev_volume:.2f}倍"
                            self.order = self.buy(size=shares)
                            if self.order:
                                # 记录买入日期和价格
                                self.buy_dates.add(self.data.datetime.date())
                                self.entry_price = current_price
                                logger.info(f"买入信号 - 数量: {shares}, 价格: {current_price:.2f}, 可用资金: {self.etf_cash:.2f}, 风险比例: {self.p.risk_ratio:.2%}, 上穿幅度: {crossover_pct:.4%}, 量能放大: {current_volume/prev_volume:.2f}倍")
                    else:
                        logger.info(f"量能不足 - 当前量能: {current_volume}, 前一日量能: {prev_volume}, 量能比: {current_volume/prev_volume:.2f}")
                else:
                    logger.info(f"上穿幅度不足 - 当前幅度: {crossover_pct:.4%}, 最小要求: {self.p.crossover_threshold:.4%}")
            
            # 新增：量能暴增且均线多头向上开仓条件
            current_volume = self.data.volume[0]
            prev_volume = self.data.volume[-1]
            
            # 检查量能是否超过前一日两倍
            if current_volume > self.p.volume_surge_threshold * prev_volume:
                # 检查快慢均线是否多头向上
                fast_ma_up = self.fast_ma[0] > self.fast_ma[-1]
                slow_ma_up = self.slow_ma[0] > self.slow_ma[-1]
                
                if fast_ma_up and slow_ma_up:
                    shares = self.calculate_trade_size(current_price)
                    
                    if shares >= 100:  # 确保至少有100股
                        # 计算实际所需资金（包括手续费）
                        required_cash = shares * current_price * 1.0003  # 包含0.03%手续费
                        
                        # 确保有足够资金
                        if required_cash > self.etf_cash * self.p.margin_ratio:
                            # 根据实际可用资金调整交易规模
                            adjusted_shares = int(self.etf_cash * self.p.margin_ratio / current_price / 1.0003 / 100) * 100
                            if adjusted_shares >= 100:
                                shares = adjusted_shares
                                logger.warning(f"资金不足，已调整交易规模 - 原始: {shares}, 调整后: {adjusted_shares}")
                            else:
                                logger.warning(f"资金不足以买入最低交易单位(100股) - 可用: {self.etf_cash:.2f}, 需要: {required_cash:.2f}")
                                return
                    
                        self.trade_reason = f"量能暴增且均线多头向上 - 量能放大: {current_volume/prev_volume:.2f}倍, 快线: {self.fast_ma[0]:.2f}↑{self.fast_ma[-1]:.2f}, 慢线: {self.slow_ma[0]:.2f}↑{self.slow_ma[-1]:.2f}"
                        self.order = self.buy(size=shares)
                        if self.order:
                            # 记录买入日期和价格
                            self.buy_dates.add(self.data.datetime.date())
                            self.entry_price = current_price
                            logger.info(f"买入信号 - 数量: {shares}, 价格: {current_price:.2f}, 可用资金: {self.etf_cash:.2f}, 风险比例: {self.p.risk_ratio:.2%}, 量能放大: {current_volume/prev_volume:.2f}倍")
                else:
                    logger.info(f"均线未多头向上 - 快线: {self.fast_ma[0]:.2f} vs {self.fast_ma[-1]:.2f}, 慢线: {self.slow_ma[0]:.2f} vs {self.slow_ma[-1]:.2f}")
        
        else:  # 有持仓
            # 检查是否可以卖出（T+1规则）
            current_date = self.data.datetime.date()
            if current_date in self.buy_dates:
                return
                
            # 计算ATR止盈止损价格
            current_atr = self.atr[0]
            stop_loss = self.entry_price - (current_atr * self.p.atr_multiplier)
            take_profit = self.entry_price + (current_atr * self.p.atr_multiplier)
            
            # 获取追踪止损价格（如果启用）
            trailing_stop_price = None
            if self.p.enable_trailing_stop:
                trailing_stop_price = self.trailing_stop[0]
            
            logger.info(f"持仓检查 - 今天日期: {current_date}, 当前价格: {current_price:.2f}, ATR止损: {stop_loss:.2f}, ATR止盈: {take_profit:.2f}")
            
            if self.p.enable_death_cross and self.crossover < 0:  # 死叉，卖出信号
                self.trade_reason = f"快线下穿慢线 ({self.p.fast_period}日均线下穿{self.p.slow_period}日均线)"
                self.order = self.close()
                if self.order:
                    logger.info(f"卖出信号 - 价格: {current_price:.2f}")
            
            # ATR止损检查
            elif current_price < stop_loss:
                # 计算损失金额 - 使用当前价格与入场价差值
                actual_loss_per_share = self.entry_price - current_price
                loss_amount = actual_loss_per_share * self.position.size
                
                # 详细记录损失计算
                logger.info(f"计算止损损失 - 入场价: {self.entry_price:.4f}, 当前价: {current_price:.4f}, "
                         f"每股损失: {actual_loss_per_share:.4f}, 持仓量: {self.position.size}, 总损失: {loss_amount:.2f}")
                
                self.trade_reason = f"触发ATR止损 (止损价: {stop_loss:.2f})"
                self.order = self.close()
                
                # 记录ATR止损触发前的数据
                pre_loss_etf_cash = self.etf_cash
                pre_loss_future_cash = self.future_cash
                
                if self.order:
                    logger.info(f"ATR止损触发 - 当前价格: {current_price:.2f}, 止损价: {stop_loss:.2f}, "
                              f"损失金额: {loss_amount:.2f}, ETF账户当前余额: {self.etf_cash:.2f}, 期货账户余额: {self.future_cash:.2f}")
                    
                    # 开启对冲 - 但不应该直接影响ETF账户资金
                    self.hedge(loss_amount)
            
            # ATR止盈检查
            elif current_price > take_profit:
                self.trade_reason = f"触发ATR止盈 (止盈价: {take_profit:.2f})"
                self.order = self.close()
                if self.order:
                    logger.info(f"ATR止盈触发 - 当前价格: {current_price:.2f}, 止盈价: {take_profit:.2f}")
            
            # 追踪止损检查（如果启用）
            elif self.p.enable_trailing_stop and current_price < trailing_stop_price:
                self.trade_reason = f"触发追踪止损 (止损价: {trailing_stop_price:.2f})"
                self.order = self.close()
                if self.order:
                    logger.info(f"追踪止损触发 - 当前价格: {current_price:.2f}, 止损价: {trailing_stop_price:.2f}, 最高价: {self.trailing_stop.max_price:.2f}")

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
            
        if order.status in [order.Completed]:
            # 检查是否是对冲订单
            is_hedge_order = (order.data == self.data1)
            
            if is_hedge_order:
                # 对冲订单处理
                if order.isbuy():  # 买入豆粕期货（平空）
                    # 检查合约是否匹配
                    current_contract = self.data1.contract_mapping[self.data1.datetime.datetime(0)]
                    if current_contract != self.hedge_contract_code:
                        logger.error(f"合约不匹配 - 开仓合约: {self.hedge_contract_code}, 平仓合约: {current_contract}")
                        
                    # 获取开仓合约在平仓日期的价格
                    if self.future_loader:
                        price_data = self.future_loader.get_contract_price(self.hedge_contract_code, self.data.datetime.date(0))
                        if not price_data:
                            logger.error(f"无法获取合约{self.hedge_contract_code}在{self.data.datetime.date(0)}的价格数据")
                            return
                            
                        # 使用获取到的价格计算对冲盈亏
                        hedge_profit = (self.hedge_entry_price - price_data['close']) * self.p.hedge_contract_size * self.p.future_contract_multiplier
                    else:
                        # 如果没有数据加载器，使用当前价格
                        logger.warning("无法获取历史价格数据，使用当前价格计算")
                        hedge_profit = (self.hedge_entry_price - order.executed.price) * self.p.hedge_contract_size * self.p.future_contract_multiplier
                    
                    # 减去开平仓手续费
                    total_fee = self.p.hedge_fee * self.p.hedge_contract_size * 2  # 开仓+平仓
                    net_profit = hedge_profit - total_fee
                    
                    # 归还保证金并添加盈亏到期货账户
                    margin_returned = self.hedge_entry_price * self.p.hedge_contract_size * self.p.future_contract_multiplier * 0.10
                    self.future_cash += (margin_returned + net_profit)
                    
                    # 更新期货账户最高净值
                    self.future_highest_value = max(self.future_highest_value, self.future_cash)
                    
                    # 计算期货账户回撤
                    future_drawdown = (self.future_highest_value - self.future_cash) / self.future_highest_value if self.future_highest_value > 0 else 0
                    
                    logger.info(f"对冲平仓 - 价格: {price_data['close']:.2f}, 盈利: {hedge_profit:.2f}, 手续费: {total_fee:.2f}, "
                               f"净盈利: {net_profit:.2f}, 期货账户余额: {self.future_cash:.2f}, 回撤: {future_drawdown:.2%}")
                    
                    # 添加交易原因
                    if hedge_profit >= self.hedge_target_profit:
                        reason = f"对冲止盈 - 盈利: {hedge_profit:.2f}, 目标: {self.hedge_target_profit:.2f}"
                    elif hedge_profit <= -self.original_loss:
                        reason = f"对冲止损 - 亏损: {-hedge_profit:.2f}, 上限: {self.original_loss:.2f}"
                    else:
                        reason = "对冲平仓"
                    
                    # 计算收益率
                    price_diff_percent = ((self.hedge_entry_price - price_data['close']) / self.hedge_entry_price) * 100
                    
                    # 记录交易信息
                    order.info = {
                        'reason': reason,
                        'original_loss': self.original_loss,
                        'hedge_profit': hedge_profit,
                        'net_profit': net_profit,
                        'future_cash': self.future_cash,
                        'execution_date': self.data.datetime.date(0),
                        'total_value': self.future_cash,  # 添加期货账户总值
                        'position_value': 0,  # 平仓后持仓价值为0
                        'position_ratio': 0.0,  # 平仓后持仓比例为0
                        'etf_code': self.data1.contract_mapping[self.data1.datetime.datetime(0)],  # 使用期货代码
                        'pnl': hedge_profit,  # 添加盈亏
                        'return': price_diff_percent  # 添加收益率
                    }
                    
                    # 重置对冲相关变量
                    self.hedge_position = None
                    self.hedge_entry_price = None
                    self.original_loss = None
                    self.hedge_target_profit = None
                    self.hedge_order = None
                    
                else:  # 卖出豆粕期货（开空）
                    # 记录对冲持仓
                    self.hedge_position = order
                    self.hedge_entry_price = order.executed.price
                    # 记录当前合约代码
                    self.hedge_contract_code = self.data1.contract_mapping[self.data1.datetime.datetime(0)]
                    
                    # 计算保证金
                    margin = order.executed.price * order.executed.size * self.p.future_contract_multiplier * 0.10
                    
                    # 记录交易信息
                    order.info = {
                        'reason': f"开启对冲 - ETF止损损失: {self.original_loss:.2f}",
                        'margin': margin,
                        'future_cash': self.future_cash,
                        'execution_date': self.data.datetime.date(0),
                        'total_value': self.future_cash,  # 添加期货账户总值
                        'position_value': abs(margin),  # 持仓价值为保证金
                        'position_ratio': margin / self.future_cash if self.future_cash > 0 else 0,  # 计算持仓比例
                        'etf_code': self.data1.contract_mapping[self.data1.datetime.datetime(0)],  # 使用期货代码
                        'pnl': 0,  # 开仓时盈亏为0
                        'return': 0  # 开仓时收益率为0
                    }
                    
                    logger.info(f"对冲开仓 - 价格: {order.executed.price:.2f}, 数量: {order.executed.size}手, 占用保证金: {margin:.2f}")
                    self.hedge_order = None
                    
                # 将期货订单也添加到订单列表
                self._orders.append(order)
            else:
                # ETF订单处理
                if order.isbuy():
                    # 买入订单执行后立即重置追踪止损，使用实际成交价
                    if self.p.enable_trailing_stop:
                        self.trailing_stop.reset(price=order.executed.price)
                    
                    # 更新平均成本 - 买入时更新
                    if self.avg_cost is None:
                        # 首次买入
                        self.avg_cost = order.executed.price
                    else:
                        # 计算新的平均成本
                        current_position = self.position.size - order.executed.size
                        current_value = current_position * self.avg_cost
                        new_value = order.executed.size * order.executed.price
                        total_position = current_position + order.executed.size
                        self.avg_cost = (current_value + new_value) / total_position
                        logger.info(f"当前持仓量: {current_position}, 当前持仓市值: {current_value}, 新买入市值: {new_value}, 总持仓量: {total_position}")
        
                    logger.info(f"买入平均成本: {self.avg_cost:.4f}")
                    # 记录入场价格用于计算止盈
                    self.entry_price = order.executed.price  # 保留最后一次买入价格
                    
                    # 更新当前持仓比例
                    self.current_position_ratio = self.get_position_value_ratio()
                    
                    # 从ETF账户扣除资金
                    trade_value = order.executed.price * order.executed.size
                    commission = trade_value * 0.00025  # 手续费0.025%
                    total_cost = trade_value + commission
                    
                    # 记录资金流出前的账户余额
                    pre_cash = self.etf_cash
                    
                    # 检查是否有足够资金，如果没有，调整交易规模
                    if total_cost > self.etf_cash:
                        logger.warning(f"ETF账户资金不足，需要{total_cost:.2f}，当前可用{self.etf_cash:.2f}，将只使用可用资金")
                        # 这种情况理论上不应该发生，因为计算交易大小时已经考虑了资金限制
                        # 但为了安全，这里再次检查并确保不会出现负值
                        self.etf_cash = 0  # 最多用完所有资金
                    else:
                        self.etf_cash -= total_cost
                    
                    # 详细记录资金变动
                    logger.info(f"ETF买入资金结算 - 买入数量: {order.executed.size}, 价格: {order.executed.price:.4f}, "
                              f"总值: {trade_value:.2f}, 手续费: {commission:.2f}, 总成本: {total_cost:.2f}, "
                              f"变动前余额: {pre_cash:.2f}, 变动后余额: {self.etf_cash:.2f}")
                    
                    # 更新ETF账户最高净值
                    etf_value = self.etf_cash + (self.position.size * self.data.close[0])
                    self.etf_highest_value = max(self.etf_highest_value, etf_value)
                    
                    # 如果没有交易原因，添加默认原因
                    if not self.trade_reason:
                        self.trade_reason = "买入信号触发"
                        
                    order.info = {'reason': self.trade_reason}  # 记录交易原因
                    # 计算总资产和持仓市值 - 使用成交价格而非最新价格，更准确反映交易时点
                    position_value = self.position.size * order.executed.price
                    total_etf_value = self.etf_cash + position_value
                    
                    order.info['total_value'] = total_etf_value  # 记录ETF总资产（含现金）
                    order.info['position_value'] = position_value  # 记录持仓市值
                    order.info['position_ratio'] = self.current_position_ratio  # 记录持仓比例
                    order.info['avg_cost'] = self.avg_cost  # 记录平均成本
                    order.info['etf_code'] = self.etf_code  # 添加ETF代码
                    order.info['execution_date'] = self.data.datetime.date(0)  # 添加执行日期
                    order.info['etf_cash'] = self.etf_cash  # 添加ETF账户现金
                    self._orders.append(order)  # 添加到订单列表
                    logger.info(f'买入执行 - 价格: {order.executed.price:.2f}, 数量: {order.executed.size}, '
                              f'仓位比例: {self.current_position_ratio:.2%}, 平均成本: {self.avg_cost:.2f}, '
                              f'ETF账户余额: {self.etf_cash:.2f}, ETF总资产: {total_etf_value:.2f}, 原因: {self.trade_reason}')
                else:
                    # 卖出 - 更新持仓相关指标
                    # 记录卖出前的平均成本（用于日志记录）
                    last_avg_cost = self.avg_cost
                    
                    # 计算卖出收益并添加到ETF账户
                    trade_value = order.executed.price * abs(order.executed.size)  # 确保使用绝对值，因为平仓size为负值
                    commission = trade_value * 0.00025  # 手续费0.025%
                    net_proceeds = trade_value - commission  # 实际收到的金额
                    
                    # 记录资金流入前的账户余额
                    pre_cash = self.etf_cash
                    
                    # 加到ETF账户
                    self.etf_cash += net_proceeds
                    
                    # 详细记录资金变动
                    logger.info(f"ETF卖出资金结算 - 卖出数量: {abs(order.executed.size)}, 价格: {order.executed.price:.4f}, "
                              f"总值: {trade_value:.2f}, 手续费: {commission:.2f}, 净收入: {net_proceeds:.2f}, "
                              f"变动前余额: {pre_cash:.2f}, 变动后余额: {self.etf_cash:.2f}")
                    
                    # 确保账户资金永远不会为负
                    if self.etf_cash < 0:
                        logger.error(f"ETF账户出现负值：{self.etf_cash:.2f}，已重置为0，请检查资金计算逻辑")
                        self.etf_cash = 0
                    
                    if not self.position or self.position.size == 0:  # 如果全部平仓
                        self.entry_price = None
                        self.avg_cost = None
                        if self.p.enable_trailing_stop:
                            self.trailing_stop.stop_tracking()
                        # 记录清仓日期
                        self.last_close_date = self.data.datetime.date(0)
                        logger.info(f"记录清仓日期: {self.last_close_date}")
                    
                    # 更新当前持仓比例
                    self.current_position_ratio = self.get_position_value_ratio()
                    
                    # 更新ETF账户最高净值
                    etf_value = self.etf_cash + (self.position.size * self.data.close[0] if self.position else 0)
                    self.etf_highest_value = max(self.etf_highest_value, etf_value)
                    
                    # 如果没有交易原因，添加默认原因
                    if not self.trade_reason:
                        self.trade_reason = "卖出信号触发"
                        
                    order.info = {'reason': self.trade_reason}  # 记录交易原因
                    # 计算总资产和持仓市值 - 使用成交价格
                    position_value = self.position.size * order.executed.price if self.position and self.position.size > 0 else 0
                    total_etf_value = self.etf_cash + position_value
                    
                    order.info['total_value'] = total_etf_value  # 记录ETF总资产（含现金）
                    order.info['position_value'] = position_value  # 记录持仓市值
                    order.info['position_ratio'] = self.current_position_ratio  # 记录持仓比例
                    order.info['avg_cost'] = last_avg_cost  # 记录卖出前的平均成本
                    order.info['etf_code'] = self.etf_code  # 添加ETF代码
                    order.info['execution_date'] = self.data.datetime.date(0)  # 添加执行日期
                    order.info['etf_cash'] = self.etf_cash  # 添加ETF账户现金
                    self._orders.append(order)  # 添加到订单列表
                    
                    # 计算卖出收益
                    # 修正计算：使用正数表示数量，避免符号导致的计算错误
                    position_size = abs(order.executed.size)
                    profit = (order.executed.price - last_avg_cost) * position_size if last_avg_cost and order.executed.price else 0
                    profit_pct = ((order.executed.price / last_avg_cost) - 1.0) * 100 if last_avg_cost and order.executed.price else 0
                    
                    # 格式化价格和成本
                    price_str = f"{order.executed.price:.2f}" if order.executed.price else "N/A"
                    cost_str = f"{last_avg_cost:.4f}" if last_avg_cost else "N/A"
                    
                    logger.info(f'卖出执行 - 价格: {price_str}, 数量: {position_size}, '
                              f'仓位比例: {self.current_position_ratio:.2%}, 平均成本: {cost_str}, '
                              f'收益: {profit:.2f}, 收益率: {profit_pct:.2f}%, ETF账户余额: {self.etf_cash:.2f}, '
                              f'ETF总资产: {total_etf_value:.2f}, 原因: {self.trade_reason}')
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            # 检查是否是对冲订单
            is_hedge_order = (order.data == self.data1)
            
            if is_hedge_order:
                # 重置对冲订单状态
                self.hedge_order = None
                logger.warning(f'对冲订单失败 - 状态: {order.getstatusname()}')
            else:
                logger.warning(f'订单失败 - 状态: {order.getstatusname()}')
            
        # 重置一般订单状态
        if order.data == self.data:  # 只重置主数据源的订单
            self.order = None
        elif order.data == self.data1:  # 重置对冲订单
            self.hedge_order = None

    def stop(self):
        """策略结束时的汇总信息"""
        try:
            # 计算ETF账户表现
            etf_value = max(0, self.etf_cash)  # 确保ETF账户不为负
            if self.position:
                etf_value += self.position.size * self.data.close[0]
            etf_returns = (etf_value / 100000.0) - 1.0
            
            # 计算期货账户表现
            future_returns = (self.future_cash / 100000.0) - 1.0
            
            # 计算总体表现 - 两个账户初始资金总和为200,000
            total_value = etf_value + self.future_cash
            total_initial = 200000.0
            total_returns = (total_value / total_initial) - 1.0
            
            logger.info(f"===== 策略结束 =====")
            logger.info(f"ETF账户: 初始资金 100000.00, 最终资金 {etf_value:.2f}, 收益率 {etf_returns:.2%}")
            logger.info(f"期货账户: 初始资金 100000.00, 最终资金 {self.future_cash:.2f}, 收益率 {future_returns:.2%}")
            logger.info(f"总体表现: 初始资金 {total_initial:.2f}, 最终资金 {total_value:.2f}, 总收益率 {total_returns:.2%}")
            
            # 添加账户信息到broker的属性中，便于回测引擎获取
            self.broker.etf_value = etf_value
            self.broker.future_value = self.future_cash
            self.broker.total_value = total_value
            self.broker.etf_returns = etf_returns
            self.broker.future_returns = future_returns
            self.broker.total_returns = total_returns
        except Exception as e:
            logger.error(f"策略结束时计算收益率出错: {str(e)}")
            # 设置默认值避免除零错误
            self.broker.etf_value = self.etf_cash
            self.broker.future_value = self.future_cash
            self.broker.total_value = self.etf_cash + self.future_cash
            self.broker.etf_returns = -1.0
            self.broker.future_returns = 0.0
            self.broker.total_returns = -0.5

    def get_position_value_ratio(self):
        """计算当前持仓市值占总资产的比例"""
        if not self.position:
            return 0.0
        
        position_value = self.position.size * self.data.close[0]
        total_value = self.broker.getvalue()
        return position_value / total_value 