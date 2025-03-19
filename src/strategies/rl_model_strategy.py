import backtrader as bt
import numpy as np
import pandas as pd
import torch
from loguru import logger
import json

class RLModelStrategy(bt.Strategy):
    params = (
        ('model_path', None),         # 模型路径
        ('config_path', None),        # 配置文件路径
        ('window_size', 10),          # 观察窗口大小
        ('risk_ratio', 0.02),         # 单次交易风险比率
        ('max_drawdown', 0.15),       # 最大回撤限制
        ('price_limit', 0.10),        # 涨跌停限制(10%)
        ('min_shares', 100),          # 最小交易股数
        ('cash_buffer', 0.95),        # 现金缓冲比例
    )

    def __init__(self):
        """初始化策略"""
        # 加载配置和模型
        if self.p.config_path:
            with open(self.p.config_path, 'r') as f:
                config_dict = json.load(f)
                # TODO: 从字典更新配置

        # 设置设备
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # 初始化智能体
        if self.p.model_path:
            self.agent = self._load_agent()
        else:
            logger.error("未提供模型路径")
            raise ValueError("请提供有效的模型路径")

        # 记录最高净值，用于计算回撤
        self.highest_value = self.broker.getvalue()
        
        # 用于跟踪订单和持仓
        self.order = None
        self.entry_price = None
        self.trade_reason = None
        self._orders = []
        
        # 技术指标
        self.atr = bt.indicators.ATR(self.data)
        
        # 价格和特征历史
        self.price_history = []
        self.feature_history = []
        
        logger.info("强化学习模型策略初始化完成")

    def _load_agent(self):
        """加载训练好的智能体"""
        # 计算状态维度
        pass

    def _get_state(self):
        """构建当前状态"""
        # 获取价格历史
        price_data = np.array([
            [self.data.open[i], self.data.high[i], self.data.low[i], 
             self.data.close[i], self.data.volume[i]]
            for i in range(-self.p.window_size + 1, 1)
        ]).flatten()

        # 生成技术指标特征
        features = []
        for i in range(-self.p.window_size + 1, 1):
            # 趋势指标
            ma5 = np.mean([self.data.close[j] for j in range(i-4, i+1)])
            ma10 = np.mean([self.data.close[j] for j in range(i-9, i+1)])
            ma20 = np.mean([self.data.close[j] for j in range(i-19, i+1)])
            
            # 动量指标
            momentum = self.data.close[i] / self.data.close[i-5] - 1
            
            # 波动率指标
            volatility = np.std([self.data.close[j] for j in range(i-4, i+1)])
            
            # 成交量指标
            volume_ma5 = np.mean([self.data.volume[j] for j in range(i-4, i+1)])
            
            features.extend([ma5, ma10, ma20, momentum, volatility, volume_ma5])

        # 账户状态
        portfolio_value = self.broker.getvalue()
        position_value = self.position.size * self.data.close[0] if self.position else 0
        position_pct = position_value / portfolio_value if portfolio_value > 0 else 0
        cash_pct = self.broker.getcash() / portfolio_value

        # 组合状态
        state = np.concatenate([
            price_data,
            features,
            [cash_pct, position_pct]
        ])

        return state.astype(np.float32)

    def round_shares(self, shares):
        """将股数调整为100的整数倍"""
        return int(shares / 100) * 100
        
    def check_price_limit(self, price):
        """检查是否触及涨跌停"""
        prev_close = self.data.close[-1]
        upper_limit = prev_close * (1 + self.p.price_limit)
        lower_limit = prev_close * (1 - self.p.price_limit)
        return lower_limit <= price <= upper_limit

    def calculate_trade_size(self, price):
        """计算可交易的股数（考虑资金、手续费和100股整数倍）"""
        cash = self.broker.getcash() * self.p.cash_buffer
        
        # 计算风险金额（使用总资产的一定比例）
        total_value = self.broker.getvalue()
        risk_amount = total_value * self.p.risk_ratio
        
        # 使用ATR计算每股风险
        current_atr = self.atr[0]
        risk_per_share = current_atr * 1.5
        
        # 根据风险计算的股数
        risk_size = risk_amount / risk_per_share if risk_per_share > 0 else 0
        
        # 根据可用资金计算的股数
        cash_size = cash / price
        
        # 取较小值并调整为100股整数倍
        shares = min(risk_size, cash_size)
        shares = self.round_shares(shares)
        
        # 再次验证金额是否超过可用资金
        if shares * price > cash:
            shares = self.round_shares(cash / price)
            
        return shares if shares >= self.p.min_shares else 0

    def next(self):
        # 如果有未完成的订单，不执行新的交易
        if self.order:
            return
            
        # 计算当前回撤
        current_value = self.broker.getvalue()
        self.highest_value = max(self.highest_value, current_value)
        drawdown = (self.highest_value - current_value) / self.highest_value
        
        # 如果回撤超过限制，不开新仓
        if drawdown > self.p.max_drawdown:
            if self.position:
                self.trade_reason = f"触发最大回撤限制 ({drawdown:.2%})"
                self.close()
                logger.info(f"触发最大回撤限制 - 当前回撤: {drawdown:.2%}, 限制: {self.p.max_drawdown:.2%}")
            return
            
        # 检查是否触及涨跌停
        if not self.check_price_limit(self.data.close[0]):
            return

        # 获取当前状态
        state = self._get_state()
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        # 使用智能体选择动作
        with torch.no_grad():
            q_values = self.agent.q_network(state_tensor)
            action = q_values.argmax().item()

        current_price = self.data.close[0]

        if action == 1:  # 买入
            if not self.position:  # 没有持仓
                shares = self.calculate_trade_size(current_price)
                if shares >= self.p.min_shares:
                    self.trade_reason = "智能体买入信号"
                    self.order = self.buy(size=shares)
                    if self.order:
                        self.entry_price = current_price
                        logger.info(f"买入信号 - 数量: {shares}, 价格: {current_price:.2f}")

        elif action == 2:  # 卖出
            if self.position:  # 有持仓
                self.trade_reason = "智能体卖出信号"
                self.order = self.close()
                if self.order:
                    logger.info(f"卖出信号 - 价格: {current_price:.2f}")

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
            
        if order.status in [order.Completed]:
            if order.isbuy():
                order.info = {
                    'reason': self.trade_reason,
                    'total_value': self.broker.getvalue(),
                    'position_value': self.position.size * order.executed.price if self.position else 0
                }
                self._orders.append(order)
                logger.info(
                    f'买入执行 - 价格: {order.executed.price:.2f}, '
                    f'数量: {order.executed.size}, '
                    f'原因: {self.trade_reason}'
                )
            else:
                self.entry_price = None
                order.info = {
                    'reason': self.trade_reason,
                    'total_value': self.broker.getvalue(),
                    'position_value': self.position.size * order.executed.price if self.position else 0
                }
                self._orders.append(order)
                logger.info(
                    f'卖出执行 - 价格: {order.executed.price:.2f}, '
                    f'数量: {order.executed.size}, '
                    f'原因: {self.trade_reason}'
                )
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            logger.warning(f'订单失败 - 状态: {order.getstatusname()}')
            
        self.order = None

    def stop(self):
        """策略结束时的汇总信息"""
        portfolio_value = self.broker.getvalue()
        returns = (portfolio_value / self.broker.startingcash) - 1.0
        logger.info(f"策略结束 - 最终资金: {portfolio_value:.2f}, 收益率: {returns:.2%}")
