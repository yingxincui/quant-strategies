import numpy as np
import pandas as pd
import gym
from gym import spaces
from gym.utils import seeding
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from stable_baselines3.common.vec_env import DummyVecEnv
from typing import List, Dict, Any, Tuple, Optional
import copy
import time
from datetime import datetime


class ETFTradingEnv(gym.Env):
    """
    ETF交易环境，专为ETF交易任务定制的强化学习环境
    
    功能特点：
    1. 支持多ETF组合交易
    2. 考虑交易成本（买卖佣金）
    3. 提供多种状态表示和奖励函数选项
    4. 支持自定义技术指标特征
    """
    
    metadata = {'render.modes': ['human']}
    
    def __init__(
        self,
        df: pd.DataFrame,
        stock_dim: int,
        hmax: int = 100,
        initial_amount: float = 10000.0,
        transaction_cost_pct: float = 0.001,
        reward_scaling: float = 1.0,
        state_space: int = None,
        action_space: int = None,
        tech_indicator_list: List[str] = None,
        turbulence_threshold: Optional[float] = None,
        day_trade: bool = False,
        risk_free_rate: float = 0.0,
        lookback: int = 1,
        reward_type: str = 'sharpe',
        cash_penalty_proportion: float = 0.1
    ):
        """
        初始化ETF交易环境
        
        参数:
            df: 包含历史市场数据的DataFrame，索引为日期，包含每个交易日每个ETF的价格和特征
            stock_dim: ETF数量
            hmax: 每次交易最大持仓变化量
            initial_amount: 初始资金
            transaction_cost_pct: 交易成本百分比
            reward_scaling: 奖励缩放因子
            state_space: 状态空间维度，如为None则自动计算
            action_space: 动作空间维度，如为None则自动设置为stock_dim
            tech_indicator_list: 技术指标列表
            turbulence_threshold: 市场波动阈值，超过该值视为高波动市场
            day_trade: 是否允许日内交易
            risk_free_rate: 无风险利率，用于计算Sharpe比率
            lookback: 回溯天数，即状态包含多少天的历史信息
            reward_type: 奖励函数类型，可选'daily_return', 'sharpe', 'sortino'等
            cash_penalty_proportion: 持有现金惩罚比例
        """
        # 保存参数
        self.df = df
        self.stock_dim = stock_dim
        self.hmax = hmax
        self.initial_amount = initial_amount
        self.transaction_cost_pct = transaction_cost_pct
        self.reward_scaling = reward_scaling
        self.tech_indicator_list = tech_indicator_list if tech_indicator_list is not None else []
        self.turbulence_threshold = turbulence_threshold
        self.day_trade = day_trade
        self.lookback = lookback
        self.reward_type = reward_type
        self.cash_penalty_proportion = cash_penalty_proportion
        
        # 获取交易日期
        self.dates = self.df.index.unique().tolist()
        self.data = self.df.reset_index()
        
        # 初始化状态
        self.terminal = False  
        self.day = 0
        self.risk_free_rate = risk_free_rate
        
        # 自动计算状态空间维度
        self.feature_dim = len(self.tech_indicator_list) + 4  # 4基本特征: OHLC
        if state_space is None:
            self.state_space = self.stock_dim * self.feature_dim + self.stock_dim + 3  # +3表示当前持有现金、净值、总资产
        else:
            self.state_space = state_space
        
        # 设置动作空间维度
        if action_space is None:
            self.action_space = spaces.Box(low=-1, high=1, shape=(self.stock_dim,))
        else:
            self.action_space = action_space
        
        # 设置观察空间
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_space,))
        
        # 初始化交易相关变量
        self.asset_memory = [self.initial_amount]
        self.actions_memory = []
        self.date_memory = [self._get_date()]
        self.trades = []
        self.rewards_memory = []
        self.total_dividend = 0  # 添加分红总额跟踪
        
        # 随机数生成器
        self.seed()
        
    def seed(self, seed=None):
        """初始化随机数生成器"""
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def _get_date(self):
        """获取当前日期"""
        if self.day < len(self.dates):
            return self.dates[self.day]
        else:
            return self.dates[-1]  # 防止索引越界
    
    def reset(self):
        """
        重置环境到初始状态
        
        返回:
            初始状态
        """
        # 重置交易状态
        self.day = 0
        self.terminal = False
        
        # 初始化账户状态
        self.asset_memory = [self.initial_amount]
        self.actions_memory = []
        self.date_memory = [self._get_date()]
        self.trades = []
        self.rewards_memory = []
        self.total_dividend = 0  # 重置分红总额
        
        # 初始化持仓和现金
        self.holdings = np.zeros(self.stock_dim)
        self.cash_balance = self.initial_amount
        self.turbulence = 0
        self.cost = 0
        self.total_asset = self.initial_amount
        self.prev_total_asset = self.initial_amount
        self.asset_memory = [self.initial_amount]
        
        # 获取初始状态
        return self._get_observation()
    
    def _get_observation(self):
        """
        构建观察空间状态
        
        返回:
            当前状态向量
        """
        # 获取当前价格数据
        current_date = self._get_date()
        current_data = self.data[self.data['date'] == current_date]
        
        # 构建状态向量
        state = []
        
        # 添加价格和技术指标特征
        for i in range(self.stock_dim):
            # 获取该ETF当天的数据
            stock_data = current_data[current_data['tic'] == current_data['tic'].unique()[i]]
            
            if not stock_data.empty:
                # 添加基本价格特征 (OHLC)
                if 'open' in stock_data.columns:
                    state.append(stock_data['open'].values[0])
                if 'high' in stock_data.columns:
                    state.append(stock_data['high'].values[0])
                if 'low' in stock_data.columns:
                    state.append(stock_data['low'].values[0])
                if 'close' in stock_data.columns:
                    state.append(stock_data['close'].values[0])
                
                # 添加技术指标
                for tech in self.tech_indicator_list:
                    if tech in stock_data.columns:
                        state.append(stock_data[tech].values[0])
            else:
                # 如果没有该日期的数据，用0填充
                # 4个基本特征 + 技术指标数量
                state.extend([0] * (4 + len(self.tech_indicator_list)))
        
        # 添加当前持仓量
        state.extend(self.holdings)
        
        # 添加现金余额
        state.append(self.cash_balance)
        
        # 添加净值
        state.append(self.total_asset)
        
        # 添加市场波动指标
        state.append(self.turbulence)
        
        return np.array(state)
    
    def _get_close_prices(self):
        """获取当天所有ETF的收盘价"""
        current_date = self._get_date()
        df_today = self.data[self.data['date'] == current_date]
        
        # 创建一个与持仓顺序一致的收盘价向量
        close_prices = []
        for i in range(self.stock_dim):
            stock_data = df_today[df_today['tic'] == df_today['tic'].unique()[i]]
            if not stock_data.empty and 'close' in stock_data.columns:
                close_prices.append(stock_data['close'].values[0])
            else:
                close_prices.append(0)
        
        return np.array(close_prices)
    
    def step(self, actions):
        """
        执行一步交易
        
        参数:
            actions: 动作向量，表示每个ETF的买入卖出动作
            
        返回:
            (state, reward, done, info) 元组
        """
        # 如果已经到了终端状态，直接返回
        if self.terminal:
            return self._get_observation(), 0, True, {}
        
        # 保存前一天的资产总值用于计算回报
        self.prev_total_asset = self.total_asset
        
        # 获取当天日期
        current_date = self._get_date()
        df_today = self.data[self.data['date'] == current_date]
        
        # 检查是否有当天的数据
        if df_today.empty:
            print(f"当前日期 {current_date} 没有交易数据")
            self.day += 1
            return self._get_observation(), 0, self.terminal, {}
        
        # 检查市场波动情况
        if 'turbulence' in df_today.columns:
            self.turbulence = df_today['turbulence'].values[0]
        
        # 获取当天的收盘价
        close_prices = self._get_close_prices()
        
        # 根据动作值调整持仓
        actions = np.clip(actions, -1, 1)  # 确保动作在[-1, 1]范围内
        
        # 如果市场波动超过阈值，全部卖出持仓
        if self.turbulence_threshold is not None and self.turbulence > self.turbulence_threshold:
            actions = -np.ones(self.stock_dim)
        
        # 记录交易活动
        trade_actions = []
        trades_list = []
        
        # 计算目标权重
        target_weights = (actions + 1) / 2  # 将动作从[-1,1]映射到[0,1]
        
        # 对每个ETF执行交易
        for i in range(self.stock_dim):
            if close_prices[i] > 0:  # 确保有效价格
                # 计算目标持仓
                current_value = self.holdings[i] * close_prices[i]
                target_value = self.total_asset * target_weights[i]
                trade_value = target_value - current_value
                
                # 计算交易量
                trade_shares = int(trade_value / close_prices[i])
                
                # 限制单次交易量
                trade_shares = np.clip(trade_shares, -self.hmax, self.hmax)
                
                # 执行交易
                if trade_shares != 0:
                    # 计算交易成本
                    transaction_cost = abs(trade_shares * close_prices[i] * self.transaction_cost_pct)
                    
                    # 更新持仓和现金
                    self.holdings[i] += trade_shares
                    self.cash_balance -= (trade_shares * close_prices[i] + transaction_cost)
                    self.cost += transaction_cost
                    
                    # 记录交易
                    trade_actions.append({
                        'date': current_date,
                        'tic': df_today['tic'].unique()[i],
                        'action': 'buy' if trade_shares > 0 else 'sell',
                        'shares': abs(trade_shares),
                        'price': close_prices[i],
                        'cost': transaction_cost
                    })
                    
                    trades_list.append(trade_actions[-1])
        
        # 更新总资产价值
        self.total_asset = self.cash_balance + np.sum(self.holdings * close_prices)
        
        # 计算回报
        reward = self._calculate_reward()
        
        # 记录到内存
        self.asset_memory.append(self.total_asset)
        self.date_memory.append(current_date)
        self.actions_memory.extend(trade_actions)
        self.rewards_memory.append(reward)
        self.trades.extend(trades_list)
        
        # 前进到下一天
        self.day += 1
        
        # 检查是否达到终止状态
        if self.day >= len(self.dates):
            self.terminal = True
        
        # 返回观察、奖励、是否完成以及附加信息
        return self._get_observation(), reward, self.terminal, {
            'date': current_date,
            'portfolio_value': self.total_asset,
            'holdings': self.holdings.tolist(),
            'cash_balance': self.cash_balance,
            'trades': trades_list,
            'cost': self.cost
        }
    
    def _calculate_reward(self):
        """
        计算奖励
        
        支持多种奖励函数：
        - daily_return: 每日回报率
        - sharpe: 夏普比率
        - sortino: 索蒂诺比率
        
        返回:
            计算的奖励值
        """
        # 计算当日收益率
        daily_return = (self.total_asset - self.prev_total_asset) / self.prev_total_asset
        
        # 根据选择的奖励类型计算奖励
        if self.reward_type == 'daily_return':
            reward = daily_return
        
        elif self.reward_type == 'sharpe':
            # 使用简化的夏普比率计算
            # 实际应用中，应累积一段时间的收益率再计算夏普
            if len(self.rewards_memory) > 1:  # 至少需要2个值来计算标准差
                returns = np.array(self.asset_memory) / self.asset_memory[0] - 1
                sharpe = (np.mean(returns) - self.risk_free_rate) / (np.std(returns) + 1e-9)
                reward = sharpe
            else:
                reward = 0
        
        elif self.reward_type == 'sortino':
            # 使用索蒂诺比率，只考虑下行风险
            if len(self.rewards_memory) > 1:
                returns = np.array(self.asset_memory) / self.asset_memory[0] - 1
                negative_returns = returns[returns < 0]
                
                if len(negative_returns) > 0:
                    downside_std = np.std(negative_returns)
                    sortino = (np.mean(returns) - self.risk_free_rate) / (downside_std + 1e-9)
                    reward = sortino
                else:
                    reward = np.mean(returns) - self.risk_free_rate  # 没有下行风险时
            else:
                reward = 0
        
        else:  # 默认使用日回报率
            reward = daily_return
        
        # 持有现金惩罚：如果持有过多现金，给予惩罚
        cash_ratio = self.cash_balance / self.total_asset
        if cash_ratio > 0.7:  # 如果现金比例超过70%
            reward -= cash_ratio * self.cash_penalty_proportion
        
        # 应用奖励缩放
        reward = reward * self.reward_scaling
        
        return reward
    
    def render(self, mode='human'):
        """
        渲染环境当前状态
        
        参数:
            mode: 渲染模式
        """
        if mode == 'human':
            # 打印当前状态信息
            current_date = self._get_date()
            print(f"日期: {current_date}")
            print(f"总资产: ${self.total_asset:.2f}")
            print(f"现金余额: ${self.cash_balance:.2f}")
            print(f"持仓数量: {self.holdings}")
            print(f"当日奖励: {self.rewards_memory[-1] if len(self.rewards_memory) > 0 else 0:.5f}")
            
            # 绘制资产曲线
            plt.figure(figsize=(15, 5))
            plt.plot(self.date_memory, self.asset_memory)
            plt.xlabel('日期')
            plt.ylabel('资产价值')
            plt.title('ETF交易策略资产曲线')
            plt.grid(True)
            
            # 格式化日期轴
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator(interval=4))
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            plt.show()
    
    def get_sb_env(self):
        """
        获取适用于Stable-Baselines的向量化环境
        
        返回:
            DummyVecEnv实例
        """
        e = DummyVecEnv([lambda: self])
        return e
    
    def save_asset_memory(self):
        """
        将资产记忆保存为DataFrame
        
        返回:
            包含日期和资产价值的DataFrame
        """
        # 确保日期内存长度与资产记忆长度一致
        min_length = min(len(self.date_memory), len(self.asset_memory))
        
        # 创建DataFrame
        df_account_value = pd.DataFrame({
            'date': self.date_memory[:min_length],
            'portfolio_value': self.asset_memory[:min_length]
        })
        df_account_value.set_index('date', inplace=True)
        
        return df_account_value
    
    def save_action_memory(self):
        """
        将交易记忆保存为DataFrame
        
        返回:
            包含交易记录的DataFrame
        """
        if not self.actions_memory:
            return pd.DataFrame()
        
        # 创建DataFrame
        df_actions = pd.DataFrame(self.actions_memory)
        
        return df_actions
    
    def get_final_stats(self):
        """
        获取策略的最终统计信息
        
        返回:
            包含策略统计指标的字典
        """
        # 计算收益率
        df_asset = self.save_asset_memory()
        df_returns = df_asset.pct_change().dropna()
        
        # 计算总回报
        total_return = (df_asset['portfolio_value'].iloc[-1] / df_asset['portfolio_value'].iloc[0]) - 1
        
        # 计算年化收益率 (假设252个交易日)
        annual_return = (1 + total_return) ** (252 / len(df_asset)) - 1
        
        # 计算波动率
        volatility = df_returns.std() * np.sqrt(252)
        
        # 计算夏普比率
        sharpe = (annual_return - self.risk_free_rate) / (volatility['portfolio_value'] + 1e-9)
        
        # 计算索提诺比率
        downside_returns = df_returns[df_returns < 0]
        if not downside_returns.empty:
            downside_volatility = downside_returns.std() * np.sqrt(252)
            sortino = (annual_return - self.risk_free_rate) / (downside_volatility['portfolio_value'] + 1e-9)
        else:
            sortino = np.nan
        
        # 计算最大回撤
        df_asset['portfolio_value_cummax'] = df_asset['portfolio_value'].cummax()
        df_asset['drawdown'] = (df_asset['portfolio_value'] / df_asset['portfolio_value_cummax']) - 1
        max_drawdown = df_asset['drawdown'].min()
        
        # 计算胜率 (正回报天数 / 总交易天数)
        win_days = (df_returns > 0).sum()
        total_days = len(df_returns)
        win_rate = win_days / total_days if total_days > 0 else 0
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility['portfolio_value'],
            'sharpe': sharpe,
            'sortino': sortino,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate['portfolio_value'] if isinstance(win_rate, pd.Series) else win_rate,
            'num_trading_days': len(df_asset),
            'final_value': df_asset['portfolio_value'].iloc[-1],
            'initial_value': df_asset['portfolio_value'].iloc[0],
            'total_trades': len(self.trades),
            'total_dividend': self.total_dividend  # 更新为分红总额
        }