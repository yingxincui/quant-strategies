import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import pandas as pd

class Plot:
    def __init__(self, strategy):
        self.strategy = strategy

    def plot(self, **kwargs):
        """使用Plotly绘制交互式回测结果"""
        # 获取策略数据
        data = self.strategy.data
        
        # 将数据转换为numpy数组，处理日期时间
        dates = [datetime.fromordinal(int(d)).strftime('%Y-%m-%d') for d in data.datetime.array]
        opens = np.array(data.open.array)
        highs = np.array(data.high.array)
        lows = np.array(data.low.array)
        closes = np.array(data.close.array)
        volumes = np.array(data.volume.array)
        
        # 检查trailing_stop是否存在
        trailing_stop = getattr(self.strategy, 'trailing_stop', None)
        if trailing_stop is not None:
            trailing_stop_vals = np.array(trailing_stop.trailing_stop.array)
        else:
            # 如果trailing_stop为None，创建一个全为0的数组
            trailing_stop_vals = np.zeros_like(closes)
        
        # 创建DataFrame以便处理数据
        df = pd.DataFrame({
            'date': dates,
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': volumes,
            'trailing_stop': trailing_stop_vals
        })
        
        # 移除volume为0的行（非交易日）
        df = df[df['volume'] > 0].copy()
        
        # 创建子图
        fig = make_subplots(rows=2, cols=1, 
                           shared_xaxes=True,
                           vertical_spacing=0.03, 
                           row_heights=[0.7, 0.3])
        
        # 添加K线图
        fig.add_trace(go.Candlestick(
            x=df['date'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='K线',
            increasing_line_color='#ff0000',  # 上涨为红色
            decreasing_line_color='#00ff00',  # 下跌为绿色
        ), row=1, col=1)
        
        # 添加追踪止损线
        # 过滤掉追踪止损为0的点
        valid_stops = df[df['trailing_stop'] > 0].copy()
        if not valid_stops.empty:
            fig.add_trace(go.Scatter(
                x=valid_stops['date'],
                y=valid_stops['trailing_stop'],
                name='追踪止损',
                line=dict(color='#f1c40f', dash='dash')
            ), row=1, col=1)
        
        # 添加买卖点标记
        if hasattr(self, 'trades_df') and not self.trades_df.empty:
            # 获取买入点
            buy_points = self.trades_df[self.trades_df['类型'] == '买入']
            if not buy_points.empty:
                fig.add_trace(go.Scatter(
                    x=buy_points['时间'].dt.strftime('%Y-%m-%d'),
                    y=buy_points['价格'],
                    mode='markers+text',
                    marker=dict(symbol='triangle-up', size=15, color='#2ecc71'),
                    text=[f"买入\n价格:{price:.2f}\n数量:{size}" 
                          for price, size in zip(buy_points['价格'], buy_points['数量'])],
                    textposition="top center",
                    name='买入点',
                    hoverinfo='text'
                ), row=1, col=1)
            
            # 获取卖出点
            sell_points = self.trades_df[self.trades_df['类型'] == '卖出']
            if not sell_points.empty:
                fig.add_trace(go.Scatter(
                    x=sell_points['时间'].dt.strftime('%Y-%m-%d'),
                    y=sell_points['价格'],
                    mode='markers+text',
                    marker=dict(symbol='triangle-down', size=15, color='#e74c3c'),
                    text=[f"卖出\n价格:{price:.2f}\n收益:{profit:.2f}" 
                          for price, profit in zip(sell_points['价格'], sell_points['累计收益'])],
                    textposition="bottom center",
                    name='卖出点',
                    hoverinfo='text'
                ), row=1, col=1)
        
        # 添加成交量图
        colors = ['#ff0000' if close >= open_ else '#00ff00' 
                 for close, open_ in zip(df['close'], df['open'])]
        
        fig.add_trace(go.Bar(
            x=df['date'],
            y=df['volume'],
            name='成交量',
            marker_color=colors,
            marker=dict(
                color=colors,
                line=dict(color=colors, width=1)
            ),
            hovertemplate='日期: %{x}<br>成交量: %{y:,.0f}<extra></extra>'
        ), row=2, col=1)
        
        # 更新布局
        fig.update_layout(
            title={'text': '回测结果', 'font': {'family': 'Arial'}},
            yaxis_title={'text': '价格', 'font': {'family': 'Arial'}},
            yaxis2_title={'text': '成交量', 'font': {'family': 'Arial'}},
            xaxis_rangeslider_visible=False,
            height=800,
            template='plotly_white',
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                font=dict(family='Arial')
            ),
            # 优化X轴显示
            xaxis=dict(
                type='category',
                rangeslider=dict(visible=False),
                showgrid=False,  # 移除X轴网格线
                gridwidth=1,
                gridcolor='lightgrey'
            ),
            xaxis2=dict(
                type='category',
                rangeslider=dict(visible=False),
                showgrid=False,  # 移除X轴网格线
                gridwidth=1,
                gridcolor='lightgrey'
            ),
            bargap=0,  # 设置柱状图之间的间隔为0
            bargroupgap=0  # 设置柱状图组之间的间隔为0
        )
        
        # 更新Y轴格式
        fig.update_yaxes(
            title_text="价格", 
            title_font=dict(family='Arial'),
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgrey',
            row=1, 
            col=1
        )
        fig.update_yaxes(
            title_text="成交量", 
            title_font=dict(family='Arial'),
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgrey',
            row=2, 
            col=1
        )
        
        return fig