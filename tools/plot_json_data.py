import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime
import os

def load_json_data(file_path):
    """从文件加载JSON数据"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        st.error(f"加载JSON文件时出错：{e}")
        return None

def process_dividend_data(data):
    """处理分红数据以准备可视化"""
    # 转换为DataFrame
    df = pd.DataFrame(data)
    
    # 将日期字符串转换为datetime对象
    df['date'] = pd.to_datetime(df['date'])
    
    # 按日期排序
    df = df.sort_values('date')
    
    # 根据'date'和'dividend'删除重复项
    df = df.drop_duplicates(subset=['date', 'dividend'])
    
    return df

def process_sentiment_data(data):
    """处理情绪数据以准备可视化"""
    # 提取情绪数据并转换为DataFrame
    sentiment_list = data.get('sentiment', [])
    if not sentiment_list:
        st.error("JSON文件中未找到情绪数据")
        return None, None
        
    # 创建日期和值的记录列表
    records = []
    indices_data = {}
    
    for item in sentiment_list:
        records.append({
            'date': item.get('date'),
            'value': item.get('value'),
            'trend': item.get('details', {}).get('trend', None),
            'close': item.get('details', {}).get('close', None),
            'change': item.get('details', {}).get('change', None),
            'volume_ratio': item.get('details', {}).get('volume_ratio', None),
            'rsi': item.get('details', {}).get('rsi', None),
            'positive_volatility': item.get('details', {}).get('positive_volatility', None),
            'negative_volatility': item.get('details', {}).get('negative_volatility', None)
        })
        
        # 提取指数数据（如果可用）
        if 'details' in item and 'indices' in item['details']:
            date = item.get('date')
            if date not in indices_data:
                indices_data[date] = []
            
            for index in item['details']['indices']:
                indices_data[date].append({
                    'code': index.get('code'),
                    'close': index.get('close'),
                    'change': index.get('change'),
                    'trend': index.get('trend')
                })
    
    df = pd.DataFrame(records)
    
    # 将日期字符串转换为datetime对象
    df['date'] = pd.to_datetime(df['date'])
    
    # 按日期排序
    df = df.sort_values('date')
    
    return df, indices_data

def plot_dividend_history(df):
    """创建分红历史图表"""
    fig = px.line(df, x='date', y='dividend', 
                 title='ETF分红历史 (510050.SH)',
                 labels={'date': '日期', 'dividend': '分红值'})
    
    fig.update_layout(
        xaxis_title='日期',
        yaxis_title='分红值',
        hovermode='x unified'
    )
    
    return fig

def plot_dividend_boxplot(df):
    """创建按年份的分红箱线图"""
    df['year'] = df['date'].dt.year
    
    fig = px.box(df, x='year', y='dividend',
                title='按年份的分红分布',
                labels={'year': '年份', 'dividend': '分红值'})
    
    fig.update_layout(
        xaxis_title='年份',
        yaxis_title='分红值'
    )
    
    return fig

def plot_dividend_heatmap(df):
    """创建按年月的分红热力图"""
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    
    # 按年月分组，计算平均分红
    heatmap_data = df.groupby(['year', 'month'])['dividend'].mean().unstack()
    
    # 创建热力图
    fig = go.Figure(data=go.Heatmap(
                   z=heatmap_data.values,
                   x=heatmap_data.columns,
                   y=heatmap_data.index,
                   colorscale='Viridis',
                   hoverongaps=False))
    
    fig.update_layout(
        title='按年月的分红热力图',
        xaxis_title='月份',
        yaxis_title='年份'
    )
    
    return fig

def plot_sentiment_history(df):
    """创建情绪历史图表"""
    fig = px.line(df, x='date', y='value', 
                 title='市场情绪历史',
                 labels={'date': '日期', 'value': '情绪值'})
    
    fig.update_layout(
        xaxis_title='日期',
        yaxis_title='情绪值',
        hovermode='x unified'
    )
    
    return fig

def plot_sentiment_vs_close(df):
    """创建情绪与收盘价对比图"""
    # 检查'close'列是否有数据
    if df['close'].isnull().all():
        st.warning("没有可用的收盘价数据进行可视化")
        return None
        
    # 创建带有次坐标轴的图表
    fig = go.Figure()
    
    # 添加数据
    fig.add_trace(
        go.Scatter(x=df['date'], y=df['value'], name="情绪值", line=dict(color='blue'))
    )
    
    fig.add_trace(
        go.Scatter(x=df['date'], y=df['close'], name="收盘价", line=dict(color='red'),
                  yaxis="y2")
    )
    
    # 设置布局
    fig.update_layout(
        title="情绪值与收盘价对比",
        xaxis_title="日期",
        yaxis=dict(
            title="情绪值",
            titlefont=dict(color='blue'),
            tickfont=dict(color='blue')
        ),
        yaxis2=dict(
            title="收盘价",
            titlefont=dict(color='red'),
            tickfont=dict(color='red'),
            anchor="x",
            overlaying="y",
            side="right"
        ),
        hovermode='x unified'
    )
    
    return fig

def plot_sentiment_components(df):
    """创建情绪组成部分图表"""
    # 选择组成部分
    components = ['positive_volatility', 'negative_volatility', 'rsi', 'volume_ratio']
    
    # 检查是否有任何组成部分的数据
    has_data = False
    for comp in components:
        if not df[comp].isnull().all():
            has_data = True
            break
            
    if not has_data:
        st.warning("没有可用的组成部分数据进行可视化")
        return None
    
    # 创建仅包含所需列的数据框
    plot_df = df[['date'] + components].copy()
    
    # 将数据框转换为单一'value'列
    plot_df = pd.melt(
        plot_df, 
        id_vars=['date'], 
        value_vars=components,
        var_name='component', 
        value_name='value'
    )
    
    # 组件名称中文映射
    component_names = {
        'positive_volatility': '正波动率',
        'negative_volatility': '负波动率',
        'rsi': 'RSI指标',
        'volume_ratio': '成交量比'
    }
    
    # 替换组件名称为中文
    plot_df['component'] = plot_df['component'].map(component_names)
    
    fig = px.line(
        plot_df, 
        x='date', 
        y='value', 
        color='component',
        title='情绪组成部分随时间变化',
        labels={'date': '日期', 'value': '值', 'component': '组成部分'}
    )
    
    fig.update_layout(
        xaxis_title='日期',
        yaxis_title='值',
        legend_title='组成部分',
        hovermode='x unified'
    )
    
    return fig

def plot_sentiment_change_scatter(df):
    """创建情绪与市场变化的散点图"""
    # 检查'change'列是否有数据
    if df['change'].isnull().all():
        st.warning("没有可用的市场变化数据进行可视化")
        return None
        
    # 创建图表
    fig = px.scatter(
        df, 
        x='value', 
        y='change',
        color='trend',
        hover_data=['date', 'close', 'volume_ratio', 'rsi'],
        title='情绪值与市场变化关系',
        labels={
            'value': '情绪值', 
            'change': '市场变化 (%)', 
            'trend': '市场趋势',
            'date': '日期',
            'close': '收盘价',
            'volume_ratio': '成交量比',
            'rsi': 'RSI指标'
        }
    )
    
    fig.update_layout(
        xaxis_title='情绪值',
        yaxis_title='市场变化 (%)',
        hovermode='closest'
    )
    
    # 添加y=0的水平线以显示正负边界
    fig.add_hline(y=0, line_width=1, line_dash="dash", line_color="gray")
    
    # 添加平均情绪值的垂直线
    avg_sentiment = df['value'].mean()
    fig.add_vline(x=avg_sentiment, line_width=1, line_dash="dash", line_color="gray")
    
    return fig

def plot_sentiment_distribution(df):
    """创建情绪值分布直方图"""
    fig = px.histogram(
        df, 
        x='value',
        nbins=30,
        title='情绪值分布',
        labels={'value': '情绪值', 'count': '频率'},
        marginal='box'  # 在边缘添加箱线图
    )
    
    fig.update_layout(
        xaxis_title='情绪值',
        yaxis_title='频率',
        bargap=0.1  # 柱间距
    )
    
    # 添加平均情绪值的垂直线
    avg_sentiment = df['value'].mean()
    fig.add_vline(x=avg_sentiment, line_width=2, line_dash="dash", line_color="red",
                 annotation_text=f"平均值: {avg_sentiment:.2f}", annotation_position="top right")
    
    return fig

def plot_sentiment_trend_analysis(df):
    """创建情绪趋势分析可视化"""
    # 添加滚动平均列
    df_copy = df.copy()
    df_copy['rolling_avg'] = df_copy['value'].rolling(window=5).mean()
    
    # 添加每日情绪变化列
    df_copy['sentiment_change'] = df_copy['value'].diff()
    
    # 创建带有次坐标轴的图表
    fig = go.Figure()
    
    # 添加数据
    fig.add_trace(
        go.Scatter(x=df_copy['date'], y=df_copy['value'], name="情绪值", line=dict(color='blue'))
    )
    
    fig.add_trace(
        go.Scatter(x=df_copy['date'], y=df_copy['rolling_avg'], name="5日移动平均", 
                  line=dict(color='red', dash='dash'))
    )
    
    fig.add_trace(
        go.Bar(x=df_copy['date'], y=df_copy['sentiment_change'], name="每日变化",
              marker_color='green', yaxis="y2", opacity=0.5)
    )
    
    # 设置布局
    fig.update_layout(
        title="情绪趋势分析",
        xaxis_title="日期",
        yaxis=dict(
            title="情绪值",
            titlefont=dict(color='blue'),
            tickfont=dict(color='blue'),
            side="left"
        ),
        yaxis2=dict(
            title="每日变化",
            titlefont=dict(color='green'),
            tickfont=dict(color='green'),
            anchor="x",
            overlaying="y",
            side="right"
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        hovermode='x unified'
    )
    
    return fig

def display_indices_data(indices_data):
    """显示指数数据"""
    if not indices_data:
        st.write("没有可用的指数数据")
        return
        
    # 获取最近的日期
    dates = list(indices_data.keys())
    dates.sort(reverse=True)
    
    # 允许用户选择日期
    selected_date = st.selectbox("选择日期查看指数数据", dates)
    
    if selected_date:
        indices = indices_data[selected_date]
        
        if not indices:
            st.write("该日期没有指数数据")
            return
            
        # 为选定日期创建DataFrame
        indices_df = pd.DataFrame(indices)
        
        # 显示数据
        st.dataframe(indices_df)
        
        # 创建变化的柱状图
        fig = px.bar(
            indices_df, 
            x='code', 
            y='change',
            color='change',
            color_continuous_scale=['red', 'green'],
            title=f'{selected_date}的指数变化',
            labels={'code': '指数代码', 'change': '变化 (%)'}
        )
        
        fig.update_layout(
            xaxis_title='指数代码',
            yaxis_title='变化 (%)'
        )
        
        # 添加y=0的水平线
        fig.add_hline(y=0, line_width=1, line_dash="dash", line_color="gray")
        
        st.plotly_chart(fig)

def convert_df_to_csv(df):
    """将DataFrame转换为CSV以供下载"""
    return df.to_csv(index=False).encode('utf-8')

def main():
    st.title("cache目录数据可视化工具")
    st.write("此应用程序可视化cache目录下的ETF分红数据和市场情绪数据。")
    
    # 文件选择器
    json_files = [f for f in os.listdir('../cache') if f.endswith('.json')]
    
    if not json_files:
        st.error("在'cache'目录中未找到JSON文件。")
        return
    
    selected_file = st.selectbox("选择要可视化的JSON文件", json_files)
    file_path = os.path.join('../cache', selected_file)
    
    # 加载数据
    data = load_json_data(file_path)
    
    if data is None:
        return
    
    # 根据文件类型处理数据
    if "dividend" in selected_file:
        df = process_dividend_data(data)
        
        # 显示基本统计信息
        st.subheader("数据统计")
        st.write(f"记录数量：{len(df)}")
        st.write(f"日期范围：{df['date'].min().date()} 至 {df['date'].max().date()}")
        st.write(f"平均分红：{df['dividend'].mean():.4f}")
        st.write(f"最小分红：{df['dividend'].min():.4f}")
        st.write(f"最大分红：{df['dividend'].max():.4f}")
        
        # 可视化选项
        viz_option = st.selectbox(
            "选择可视化类型",
            ["分红历史", "按年份的分红箱线图", "分红热力图"]
        )
        
        if viz_option == "分红历史":
            fig = plot_dividend_history(df)
            st.plotly_chart(fig)
        elif viz_option == "按年份的分红箱线图":
            fig = plot_dividend_boxplot(df)
            st.plotly_chart(fig)
        elif viz_option == "分红热力图":
            fig = plot_dividend_heatmap(df)
            st.plotly_chart(fig)
        
        # 显示原始数据（如果用户需要）
        if st.checkbox("显示原始数据"):
            st.subheader("原始数据")
            st.dataframe(df)
            
            # 添加下载按钮
            csv = convert_df_to_csv(df)
            st.download_button(
                label="下载数据为CSV",
                data=csv,
                file_name=f'dividend_data_{datetime.now().strftime("%Y%m%d")}.csv',
                mime='text/csv',
            )
    
    elif "sentiment" in selected_file:
        df, indices_data = process_sentiment_data(data)
        
        if df is not None:
            # 显示基本统计信息
            st.subheader("数据统计")
            st.write(f"记录数量：{len(df)}")
            st.write(f"日期范围：{df['date'].min().date()} 至 {df['date'].max().date()}")
            st.write(f"平均情绪值：{df['value'].mean():.4f}")
            st.write(f"最小情绪值：{df['value'].min():.4f}")
            st.write(f"最大情绪值：{df['value'].max():.4f}")
            
            # 可视化选项
            viz_option = st.selectbox(
                "选择可视化类型",
                ["情绪历史", "情绪与收盘价对比", "情绪组成部分", 
                 "情绪与市场变化关系", "情绪分布", "情绪趋势分析"]
            )
            
            if viz_option == "情绪历史":
                fig = plot_sentiment_history(df)
                st.plotly_chart(fig)
            elif viz_option == "情绪与收盘价对比":
                fig = plot_sentiment_vs_close(df)
                if fig:
                    st.plotly_chart(fig)
            elif viz_option == "情绪组成部分":
                fig = plot_sentiment_components(df)
                if fig:
                    st.plotly_chart(fig)
            elif viz_option == "情绪与市场变化关系":
                fig = plot_sentiment_change_scatter(df)
                if fig:
                    st.plotly_chart(fig)
            elif viz_option == "情绪分布":
                fig = plot_sentiment_distribution(df)
                st.plotly_chart(fig)
            elif viz_option == "情绪趋势分析":
                fig = plot_sentiment_trend_analysis(df)
                st.plotly_chart(fig)
            
            # 高级分析选项
            if st.checkbox("显示高级分析"):
                st.subheader("高级分析")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("按趋势的情绪统计")
                    trend_stats = df.groupby('trend')['value'].agg(['mean', 'min', 'max', 'count'])
                    trend_stats.columns = ['平均值', '最小值', '最大值', '数量']
                    st.dataframe(trend_stats)
                
                with col2:
                    st.write("相关性矩阵")
                    corr_columns = ['value', 'close', 'change', 'volume_ratio', 'rsi']
                    column_names = {
                        'value': '情绪值',
                        'close': '收盘价',
                        'change': '变化',
                        'volume_ratio': '成交量比',
                        'rsi': 'RSI指标'
                    }
                    corr_df = df[corr_columns].corr()
                    corr_df.columns = [column_names[col] for col in corr_df.columns]
                    corr_df.index = [column_names[idx] for idx in corr_df.index]
                    st.dataframe(corr_df.style.highlight_max(axis=0))
            
            # 显示指数数据（如果可用）
            if indices_data:
                st.subheader("指数数据")
                display_indices_data(indices_data)
            
            # 显示原始数据（如果用户需要）
            if st.checkbox("显示原始数据"):
                st.subheader("原始数据")
                st.dataframe(df)
                
                # 添加下载按钮
                csv = convert_df_to_csv(df)
                st.download_button(
                    label="下载数据为CSV",
                    data=csv,
                    file_name=f'sentiment_data_{datetime.now().strftime("%Y%m%d")}.csv',
                    mime='text/csv',
                )
    
    else:
        st.error("当前不支持此文件类型的可视化。")

if __name__ == "__main__":
    main() 