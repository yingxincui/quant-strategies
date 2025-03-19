import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime
import os

def load_json_data(file_path=None, uploaded_file=None):
    """从文件或上传的数据加载JSON数据"""
    try:
        if uploaded_file is not None:
            data = json.load(uploaded_file)
        elif file_path is not None:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            st.error("未提供数据源")
            return None
        return data
    except Exception as e:
        st.error(f"加载JSON文件时出错：{e}")
        return None

def flatten_json(data, parent_key='', sep='_'):
    """展平嵌套的JSON数据结构"""
    items = []
    for k, v in data.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_json(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            if len(v) > 0 and isinstance(v[0], dict):
                # 处理字典列表
                for i, item in enumerate(v):
                    items.extend(flatten_json(item, f"{new_key}{sep}{i}", sep=sep).items())
            else:
                items.append((new_key, str(v)))
        else:
            items.append((new_key, v))
    return dict(items)

def process_json_data(data):
    """处理JSON数据以准备可视化"""
    if isinstance(data, list):
        # 如果是列表，尝试将其转换为DataFrame
        if len(data) > 0 and isinstance(data[0], dict):
            df = pd.DataFrame(data)
        else:
            df = pd.DataFrame({"值": data})
    elif isinstance(data, dict):
        # 如果是字典，展平它
        flat_data = flatten_json(data)
        df = pd.DataFrame([flat_data])
    else:
        st.error("不支持的JSON数据格式")
        return None
    
    # 尝试转换日期列
    for col in df.columns:
        if df[col].dtype == object:
            try:
                df[col] = pd.to_datetime(df[col])
            except:
                pass
    
    return df

def get_numeric_columns(df):
    """获取数值类型的列"""
    return df.select_dtypes(include=[np.number]).columns.tolist()

def get_datetime_columns(df):
    """获取日期时间类型的列"""
    return df.select_dtypes(include=['datetime64']).columns.tolist()

def get_categorical_columns(df):
    """获取分类类型的列"""
    return df.select_dtypes(include=['object', 'category']).columns.tolist()

def plot_time_series(df, x_col, y_col, title=None):
    """创建时间序列图表"""
    fig = px.line(df, x=x_col, y=y_col,
                 title=title or f'{y_col}随{x_col}的变化',
                 labels={x_col: x_col, y_col: y_col})
    
    fig.update_layout(
        xaxis_title=x_col,
        yaxis_title=y_col,
        hovermode='x unified'
    )
    
    return fig

def plot_scatter(df, x_col, y_col, color_col=None, title=None):
    """创建散点图"""
    fig = px.scatter(df, x=x_col, y=y_col, color=color_col,
                   title=title or f'{x_col}与{y_col}的关系',
                   labels={x_col: x_col, y_col: y_col})
    
    fig.update_layout(
        xaxis_title=x_col,
        yaxis_title=y_col,
        hovermode='closest'
    )
    
    return fig

def plot_bar(df, x_col, y_col, title=None):
    """创建柱状图"""
    fig = px.bar(df, x=x_col, y=y_col,
                title=title or f'{x_col}的{y_col}分布',
                labels={x_col: x_col, y_col: y_col})
    
    fig.update_layout(
        xaxis_title=x_col,
        yaxis_title=y_col,
        bargap=0.2
    )
    
    return fig

def plot_histogram(df, col, title=None):
    """创建直方图"""
    fig = px.histogram(df, x=col,
                     title=title or f'{col}的分布',
                     labels={col: col})
    
    fig.update_layout(
        xaxis_title=col,
        yaxis_title='频率',
        bargap=0.1
    )
    
    return fig

def plot_box(df, x_col, y_col, title=None):
    """创建箱线图"""
    fig = px.box(df, x=x_col, y=y_col,
                title=title or f'{y_col}按{x_col}的分布',
                labels={x_col: x_col, y_col: y_col})
    
    fig.update_layout(
        xaxis_title=x_col,
        yaxis_title=y_col
    )
    
    return fig

def plot_heatmap(df, x_col, y_col, values_col, title=None):
    """创建热力图"""
    heatmap_data = df.pivot_table(
        values=values_col,
        index=y_col,
        columns=x_col,
        aggfunc='mean'
    )
    
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=heatmap_data.columns,
        y=heatmap_data.index,
        colorscale='Viridis'
    ))
    
    fig.update_layout(
        title=title or f'{x_col}和{y_col}的{values_col}热力图',
        xaxis_title=x_col,
        yaxis_title=y_col
    )
    
    return fig

def main():
    st.title("通用JSON数据可视化工具")
    st.write("此应用程序可以可视化任何结构的JSON数据。")
    
    # 数据输入选项
    data_source = st.radio(
        "选择数据来源",
        ["从cache目录加载", "上传JSON文件"]
    )
    
    data = None
    if data_source == "从cache目录加载":
        json_files = [f for f in os.listdir('../cache') if f.endswith('.json')]
        if not json_files:
            st.error("在'cache'目录中未找到JSON文件。")
            return
        
        selected_file = st.selectbox("选择要可视化的JSON文件", json_files)
        file_path = os.path.join('../cache', selected_file)
        data = load_json_data(file_path=file_path)
    else:
        uploaded_file = st.file_uploader("上传JSON文件", type=['json'])
        if uploaded_file is not None:
            data = load_json_data(uploaded_file=uploaded_file)
    
    if data is None:
        return
    
    # 处理数据
    df = process_json_data(data)
    if df is None:
        return
    
    # 显示基本信息
    st.subheader("数据概览")
    st.write(f"记录数量：{len(df)}")
    st.write(f"列数量：{len(df.columns)}")
    
    # 获取不同类型的列
    numeric_cols = get_numeric_columns(df)
    datetime_cols = get_datetime_columns(df)
    categorical_cols = get_categorical_columns(df)
    
    # 可视化选项
    viz_type = st.selectbox(
        "选择可视化类型",
        ["时间序列图", "散点图", "柱状图", "直方图", "箱线图", "热力图"]
    )
    
    if viz_type == "时间序列图":
        if not datetime_cols:
            st.warning("数据中没有日期时间列，无法创建时间序列图。")
            return
            
        x_col = st.selectbox("选择时间列", datetime_cols)
        y_col = st.selectbox("选择数值列", numeric_cols)
        
        fig = plot_time_series(df, x_col, y_col)
        st.plotly_chart(fig)
        
    elif viz_type == "散点图":
        x_col = st.selectbox("选择X轴", numeric_cols)
        y_col = st.selectbox("选择Y轴", numeric_cols)
        color_col = st.selectbox("选择颜色分类（可选）", ["无"] + categorical_cols)
        
        fig = plot_scatter(df, x_col, y_col, 
                         color_col if color_col != "无" else None)
        st.plotly_chart(fig)
        
    elif viz_type == "柱状图":
        x_col = st.selectbox("选择分类列", categorical_cols)
        y_col = st.selectbox("选择数值列", numeric_cols)
        
        fig = plot_bar(df, x_col, y_col)
        st.plotly_chart(fig)
        
    elif viz_type == "直方图":
        col = st.selectbox("选择数值列", numeric_cols)
        
        fig = plot_histogram(df, col)
        st.plotly_chart(fig)
        
    elif viz_type == "箱线图":
        x_col = st.selectbox("选择分组列", categorical_cols)
        y_col = st.selectbox("选择数值列", numeric_cols)
        
        fig = plot_box(df, x_col, y_col)
        st.plotly_chart(fig)
        
    elif viz_type == "热力图":
        x_col = st.selectbox("选择X轴分类", categorical_cols)
        y_col = st.selectbox("选择Y轴分类", categorical_cols)
        values_col = st.selectbox("选择数值列", numeric_cols)
        
        fig = plot_heatmap(df, x_col, y_col, values_col)
        st.plotly_chart(fig)
    
    # 显示数据统计
    if st.checkbox("显示数据统计"):
        st.subheader("数据统计")
        st.write("数值列统计：")
        st.dataframe(df[numeric_cols].describe())
        
        if categorical_cols:
            st.write("分类列统计：")
            for col in categorical_cols:
                st.write(f"\n{col}的唯一值数量：{df[col].nunique()}")
                st.dataframe(df[col].value_counts().head())
    
    # 显示原始数据
    if st.checkbox("显示原始数据"):
        st.subheader("原始数据")
        st.dataframe(df)
        
        # 添加下载按钮
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="下载数据为CSV",
            data=csv,
            file_name=f'data_{datetime.now().strftime("%Y%m%d")}.csv',
            mime='text/csv',
        )

if __name__ == "__main__":
    main() 