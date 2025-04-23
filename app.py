import streamlit as st
import os
from src.utils.logger import setup_logger
from ui.pages.sidebar import render_sidebar
from ui.pages.backtest import render_backtest
from ui.pages.market import render_market
from ui.pages.settings import render_settings

# 设置日志
logger = setup_logger()

# 设置页面
st.set_page_config(
    page_title="量化策略回测系统",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    st.title("量化策略回测系统")
    
    # 渲染侧边栏并获取参数
    params = render_sidebar()
    if params is None:
        return
        
    # 创建标签页
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["系统介绍", "回测", "实盘记录", "文章", "系统设置"])

    # 系统介绍标签页
    with tab1:
        st.header("系统介绍")
        try:
            with open("ui/help/intro.md", "r", encoding="utf-8") as f:
                st.markdown(f.read())
        except FileNotFoundError:
            st.error("找不到系统介绍文件")
        except Exception as e:
            st.error(f"读取系统介绍文件时出错: {str(e)}")
    
    # 回测标签页
    with tab2:
        render_backtest(params)
    
    # 实盘记录标签页
    with tab3:
        render_market(params)
    
    # 文章标签页
    with tab4:
        st.header("文章列表")
        try:
            # 获取articles目录下的所有md文件
            articles_dir = "ui/articles"
            articles = [f for f in os.listdir(articles_dir) if f.endswith('.md')]
            
            # 创建两列布局
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.subheader("目录")
                # 创建文章列表
                for article in articles:
                    if st.button(article.replace('.md', ''), key=f"article_{article}"):
                        st.session_state.selected_article = article
            
            with col2:
                # 显示选中的文章内容
                selected_article = getattr(st.session_state, 'selected_article', None)
                if selected_article:
                    with open(os.path.join(articles_dir, selected_article), "r", encoding="utf-8") as f:
                        st.markdown(f.read())
                else:
                    st.info("请从左侧选择要阅读的文章")
        except Exception as e:
            st.error(f"读取文章时出错: {str(e)}")
    
    # 系统设置标签页
    with tab5:
        render_settings()

if __name__ == "__main__":
    main() 