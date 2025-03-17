from loguru import logger
import sys
import os
from datetime import datetime

def setup_logger():
    """配置日志记录器"""
    # 移除默认的处理程序
    logger.remove()
    
    # 获取当前日期作为日志文件名的一部分
    current_date = datetime.now().strftime("%Y%m%d")
    log_file = f"logs/backtest_{current_date}.log"
    
    # 确保日志目录存在
    os.makedirs("logs", exist_ok=True)
    
    # 添加控制台输出
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO"
    )
    
    # 添加文件输出
    logger.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="DEBUG",
        rotation="1 day",    # 每天轮换一次日志文件
        retention="30 days", # 保留30天的日志
        compression="zip"    # 压缩旧的日志文件
    )
    
    return logger 