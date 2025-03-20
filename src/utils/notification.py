import json
import os
import requests
from src.utils.logger import setup_logger

logger = setup_logger()

def load_settings():
    """加载通知设置"""
    settings_file = "config/settings.json"
    if os.path.exists(settings_file):
        with open(settings_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"sms": {"enabled": False}, "wechat": {"enabled": False}}

def send_sms(message: str, api_key: str, phone_number: str):
    """发送短信通知"""
    # 这里需要根据实际使用的短信服务商API来实现
    # 示例使用阿里云短信服务
    try:
        # TODO: 实现实际的短信发送逻辑
        logger.info(f"发送短信到 {phone_number}: {message}")
    except Exception as e:
        logger.error(f"发送短信失败: {str(e)}")

def send_wechat(message: str, webhook_url: str):
    """发送企业微信通知"""
    try:
        data = {
            "msgtype": "text",
            "text": {
                "content": message
            }
        }
        response = requests.post(webhook_url, json=data)
        response.raise_for_status()
        logger.info(f"发送企业微信通知成功: {message}")
    except Exception as e:
        logger.error(f"发送企业微信通知失败: {str(e)}")

def send_notification(message: str):
    """发送通知"""
    settings = load_settings()
    
    # 发送短信通知
    if settings["sms"]["enabled"]:
        send_sms(
            message,
            settings["sms"]["api_key"],
            settings["sms"]["phone_number"]
        )
    
    # 发送企业微信通知
    if settings["wechat"]["enabled"]:
        send_wechat(
            message,
            settings["wechat"]["webhook_url"]
        ) 