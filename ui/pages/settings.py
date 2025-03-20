import streamlit as st
import json
import os
from src.utils.logger import setup_logger

logger = setup_logger()

def load_settings():
    settings_file = "config/settings.json"
    if os.path.exists(settings_file):
        with open(settings_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "sms": {
            "enabled": False,
            "api_key": "",
            "phone_number": ""
        },
        "wechat": {
            "enabled": False,
            "webhook_url": ""
        }
    }

def save_settings(settings):
    os.makedirs("config", exist_ok=True)
    settings_file = "config/settings.json"
    with open(settings_file, "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=4, ensure_ascii=False)

def render_settings():
    st.header("系统设置")
    
    # 加载设置
    settings = load_settings()
    
    # 短信通知设置
    st.subheader("短信通知设置")
    sms_enabled = st.checkbox("启用短信通知", settings["sms"]["enabled"])
    if sms_enabled:
        sms_api_key = st.text_input("短信API密钥", settings["sms"]["api_key"])
        sms_phone = st.text_input("接收手机号", settings["sms"]["phone_number"])
        settings["sms"].update({
            "enabled": True,
            "api_key": sms_api_key,
            "phone_number": sms_phone
        })
    else:
        settings["sms"]["enabled"] = False
    
    # 微信通知设置
    st.subheader("微信通知设置")
    wechat_enabled = st.checkbox("启用微信通知", settings["wechat"]["enabled"])
    if wechat_enabled:
        webhook_url = st.text_input("企业微信Webhook地址", settings["wechat"]["webhook_url"])
        settings["wechat"].update({
            "enabled": True,
            "webhook_url": webhook_url
        })
    else:
        settings["wechat"]["enabled"] = False
    
    # 保存按钮
    if st.button("保存设置"):
        try:
            save_settings(settings)
            st.success("设置保存成功！")
        except Exception as e:
            st.error(f"保存设置时出错: {str(e)}")
            logger.error(f"保存设置时出错: {str(e)}")
