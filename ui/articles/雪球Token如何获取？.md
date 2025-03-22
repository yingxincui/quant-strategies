# 雪球Token如何获取？

雪球实时行情数据可以通过以下接口获取：
```
https://stock.xueqiu.com/v5/stock/quote.json?symbol=SH000001&extend=detail
```
注意：此接口需要token认证才能访问。

## 接口详情

### AKShare集成接口
- 接口名称: stock_individual_spot_xq
- 接口类型: 雪球-行情中心-个股
- 目标地址: https://xueqiu.com/S/SH513520
- 访问限制: 单次获取指定 symbol 的最新行情数据

### 认证信息获取步骤

获取token的具体方法如下：

1. 登录雪球官方网站 (https://xueqiu.com)
2. 打开浏览器开发者工具（按F12键）
3. 在浏览器中请求测试接口：
   ```
   https://stock.xueqiu.com/v5/stock/quote.json?symbol=SH000001&extend=detail
   ```
4. 在开发者工具的"网络"(Network)标签页中，找到该请求
5. 在请求的标头(Headers)信息中找到cookie字段，内容示例如下：
   ```
   cookie: cookiesu=631719021518417; device_id=c37e94779eede2e3f0250482d804ff81; s=af1w3zgebe; bid=03b31a66824a4a426b80229356033a8c_m4hqnffb; Hm_lvt_1db88642e346389874251b5a1eded6e3=1742519317; HMACCOUNT=2436B2ECFC061307; xq_a_token=d679467b716fd5b0a0af195f7e8143774d271a41; [... 其他cookie值 ...]
   ```
6. 从cookie中提取`xq_a_token`字段的值，这就是我们需要的token

### Token示例
```
xq_a_token=d679467b716fd5b0a0af195f7e8143774d271a41
```

## 使用注意事项
- Token具有时效性，一般有效期为7天，需要定期更新
- 请合理控制接口调用频率，避免触发雪球的访问限制
- 建议在程序中实现token自动更新机制
- 不要公开分享或传播你的个人token
- 使用token时建议通过环境变量或配置文件管理，避免硬编码

## 常见问题
- 如遇到"未授权"错误，请检查token是否过期
- 如遇到访问限制，请适当降低请求频率
- Token仅对获取它时登录的账号有效