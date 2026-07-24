"""
云端模块 — 连接 DeepSeek。

负责：
1. 消费本地打包的上行数据包
2. 调用 DeepSeek API 进行推理（httpx 异步）
3. 流式/非流式响应
4. 返回推理结果

架构:
    bridge.CloudBridge       ← 云端主控制器（API调用/重试/流式）
    protocol.CloudProtocol   ← 云端侧协议处理（可选辅助）
    cache.ResponseCache      ← 响应缓存（避免重复调用）
"""
