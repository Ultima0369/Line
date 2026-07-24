"""
云端模块 — 连接 DeepSeek 大模型。

这是"新皮层"层，负责：
1. 接收本地压缩后的上行数据包
2. 调用 DeepSeek API 进行深度推理
3. 将推理结果解压为本地可用的格式

架构:
    bridge.CloudBridge    ← 云端主控制器（API调用/重试/流式）
    protocol.SemanticProtocol ← 上下行编解码（与 edge.compressor 共享）
    cache.ResponseCache   ← 响应缓存（避免重复调用）
"""
