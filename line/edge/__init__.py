"""
边缘处理模块 — 本地侧：注意力调度 + 上下文打包 + 对话管理。

负责：
1. 实时处理传感器数据流
2. 注意力预筛选（判断当前什么信息重要）
3. 上下文打包（把传感器/注意力/对话状态编码为紧凑串）
4. 上下文管理（维护对话历史并自动摘要）

架构:
    processor.EdgeProcessor   ← 主控制器
    attention.AttentionFilter ← 注意力预筛选
    compressor.SemanticProtocol ← 上下文打包协议
    context.ContextManager   ← 上下文管理
"""
