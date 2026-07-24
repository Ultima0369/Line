"""
边缘处理模块 — 本地小模型 + 注意力调度 + 语义压缩。

这是"感官皮层"层，负责：
1. 实时处理传感器数据流
2. 注意力预筛选（判断当前什么信息重要）
3. 语义压缩（把原始数据压缩为特征向量）
4. 上下文管理（维护对话历史并自动摘要）

架构:
    processor.EdgeProcessor  ← 主控制器
    attention.AttentionFilter ← 注意力预筛选
    compressor.Compressor    ← 语义压缩/解压
    context.ContextManager   ← 上下文管理
"""
