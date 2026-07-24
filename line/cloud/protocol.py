"""
云端语义协议 — 与 edge.compressor 共享编解码逻辑。

这里主要负责"云端侧"的特殊处理：
- 解包上行数据（接收本地发来的压缩包）
- 提取传感器上下文和注意力状态
- 生成适合本地处理的下行格式

实际编解码核心在 edge/compressor.py，这里引用并补充云端视角。
"""

from typing import Any, Dict, Optional

from ..edge.compressor import SemanticProtocol


class CloudProtocol:
    """云端协议处理器。"""

    def __init__(self):
        self.protocol = SemanticProtocol()

    def unpack_upstream(self, payload: Dict) -> Dict:
        """解包上行数据。"""
        return {
            "user_input": payload.get("user", ""),
            "sensor_data": payload.get("sensor", ""),
            "attention": payload.get("attention", ""),
            "context": payload.get("context", ""),
        }

    def build_system_context(self, upacked: Dict) -> str:
        """构建系统上下文文本（注入到系统提示词中）。"""
        parts = []
        
        sensor = upacked.get("sensor_data", "")
        if sensor:
            parts.append(f"[环境] {sensor}")
        
        attention = upacked.get("attention", "")
        if attention:
            parts.append(f"[注意力] {attention}")
        
        return "\n".join(parts)

    def pack_downstream(self, content: str, reasoning: str = "") -> Dict:
        """打包下行数据。"""
        return {
            "content": content,
            "reasoning": reasoning,
            "confidence": 1.0,
        }
