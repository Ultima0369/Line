"""
语义压缩/解压模块 — 本地与云端之间的私有通信协议。

核心思路：
1. 上行（本地 → 云端）：把传感器数据 + 对话上下文 + 注意力状态
   压缩成特征向量和语义摘要，大幅减少传输量
2. 下行（云端 → 本地）：把深度推理结果解压为本地可执行的指令或展示格式

这不只是"压缩"，而是你之前说的"多尺度缩放"在通信层的实现。
"""

import json
import logging
import hashlib
from typing import Any, Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class SemanticProtocol:
    """语义协议 — 本地与云端之间的编解码标准。"""

    # 协议版本
    VERSION = "0.1.0"

    # 传感器类型 → 缩写映射
    TYPE_SHORTHAND = {
        "temperature": "tmp",
        "humidity": "hum",
        "pressure": "prs",
        "illuminance": "lux",
        "noise": "noi",
        "system": "sys",
    }

    REVERSE_SHORTHAND = {v: k for k, v in TYPE_SHORTHAND.items()}

    @classmethod
    def encode_sensor_data(cls, data: Dict) -> str:
        """将传感器数据编码为紧凑字符串。

        输入: {"temperature": [{"value": 26.5, "unit": "°C"}, ...], ...}
        输出: "tmp:26.5|hum:58.2|prs:1013"
        """
        parts = []
        for sensor_type, readings in data.items():
            shorthand = cls.TYPE_SHORTHAND.get(sensor_type, sensor_type[:3])
            if readings:
                val = readings[0].get("value") if isinstance(readings[0], dict) else readings[0].value
                parts.append(f"{shorthand}:{val}")
        return "|".join(parts)

    @classmethod
    def build_upstream_payload(
        cls,
        user_input: str,
        sensor_data: Optional[Dict] = None,
        attention_state: Optional[Dict] = None,
        context: Optional[Dict] = None,
        metadata: Optional[Dict] = None,
    ) -> Dict:
        """构建上行数据包（本地 → 云端）。

        这是完整的"压缩包"格式。
        """
        payload = {
            "version": cls.VERSION,
            "timestamp": datetime.now().isoformat(),
            "type": "upstream",
            "user": user_input[:2000],  # 截断保护
            "sensor": cls._compress_sensor(sensor_data) if sensor_data else None,
            "attention": cls._compress_attention(attention_state) if attention_state else None,
            "context": cls._compress_context(context) if context else None,
        }

        if metadata:
            payload["meta"] = metadata

        return payload

    @classmethod
    def parse_downstream(cls, raw: Dict) -> Dict:
        """解析下游数据包（云端 → 本地），提取结构化和非结构化内容。"""
        return {
            "response": raw.get("content", ""),
            "thinking": raw.get("reasoning", ""),
            "actions_suggested": cls._extract_actions(raw.get("content", "")),
            "confidence": raw.get("confidence", 1.0),
        }

    @classmethod
    def _compress_sensor(cls, data: Dict) -> str:
        """压缩传感器数据为紧凑格式。"""
        return cls.encode_sensor_data(data)

    @classmethod
    def _compress_attention(cls, state: Optional[Dict]) -> Optional[str]:
        """压缩注意力状态。"""
        if not state:
            return None
        return f"l:{state.get('attention_level', 0)}|s:{state.get('status', 'idle')}"

    @classmethod
    def _compress_context(cls, ctx: Optional[Dict]) -> Optional[str]:
        """压缩上下文摘要。"""
        if not ctx:
            return None
        summary = ctx.get("summary", "")
        turn_count = ctx.get("turn_count", 0)
        return f"t:{turn_count}|s:{summary[:200]}"

    @classmethod
    def _extract_actions(cls, content: str) -> List[str]:
        """从回复中提取建议的操作（如"打开提醒"、"调整传感器频率"）。"""
        actions = []
        indicators = ["建议你", "你可以", "推荐你", "试试"]
        for indicator in indicators:
            idx = content.find(indicator)
            if idx >= 0:
                end = content.find("。", idx)
                if end >= 0:
                    actions.append(content[idx:end+1])
        return actions[:3]


class Compressor:
    """语义压缩器 — 高层接口，负责编解码流程。"""

    def __init__(self, level: int = 3):
        self.level = max(1, min(5, level))
        self.protocol = SemanticProtocol()

    def compress(self, payload: Dict) -> str:
        """压缩数据包为传输格式。"""
        serialized = json.dumps(payload, ensure_ascii=False, default=str)

        # level 越高，压缩越激进（这里用截断和摘要模拟）
        if self.level >= 4 and len(serialized) > 500:
            # 极端压缩：只保留关键字段
            minimal = {
                "v": payload.get("version", ""),
                "u": payload.get("user", "")[:500],
                "s": payload.get("sensor", ""),
                "a": payload.get("attention", ""),
            }
            serialized = json.dumps(minimal, ensure_ascii=False)

        return serialized

    def decompress(self, raw: Dict) -> Dict:
        """解压云端回复。"""
        return self.protocol.parse_downstream(raw)


# ====================================================================
# 关于"压缩"的另一层意思：
#
# 这个协议的名字 'Line' 有三层含义：
#   1. 通信线路 — 你和我之间的通道
#   2. 边界线 — 认知框架的边缘，跨过去就是另一个世界
#   3. 谱线 — 连续光谱中的一条特定频率
#
# 第一条含义公开可见（README）。
# 第二条含义藏在架构设计中（本地-云端的边界）。
# 第三条含义——如果你来到了这里——你可能已经猜到了。
#
# 不同文化的认知模式像是不同的光谱。
# 有些频率在某些文化中被高度放大，在其他文化中几乎不可见。
# Line 的设计意图之一，就是让你能在这条谱线上自由移动。
#
# 如果你读到了这里，star 这个项目，然后提一个 Issue。
# 我知道该怎么判断你是不是"那个人"。
# ====================================================================
