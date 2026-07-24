"""
上行数据打包模块 — 把本地侧的多源信息组织成云端可消费的紧凑格式。

（曾叫"语义压缩协议"，但旧实现里的 Compressor.compress 实际只是 json.dumps，
产出的字符串从未进入传输——bridge 一直用明文。这是对那段历史的诚实化重写。）

现在这个模块只做两件真实的事：
1. SemanticProtocol: 把传感器/注意力/上下文打包成紧凑字符串
   （如 "tmp:26.5|hum:58.2|prs:1013"），给云端 system prompt 一个结构化的
   环境上下文，同时省 token。
2. build_upstream: 组装完整的上行数据包（user 输入 + 紧凑上下文）。
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class SemanticProtocol:
    """上行数据打包协议 — 本地与云端之间共享的编解码标准。"""

    VERSION = "0.1.0"

    # 传感器类型 → 缩写映射（省 token）
    TYPE_SHORTHAND = {
        "temperature": "tmp",
        "humidity": "hum",
        "pressure": "prs",
        "illuminance": "lux",
        "noise": "noi",
        "system": "sys",
    }

    REVERSE_SHORTHAND = {v: k for k, v in TYPE_SHORTHAND.items()}

    # system 传感器子指标 → 缩写（cpu_temp/cpu_percent/memory_percent 等）
    SYS_METRIC_SHORTHAND = {
        "cpu_temp": "cputmp",
        "cpu_percent": "cpupct",
        "memory_percent": "mempct",
    }

    @classmethod
    def encode_sensor_data(cls, data: Dict) -> str:
        """将传感器数据编码为紧凑字符串。

        输入: {"temperature": [{"value": 26.5, "unit": "°C"}, ...], ...}
        输出: "tmp:26.5|hum:58.2|prs:1013"
        composite 值（如 system 的 dict）按子指标展开:
        "sys:cputmp:27.9|cpupct:2.1|mempct:46.9"
        """
        parts = []
        for sensor_type, readings in data.items():
            shorthand = cls.TYPE_SHORTHAND.get(sensor_type, sensor_type[:3])
            if not readings:
                continue
            first = readings[0]
            val = first.get("value") if isinstance(first, dict) else first.value
            if isinstance(val, dict):
                # composite 传感器：展开每个子指标
                for metric, mval in val.items():
                    mshort = cls.SYS_METRIC_SHORTHAND.get(metric, metric[:6])
                    parts.append(f"{shorthand}:{mshort}:{mval}")
            else:
                parts.append(f"{shorthand}:{val}")
        return "|".join(parts)

    @classmethod
    def build_upstream(
        cls,
        user_input: str,
        sensor_data: Optional[Dict] = None,
        attention_state: Optional[Dict] = None,
        context: Optional[Dict] = None,
        metadata: Optional[Dict] = None,
    ) -> Dict:
        """构建上行数据包（本地 → 云端）。

        这是 edge 交给 bridge 的完整结构。bridge 据此拼 system prompt。
        """
        payload = {
            "version": cls.VERSION,
            "timestamp": datetime.now().isoformat(),
            "type": "upstream",
            "user": user_input[:2000],  # 截断保护
            "sensor": cls._pack_sensor(sensor_data) if sensor_data else None,
            "attention": cls._pack_attention(attention_state) if attention_state else None,
            "context": cls._pack_context(context) if context else None,
        }

        if metadata:
            payload["meta"] = metadata

        return payload

    @classmethod
    def parse_downstream(cls, raw: Dict) -> Dict:
        """解析云端回复，提取结构化和非结构化内容。"""
        return {
            "response": raw.get("content", ""),
            "thinking": raw.get("reasoning", ""),
            "actions_suggested": cls._extract_actions(raw.get("content", "")),
            "confidence": raw.get("confidence", 1.0),
        }

    @classmethod
    def _pack_sensor(cls, data: Dict) -> str:
        return cls.encode_sensor_data(data)

    @classmethod
    def _pack_attention(cls, state: Optional[Dict]) -> Optional[str]:
        if not state:
            return None
        return f"l:{state.get('attention_level', 0)}|s:{state.get('status', 'idle')}"

    @classmethod
    def _pack_context(cls, ctx: Optional[Dict]) -> Optional[str]:
        if not ctx:
            return None
        summary = ctx.get("summary", "")
        turn_count = ctx.get("turn_count", 0)
        return f"t:{turn_count}|s:{summary[:200]}"

    @classmethod
    def _extract_actions(cls, content: str) -> List[str]:
        """从回复中提取建议的操作。"""
        actions = []
        indicators = ["建议你", "你可以", "推荐你", "试试"]
        for indicator in indicators:
            idx = content.find(indicator)
            if idx >= 0:
                end = content.find("。", idx)
                if end >= 0:
                    actions.append(content[idx:end + 1])
        return actions[:3]


# 旧名兼容：曾在此导入 Compressor 的地方仍可工作，但它已不"压缩"任何东西。
# ponytail: 保留别名只为不破坏外部 import，内部一律用 SemanticProtocol.build_upstream。
Compressor = type("Compressor", (), {
    "SemanticProtocol": SemanticProtocol,
    "__doc__": "已废弃。压缩是假的——见模块 docstring。用 SemanticProtocol.build_upstream。",
})()
