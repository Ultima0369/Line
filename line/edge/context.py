"""
上下文管理器 — 维护对话历史，自动摘要，生命周期管理。

功能：
- 记录对话轮次
- 超过阈值后自动进行摘要压缩（用本地或云端模型）
- 为语义协议提供上下文摘要
"""

import logging
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class Turn:
    """一轮对话。"""
    role: str  # "user" | "assistant" | "system"
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict = field(default_factory=dict)


class ContextManager:
    """上下文管理器。
    
    用法:
        ctx = ContextManager(max_turns=20)
        ctx.add_turn("user", "今天天气如何？")
        ctx.add_turn("assistant", "根据传感器数据，26.5°C...")
        summary = ctx.get_summary()
    """

    def __init__(self, max_turns: int = 20, summary_threshold: int = 10):
        self.max_turns = max_turns
        self.summary_threshold = summary_threshold
        self._turns: List[Turn] = []
        self._summary: Optional[str] = None

    def add_turn(self, role: str, content: str, metadata: Optional[Dict] = None):
        """添加一轮对话。"""
        turn = Turn(role=role, content=content, metadata=metadata or {})
        self._turns.append(turn)
        
        # 超出阈值时，触发自动摘要
        if len(self._turns) >= self.summary_threshold and self._summary is None:
            logger.info(f"对话已超过 {self.summary_threshold} 轮，准备摘要压缩")
        
        # 超出最大轮次时，移除最早的
        while len(self._turns) > self.max_turns * 2:
            removed = self._turns.pop(0)
            logger.debug(f"移除旧对话: {removed.content[:50]}...")

    def get_recent(self, n: int = 5) -> List[Turn]:
        """获取最近 n 轮对话。"""
        return self._turns[-n:]

    def get_all(self) -> List[Turn]:
        return self._turns

    def get_summary(self) -> Dict:
        """获取上下文摘要（用于语义协议）。"""
        recent = self.get_recent(5)
        
        # 简单的自动摘要：拼接最近几轮的核心内容
        if not recent:
            return {"summary": "", "turn_count": 0, "recent_topics": []}
        
        # 提取最近的话题
        recent_topics = []
        for t in recent[-3:]:
            if t.role == "user":
                # 提取用户输入的前20个字作为话题线索
                topic = t.content[:30].strip()
                if topic:
                    recent_topics.append(topic)
        
        # 如果已有摘要，合并
        summary_parts = []
        if self._summary:
            summary_parts.append(f"[摘要] {self._summary}")
        
        # 加上最近对话线索
        if recent_topics:
            summary_parts.append(f"[最近] {' → '.join(recent_topics)}")
        
        return {
            "summary": " | ".join(summary_parts) if summary_parts else "",
            "turn_count": len(self._turns),
            "recent_topics": recent_topics,
        }

    def set_summary(self, summary: str):
        """设置人工或外部生成的摘要。"""
        self._summary = summary

    def clear(self):
        """清空上下文。"""
        self._turns.clear()
        self._summary = None

    @property
    def turn_count(self) -> int:
        return len(self._turns)

    def to_dict(self) -> Dict:
        """序列化为字典（用于日志/调试）。"""
        return {
            "turn_count": self.turn_count,
            "summary": self._summary,
            "recent": [{"role": t.role, "content": t.content[:50]} for t in self._turns[-5:]],
        }
