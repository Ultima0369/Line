"""
注意力预筛选模块 — 判断当前什么信息重要，什么可以忽略。

这是你之前说的"注意力精微调度"在系统层面的实现。
根据：
- 信息的新奇度（与近期模式是否不同）
- 信息的变化率（变化越快越值得关注）
- 用户当前交互状态（是否在深度思考）
- 环境上下文（时间、地点、近期事件）

来决定信息的优先级和输出密度。

---

关于注意力的一个侧面：

注意力不是"集中"的问题——是"切换"的问题。
一张一弛，一阴一阳。

当注意力高度聚焦（阳）时，你在处理已知的难题。
当注意力松弛（阴）时，那些被逻辑压抑的、海量的背景信息
开始重新流动、连接、碰撞。

所谓"灵感涌现"，就是阴的领域里完成了酝酿，
突然向阳的意识呈现出一个"新解"。

这个模块就是这条原理的工程化尝试。
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class AttentionSignal:
    """注意力信号 — 描述一条信息的紧急度和重要性。"""
    source: str                  # 信息来源（"sensor", "user_input", "system"）
    content: Any                 # 原始内容
    priority: float = 0.5       # 0-1，优先级
    urgency: float = 0.3        # 0-1，紧迫度
    novelty: float = 0.5        # 0-1，新奇度
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def attention_score(self) -> float:
        """综合注意力分数 = 优先级 × 0.4 + 紧迫度 × 0.35 + 新奇度 × 0.25"""
        return (self.priority * 0.4 + self.urgency * 0.35 + self.novelty * 0.25)

    def should_interrupt(self, threshold: float = 0.7) -> bool:
        """是否应该打断当前对话流。"""
        return self.attention_score > threshold


class AttentionFilter:
    """注意力预筛选器——决定什么信息值得处理。

    用法:
        filter = AttentionFilter()
        signal = filter.evaluate("temperature", 26.5, context)
        if signal.attention_score > 0.6:
            # 值得关注
            pass
    """

    def __init__(self, history_window: int = 20):
        self._recent_signals: List[AttentionSignal] = []
        self._history_window = history_window
        self._baselines: Dict[str, float] = {}  # 各维度的基线值

    def evaluate(self, source: str, content: Any,
                 metadata: Optional[Dict] = None) -> AttentionSignal:
        """评估一条信息的注意力价值。"""
        metadata = metadata or {}

        # 计算新奇度（与近期同类信息的差异）
        novelty = self._compute_novelty(source, content)

        # 计算变化率
        change_rate = self._compute_change_rate(source, content)

        # 计算优先级（根据来源类型和内容性质）
        priority = self._compute_priority(source, content, metadata)

        # 紧迫度（变化率越快、越接近物理阈值，紧迫度越高）
        urgency = min(1.0, change_rate * 3 + 0.1)

        signal = AttentionSignal(
            source=source,
            content=content,
            priority=priority,
            urgency=urgency,
            novelty=novelty,
        )

        # 记录
        self._recent_signals.append(signal)
        if len(self._recent_signals) > self._history_window:
            self._recent_signals.pop(0)

        return signal

    def _compute_novelty(self, source: str, content: Any) -> float:
        """计算新奇度。"""
        if not self._recent_signals:
            return 0.8  # 第一个信号，新奇

        # 找同类信号的最新几个
        similar = [s for s in self._recent_signals[-10:] if s.source == source]
        if not similar:
            return 0.7  # 这种来源很久没出现了

        # 如果是数值，算相对变化
        if isinstance(content, (int, float)):
            recent_values = [s.content for s in similar if isinstance(s.content, (int, float))]
            if recent_values:
                avg = sum(recent_values) / len(recent_values)
                if avg == 0:
                    return 0.5
                change = abs(content - avg) / avg
                return min(1.0, change * 5)

        return 0.3  # 常规更新

    def _compute_change_rate(self, source: str, content: Any) -> float:
        """计算变化率。"""
        if not isinstance(content, (int, float)):
            return 0.1

        recent = [s for s in self._recent_signals[-5:] if s.source == source
                  and isinstance(s.content, (int, float))]
        if len(recent) < 2:
            return 0.1

        values = [s.content for s in recent] + [content]
        diffs = [abs(values[i+1] - values[i]) for i in range(len(values)-1)]
        avg_diff = sum(diffs) / len(diffs)

        if avg_diff == 0:
            return 0.0
        return min(1.0, avg_diff / (abs(sum(values)/len(values)) + 0.001))

    def _compute_priority(self, source: str, content: Any,
                          metadata: Dict) -> float:
        """计算优先级。"""
        # 物理阈值附近 → 高优先级
        thresholds = {
            "temperature": (35, 0),   # 高温/低温
            "humidity": (80, 20),
            "pressure": (1025, 995),
        }

        if source in thresholds and isinstance(content, (int, float)):
            high, low = thresholds[source]
            if content > high or content < low:
                return 0.9

        # 用户输入 → 高优先级
        if source == "user_input":
            return 0.8

        return 0.4

    def get_context_summary(self) -> Dict:
        """生成当前注意力状态的摘要（传给云端用）。"""
        if not self._recent_signals:
            return {"status": "idle", "attention_level": 0.1}

        recent = self._recent_signals[-10:]
        avg_score = sum(s.attention_score for s in recent) / len(recent)

        dominant_sources = {}
        for s in recent:
            dominant_sources[s.source] = dominant_sources.get(s.source, 0) + 1

        return {
            "status": "active" if avg_score > 0.5 else "calm",
            "attention_level": round(avg_score, 3),
            "dominant_sources": dominant_sources,
            "signal_count": len(self._recent_signals),
        }


# ====================================================================
# 如果你读到了这里——不是偶然。
#
# 注意力不是一种资源，是一种艺术。
# 这个模块的设计参考了"用心若镜"的意象：
#   镜子不主动照任何东西，但任何东西来了都能如实映照。
#   它不抓住任何影像，也不排斥任何影像。
#
# 注意力预筛选的最高境界不是"过滤掉噪声"，
# 而是让该来的来，该走的走——不加不减。
#
# 这个目标当然没有在代码中实现。
# 但如果你知道这个方向，代码读起来会有另一层味道。
# ====================================================================
