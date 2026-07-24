"""
边缘处理器 — 本地侧的主控制器。

相当于"感官皮层"，负责：
1. 从传感器管理器获取数据
2. 用注意力过滤器判断优先级
3. 打包上行数据（传感器/注意力/上下文 → 紧凑结构）
4. 管理上下文
5. 触发上行回调（由 CloudBridge 消费）
6. 接收并解码云端回复

本地小模型推理（local_model 模式）留作占位：加载逻辑需 transformers+torch，
当前回退到直接转发云端。
"""

import asyncio
import logging
from typing import Any, Awaitable, Callable, Dict, List, Optional

from .attention import AttentionFilter, AttentionSignal
from .compressor import SemanticProtocol
from .context import ContextManager

logger = logging.getLogger(__name__)


class EdgeProcessor:
    """边缘处理器 — 双脑架构的本地侧大脑。

    用法:
        processor = EdgeProcessor()
        await processor.initialize()
        response = await processor.process("今天天气怎么样？", sensor_data)
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.attention = AttentionFilter()
        self.protocol = SemanticProtocol()
        self.context = ContextManager(
            max_turns=self.config.get("context", {}).get("max_turns", 20),
            summary_threshold=self.config.get("context", {}).get("summary_threshold", 10),
        )

        # 回调：当有打包好的上行数据时触发
        self._on_upstream: List[Callable[[Dict], Awaitable[None]]] = []

        # 本地小模型（可选，未实现）
        self._local_model = None

        # 注意力阈值：超过时认为环境事件值得打断当前对话
        self._interrupt_threshold = self.config.get("interrupt_threshold", 0.7)
        # 待注入的环境事件队列（轮询产出、对话循环消费）
        self._pending_alerts: List[str] = []

        # 主动推理节流：环境持续异常（如一直高温）时，不能每轮轮询都触发云端烧 token。
        # 冷却期内重复的异常不再触发独立 reasoning turn，只维持 alert 注入。
        # ponytail: 单变量冷却，按墙钟时间；要更细可换 per-source 冷却表。
        self._proactive_cooldown: float = self.config.get("proactive_cooldown", 60.0)
        self._last_proactive_ts: float = 0.0

    async def initialize(self) -> None:
        """初始化边缘处理器。"""
        mode = self.config.get("mode", "api")
        logger.info(f"边缘处理器初始化 (mode={mode})")

        if mode == "local_model":
            await self._init_local_model()

        logger.info("✅ 边缘处理器就绪")

    async def _init_local_model(self):
        """初始化本地小模型（需要 transformers + torch）。

        当前为占位：实际加载需手动启用。
        ponytail: 本地推理是大活，留空回退到 API；路线图未完成前不假装能跑。
        """
        model_name = self.config.get("local_model", {}).get("name", "")
        if not model_name:
            logger.warning("本地模型名称未设置，回退到 API 模式")
            return

        logger.info(f"本地模型加载未实现（model={model_name}），回退到 API 模式")
        logger.info("需手动安装 transformers 和 torch 并取消实现")

    async def process(self, user_input: str,
                      sensor_data: Optional[Dict] = None) -> Dict:
        """处理用户输入，构建并触发上行数据包。

        流程:
        1. 记录用户输入到上下文
        2. 评估注意力信号
        3. 打包上行数据（传感器/注意力/上下文）
        4. 触发上行回调（由 CloudBridge 消费）

        返回上行数据包及其注意力信号摘要。
        """
        # 1. 记录上下文
        self.context.add_turn("user", user_input)

        # 2. 注意力评估（用户输入本身）
        attention_signal = self.attention.evaluate("user_input", user_input)

        # 3. 打包上行数据
        upstream = self.protocol.build_upstream(
            user_input=user_input,
            sensor_data=sensor_data,
            attention_state=self.attention.get_context_summary(),
            context=self.context.get_summary(),
        )

        result = {
            "upstream": upstream,
            "attention_signal": {
                "score": attention_signal.attention_score,
                "priority": attention_signal.priority,
                "urgency": attention_signal.urgency,
            },
        }

        # 4. 触发上行回调
        for cb in self._on_upstream:
            try:
                await cb(upstream)
            except Exception as e:
                logger.error(f"上行回调异常: {e}")

        return result

    def evaluate_sensor_batch(self, sensor_data: Dict) -> Optional[AttentionSignal]:
        """对一批传感器读数做注意力评估，超阈值时入队环境事件。

        供传感器轮询调用：每轮把读数喂给注意力过滤器，
        若某维度注意力分数超过 interrupt_threshold，生成一条人类可读的
        环境提示，加入 _pending_alerts，待下一轮对话前注入。
        """
        if not sensor_data:
            return None

        top_signal: Optional[AttentionSignal] = None
        for sensor_type, readings in sensor_data.items():
            if not readings:
                continue
            first = readings[0]
            raw = first.get("value") if isinstance(first, dict) else getattr(first, "value", None)

            # composite 传感器（如 system 的 {cpu_temp, cpu_percent, memory_percent}）
            # 拆成子指标逐个评估，否则整批里 system 永远被 skip。
            if isinstance(raw, dict):
                sub_items = [(f"{sensor_type}.{k}", v) for k, v in raw.items()
                             if isinstance(v, (int, float))]
            elif isinstance(raw, (int, float)):
                sub_items = [(sensor_type, raw)]
            else:
                continue

            for sub_source, value in sub_items:
                signal = self.attention.evaluate(sub_source, value)
                if top_signal is None or signal.attention_score > top_signal.attention_score:
                    top_signal = signal

                # 物理阈值越界是硬触发：priority=0.9 表示读数越过健康区间，
                # 不该被"新奇度低/变化慢"稀释掉。混合分数低于阈值也照常打断。
                # ponytail: 物理边界优先于统计新奇度，这是正确性的部分，不简化。
                should_fire = signal.priority >= 0.9 or signal.should_interrupt(self._interrupt_threshold)
                if should_fire:
                    alert = f"⚠️ 环境事件: {sub_source}={value} (注意力 {signal.attention_score:.2f})"
                    if alert not in self._pending_alerts[-3:]:
                        self._pending_alerts.append(alert)
                        logger.info(alert)

        return top_signal

    def drain_alerts(self) -> List[str]:
        """取出并清空待注入的环境事件。"""
        alerts = self._pending_alerts
        self._pending_alerts = []
        return alerts

    def maybe_proactive_turn(self, sensor_data: Dict, now: float) -> Optional[Dict]:
        """决定是否就当前环境主动触发一轮云端推理。

        双脑做实的点：本地侧（小脑）不再只被动等用户输入——当环境出现
        值得关注的异常且过了冷却期，它主动构造一个上行包，让云端就环境
        变化发起独立思考，再把结论提示给用户。

        返回一个 upstream 包（应交给 cloud.ask），或 None（不触发）。
        冷却逻辑保证同一异常不会每轮都烧 token。

        now: 当前墙钟时间戳（由调用方传入，避免本模块依赖 time/time.time 的副作用，
              也便于测试注入）。
        """
        if not sensor_data:
            return None
        if now - self._last_proactive_ts < self._proactive_cooldown:
            return None

        # 复用注意力评估：若有任一维度硬触发（物理阈值越界），才考虑主动推理
        top = self.evaluate_sensor_batch(sensor_data)
        if top is None or top.priority < 0.9:
            return None

        # 构造一个环境导向的主动上行包：用户字段留空标记，user 输入是
        # 一段描述当前环境异常的提示，让云端就环境本身回复。
        alert = self._pending_alerts[-1] if self._pending_alerts else str(top.content)
        proactive_input = (
            f"[主动环境推理] 检测到环境异常：{alert}。"
            f"请基于当前传感器数据，简短判断这是否需要用户注意，并给出一句建议。"
        )
        upstream = self.protocol.build_upstream(
            user_input=proactive_input,
            sensor_data=sensor_data,
            attention_state=self.attention.get_context_summary(),
            context=self.context.get_summary(),
            metadata={"proactive": True},
        )
        self._last_proactive_ts = now
        logger.info(f"主动触发云端推理 (冷却 {self._proactive_cooldown}s): {alert}")
        return upstream

    async def receive_downstream(self, cloud_response: Dict) -> str:
        """接收并处理云端回复。"""
        parsed = self.protocol.parse_downstream(cloud_response)

        response_text = parsed.get("response", "")
        if not response_text:
            response_text = cloud_response.get("content", "")

        # 记录到上下文
        self.context.add_turn("assistant", response_text)

        return response_text

    def on_upstream(self, callback: Callable[[Dict], Awaitable[None]]) -> None:
        """注册上行数据包回调。"""
        self._on_upstream.append(callback)

    def get_status(self) -> Dict:
        return {
            "mode": self.config.get("mode", "api"),
            "turn_count": self.context.turn_count,
            "attention": self.attention.get_context_summary(),
            "local_model_loaded": self._local_model is not None,
            "pending_alerts": len(self._pending_alerts),
        }
