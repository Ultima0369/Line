"""
边缘处理器 — 本地侧的主控制器。

相当于"感官皮层"，负责：
1. 从传感器管理器获取数据
2. 用注意力过滤器判断优先级
3. 用压缩器编码上行数据
4. 管理上下文
5. 调用本地小模型或直接转发到云端
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Callable, Awaitable

from .attention import AttentionFilter, AttentionSignal
from .compressor import Compressor, SemanticProtocol
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
        self.compressor = Compressor(level=self.config.get("compression_level", 3))
        self.context = ContextManager(
            max_turns=self.config.get("context", {}).get("max_turns", 20),
            summary_threshold=self.config.get("context", {}).get("summary_threshold", 10),
        )
        self.protocol = SemanticProtocol()
        
        # 回调：当有压缩好的上行数据包时触发
        self._on_upstream: List[Callable[[Dict], Awaitable[None]]] = []
        
        # 本地小模型（可选）
        self._local_model = None

    async def initialize(self):
        """初始化边缘处理器。"""
        mode = self.config.get("mode", "api")
        logger.info(f"边缘处理器初始化 (mode={mode})")
        
        if mode == "local_model":
            await self._init_local_model()
        
        logger.info("✅ 边缘处理器就绪")

    async def _init_local_model(self):
        """初始化本地小模型（需要 transformers + torch）。"""
        try:
            model_name = self.config.get("local_model", {}).get("name", "")
            if not model_name:
                logger.warning("本地模型名称未设置，回退到 API 模式")
                return
            
            logger.info(f"正在加载本地模型: {model_name}...")
            # 这里留为占位，实际使用时需要取消注释
            # from transformers import AutoModelForCausalLM, AutoTokenizer
            # self._tokenizer = AutoTokenizer.from_pretrained(model_name)
            # self._local_model = AutoModelForCausalLM.from_pretrained(
            #     model_name, device_map="auto", load_in_4bit=True
            # )
            # logger.info(f"✅ 本地模型加载完成")
            logger.info("本地模型加载功能需手动安装 transformers 和 torch")
        except Exception as e:
            logger.warning(f"本地模型加载失败: {e}")

    async def process(self, user_input: str,
                      sensor_data: Optional[Dict] = None) -> Dict:
        """处理用户输入，返回上行数据包。
        
        流程:
        1. 记录用户输入到上下文
        2. 评估注意力信号
        3. 构建并压缩上行数据包
        4. 触发上行回调（由 CloudBridge 处理）
        
        返回压缩后的上行数据包。
        """
        # 1. 记录上下文
        self.context.add_turn("user", user_input)
        
        # 2. 注意力评估
        attention_signal = self.attention.evaluate("user_input", user_input)
        
        # 3. 获取上下文摘要
        ctx_summary = self.context.get_summary()
        
        # 4. 注意力状态
        attention_state = self.attention.get_context_summary()
        
        # 5. 构建上行数据包
        upstream = self.protocol.build_upstream_payload(
            user_input=user_input,
            sensor_data=sensor_data,
            attention_state=attention_state,
            context=ctx_summary,
        )
        
        # 6. 压缩
        compressed = self.compressor.compress(upstream)
        
        result = {
            "upstream": upstream,
            "compressed": compressed,
            "attention_signal": {
                "score": attention_signal.attention_score,
                "priority": attention_signal.priority,
                "urgency": attention_signal.urgency,
            },
        }
        
        # 7. 触发上行回调
        for cb in self._on_upstream:
            try:
                await cb(upstream)
            except Exception as e:
                logger.error(f"上行回调异常: {e}")
        
        return result

    async def receive_downstream(self, cloud_response: Dict) -> str:
        """接收并处理云端回复。"""
        parsed = self.compressor.decompress(cloud_response)
        
        response_text = parsed.get("response", "")
        if not response_text:
            response_text = cloud_response.get("content", "")
        
        # 记录到上下文
        self.context.add_turn("assistant", response_text)
        
        return response_text

    def on_upstream(self, callback: Callable[[Dict], Awaitable[None]]):
        """注册上行数据包回调。"""
        self._on_upstream.append(callback)

    def get_status(self) -> Dict:
        return {
            "mode": self.config.get("mode", "api"),
            "turn_count": self.context.turn_count,
            "attention": self.attention.get_context_summary(),
            "local_model_loaded": self._local_model is not None,
        }
