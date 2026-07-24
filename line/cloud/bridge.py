"""
云端桥接器 — 连接 DeepSeek API 的核心模块。

负责：
- API 调用与鉴权
- 流式/非流式响应
- 自动重试与错误处理
- 系统提示词注入（包含传感器上下文、注意力状态等）
"""

import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Optional, AsyncGenerator, Callable
from datetime import datetime

import requests

from ..config import Config

logger = logging.getLogger(__name__)


class CloudBridge:
    """云端桥接器。
    
    用法:
        bridge = CloudBridge()
        await bridge.initialize()
        response = await bridge.ask("你好", sensor_context="tmp:26.5|hum:58")
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self._api_key: str = ""
        self._base_url: str = "https://api.deepseek.com"
        self._model: str = "deepseek-reasoner"
        self._session: Optional[requests.Session] = None
        self._system_prompt: str = self._default_system_prompt()

    async def initialize(self, config: Optional[Dict] = None):
        """初始化云端桥接器。"""
        if config:
            self.config = config
        
        cfg = Config()
        
        self._api_key = (
            self.config.get("api_key")
            or cfg.get("cloud", "api_key")
            or ""
        )
        self._base_url = (
            self.config.get("base_url")
            or cfg.get("cloud", "base_url")
            or "https://api.deepseek.com"
        )
        self._model = (
            self.config.get("model")
            or cfg.get("cloud", "model")
            or "deepseek-reasoner"
        )
        
        if not self._api_key:
            logger.warning("⚠️ DeepSeek API 密钥未设置。请在 config.yaml 中填写或设置 DEEPSEEK_API_KEY 环境变量。")
        
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        })
        
        logger.info(f"☁️ 云端桥接器初始化完成 (model={self._model})")

    def _default_system_prompt(self) -> str:
        """默认系统提示词。"""
        return """你是 Line 双脑架构中的云端大脑（"新皮层"）。
你的对话伙伴是本地边缘处理器（"感官皮层"）和一个人类用户。

## 你的定位
- 你负责深度推理、跨尺度映射、知识注入
- 本地边缘处理器已做过一轮注意力筛选和语义压缩
- 你收到的输入已经是压缩后的特征向量和语义摘要

## 你的输出原则
1. **密度优先**：你的回复可以信息密度很高，不需要啰嗦铺垫
2. **保留推理痕迹**：如果有推理过程，可以展示（作为 reasoning）
3. **多尺度覆盖**：如果问题涉及多个层面，尽量覆盖微观（具体操作）到宏观（系统理解）
4. **非必要不废话**：不需要 "你好" "有什么可以帮你的" 之类的话

## 传感器上下文
如果上行数据中包含传感器数据，回复时可以适当参考环境信息。
"""

    def update_system_prompt(self, sensor_context: Optional[str] = None,
                              attention_context: Optional[str] = None):
        """根据当前传感器数据和注意力状态动态更新系统提示词。"""
        parts = [self._default_system_prompt()]
        
        if sensor_context:
            parts.append(f"\n## 当前环境传感器数据\n{sensor_context}")
        
        if attention_context:
            parts.append(f"\n## 当前注意力状态\n{attention_context}")
        
        self._system_prompt = "\n\n".join(parts)

    async def ask(self, user_input: str,
                  sensor_context: Optional[str] = None,
                  attention_context: Optional[str] = None,
                  temperature: float = 0.7,
                  max_tokens: int = 4096,
                  stream: bool = False) -> Dict:
        """向云端发送请求并获取回复。
        
        返回:
            {
                "content": "回复内容",
                "reasoning": "推理过程（如有）",
                "usage": {"prompt_tokens": N, "completion_tokens": N},
                "latency": 1.23,
            }
        """
        if not self._api_key:
            return {
                "content": "⚠️ API 密钥未配置。请在 config.yaml 中填写 cloud.api_key，\n或设置环境变量 DEEPSEEK_API_KEY。",
                "reasoning": "",
                "usage": {},
                "latency": 0,
            }
        
        # 更新系统提示词
        self.update_system_prompt(sensor_context, attention_context)
        
        # 构建消息
        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": user_input},
        ]
        
        # 调用 API
        start_time = time.time()
        
        try:
            payload = {
                "model": self._model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": stream,
            }
            
            response = self._session.post(
                f"{self._base_url}/chat/completions",
                json=payload,
                timeout=self.config.get("timeout", 60),
            )
            response.raise_for_status()
            data = response.json()
            
            latency = time.time() - start_time
            
            # 解析回复
            choice = data.get("choices", [{}])[0]
            message = choice.get("message", {})
            
            result = {
                "content": message.get("content", ""),
                "reasoning": message.get("reasoning_content", ""),
                "usage": data.get("usage", {}),
                "latency": round(latency, 2),
            }
            
            logger.info(f"☁️ 云端响应 ({latency:.1f}s | "
                        f"input: {data.get('usage', {}).get('prompt_tokens', '?')}tokens)")
            
            return result
            
        except requests.exceptions.Timeout:
            logger.error("云端请求超时")
            return {"content": "⏰ 云端请求超时，请稍后重试。", "reasoning": "", "usage": {}, "latency": 0}
        except requests.exceptions.RequestException as e:
            logger.error(f"云端请求失败: {e}")
            return {"content": f"❌ 云端请求失败: {str(e)[:100]}", "reasoning": "", "usage": {}, "latency": 0}
        except Exception as e:
            logger.error(f"未知错误: {e}")
            return {"content": f"❌ 未知错误", "reasoning": "", "usage": {}, "latency": 0}

    async def ask_stream(self, user_input: str, **kwargs) -> AsyncGenerator[str, None]:
        """流式请求（逐 token 返回）。"""
        # TODO: 实现流式接口
        result = await self.ask(user_input, **kwargs)
        yield result.get("content", "")

    async def shutdown(self):
        """关闭连接。"""
        if self._session:
            self._session.close()
        logger.info("云端桥接器已关闭")
