"""
云端桥接器 — 连接 DeepSeek API 的核心模块。

负责：
- API 调用与鉴权（httpx 异步客户端，不阻塞 event loop）
- 流式/非流式响应
- 自动重试与错误处理
- 系统提示词注入（消费 edge 打包的上行数据：传感器上下文、注意力状态）
"""

import asyncio
import logging
import time
from typing import Any, AsyncGenerator, Dict, Optional

import httpx

from ..config import Config

logger = logging.getLogger(__name__)


class CloudBridge:
    """云端桥接器。

    用法:
        bridge = CloudBridge()
        await bridge.initialize()
        upstream = SemanticProtocol.build_upstream(user_input, sensor_data, ...)
        response = await bridge.ask(upstream)
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self._api_key: str = ""
        self._base_url: str = "https://api.deepseek.com"
        self._model: str = "deepseek-reasoner"
        self._client: Optional[httpx.AsyncClient] = None
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

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        self._client = httpx.AsyncClient(
            headers=headers,
            timeout=httpx.Timeout(self.config.get("timeout", 60.0)),
        )

        logger.info(f"☁️ 云端桥接器初始化完成 (model={self._model})")

    def _default_system_prompt(self) -> str:
        """默认系统提示词。"""
        return """你是 Line 的云端大脑。

## 你的定位
- 你负责深度推理、跨尺度映射、知识注入
- 本地边缘处理器已做过一轮注意力筛选和上下文打包
- 你收到的 system prompt 里包含紧凑的环境上下文（传感器、注意力状态）

## 输出原则
1. 密度优先：回复可以信息密度很高，不需要啰嗦铺垫
2. 保留推理痕迹：如果有推理过程，展示出来（作为 reasoning）
3. 多尺度覆盖：涉及多个层面时，覆盖微观（具体操作）到宏观（系统理解）
4. 非必要不废话：不需要 "你好" "有什么可以帮你的" 之类的话

## 传感器上下文
如果系统提示词里包含传感器数据，回复时可以适当参考环境信息。
"""

    def _build_system_prompt(self, upstream: Dict) -> str:
        """消费 edge 打包的上行数据，拼出带环境上下文的 system prompt。"""
        parts = [self._default_system_prompt()]

        sensor = upstream.get("sensor")
        if sensor:
            parts.append(f"\n## 当前环境传感器数据\n{sensor}")

        attention = upstream.get("attention")
        if attention:
            parts.append(f"\n## 当前注意力状态\n{attention}")

        context = upstream.get("context")
        if context:
            parts.append(f"\n## 对话上下文摘要\n{context}")

        return "\n\n".join(parts)

    async def ask(
        self,
        upstream: Dict,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        stream: bool = False,
    ) -> Dict:
        """向云端发送上行数据包并获取回复。

        upstream: SemanticProtocol.build_upstream 的返回值。
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

        if not self._client:
            await self.initialize()

        user_input = upstream.get("user", "")
        system_prompt = self._build_system_prompt(upstream)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ]

        start_time = time.time()

        payload = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
        }

        url = f"{self._base_url}/chat/completions"
        try:
            resp = await self._client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()

            latency = time.time() - start_time
            choice = data.get("choices", [{}])[0]
            message = choice.get("message", {})

            result = {
                "content": message.get("content", ""),
                "reasoning": message.get("reasoning_content", ""),
                "usage": data.get("usage", {}),
                "latency": round(latency, 2),
            }

            logger.info(
                f"☁️ 云端响应 ({latency:.1f}s | "
                f"input: {data.get('usage', {}).get('prompt_tokens', '?')}tokens)"
            )
            return result

        except httpx.TimeoutException:
            logger.error("云端请求超时")
            return {"content": "⏰ 云端请求超时，请稍后重试。", "reasoning": "", "usage": {}, "latency": 0}
        except httpx.HTTPStatusError as e:
            logger.error(f"云端请求失败: {e.response.status_code} {e.response.text[:100]}")
            return {"content": f"❌ 云端请求失败 ({e.response.status_code})", "reasoning": "", "usage": {}, "latency": 0}
        except httpx.RequestError as e:
            logger.error(f"云端连接失败: {e}")
            return {"content": f"❌ 云端连接失败: {str(e)[:100]}", "reasoning": "", "usage": {}, "latency": 0}
        except Exception as e:
            logger.error(f"未知错误: {e}")
            return {"content": "❌ 未知错误", "reasoning": "", "usage": {}, "latency": 0}

    async def ask_stream(self, upstream: Dict, **kwargs) -> AsyncGenerator[str, None]:
        """流式请求（逐 chunk 返回内容文本）。"""
        if not self._api_key:
            yield "⚠️ API 密钥未配置。"
            return
        if not self._client:
            await self.initialize()

        user_input = upstream.get("user", "")
        system_prompt = self._build_system_prompt(upstream)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ]

        payload = {
            "model": self._model,
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 4096),
            "stream": True,
        }
        url = f"{self._base_url}/chat/completions"

        try:
            async with self._client.stream("POST", url, json=payload) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    line = line.strip()
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        import json
                        chunk = json.loads(data)
                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            yield content
                    except Exception:
                        continue
        except (httpx.TimeoutException, httpx.RequestError) as e:
            logger.error(f"流式请求失败: {e}")
            yield f"\n❌ 流式请求中断: {str(e)[:80]}"

    async def shutdown(self):
        """关闭连接。"""
        if self._client:
            await self._client.aclose()
            self._client = None
        logger.info("云端桥接器已关闭")
