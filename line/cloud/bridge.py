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

    async def initialize(self, config: Optional[Dict] = None) -> None:
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

    def _sample_params_from_attention(self, upstream: Dict) -> Dict[str, float]:
        """根据上行数据里的注意力状态推导采样参数。

        这是"认知架构"真正落地的点：注意力分数不再只塞一句话进 prompt，
        而是改变云端推理的采样行为。

        映射依据（可调）:
        - active（高注意力，环境/输入异常，已在聚焦难题）→ 降 temperature，
          让模型更确定、更聚焦于手头问题，少发散。
        - calm（低注意力，环境平稳，认知可松弛）→ 升 temperature，
          让模型发散，利于背景信息流动碰撞（对应 attention.py 里"阴的领域酝酿"）。

        注意力级别是 0-1 的浮点，映射到 temperature 区间 [0.3, 1.0]。
        ponytail: 线性映射 + 钳位，够用且可解释；要更精细可换成分段。
        """
        attention = upstream.get("attention") or ""
        level = self._parse_attention_level(attention)

        # level 越高 → 越聚焦 → temperature 越低
        # level=1.0 → temp 0.3 ; level=0.0 → temp 1.0
        temp = round(1.0 - level * 0.7, 2)
        temp = max(0.3, min(1.0, temp))
        return {"temperature": temp}

    @staticmethod
    def _parse_attention_level(attention: str) -> float:
        """从紧凑注意力串 'l:0.6|s:active' 解析出 level，失败回中性 0.5。"""
        if not attention:
            return 0.5
        for part in attention.split("|"):
            if part.startswith("l:"):
                try:
                    return float(part[2:])
                except ValueError:
                    return 0.5
        return 0.5

    async def ask(
        self,
        upstream: Dict,
        temperature: Optional[float] = None,
        max_tokens: int = 4096,
        stream: bool = False,
    ) -> Dict:
        """向云端发送上行数据包并获取回复。

        upstream: SemanticProtocol.build_upstream 的返回值。
        temperature: 显式覆盖。None 时由注意力状态推导（高注意力→低温度聚焦）。
        返回:
            {
                "content": "回复内容",
                "reasoning": "推理过程（如有）",
                "usage": {"prompt_tokens": N, "completion_tokens": N},
                "latency": 1.23,
                "temperature": 实际使用的采样温度,
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

        # 注意力驱动的采样温度（认知架构落地：分数改变推理行为）
        if temperature is None:
            temperature = self._sample_params_from_attention(upstream)["temperature"]

        payload = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
        }

        url = f"{self._base_url}/chat/completions"
        data = await self._post_with_retry(url, payload)
        if data is None:
            return {"content": "❌ 云端请求失败（已重试）。", "reasoning": "", "usage": {}, "latency": 0}

        latency = time.time() - start_time
        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {})

        result = {
            "content": message.get("content", ""),
            "reasoning": message.get("reasoning_content", ""),
            "usage": data.get("usage", {}),
            "latency": round(latency, 2),
            "temperature": temperature,
        }

        logger.info(
            f"☁️ 云端响应 ({latency:.1f}s | "
            f"input: {data.get('usage', {}).get('prompt_tokens', '?')}tokens)"
        )
        return result

    async def _post_with_retry(self, url: str, payload: Dict) -> Optional[Dict]:
        """POST 并对可重试错误（超时/429/5xx）做指数退避重试。

        4xx（除 429）不重试——那是请求本身的问题。
        返回解析后的 JSON dict，或 None（彻底失败）。
        """
        max_retries = int(self.config.get("max_retries", 3))
        backoff = 0.5

        for attempt in range(max_retries + 1):
            try:
                resp = await self._client.post(url, json=payload)
                resp.raise_for_status()
                return resp.json()
            except httpx.TimeoutException:
                retryable = True
                logger.warning(f"云端超时 (attempt {attempt + 1}/{max_retries + 1})")
            except httpx.HTTPStatusError as e:
                code = e.response.status_code
                # 429 (限流) 和 5xx (服务端) 可重试；其余 4xx 是请求问题，不重试
                retryable = code == 429 or 500 <= code < 600
                logger.warning(f"云端 HTTP {code} (attempt {attempt + 1}/{max_retries + 1}, retryable={retryable})")
            except httpx.RequestError as e:
                retryable = True
                logger.warning(f"云端连接失败: {e} (attempt {attempt + 1}/{max_retries + 1})")
            except Exception as e:
                logger.error(f"未知错误（不重试）: {e}")
                return None

            if not retryable or attempt == max_retries:
                return None
            await asyncio.sleep(backoff)
            backoff *= 2
        return None

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

    async def shutdown(self) -> None:
        """关闭连接。"""
        if self._client:
            await self._client.aclose()
            self._client = None
        logger.info("云端桥接器已关闭")
