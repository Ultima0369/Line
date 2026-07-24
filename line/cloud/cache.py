"""
响应缓存 — 避免重复调用云端 API。

对相似的输入（语义相似度高于阈值）直接返回缓存结果，
节省 API 配额和延迟。
"""

import hashlib
import json
import logging
import time
from typing import Any, Dict, Optional, Tuple
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class ResponseCache:
    """简单的响应缓存（基于输入哈希）。
    
    用法:
        cache = ResponseCache(ttl=3600, max_size=100)
        cache.put("用户输入", {"content": "..."})
        cached = cache.get("用户输入")
    """

    def __init__(self, ttl: int = 3600, max_size: int = 100):
        self.ttl = ttl
        self.max_size = max_size
        self._cache: Dict[str, Tuple[float, Any]] = {}  # key -> (timestamp, data)

    def _make_key(self, user_input: str, sensor_context: str = "") -> str:
        """生成缓存键。"""
        raw = f"{user_input}|{sensor_context}"
        return hashlib.md5(raw.encode()).hexdigest()

    def get(self, user_input: str, sensor_context: str = "") -> Optional[Dict]:
        """获取缓存。"""
        key = self._make_key(user_input, sensor_context)
        entry = self._cache.get(key)
        
        if entry is None:
            return None
        
        timestamp, data = entry
        
        # 检查过期
        if time.time() - timestamp > self.ttl:
            del self._cache[key]
            return None
        
        logger.debug(f"缓存命中: {user_input[:30]}...")
        return data

    def put(self, user_input: str, data: Dict, sensor_context: str = ""):
        """写入缓存。"""
        key = self._make_key(user_input, sensor_context)
        self._cache[key] = (time.time(), data)
        
        # 限制缓存大小
        if len(self._cache) > self.max_size:
            # 移除最旧的条目
            oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k][0])
            del self._cache[oldest_key]
        
        logger.debug(f"缓存写入: {user_input[:30]}...")

    def clear(self):
        """清空缓存。"""
        self._cache.clear()

    @property
    def size(self) -> int:
        return len(self._cache)
