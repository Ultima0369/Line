"""
传感器抽象基类 — 所有传感器的统一接口。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from datetime import datetime


@dataclass
class SensorReading:
    """一次传感器读数的标准格式。"""
    sensor_id: str                # 传感器唯一标识
    sensor_type: str              # 传感器类型（"temperature", "pressure" 等）
    value: Any                    # 读数主值
    unit: str                     # 单位
    timestamp: datetime = field(default_factory=datetime.now)
    raw: Optional[Dict] = None    # 原始数据
    confidence: float = 1.0       # 置信度 0-1

    def to_dict(self) -> Dict:
        return {
            "sensor_id": self.sensor_id,
            "sensor_type": self.sensor_type,
            "value": self.value,
            "unit": self.unit,
            "timestamp": self.timestamp.isoformat(),
            "confidence": self.confidence,
        }

    def to_feature_vector(self) -> Dict:
        """压缩为特征向量格式（供语义协议传输）。"""
        return {
            "t": self.sensor_type[:4],      # 类型缩写
            "v": round(self.value, 2),      # 值
            "u": self.unit[0],              # 单位首字母
            "c": round(self.confidence, 2),
        }


class Sensor(ABC):
    """传感器抽象基类。"""

    def __init__(self, sensor_id: str, config: Optional[Dict] = None):
        self.sensor_id = sensor_id
        self.config = config or {}
        self._is_running = False
        self._last_reading: Optional[SensorReading] = None

    @property
    @abstractmethod
    def sensor_type(self) -> str:
        """传感器类型标识，如 'temperature'。"""
        pass

    @abstractmethod
    async def read(self) -> Optional[SensorReading]:
        """执行一次读取，返回读数。失败时返回 None。"""
        pass

    async def initialize(self) -> bool:
        """初始化传感器（如打开串口、I2C总线）。返回是否成功。"""
        self._is_running = True
        return True

    async def shutdown(self):
        """关闭传感器。"""
        self._is_running = False

    @property
    def is_running(self) -> bool:
        return self._is_running

    @property
    def last_reading(self) -> Optional[SensorReading]:
        return self._last_reading

    def __repr__(self) -> str:
        return f"<Sensor {self.sensor_id} ({self.sensor_type})>"
