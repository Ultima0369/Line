"""
模拟传感器 — 无硬件时用于开发和演示。

通过随机波动模拟真实环境数据。
你可以在控制台看到类似这样的输出：
    🌡️  温度: 26.3°C  |  💧 湿度: 58.7%  |  🌬️ 气压: 1013.2hPa
    ☀️  光照: 423.8lux |  🔊 噪声: 42.1dB
"""

import asyncio
import random
import math
from datetime import datetime
from typing import List, Optional

from .base import Sensor, SensorReading


class MockSensor(Sensor):
    """模拟传感器基类，产生带周期性波动的模拟数据。"""

    def __init__(self, sensor_id: str, sensor_type: str,
                 base_value: float, variance: float, unit: str,
                 cycle_period: float = 60.0):
        super().__init__(sensor_id)
        self._type = sensor_type
        self.base_value = base_value
        self.variance = variance
        self._unit = unit
        self.cycle_period = cycle_period  # 模拟日周期（秒）
        self._start_time = datetime.now()
        self._drift = random.uniform(-0.1, 0.1)

    @property
    def sensor_type(self) -> str:
        return self._type

    async def read(self) -> Optional[SensorReading]:
        """生成模拟读数，包含：日周期波动 + 随机噪声 + 缓慢漂移。"""
        elapsed = (datetime.now() - self._start_time).total_seconds()
        
        # 日周期模拟（正弦波）
        cycle = math.sin(elapsed * 2 * math.pi / self.cycle_period)
        
        # 随机噪声
        noise = random.gauss(0, self.variance * 0.15)
        
        # 缓慢漂移
        self._drift += random.gauss(0, 0.01)
        self._drift = max(-0.5, min(0.5, self._drift))
        
        value = self.base_value + cycle * self.variance * 0.5 + noise + self._drift
        value = round(max(0, value), 1)
        
        return SensorReading(
            sensor_id=self.sensor_id,
            sensor_type=self._type,
            value=value,
            unit=self._unit,
            confidence=round(random.uniform(0.85, 0.99), 2),
        )


class MockSensorGroup:
    """一组模拟传感器，覆盖常见的环境监测维度。"""

    def __init__(self, prefix: str = "mock"):
        self.sensors: List[MockSensor] = [
            MockSensor(f"{prefix}_temp", "temperature", 26.0, 3.0, "°C", cycle_period=120),
            MockSensor(f"{prefix}_humidity", "humidity", 60.0, 10.0, "%", cycle_period=180),
            MockSensor(f"{prefix}_pressure", "pressure", 1013.0, 5.0, "hPa", cycle_period=300),
            MockSensor(f"{prefix}_light", "illuminance", 500.0, 200.0, "lux", cycle_period=90),
            MockSensor(f"{prefix}_noise", "noise", 40.0, 10.0, "dB", cycle_period=45),
        ]
