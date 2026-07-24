"""
传感器管理器 — 统一注册、轮询、数据聚合。
"""

import asyncio
import logging
from typing import Dict, List, Optional, Callable, Awaitable
from datetime import datetime

from .base import Sensor, SensorReading

logger = logging.getLogger(__name__)


class SensorManager:
    """传感器管理器。
    
    用法:
        manager = SensorManager()
        await manager.register(TemperatureSensor("temp_01"))
        await manager.scan()          # 扫描并初始化所有传感器
        data = await manager.read_all()  # 读取所有传感器
        print(data["temperature"][0].value)  # 28.5
    """

    def __init__(self):
        self._sensors: Dict[str, Sensor] = {}
        self._callbacks: List[Callable[[SensorReading], Awaitable[None]]] = []
        self._batch_callbacks: List[Callable[[Dict[str, List[SensorReading]]], Awaitable[None]]] = []
        self._polling_task: Optional[asyncio.Task] = None

    async def register(self, sensor: Sensor) -> bool:
        """注册一个传感器并初始化。"""
        if sensor.sensor_id in self._sensors:
            logger.warning(f"传感器 {sensor.sensor_id} 已注册，跳过")
            return False
        
        ok = await sensor.initialize()
        if ok:
            self._sensors[sensor.sensor_id] = sensor
            logger.info(f"✅ 传感器注册成功: {sensor}")
        else:
            logger.warning(f"❌ 传感器初始化失败: {sensor}")
        return ok

    def unregister(self, sensor_id: str) -> None:
        """注销一个传感器。"""
        sensor = self._sensors.pop(sensor_id, None)
        if sensor:
            asyncio.ensure_future(sensor.shutdown())

    async def scan(self, config: Optional[Dict] = None) -> int:
        """根据配置自动扫描并注册传感器。返回注册数量。"""
        count = 0
        config = config or {}
        
        # 系统传感器（跨平台可用）
        if config.get("system", {}).get("enabled", True):
            try:
                from .system import SystemSensor
                await self.register(SystemSensor("system_01"))
                count += 1
            except Exception as e:
                logger.warning(f"系统传感器初始化失败: {e}")
        
        # Mock 模式（开发/演示用）
        if config.get("mock", True):
            from .mock import MockSensorGroup
            mock_group = MockSensorGroup("mock_group")
            for s in mock_group.sensors:
                await self.register(s)
                count += 1
        
        logger.info(f"传感器扫描完成，共注册 {count} 个")
        return count

    async def read(self, sensor_id: str) -> Optional[SensorReading]:
        """读取指定传感器。"""
        sensor = self._sensors.get(sensor_id)
        if not sensor:
            logger.warning(f"传感器 {sensor_id} 未注册")
            return None
        return await sensor.read()

    async def read_all(self) -> Dict[str, List[SensorReading]]:
        """读取所有传感器，按类型分组。"""
        results: Dict[str, List[SensorReading]] = {}
        
        tasks = []
        sensor_list = list(self._sensors.values())
        
        for sensor in sensor_list:
            tasks.append(self._safe_read(sensor))
        
        readings = await asyncio.gather(*tasks)
        
        for reading in readings:
            if reading:
                t = reading.sensor_type
                if t not in results:
                    results[t] = []
                results[t].append(reading)
                
                # 触发回调
                for cb in self._callbacks:
                    try:
                        await cb(reading)
                    except Exception as e:
                        logger.error(f"回调异常: {e}")

        # 触发批量回调（注意力评估等需要整批上下文的消费者）
        for cb in self._batch_callbacks:
            try:
                await cb(results)
            except Exception as e:
                logger.error(f"批量回调异常: {e}")

        return results

    async def _safe_read(self, sensor: Sensor) -> Optional[SensorReading]:
        try:
            return await sensor.read()
        except Exception as e:
            logger.error(f"读取传感器 {sensor.sensor_id} 失败: {e}")
            return None

    def on_reading(self, callback: Callable[[SensorReading], Awaitable[None]]) -> None:
        """注册读数回调（每次读取完成后触发）。"""
        self._callbacks.append(callback)

    def on_batch(self, callback: Callable[[Dict[str, List[SensorReading]]], Awaitable[None]]) -> None:
        """注册批量读数回调（每次完整轮询后触发，收全部分组结果）。"""
        self._batch_callbacks.append(callback)

    def start_polling(self, interval: float = 5.0) -> None:
        """启动持续轮询。"""
        if self._polling_task:
            return
        
        async def _poll():
            while True:
                try:
                    await self.read_all()
                except Exception as e:
                    logger.error(f"轮询异常: {e}")
                await asyncio.sleep(interval)
        
        self._polling_task = asyncio.create_task(_poll())
        logger.info(f"🔄 传感器轮询已启动 (间隔 {interval}s)")

    def stop_polling(self) -> None:
        if self._polling_task:
            self._polling_task.cancel()
            self._polling_task = None

    async def shutdown_all(self) -> None:
        """关闭所有传感器。"""
        self.stop_polling()
        for sensor in self._sensors.values():
            await sensor.shutdown()
        self._sensors.clear()
        logger.info("所有传感器已关闭")

    @property
    def sensor_count(self) -> int:
        return len(self._sensors)

    @property
    def sensor_list(self) -> List[Sensor]:
        return list(self._sensors.values())
