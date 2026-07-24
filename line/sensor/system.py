"""
系统传感器 — 读取本机硬件状态（CPU温度、内存、负载等）。

跨平台实现：
- Windows: powershell / psutil
- Linux: /sys/class/thermal / psutil
- macOS: sysctl / psutil
"""

import asyncio
import logging
import platform
from typing import Optional

from .base import Sensor, SensorReading

logger = logging.getLogger(__name__)


class SystemSensor(Sensor):
    """系统状态传感器，读取 CPU、内存、负载等信息。"""

    def __init__(self, sensor_id: str = "system_01"):
        super().__init__(sensor_id)
        self._os = platform.system().lower()

    @property
    def sensor_type(self) -> str:
        return "system"

    async def read(self) -> Optional[SensorReading]:
        try:
            cpu_temp = await self._get_cpu_temperature()
            cpu_percent = await self._get_cpu_percent()
            memory_percent = await self._get_memory_percent()

            return SensorReading(
                sensor_id=self.sensor_id,
                sensor_type="system",
                value={
                    "cpu_temp": cpu_temp,
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory_percent,
                },
                unit="composite",
                confidence=0.9,
            )
        except Exception as e:
            logger.debug(f"系统传感器读取失败: {e}")
            return None

    async def _get_cpu_temperature(self) -> Optional[float]:
        """获取 CPU 温度（各平台方法不同）。"""
        try:
            if self._os == "linux":
                try:
                    with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
                        return round(int(f.read().strip()) / 1000, 1)
                except (FileNotFoundError, PermissionError):
                    # 有些容器环境没有 thermal zone
                    try:
                        import subprocess
                        result = subprocess.run(
                            ["sensors", "-u"], capture_output=True, text=True, timeout=3
                        )
                        for line in result.stdout.split("\n"):
                            if "temp1_input" in line:
                                return round(float(line.split(":")[1].strip()), 1)
                    except:
                        pass
            elif self._os == "windows":
                try:
                    import subprocess
                    result = subprocess.run(
                        ["powershell", "-Command",
                         "Get-CimInstance -Namespace root/wmi -ClassName MSAcpi_ThermalZoneTemperature "
                         "| Select-Object -ExpandProperty CurrentTemperature"],
                        capture_output=True, text=True, timeout=5
                    )
                    if result.returncode == 0 and result.stdout.strip():
                        temp_kelvin_x10 = float(result.stdout.strip())
                        return round(temp_kelvin_x10 / 10 - 273.15, 1)
                except:
                    pass
            elif self._os == "darwin":
                try:
                    import subprocess
                    result = subprocess.run(
                        ["powermetrics", "--samplers", "smc", "-i", "1", "-n", "1"],
                        capture_output=True, text=True, timeout=5
                    )
                    # 解析输出示例: "CPU Thermal level: 0"
                    for line in result.stdout.split("\n"):
                        if "CPU Thermal" in line:
                            return 0.0  # macOS 很难直接读数值
                except:
                    pass
        except:
            pass
        return None

    async def _get_cpu_percent(self) -> float:
        try:
            import psutil
            return psutil.cpu_percent(interval=0.3)
        except ImportError:
            return 0.0

    async def _get_memory_percent(self) -> float:
        try:
            import psutil
            return psutil.virtual_memory().percent
        except ImportError:
            return 0.0

    async def initialize(self) -> bool:
        logger.info(f"系统传感器初始化 (OS: {self._os})")
        return True
