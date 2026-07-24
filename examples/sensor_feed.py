"""
示例 2: 传感器数据流 — 测试传感器。

显示模拟传感器的实时数据。
不需要 API Key。

用法:
    python -m examples.sensor_feed
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from line.sensor.manager import SensorManager
from line.utils.helpers import format_sensor_for_display


async def main():
    manager = SensorManager()
    
    print("📡 传感器数据流测试\n")
    
    # 注册模拟传感器
    await manager.scan({"mock": True, "system": {"enabled": True}})
    print(f"已注册 {manager.sensor_count} 个传感器\n")
    
    try:
        for i in range(20):
            data = await manager.read_all()
            display = format_sensor_for_display(data)
            print(f"  [{i+1:02d}] {display}")
            await asyncio.sleep(2)
    except KeyboardInterrupt:
        pass
    finally:
        await manager.shutdown_all()
        print("\n已关闭")


if __name__ == "__main__":
    asyncio.run(main())
