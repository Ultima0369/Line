"""
示例 3: 双脑模式 — 边缘处理器 + 云端桥接器 完整联动。

展示了完整的流程：
1. 传感器数据流入边缘处理器
2. 注意力预筛选
3. 语义压缩上行
4. 云端深度推理
5. 结果解压返回

用法:
    python -m examples.dual_brain
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from line.sensor.manager import SensorManager
from line.edge.processor import EdgeProcessor
from line.cloud.bridge import CloudBridge


async def main():
    print("""
    ╔═══════════════════════════╗
    ║  双脑模式 · 完整演示      ║
    ║  感官皮层 → 语义压缩 → 新皮层  ║
    ╚═══════════════════════════╝
    """)
    
    # 初始化各模块
    sensor = SensorManager()
    edge = EdgeProcessor({"compression_level": 3})
    cloud = CloudBridge()
    
    await sensor.scan({"mock": True})
    await edge.initialize()
    await cloud.initialize()
    
    # 连接：边缘 → 云端
    async def on_upstream(upstream: dict):
        if not upstream.get("user", "").strip():
            return

        print(f"\n  ┌─ 📤 上行数据包")
        print(f"  │ 传感器: {upstream.get('sensor') or '(无)'}")
        print(f"  │ 注意力: {upstream.get('attention') or '(无)'}")
        print(f"  └─ 发送至云端...")

        response = await cloud.ask(upstream=upstream)
        
        result = await edge.receive_downstream(response)
        print(f"\n  🤖 {result}\n")
    
    edge.on_upstream(on_upstream)
    
    # 启动传感器轮询
    sensor.start_polling(interval=3.0)
    
    print("系统就绪！输入消息开始对话。输入 /exit 退出。\n")
    
    try:
        while True:
            user_input = input("👤 ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("/exit", "exit", "quit"):
                break
            
            sensor_data = await sensor.read_all()
            await edge.process(user_input, sensor_data)
    finally:
        sensor.stop_polling()
        await sensor.shutdown_all()
        await cloud.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
