"""
示例 1: 基础对话 — 最简单的双脑模式。

直接与云端对话，不接传感器，不压缩。
适合：首次使用，验证 API 连接。

用法:
    python -m examples.basic_chat
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from line.cloud.bridge import CloudBridge


async def main():
    bridge = CloudBridge()
    await bridge.initialize()
    
    print("Line 双脑 · 基础对话模式\n")
    
    while True:
        user_input = input("👤 ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            break
        
        response = await bridge.ask(user_input)
        print(f"\n🤖 {response['content']}\n")


if __name__ == "__main__":
    asyncio.run(main())
