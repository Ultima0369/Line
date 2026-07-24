#!/usr/bin/env python3
"""
Line 快捷启动脚本。
支持: python run.py [mode]

mode:
  (空)     — 交互式双脑对话
  sensors  — 仅测试传感器
  config   — 查看配置
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

if __name__ == "__main__":
    from main import main
    main()
