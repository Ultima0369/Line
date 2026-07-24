"""
辅助工具函数。
"""

import os
import platform
from typing import Dict, Optional


def get_system_info() -> Dict:
    """获取系统信息摘要。"""
    return {
        "os": platform.system(),
        "os_version": platform.version(),
        "python_version": platform.python_version(),
        "hostname": platform.node(),
        "architecture": platform.machine(),
    }


def check_hardware_capability() -> Dict:
    """检查硬件能力（用于判断能否跑本地模型）。"""
    info = {"can_run_local_model": False, "reason": ""}
    
    try:
        import psutil
        mem = psutil.virtual_memory()
        total_gb = mem.total / (1024**3)
        
        if total_gb < 8:
            info["reason"] = f"内存不足 ({total_gb:.1f}GB < 8GB)"
        else:
            info["can_run_local_model"] = total_gb >= 16
            info["reason"] = f"内存 {total_gb:.1f}GB，{'可' if info['can_run_local_model'] else '勉强可'}运行小型量化模型"
    except ImportError:
        info["reason"] = "未安装 psutil，无法检测"
    
    return info


def format_sensor_for_display(data: Dict) -> str:
    """将传感器数据格式化为可读字符串。"""
    lines = []
    for sensor_type, readings in data.items():
        icons = {
            "temperature": "🌡️",
            "humidity": "💧",
            "pressure": "🌬️",
            "illuminance": "☀️",
            "noise": "🔊",
            "system": "💻",
        }
        icon = icons.get(sensor_type, "📡")
        
        if readings:
            r = readings[0]
            if isinstance(r, dict):
                val = r.get("value", "?")
                unit = r.get("unit", "")
            else:
                val = getattr(r, "value", "?")
                unit = getattr(r, "unit", "")
            lines.append(f"{icon} {sensor_type}: {val}{unit}")
    
    return "  |  ".join(lines)


def truncate(text: str, max_len: int = 100) -> str:
    """截断文本到指定长度。"""
    if len(text) <= max_len:
        return text
    return text[:max_len-3] + "..."
