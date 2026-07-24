"""
日志配置 — 统一日志格式和输出。
"""

import logging
import sys
from typing import Optional
from pathlib import Path


def setup_logger(level: str = "INFO", log_file: Optional[str] = None,
                 console: bool = True):
    """配置全局日志。
    
    用法:
        from line.utils.logger import setup_logger
        setup_logger(level="DEBUG", log_file="line.log")
    """
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    
    # 清空已有处理器
    logger.handlers.clear()
    
    # 格式
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-5s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    
    # 控制台输出
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # 文件输出
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger
