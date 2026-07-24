"""
配置管理器 — 负责加载、验证、合并配置。

加载优先级：
1. config.yaml（用户自定义，含API Key，已加入 .gitignore）
2. config.example.yaml（示例配置，不含真实API Key，可安全提交到GitHub）
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional


DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"
EXAMPLE_CONFIG_PATH = Path(__file__).parent.parent / "config.example.yaml"


class Config:
    """配置管理器（单例模式）。"""

    _instance: Optional["Config"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            # 实例属性而非类变量，避免跨实例/跨测试的状态泄漏
            cls._instance._data: Dict[str, Any] = {}
        return cls._instance

    def load(self, path: Optional[str] = None) -> Dict[str, Any]:
        """加载配置文件。

        先找 config.yaml（用户配置），找不到则用 config.example.yaml。
        """
        if path:
            config_path = Path(path)
        elif DEFAULT_CONFIG_PATH.exists():
            config_path = DEFAULT_CONFIG_PATH
        elif EXAMPLE_CONFIG_PATH.exists():
            config_path = EXAMPLE_CONFIG_PATH
        else:
            raise FileNotFoundError(
                "配置文件不存在。请将 config.example.yaml 复制为 config.yaml 并填写API Key。"
            )

        with open(config_path, "r", encoding="utf-8") as f:
            self._data = yaml.safe_load(f)

        # 环境变量覆盖（API密钥优先从环境变量读取，避免明文写文件）
        self._apply_env_overrides()

        return self._data

    def _apply_env_overrides(self):
        """环境变量覆盖配置项。"""
        env_key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("LINE_API_KEY")
        if env_key and not self._data.get("cloud", {}).get("api_key"):
            self._data.setdefault("cloud", {})["api_key"] = env_key

    def get(self, *keys: str, default: Any = None) -> Any:
        """安全地获取嵌套配置项。
        
        用法:
            config.get("cloud", "api_key")
            config.get("sensor", "dht", "enabled", default=False)
        """
        current = self._data
        for key in keys:
            if isinstance(current, dict):
                current = current.get(key)
                if current is None:
                    return default
            else:
                return default
        return current if current is not None else default

    @property
    def data(self) -> Dict[str, Any]:
        return self._data

    def validate(self) -> list:
        """验证配置完整性，返回缺失项列表。"""
        issues = []
        if not self.get("cloud", "api_key"):
            issues.append(
                "cloud.api_key 未设置。有两种方式：\n"
                "  方式A（推荐）: set DEEPSEEK_API_KEY=sk-你的key\n"
                "  方式B: 将 config.example.yaml 复制为 config.yaml 并填写 api_key"
            )
        if self.get("edge", "mode") == "local_model":
            if not self.get("edge", "local_model", "name"):
                issues.append("edge.local_model.name 未设置。本地模型模式需要指定模型名称。")
        return issues
