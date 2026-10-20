"""藏识 Alaya — 长期陪伴 AI 的记忆器官。

对外三层入口：
- `alaya.open(db_path, config)` → `Handle`（稳定公共 API v1，见 alaya/api.py）
- `Memory`（进程内库；高级/兼容用法）
- `Harness`（我们管对话循环的形态）
服务形态见 alaya.service.app。
"""

__version__ = "1.1.1"
api_version = "1"

__all__ = [
    "Memory",
    "Config",
    "Harness",
    "TOOLS",
    "dispatch",
    "open",
    "Handle",
    "AlayaError",
    "ApiError",
    "__version__",
    "api_version",
]

from .config import Config


def __getattr__(name):
    if name == "Memory":
        from .core.memory import Memory

        return Memory
    if name == "Harness":
        from .harness import Harness

        return Harness
    if name in ("TOOLS", "dispatch"):
        from . import tools

        return getattr(tools, name)
    if name in ("open", "Handle", "AlayaError", "ApiError"):
        from . import api

        return getattr(api, name)
    raise AttributeError(name)

