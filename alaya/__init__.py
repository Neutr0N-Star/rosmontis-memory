"""藏识 Alaya — 长期陪伴 AI 的记忆器官。

对外 API 只暴露三个东西：Memory（进程内库）、ingest/recall 两个动作。
服务形态见 alaya.service.app。
"""

__version__ = "1.0.0"
__all__ = ["Memory", "Config", "__version__"]

from .config import Config


def __getattr__(name):
    if name == "Memory":
        from .core.memory import Memory

        return Memory
    raise AttributeError(name)
