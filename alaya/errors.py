"""统一的对外错误类型。

放在这里而不是 `alaya/api.py`：`alaya/core/*` 需要在故障现场直接抛带
`code` / `retryable` 的错误（例如 embedding provider 不可用、库被锁超时），
而 core 反向 import api 会成环。`alaya.api` 继续原样导出这两个名字，
对调用方来说 `alaya.AlayaError is alaya.api.AlayaError` 仍成立。
"""

from __future__ import annotations


class AlayaError(Exception):
    """公共 API 的统一异常：带 code / message / retryable。"""

    def __init__(self, code: str, message: str, retryable: bool = False):
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable

    def as_dict(self) -> dict:
        return {
            "ok": False,
            "data": None,
            "error": {"code": self.code, "message": self.message, "retryable": self.retryable},
        }


ApiError = AlayaError
