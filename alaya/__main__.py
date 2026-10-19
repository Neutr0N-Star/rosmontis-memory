"""`python -m alaya` 与 `alaya` 命令等价。

给不想/不能把 console_scripts 放进 PATH 的场景（虚拟环境、容器、CI）留一个稳定入口。
"""

from __future__ import annotations

import sys

from alaya.cli import main

if __name__ == "__main__":
    sys.exit(main())
