#!/usr/bin/env python
"""藏识 Alaya 启动器。

用法：
    python run.py serve            启动服务（默认 8770）
    python run.py demo             跑三个月对话演示
    python run.py audit            体检
    python run.py recall "问题"    命令行检索

依赖缺失会给出明确提示，不抛栈。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

BANNER = r"""
  ┌─────────────────────────────────────────┐
  │  藏识 Alaya · 长期记忆器官               │
  │  五层记忆 · 三档注入 · 预算约束检索       │
  └─────────────────────────────────────────┘
"""


def check() -> list[str]:
    problems = []
    if sys.version_info < (3, 10):
        problems.append(f"Python 需要 3.10+，当前 {sys.version.split()[0]}")
    try:
        import sqlite3

        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE VIRTUAL TABLE t USING fts5(x)")
        conn.close()
    except Exception:
        problems.append("SQLite 缺少 FTS5 全文索引支持")
    return problems


def main() -> int:
    if len(sys.argv) < 2:
        print(BANNER)
        print("用法:")
        print("  python run.py serve   [--port 8770]   启动 HTTP 服务 + 可视化界面")
        print("  python run.py demo    [--reset]       跑三个月对话演示")
        print("  python run.py audit                    体检")
        print("  python run.py recall  \"问题\"          命令行检索")
        print("  python run.py stats                    统计")
        print()
        print("环境变量（可选）:")
        print("  ALAYA_DB=alaya.db              数据库路径")
        print("  ALAYA_EMBED=ollama|hash        向量来源")
        print("  ALAYA_LLM_HOT=ollama|openai|rule  抽取模型")
        return 0

    problems = check()
    if problems:
        print("✗ 环境检查未通过:")
        for p in problems:
            print("   -", p)
        return 1

    cmd = sys.argv[1]
    argv = sys.argv[1:]

    if cmd == "serve":
        try:
            import uvicorn  # noqa: F401
            import fastapi  # noqa: F401
        except ImportError:
            print("缺少 fastapi / uvicorn，安装：")
            print("  pip install fastapi uvicorn")
            return 1
        print(BANNER)
        from alaya.cli import main as cli_main

        port = 8770
        if "--port" in argv:
            i = argv.index("--port")
            if i + 1 < len(argv):
                port = int(argv[i + 1])
        return cli_main(["serve", "--port", str(port)])

    from alaya.cli import main as cli_main

    return cli_main(argv)


if __name__ == "__main__":
    sys.exit(main())
