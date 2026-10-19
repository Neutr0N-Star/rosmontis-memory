"""把 OpenAPI 契约快照导出到仓库根 `openapi.json`（不需要起服务）。

用法：
    pip install -e ".[serve]"
    python scripts/export_openapi.py

为什么提交这份快照：前端/第三方不用先跑起服务就能看到契约；CI 里 diff 它，
接口一改就看得见，避免"文档说一套、实现另一套"。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    try:
        from alaya.service.app import create_app
    except ImportError as e:  # 没装 fastapi 时给出人话提示，不抛栈
        print("导出失败：需要服务依赖。先跑 pip install -e \".[serve]\"")
        print(f"原因：{e}")
        return 1

    app = create_app()
    spec = app.openapi()
    out = ROOT / "openapi.json"
    out.write_text(
        json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n"
    )
    print(f"已写出 {out}（{len(spec.get('paths', {}))} 个路径）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
