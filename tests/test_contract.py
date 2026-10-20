"""契约测试：版本号一致、公共面可导入、稳定签名不漂移。

这些断言防的是"发布物与源码对不上"和"改公开面却没升版本"。
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import rosmontis  # noqa: E402


def _pyproject() -> dict:
    try:
        import tomllib
    except ModuleNotFoundError:  # Python 3.10：标准库没有 tomllib，用 tomli
        import tomli as tomllib
    with open(ROOT / "pyproject.toml", "rb") as fh:
        return tomllib.load(fh)


class TestVersionConsistency(unittest.TestCase):
    def test_pyproject_matches_package(self):
        self.assertEqual(_pyproject()["project"]["version"], rosmontis.__version__)

    def test_openapi_matches_package(self):
        spec = json.loads((ROOT / "openapi.json").read_text(encoding="utf-8"))
        self.assertEqual(spec["info"]["version"], rosmontis.__version__)

    def test_api_version_declared(self):
        self.assertEqual(rosmontis.api_version, "1")


class TestPublicSurface(unittest.TestCase):
    def test_all_exported_names_importable(self):
        missing = [n for n in rosmontis.__all__ if not hasattr(rosmontis, n)]
        self.assertEqual(missing, [])

    def test_stable_handle_methods(self):
        handle_cls = rosmontis.Handle
        for name in ("wake", "context", "turn", "observe", "settle", "settle_now", "close"):
            self.assertTrue(hasattr(handle_cls, name), f"Handle 缺少 {name}")

    def test_tool_names_frozen(self):
        from rosmontis.tools import TOOLS

        names = {t["function"]["name"] for t in TOOLS}
        self.assertEqual(
            names,
            {"rosmontis_remember", "rosmontis_recall", "rosmontis_promote", "rosmontis_revise", "rosmontis_skip"},
        )

    def test_openapi_core_routes_present(self):
        spec = json.loads((ROOT / "openapi.json").read_text(encoding="utf-8"))
        paths = set(spec.get("paths", {}))
        for p in (
            "/v1/turn",
            "/v1/turn/stream",
            "/v1/context",
            "/v1/observe",
            "/v1/settle",
            "/v1/jobs",
            "/v1/state/{scope}",
        ):
            self.assertIn(p, paths)


if __name__ == "__main__":
    unittest.main(verbosity=2)
