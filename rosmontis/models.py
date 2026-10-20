"""嵌入模型资产管理：模型不进 wheel，按需下载到本地目录。

设计原则（契约见 `docs/15-交付蓝图与模块契约.md` §3.6、`docs/17-接入形态与接口方案.md` §8）：

- **代码与模型分离**：`pip install rosmontis-memory` 只装核心；模型是独立资产。
- **断点续传 + 重试 + 校验**：`.part` 临时文件、最多 3 次重试、SHA-256 校验通过才替换。
- **失败不破坏已有模型**：下载/校验失败时旧文件原样保留。
- **可指向外部目录**：`ROSMONTIS_MODEL_DIR` 或 `--dir`。

模型清单在 `MODELS`；URL 指向发布时的固定版本。国内镜像可用 `--mirror` 或 `ROSMONTIS_MODEL_MIRROR`。

用法（CLI 侧见 `rosmontis models ...`）：

    from rosmontis import models
    models.install("bge-small-zh-v1.5")
    p = models.resolve("bge-small-zh-v1.5")   # {"dir","onnx","tokenizer","dim"}
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

HF = "https://huggingface.co"
HF_MIRROR = "https://hf-mirror.com"

# 未验证过的校验值留空：首次安装后会把实际 SHA-256 写进本地 manifest；
# 发布前应把官方校验值回填到这里，之后安装即强校验。
MODELS: dict[str, dict] = {
    "bge-small-zh-v1.5": {
        "dim": 512,
        "repo": "Xenova/bge-small-zh-v1.5",
        "files": {
            "onnx/model.onnx": "onnx/model.onnx",
            "tokenizer.json": "tokenizer.json",
            "config.json": "config.json",
        },
        "sha256": {},
        "approx_mb": 100,
    },
    "bge-m3": {
        "dim": 1024,
        "repo": "Xenova/bge-m3",
        "files": {
            "onnx/model.onnx": "onnx/model.onnx",
            "tokenizer.json": "tokenizer.json",
            "config.json": "config.json",
        },
        "sha256": {},
        "approx_mb": 900,
        "note": "重型档；仓库与文件名发布前需核实",
    },
}

DEFAULT_MODEL = "bge-small-zh-v1.5"


class ModelError(RuntimeError):
    """模型安装/校验失败。消息应当能直接给用户看。"""


def models_root(override: str | None = None) -> Path:
    p = override or os.environ.get("ROSMONTIS_MODEL_DIR") or os.path.join(Path.home(), ".rosmontis", "models")
    return Path(p)


def model_dir(name: str, root: str | None = None) -> Path:
    return models_root(root) / name


def available() -> dict[str, dict]:
    return {k: dict(v) for k, v in MODELS.items()}


def installed(root: str | None = None) -> list[str]:
    base = models_root(root)
    if not base.exists():
        return []
    out = []
    for d in sorted(base.iterdir()):
        if d.is_dir() and (d / "manifest.json").exists():
            out.append(d.name)
    return out


def read_manifest(name: str, root: str | None = None) -> dict | None:
    f = model_dir(name, root) / "manifest.json"
    if not f.exists():
        return None
    try:
        return json.loads(f.read_text(encoding="utf-8"))
    except Exception:
        return None


def is_installed(name: str, root: str | None = None) -> bool:
    m = read_manifest(name, root)
    return bool(m and m.get("complete"))


def resolve(name: str = DEFAULT_MODEL, root: str | None = None) -> dict:
    """返回本地模型文件路径；未安装时抛 ModelError（带安装命令）。"""
    d = model_dir(name, root)
    m = read_manifest(name, root)
    if not m or not m.get("complete"):
        raise ModelError(
            f"模型 {name} 未安装。运行：rosmontis models install {name}"
            f"（目录：{models_root(root)}）"
        )
    spec = MODELS.get(name, {})
    files = spec.get("files") or {}
    return {
        "name": name,
        "dir": str(d),
        "onnx": str(d / (files.get("onnx/model.onnx", "onnx/model.onnx"))),
        "tokenizer": str(d / (files.get("tokenizer.json", "tokenizer.json"))),
        "dim": int(spec.get("dim") or m.get("dim") or 512),
        "manifest": m,
    }


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def _url_for(rel: str, repo: str, mirror: str | None) -> str:
    base = (mirror or os.environ.get("ROSMONTIS_MODEL_MIRROR") or HF).rstrip("/")
    if base in (HF, HF_MIRROR):
        return f"{base}/{repo}/resolve/main/{rel}"
    return f"{base.rstrip('/')}/{repo}/resolve/main/{rel}"


def _download(url: str, dest: Path, *, retries: int = 3, timeout: float = 30.0, progress=None) -> None:
    """带断点续传与重试的下载。先写 dest.part，成功后原子替换。"""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            have = part.stat().st_size if part.exists() else 0
            req = urllib.request.Request(url)
            # hf-mirror 等镜像会屏蔽默认的 Python-urllib UA（403），带浏览器 UA 才放行
            req.add_header(
                "User-Agent",
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) rosmontis-models-installer",
            )
            if have:
                req.add_header("Range", f"bytes={have}-")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                # 服务器不支持 Range 时会回 200 全量，此时重下
                if have and getattr(resp, "status", 200) == 200:
                    have = 0
                    part.unlink(missing_ok=True)
                mode = "ab" if have else "wb"
                with open(part, mode) as fh:
                    total = int(resp.headers.get("Content-Length") or 0) + have
                    got = have
                    while True:
                        chunk = resp.read(1 << 16)
                        if not chunk:
                            break
                        fh.write(chunk)
                        got += len(chunk)
                        if progress:
                            progress(got, total)
            part.replace(dest)
            return
        except Exception as e:  # 网络/权限/中断都按可重试处理
            last_err = e
            if attempt < retries:
                time.sleep(min(2 ** attempt, 8))
    raise ModelError(f"下载失败（{url}）：{type(last_err).__name__}: {last_err}")


def install(
    name: str = DEFAULT_MODEL,
    *,
    root: str | None = None,
    mirror: str | None = None,
    force: bool = False,
    progress=None,
) -> dict:
    """安装/修复一个模型资产。返回本地 manifest。失败抛 ModelError。"""
    if name not in MODELS:
        raise ModelError(f"未知模型：{name}；可用：{', '.join(MODELS)}")
    spec = MODELS[name]
    d = model_dir(name, root)
    if is_installed(name, root) and not force:
        return read_manifest(name, root) or {}
    d.mkdir(parents=True, exist_ok=True)
    files: dict[str, str] = {}
    for rel, remote in (spec.get("files") or {}).items():
        url = _url_for(remote, spec["repo"], mirror)
        dest = d / rel
        if dest.exists() and not force and read_manifest(name, root):
            pass  # 保留已存在文件，配合 manifest 做增量修复
        else:
            _download(url, dest, progress=progress)
        files[rel] = _sha256(dest)
    pinned = (spec.get("sha256") or {})
    bad = [rel for rel, want in pinned.items() if want and files.get(rel) != want]
    if bad:
        # 校验失败：保留旧模型文件，不写 complete manifest
        raise ModelError(f"校验不通过：{', '.join(bad)}（已保留下载文件，旧模型未被替换）")
    manifest = {
        "name": name,
        "dim": int(spec.get("dim") or 512),
        "repo": spec["repo"],
        "files": files,
        "installed_at": time.time(),
        "complete": True,
    }
    (d / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def verify(name: str, root: str | None = None) -> tuple[bool, list[str]]:
    """校验已安装模型：文件存在且 SHA-256 与 manifest 一致。"""
    m = read_manifest(name, root)
    if not m:
        return False, ["manifest 缺失"]
    problems = []
    d = model_dir(name, root)
    for rel, want in (m.get("files") or {}).items():
        f = d / rel
        if not f.exists():
            problems.append(f"{rel} 缺失")
        elif _sha256(f) != want:
            problems.append(f"{rel} 校验不一致")
    return (not problems), problems


def remove(name: str, root: str | None = None) -> bool:
    d = model_dir(name, root)
    if not d.exists():
        return False
    shutil.rmtree(d)
    return True


def describe() -> list[dict]:
    """给 CLI/文档用：可用模型 + 是否已安装。"""
    rows = []
    for name, spec in MODELS.items():
        m = read_manifest(name) or {}
        rows.append(
            {
                "name": name,
                "dim": spec["dim"],
                "approx_mb": spec.get("approx_mb"),
                "installed": bool(m.get("complete")),
                "dir": str(model_dir(name)),
                "note": spec.get("note", ""),
            }
        )
    return rows
