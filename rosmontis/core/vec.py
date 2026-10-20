"""向量工具：编解码、余弦、确定性哈希兜底向量。

numpy 缺失时退化为纯 Python 实现（慢但正确）。
"""

from __future__ import annotations

import hashlib
import math
import struct

try:
    import numpy as np  # type: ignore

    HAS_NUMPY = True
except Exception:  # pragma: no cover
    np = None  # type: ignore
    HAS_NUMPY = False


def pack(vec: list[float] | "np.ndarray") -> bytes:
    """float32 打包成 BLOB。"""
    if HAS_NUMPY:
        arr = np.asarray(vec, dtype=np.float32)
        return arr.tobytes()
    return struct.pack(f"<{len(vec)}f", *[float(x) for x in vec])


def unpack(blob: bytes | None, dim: int) -> list[float]:
    if not blob:
        return [0.0] * dim
    if HAS_NUMPY:
        return np.frombuffer(blob, dtype=np.float32).tolist()
    n = len(blob) // 4
    return list(struct.unpack(f"<{n}f", blob))


def to_array(blob: bytes | None, dim: int):
    """返回 numpy 数组（无 numpy 时返回 list）。"""
    if HAS_NUMPY:
        if not blob:
            return np.zeros(dim, dtype=np.float32)
        return np.frombuffer(blob, dtype=np.float32)
    return unpack(blob, dim)


def l2_normalize(vec: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / n for x in vec]


def cosine(a: bytes | list[float], b: bytes | list[float], dim: int = 1024) -> float:
    """余弦相似度。入参可以是 BLOB 或 list。"""
    va = unpack(a, dim) if isinstance(a, (bytes, bytearray)) else a
    vb = unpack(b, dim) if isinstance(b, (bytes, bytearray)) else b
    if not va or not vb:
        return 0.0
    if HAS_NUMPY:
        x = np.asarray(va, dtype=np.float32)
        y = np.asarray(vb, dtype=np.float32)
        nx = float(np.linalg.norm(x))
        ny = float(np.linalg.norm(y))
        if nx == 0 or ny == 0:
            return 0.0
        return float(x.dot(y) / (nx * ny))
    dot = sum(p * q for p, q in zip(va, vb))
    na = math.sqrt(sum(p * p for p in va))
    nb = math.sqrt(sum(q * q for q in vb))
    return dot / (na * nb) if na and nb else 0.0


def cosine_batch(query: bytes, blobs: list[bytes], dim: int = 1024) -> list[float]:
    """一次算多条。有 numpy 时走矩阵乘法。"""
    if not blobs:
        return []
    if HAS_NUMPY:
        q = np.frombuffer(query, dtype=np.float32) if isinstance(query, (bytes, bytearray)) else np.asarray(query, dtype=np.float32)
        nq = float(np.linalg.norm(q)) or 1.0
        mat = np.frombuffer(b"".join(blobs), dtype=np.float32).reshape(len(blobs), -1)
        norms = np.linalg.norm(mat, axis=1)
        norms[norms == 0] = 1.0
        return (mat @ q / (norms * nq)).astype(float).tolist()
    return [cosine(query, b, dim) for b in blobs]


def deterministic_hash_vector(text: str, dim: int = 1024) -> list[float]:
    """无 embedding 服务时的兜底：字符 n-gram 哈希到固定维度。

    这不是语义向量，但它是**确定性且可复现**的，能让系统在无外部依赖时
    仍然保持"完全相同文本得到相同向量"的基本性质。降级状态会显式上报。
    """
    vec = [0.0] * dim
    t = (text or "").strip().lower()
    if not t:
        return vec
    grams = [t[i : i + 2] for i in range(max(1, len(t) - 1))]
    for g in grams:
        h = int.from_bytes(hashlib.blake2b(g.encode("utf-8"), digest_size=8).digest(), "little")
        idx = h % dim
        sign = 1.0 if (h >> 63) & 1 else -1.0
        vec[idx] += sign
    return l2_normalize(vec)


def quantile_normalize(scores: list[float]) -> list[float]:
    """分位数归一化：把一组分数映射到 [0,1] 的经验分布。

    比 min-max 更抗长尾：即使只有一条特别高分，其余也不会被压成 0。
    这是解决"召回二值性"的关键一步——分数保持连续可分辨。
    """
    n = len(scores)
    if n == 0:
        return []
    if n == 1:
        return [1.0]
    order = sorted(range(n), key=lambda i: scores[i])
    out = [0.0] * n
    for rank, i in enumerate(order):
        out[i] = rank / (n - 1)
    return out


def rrf_fuse(rankings: list[list[str]], k: float = 60.0) -> dict[str, float]:
    """Reciprocal Rank Fusion：多路召回融合，只依赖排名不依赖分数量纲。"""
    fused: dict[str, float] = {}
    for ranks in rankings:
        for pos, doc_id in enumerate(ranks):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + pos + 1)
    return fused
