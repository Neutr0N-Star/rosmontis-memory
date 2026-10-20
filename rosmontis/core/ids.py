"""确定性 ID：同样的输入永远得到同样的 id，保证写入幂等。"""

from __future__ import annotations

import hashlib
import re
import unicodedata

_WS = re.compile(r"\s+")


def _digest(*parts: object) -> str:
    raw = "\x1f".join("" if p is None else str(p) for p in parts)
    return hashlib.blake2b(raw.encode("utf-8"), digest_size=12).hexdigest()


def episode_id(session_id: str, role: str, text: str, t_event: float | int) -> str:
    """事件 id：同一会话、同一角色、同一句话，在 5 分钟窗口内视为同一条。

    窗口不能再小。之前用 `int(t_event)`（1 秒窗口），结果是同一句话隔两秒
    发两次就产生两条记忆——幂等的承诺等于没有。
    也不能无限大：隔一天再说同一句话，该是新的一条。
    """
    bucket = int(float(t_event) // 300)
    return "ep_" + _digest(session_id, role, normalize(text), bucket)


def entity_id(session_id: str, norm_name: str) -> str:
    return "en_" + _digest(session_id, norm_name)


def fact_id(
    session_id: str,
    subject_norm: str,
    predicate: str,
    object_text: str,
    t_valid_from: float | int,
    polarity: int = 1,
) -> str:
    """事实 id。

    毫秒精度 + **极性参与哈希**。少了极性，同一秒内先「喜欢猫」再「不喜欢猫」
    会撞同一个 id，后者被 INSERT OR IGNORE 静默丢弃——逆向事实就消失了。
    """
    return "fa_" + _digest(
        session_id, subject_norm, predicate, normalize(object_text),
        int(float(t_valid_from) * 1000), int(polarity),
    )


def edge_id(session_id: str, kind: str, label: str) -> str:
    return "he_" + _digest(session_id, kind, normalize(label))


def slot_id(session_id: str, scope: str, slot_key: str, since: float | int, value: str = "") -> str:
    """状态槽 id。

    毫秒精度 + 值参与哈希：同一毫秒内连续两次状态变更（比如一句话里说了
    "从北京搬到上海"）也不会撞 id 而被静默忽略。
    """
    return "sl_" + _digest(session_id, scope, slot_key, int(float(since) * 1000), normalize(value))


def normalize(text: str) -> str:
    """统一大小写、全半角、空白，用于去重与匹配。"""
    if not text:
        return ""
    t = unicodedata.normalize("NFKC", text).strip().lower()
    t = _WS.sub(" ", t)
    return t.strip("。！？，、；：\"'（）()[]【】…—-~ ")


if __name__ == "__main__":
    assert episode_id("s", "user", "你好", 1) == episode_id("s", "user", " 你好 ", 1)
    assert episode_id("s", "user", "你好", 1) != episode_id("s", "user", "你好", 2)
    print("ids ok")
