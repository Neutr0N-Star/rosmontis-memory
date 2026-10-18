"""给模型的工具集。

五个，少而够用。工具太多她会变成一台到处乱记的机器。

每个 dispatch 的返回都必须是**模型能读懂的结果**——包括失败原因。
静默失败会让模型重复犯同一个错。
"""

from __future__ import annotations

import logging

log = logging.getLogger("alaya.tools")

# ── JSON Schema（OpenAI function calling 格式，别的协议也能直接映射）──
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "alaya_remember",
            "description": (
                "把一件事留给以后的自己。不是记流水账——只记你觉得将来还用得上的。"
                "如果只有『发生了什么』，它会进速记层，之后可能被丢弃；"
                "如果你还写了『我注意到什么』或『下次该怎么做』，它才会成为真正的笔记。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "what": {"type": "string", "description": "发生了什么，客观陈述"},
                    "noticed": {
                        "type": "string",
                        "description": "你注意到什么（别人可能没注意的细节）。可空。",
                    },
                    "means": {"type": "string", "description": "这对你来说意味着什么。可空。"},
                    "next": {
                        "type": "string",
                        "description": "下次遇到类似情况你该怎么做。可空，但这一项最有价值。",
                    },
                    "topic": {"type": "string", "description": "主题标签，用来归组"},
                },
                "required": ["what"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "alaya_recall",
            "description": "主动查一件事。你已经拿到的上下文里没有再查，别重复查同一个问题。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "你要找什么"},
                    "budget": {"type": "integer", "description": "最多花多少 token，默认 600"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "alaya_promote",
            "description": (
                "把一条笔记变成『我就是这样』。只有当一条判断你反复想到、"
                "并且觉得它已经是你的一部分时才用。这会被系统限制频率。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "note_id": {"type": "string", "description": "哪条笔记"},
                    "section": {
                        "type": "string",
                        "enum": ["identity", "rule", "speech", "trait", "taste", "free"],
                        "description": "放进人格的哪个分区",
                    },
                },
                "required": ["note_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "alaya_revise",
            "description": "订正一条你以前记错的笔记。原文会保留，不是覆盖。",
            "parameters": {
                "type": "object",
                "properties": {
                    "note_id": {"type": "string", "description": "要改哪条"},
                    "new_text": {"type": "string", "description": "新的说法"},
                    "reason": {"type": "string", "description": "为什么改，必填"},
                },
                "required": ["note_id", "new_text", "reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "alaya_skip",
            "description": "某件事你不想再提了。它不会被删除，但会沉下去，不再主动浮现。",
            "parameters": {
                "type": "object",
                "properties": {
                    "item_id": {"type": "string", "description": "哪条记忆或笔记"},
                    "reason": {"type": "string", "description": "为什么不想再提"},
                },
                "required": ["item_id"],
            },
        },
    },
]


TOOL_NAMES = {t["function"]["name"] for t in TOOLS}


def dispatch(mem, name: str, args: dict, session_id: str) -> dict:
    """执行一次工具调用。

    `mem` 是 Memory 实例。所有失败都返回 {ok: false, reason}，
    让模型看到原因并自己调整——不静默吞掉。
    """
    args = args or {}
    try:
        if name == "alaya_remember":
            what = (args.get("what") or "").strip()
            if not what:
                return {"ok": False, "reason": "what 不能为空"}
            r = mem.notebook_write(
                what=what,
                session_id=session_id,
                noticed=args.get("noticed", "") or "",
                means=args.get("means", "") or "",
                next_step=args.get("next", "") or "",
                topic=args.get("topic", "") or "",
            )
            return {
                "ok": True,
                "id": r.get("id"),
                "layer": "笔记" if r.get("has_second_layer") else "速记",
                "note": r.get("note"),
            }

        if name == "alaya_recall":
            q = (args.get("query") or "").strip()
            if not q:
                return {"ok": False, "reason": "query 不能为空"}
            r = mem.recall(q, session_id=session_id, budget=int(args.get("budget") or 600))
            if r.get("abstained"):
                return {"ok": True, "found": False, "reason": r.get("reason")}
            return {
                "ok": True,
                "found": True,
                "prompt": r.get("prompt", ""),
                "count": len(r.get("items", [])),
            }

        if name == "alaya_promote":
            nid = (args.get("note_id") or "").strip()
            if not nid:
                return {"ok": False, "reason": "note_id 不能为空"}
            r = mem.notebook_promote(
                nid, target="persona", section=args.get("section", "free"), session_id=session_id
            )
            if not r.get("ok"):
                return {"ok": False, "reason": r.get("reason", "升格被拒绝")}
            return {"ok": True, "id": r.get("id"), "text": r.get("text"), "note": r.get("note")}

        if name == "alaya_revise":
            nid = (args.get("note_id") or "").strip()
            new_text = (args.get("new_text") or "").strip()
            if not (nid and new_text):
                return {"ok": False, "reason": "note_id 与 new_text 都必须给"}
            r = mem.notebook_revise(nid, new_text, args.get("reason", ""))
            if not r.get("ok"):
                return {"ok": False, "reason": r.get("reason", "订正失败")}
            return {"ok": True, "new_id": r.get("new_id"), "note": r.get("note")}

        if name == "alaya_skip":
            iid = (args.get("item_id") or "").strip()
            if not iid:
                return {"ok": False, "reason": "item_id 不能为空"}
            import time as _t

            from .core import db as DB

            with DB.tx(mem.cfg.db_path) as conn:
                conn.execute(
                    "UPDATE episode SET muted=1, cooldown_until=? WHERE id=?",
                    (_t.time() + 86400 * 365 * 10, iid),
                )
            return {"ok": True, "note": "不再主动提起了，但仍然存在"}

        return {"ok": False, "reason": f"没有这个工具：{name}"}

    except Exception as e:
        log.warning("tool %s failed: %s", name, e)
        return {"ok": False, "reason": f"执行出错：{type(e).__name__}"}
