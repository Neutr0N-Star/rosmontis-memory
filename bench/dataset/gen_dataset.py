"""生成可复现评测集：200 段合成对话 + 四类题集（时序/多跳/知识更新/偏好）+ 拒答题。

约束
- 固定随机种子 ``SEED = 20261019``；同一份代码重跑得到逐字节相同的文件。
- 纯标准库，不联网、不依赖任何模型（Ollama / 云端都不需要）。
- 金标由生成器在生成时同步写出（程序化标注，不是人工标注）；口径见
  ``bench/report.md``「数据集与标注口径」一节。

产物（均在本目录）
- ``conversations.jsonl``  200 行，每行一段「用户 + 助手」对话，含 gold 抽取金标
- ``sessions.json``        每个会话的时间线、当前状态期望、最终有效事实期望
- ``questions.jsonl``      四类题集 + 拒答题，带金标判定字段
- ``manifest.json``        种子、规模、文件 sha256、生成命令

运行：
    PYTHONIOENCODING=utf-8 python bench/dataset/gen_dataset.py
"""

from __future__ import annotations

import hashlib
import json
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

# ── 固定参数 ──────────────────────────────────────────────────
SEED = 20261019
N_SESSIONS = 5
UNITS_PER_SESSION = 40          # 20 个记忆事件 + 20 段闲聊（含助手回复）
HORIZON_DAYS = 180
DAY = 86400.0
TZ = timezone(timedelta(hours=8))
BASE = datetime(2026, 1, 5, 9, 0, 0, tzinfo=TZ)
BASE_TS = BASE.timestamp()
# 全量时刻：所有对话都已写入之后的查询时刻（知识更新/偏好/多跳/拒答题用它）
NOW_TS = BASE_TS + (HORIZON_DAYS + 1) * DAY

HERE = Path(__file__).resolve().parent
CONV_PATH = HERE / "conversations.jsonl"
SESS_PATH = HERE / "sessions.json"
QUEST_PATH = HERE / "questions.jsonl"
MANIFEST_PATH = HERE / "manifest.json"

# ── 取词池：同一会话内不重复，池间尽量不相交（避免答案串味）──────
USER_NAMES = ["阿禾", "沈禾", "陆川", "林晚", "周野"]
HOMETOWNS = ["杭州", "洛阳", "昆明", "青岛", "合肥"]
ROLES = ["后端开发", "前端开发", "数据分析", "产品经理", "测试开发"]
CITIES = ["杭州", "南京", "武汉", "西安", "苏州", "长沙", "厦门", "重庆", "天津", "宁波"]
COMPANIES = ["字节跳动", "蔚来", "小米", "美团", "网易", "蚂蚁集团", "京东", "快手", "联想", "携程"]
PERSON_NAMES = ["陈默", "许嘉", "唐棠", "赵屿", "林知", "何川", "苏禾", "顾言", "江眠", "温屿"]
PERSON_CITIES = ["成都", "长沙", "厦门", "苏州", "西安", "重庆", "合肥", "昆明", "南昌", "贵阳"]
PERSON_COMPANIES = ["腾讯", "阿里云", "百度", "华为", "滴滴", "哔哩哔哩", "拼多多", "大疆", "中兴", "用友"]
PETS = [("狗", "团子"), ("猫", "年糕"), ("狗", "可乐"), ("猫", "汤圆"), ("狗", "麦麦")]
FOODS_LIKE = ["咖啡", "火锅", "椰子鸡", "烤鱼", "抹茶", "小笼包", "烧烤", "螺蛳粉", "提拉米苏", "生煎"]
FOODS_HATE = ["榴莲", "折耳根", "肥肉", "苦瓜", "皮蛋", "韭菜", "内脏", "百香果", "姜", "薄荷"]
HOBBIES = ["徒步", "骑行", "摄影", "游泳", "爬山", "打羽毛球", "看展", "弹吉他", "露营", "跑步"]
FEARS = ["打雷", "坐飞机", "黑暗", "蟑螂", "牙医", "高处", "蛇", "人群", "体检", "独处"]
PLACES = ["青岛", "敦煌", "桂林", "大理", "西安", "厦门", "长白山", "呼伦贝尔", "婺源", "泉州"]
PLANS = ["辞职去做独立开发", "考一个潜水证", "攒钱买一台相机", "明年去日本旅行", "换一个更大的房子"]

CHITCHAT = [
    "今天有点累，下班回来只想躺着。",
    "周末在家看了部老电影，还不错。",
    "早上地铁特别挤，站了一路。",
    "中午吃了楼下的面，味道一般。",
    "最近睡眠不太好，老是醒。",
    "天气突然变冷了，出门忘了加衣服。",
    "今天开会开到六点，脑子有点木。",
    "养的多肉终于冒新芽了，挺开心的。",
    "晚上去江边走了走，风很舒服。",
    "手机屏幕摔裂了，还在纠结要不要修。",
    "今天试着做了顿饭，盐放多了。",
    "下班路上堵了快一个小时。",
    "新买的耳机到了，音质还行。",
    "这周终于把拖了很久的报告交了。",
    "楼下咖啡店换老板了，味道变了。",
    "今天下雨，伞还忘在公司了。",
    "办公室空调太冷，披着外套坐了一天。",
    "今天状态不太好，什么都不想干。",
    "晚上和同事去吃了顿烧烤。",
    "最近在追一部剧，节奏有点慢。",
    "想请一天假，什么都不安排。",
    "今天路过公园，花都开了。",
]

ASSISTANT_REPLIES = [
    "辛苦了，先歇一会儿吧。",
    "听起来还不错，是什么电影？",
    "那确实挺累的，记得早点休息。",
    "嗯，我记下了。",
    "慢慢来，不用急着一次做完。",
    "那挺好的，给自己一点放松的时间。",
    "听起来今天不太顺，先喝口热水。",
    "好，有需要随时跟我说。",
]

REVERSAL_OBJECT = "香菜"


# ── 小工具 ────────────────────────────────────────────────────
def _pick(rng: random.Random, pool: list[str], n: int) -> list[str]:
    vals = list(pool)
    rng.shuffle(vals)
    return vals[:n]


def _norm(s: str) -> str:
    """与 rosmontis.core.ids.normalize 同口径的轻量实现（生成器不依赖核心代码）。"""
    return "".join(ch for ch in str(s or "").lower() if not ch.isspace())


def _date_zh(ts: float) -> str:
    dt = datetime.fromtimestamp(ts, TZ)
    return f"{dt.year}年{dt.month}月{dt.day}日"


def _t(day: int, hour: int = 10) -> float:
    return BASE_TS + day * DAY + hour * 3600.0


def _entity(name: str, typ: str) -> dict:
    return {"name": name, "type": typ}


def _fact(subject: str, predicate: str, obj: str, polarity: int = 1, memory_type: str = "fact") -> dict:
    return {
        "subject": subject,
        "predicate": predicate,
        "object": obj,
        "polarity": polarity,
        "confidence": 0.9,
        "memory_type": memory_type,
    }


def _state(slot: str, value: str) -> dict:
    return {"scope": "我", "slot": slot, "value": value}


def _gold(entities=None, facts=None, states=None, theme=None) -> dict:
    return {
        "entities": list(entities or []),
        "facts": list(facts or []),
        "state_changes": list(states or []),
        "theme": theme,
    }


def _dialogue(sid: str, day: int, hour: int, user_text: str, assistant_text: str, gold: dict, tag: str) -> dict:
    ts = _t(day, hour)
    return {
        "id": f"{sid}_d{day:03d}_{tag}",
        "session_id": sid,
        "day": day,
        "t_event": round(ts, 3),
        "user_text": user_text,
        "assistant_text": assistant_text,
        "gold": gold,
    }


# ── 单个会话 ──────────────────────────────────────────────────
def build_session(idx: int) -> tuple[dict, list[dict], list[dict]]:
    """返回 (会话摘要, 对话列表, 题目列表)。"""
    rng = random.Random(SEED * 100 + idx)
    sid = f"s{idx + 1:02d}"
    name = USER_NAMES[idx]
    hometown = HOMETOWNS[idx]
    role = ROLES[idx]

    city1, city2 = _pick(rng, CITIES, 2)
    c_job0, c_job1, c_job2 = _pick(rng, COMPANIES, 3)
    p_room, p_gf, p_friend = _pick(rng, PERSON_NAMES, 3)
    room_city = _pick(rng, PERSON_CITIES, 1)[0]
    gf_company, friend_company = _pick(rng, PERSON_COMPANIES, 2)
    pet_species, pet_name = PETS[idx]
    like1, like2 = _pick(rng, FOODS_LIKE, 2)
    hate1, hate2 = _pick(rng, FOODS_HATE, 2)
    hobby = _pick(rng, HOBBIES, 1)[0]
    fear = _pick(rng, FEARS, 1)[0]
    place = _pick(rng, PLACES, 1)[0]
    plan = _pick(rng, PLANS, 1)[0]

    # 事件日（固定表；相邻状态变更之间留出足够窗口供时间旅行）
    event_days = {
        "identity": 0,
        "job0": 8,
        "pref_like1": 20,
        "reversal_old": 30,
        "relationship": 33,
        "city1": 42,
        "pref_hate1": 48,
        "job1": 58,
        "roommate": 63,
        "pet": 68,
        "pref_hobby": 76,
        "friend": 84,
        "travel": 96,
        "pref_fear": 99,
        "city2": 112,
        "job2": 127,
        "pref_like2": 135,
        "reversal_new": 146,
        "plan": 155,
        "pref_hate2": 162,
    }
    city_at_pet = city1  # 68 天落在 city1 区间内

    convs: list[dict] = []
    hours = {k: 9 + (i % 10) for i, k in enumerate(sorted(event_days, key=lambda k: event_days[k]))}
    # 助手回复按固定顺序轮转，避免随机性影响可读性
    reply_i = idx

    def add(tag: str, user_text: str, gold: dict) -> None:
        nonlocal reply_i
        day = event_days[tag]
        reply = ASSISTANT_REPLIES[reply_i % len(ASSISTANT_REPLIES)]
        reply_i += 1
        convs.append(_dialogue(sid, day, hours[tag], user_text, reply, gold, tag))

    add(
        "identity",
        f"我叫{name}，家在{hometown}，现在也住在这儿。",
        _gold(
            [_entity(name, "person"), _entity(hometown, "place")],
            [_fact("我", "叫", name), _fact("我", "住在", hometown)],
            [_state("city", hometown)],
            "初次见面",
        ),
    )
    add(
        "job0",
        f"我在{c_job0}上班，做{role}。",
        _gold([_entity(c_job0, "org")], [_fact("我", "工作在", c_job0)], [_state("job", c_job0)], "工作"),
    )
    add(
        "pref_like1",
        f"我特别喜欢{like1}，一周要吃好几次。",
        _gold([_entity(like1, "thing")], [_fact("我", "喜欢", like1, memory_type="preference")], None, "偏好"),
    )
    add(
        "reversal_old",
        f"我以前不太喜欢{REVERSAL_OBJECT}，闻到就皱眉。",
        _gold(
            [_entity(REVERSAL_OBJECT, "thing")],
            [_fact("我", "喜欢", REVERSAL_OBJECT, polarity=-1, memory_type="preference")],
            None,
            "偏好变化",
        ),
    )
    add(
        "relationship",
        f"我女朋友{p_gf}在{gf_company}上班。",
        _gold(
            [_entity(p_gf, "person"), _entity(gf_company, "org")],
            [_fact("我", "关系是", p_gf, memory_type="fact"), _fact(p_gf, "工作在", gf_company)],
            [_state("relationship", p_gf)],
            "关系",
        ),
    )
    add(
        "city1",
        f"我搬到{city1}了，这边房租便宜点。",
        _gold([_entity(city1, "place")], [_fact("我", "住在", city1)], [_state("city", city1)], "搬家"),
    )
    add(
        "pref_hate1",
        f"我受不了{hate1}的味道，完全不吃。",
        _gold([_entity(hate1, "thing")], [_fact("我", "讨厌", hate1, memory_type="preference")], None, "偏好"),
    )
    add(
        "job1",
        f"我跳槽到{c_job1}了，还是做{role}。",
        _gold([_entity(c_job1, "org")], [_fact("我", "工作在", c_job1)], [_state("job", c_job1)], "工作变动"),
    )
    add(
        "roommate",
        f"我大学室友{p_room}搬去{room_city}了，说那边机会多。",
        _gold(
            [_entity(p_room, "person"), _entity(room_city, "place")],
            # 用「认识」而不是「关系是」：核心把关系是 视为互斥状态谓词，
            # 会和女朋友那条互相作废；室友只是熟人关系，不是状态槽。
            [_fact("我", "认识", p_room), _fact(p_room, "住在", room_city)],
            None,
            "朋友",
        ),
    )
    add(
        "pet",
        f"搬到{city_at_pet}之后我养了一只{pet_species}，叫{pet_name}，特别粘人。",
        _gold(
            [_entity(pet_name, "thing"), _entity(pet_species, "thing")],
            [_fact("我", "有", pet_name), _fact(pet_name, "是", pet_species)],
            None,
            "养宠物",
        ),
    )
    add(
        "pref_hobby",
        f"最近迷上了{hobby}，周末基本都在弄这个。",
        _gold([_entity(hobby, "concept")], [_fact("我", "喜欢", hobby, memory_type="preference")], None, "偏好"),
    )
    add(
        "friend",
        f"我朋友{p_friend}在{friend_company}工作，做的是销售。",
        _gold(
            [_entity(p_friend, "person"), _entity(friend_company, "org")],
            [_fact("我", "认识", p_friend), _fact(p_friend, "工作在", friend_company)],
            None,
            "朋友",
        ),
    )
    add(
        "travel",
        f"上个月我去了{place}，挺喜欢的，想再去一次。",
        _gold([_entity(place, "place")], [_fact("我", "去过", place)], None, "旅行"),
    )
    add(
        "pref_fear",
        f"我特别怕{fear}。",
        _gold([_entity(fear, "thing")], [_fact("我", "害怕", fear, memory_type="emotion")], None, "偏好"),
    )
    add(
        "city2",
        f"上个月我又搬回{city2}了，老朋友都在那边。",
        _gold([_entity(city2, "place")], [_fact("我", "住在", city2)], [_state("city", city2)], "搬家"),
    )
    add(
        "job2",
        f"我换了工作，现在在{c_job2}上班。",
        _gold([_entity(c_job2, "org")], [_fact("我", "工作在", c_job2)], [_state("job", c_job2)], "工作变动"),
    )
    add(
        "pref_like2",
        f"最近发现{like2}也不错，慢慢喜欢上了。",
        _gold([_entity(like2, "thing")], [_fact("我", "喜欢", like2, memory_type="preference")], None, "偏好"),
    )
    add(
        "reversal_new",
        f"现在居然能接受{REVERSAL_OBJECT}了，吃火锅还会主动加。",
        _gold(
            [_entity(REVERSAL_OBJECT, "thing")],
            [_fact("我", "喜欢", REVERSAL_OBJECT, memory_type="preference")],
            None,
            "偏好变化",
        ),
    )
    add(
        "plan",
        f"我最近在考虑{plan}，还没想好。",
        _gold([], [_fact("我", "计划", plan, memory_type="plan")], None, "计划"),
    )
    add(
        "pref_hate2",
        f"我一直不太喜欢{hate2}。",
        _gold(
            [_entity(hate2, "thing")],
            [_fact("我", "喜欢", hate2, polarity=-1, memory_type="preference")],
            None,
            "偏好",
        ),
    )

    # ── 闲聊（无金标事实；用于测抽取精度与拒答）──
    used_days = set(event_days.values())
    free_days = [d for d in range(1, HORIZON_DAYS) if d not in used_days]
    rng.shuffle(free_days)
    chat_days = sorted(free_days[: UNITS_PER_SESSION - len(event_days)])
    for i, d in enumerate(chat_days):
        text = CHITCHAT[(idx * 7 + i) % len(CHITCHAT)]
        reply = ASSISTANT_REPLIES[(idx + i) % len(ASSISTANT_REPLIES)]
        convs.append(_dialogue(sid, d, 9 + (i % 10), text, reply, _gold(), "chat"))

    convs.sort(key=lambda c: (c["day"], c["t_event"]))
    assert len(convs) == UNITS_PER_SESSION, (sid, len(convs))

    # ── 期望状态 / 最终有效事实（模拟 docs/03 §4.1 的 ADD/NOOP/UPDATE/INVALIDATE）──
    expected_active = _simulate_active_facts(convs)
    timeline = {
        "city": [
            {"t": _t(0), "value": hometown},
            {"t": _t(event_days["city1"]), "value": city1},
            {"t": _t(event_days["city2"]), "value": city2},
        ],
        "job": [
            {"t": _t(0), "value": c_job0},
            {"t": _t(event_days["job1"]), "value": c_job1},
            {"t": _t(event_days["job2"]), "value": c_job2},
        ],
        "relationship": [{"t": _t(event_days["relationship"]), "value": p_gf}],
    }
    session = {
        "session_id": sid,
        "user_name": name,
        "hometown": hometown,
        "role": role,
        "timeline": timeline,
        "expected_state_now": {"job": c_job2, "city": city2, "relationship": p_gf},
        "expected_active_facts": expected_active,
        "facts": {
            "job": [c_job0, c_job1, c_job2],
            "city": [hometown, city1, city2],
            "roommate": {"name": p_room, "city": room_city},
            "girlfriend": {"name": p_gf, "company": gf_company},
            "friend": {"name": p_friend, "company": friend_company},
            "pet": {"species": pet_species, "name": pet_name, "city": city_at_pet},
            "travel_place": place,
            "plan": plan,
            "preferences": {
                "like1": like1,
                "hate1": hate1,
                "hobby": hobby,
                "fear": fear,
                "like2": like2,
                "hate2": hate2,
                "reversal_object": REVERSAL_OBJECT,
            },
        },
    }
    questions = _build_questions(session, convs, rng)
    return session, convs, questions


def _simulate_active_facts(convs: list[dict]) -> list[dict]:
    """按 docs/03 §4.1 的消解规则推出「最终应处于 active 的事实」。

    规则：同 (subject,predicate,object,polarity) → NOOP；状态谓词（住在/工作在）
    不同客体 → UPDATE（旧条作废）；同客体相反极性 → INVALIDATE（旧条作废）。
    """
    active: dict[tuple, dict] = {}
    order = sorted(convs, key=lambda c: (c["t_event"], c["day"]))
    for c in order:
        for f in c["gold"]["facts"]:
            key = (_norm(f["subject"]), f["predicate"], _norm(f["object"]))
            pol = int(f.get("polarity", 1))
            cur = active.get(key)
            if cur is not None and int(cur["polarity"]) == pol:
                continue  # NOOP
            if f["predicate"] in ("住在", "工作在"):
                for k in [k for k in active if k[0] == key[0] and k[1] == key[1]]:
                    del active[k]
                active[key] = dict(f)
            elif cur is not None:
                del active[key]
                active[key] = dict(f)
            else:
                active[key] = dict(f)
    out = [
        {
            "subject": v["subject"],
            "predicate": v["predicate"],
            "object": v["object"],
            "polarity": int(v.get("polarity", 1)),
        }
        for v in active.values()
    ]
    out.sort(key=lambda x: (_norm(x["subject"]), x["predicate"], _norm(x["object"]), x["polarity"]))
    return out


# ── 题目 ──────────────────────────────────────────────────────
ABSTAIN_POOL = [
    "量子纠缠为什么会表现出超距作用？",
    "红烧肉要炖多久才会软烂？",
    "Python 的全局解释器锁到底解决什么问题？",
    "今天那场篮球赛最后是谁投进的绝杀？",
    "等额本息和等额本金，房贷利息差多少？",
    "感冒发烧的时候应该吃抗生素吗？",
    "怎么把一张照片的噪点降下去？",
    "大语言模型的注意力机制是怎么计算的？",
    "今年诺贝尔物理学奖给了谁？",
    "怎么判断一只股票的市盈率是高还是低？",
    "高铁座位是怎么定价的？",
    "为什么天空是蓝色的？",
]


def _q(qid, sid, category, query, as_of, match, answer_atoms=None, bridge_atoms=None,
       stale_atoms=None, gold_note="") -> dict:
    return {
        "id": qid,
        "session_id": sid,
        "category": category,
        "query": query,
        "as_of": round(float(as_of), 3),
        "match": match,
        "answer_atoms": list(answer_atoms or []),
        "bridge_atoms": list(bridge_atoms or []),
        "stale_atoms": list(stale_atoms or []),
        "gold_note": gold_note,
    }


def _mid_ts(start_day: int, end_day: int) -> float:
    mid = max(start_day + 1, start_day + (end_day - start_day) // 2)
    return _t(mid, 12)


def _build_questions(session: dict, convs: list[dict], rng: random.Random) -> list[dict]:
    sid = session["session_id"]
    f = session["facts"]
    tl = session["timeline"]
    out: list[dict] = []

    city = tl["city"]
    job = tl["job"]
    bounds = [(0, city[1]["t"]), (city[1]["t"], city[2]["t"]), (city[2]["t"], BASE_TS + (HORIZON_DAYS + 1) * DAY)]
    jobs = [(0, job[1]["t"]), (job[1]["t"], job[2]["t"]), (job[2]["t"], BASE_TS + (HORIZON_DAYS + 1) * DAY)]

    # 时序类 6 题：3 个城市时刻 + 3 个工作时刻，as_of 落在槽有效窗口内
    for i, (a, b) in enumerate(bounds):
        day_a = max(0, int(round((a - BASE_TS) / DAY)))
        day_b = max(day_a + 2, int(round((b - BASE_TS) / DAY)))
        as_of = _mid_ts(day_a, day_b)
        out.append(
            _q(
                f"{sid}_q_temporal_city_{i + 1}",
                sid,
                "temporal",
                f"{_date_zh(as_of)}的时候我住在哪个城市？",
                as_of,
                {"type": "value"},
                [city[i]["value"]],
                gold_note=f"state_slot city 在 {_date_zh(as_of)} 的取值（第 {i + 1} 段城市）",
            )
        )
    for i, (a, b) in enumerate(jobs):
        day_a = max(0, int(round((a - BASE_TS) / DAY)))
        day_b = max(day_a + 2, int(round((b - BASE_TS) / DAY)))
        as_of = _mid_ts(day_a, day_b)
        out.append(
            _q(
                f"{sid}_q_temporal_job_{i + 1}",
                sid,
                "temporal",
                f"{_date_zh(as_of)}的时候我在哪家公司上班？",
                as_of,
                {"type": "value"},
                [job[i]["value"]],
                gold_note=f"state_slot job 在 {_date_zh(as_of)} 的取值（第 {i + 1} 段工作）",
            )
        )

    # 知识更新类 4 题：当前值必须压过旧值，或极性已翻转
    out.append(
        _q(
            f"{sid}_q_update_job",
            sid,
            "update",
            "我想确认下，我现在在哪个公司上班？",
            NOW_TS,
            {"type": "value"},
            [f["job"][2]],
            stale_atoms=[f["job"][0], f["job"][1]],
            gold_note="最新 job 槽；旧公司（c_job0/c_job1）不得排在前面",
        )
    )
    out.append(
        _q(
            f"{sid}_q_update_city",
            sid,
            "update",
            "我想确认下，我现在住在哪个城市？",
            NOW_TS,
            {"type": "value"},
            [f["city"][2]],
            stale_atoms=[f["city"][0], f["city"][1]],
            gold_note="最新 city 槽；旧城市不得排在前面",
        )
    )
    out.append(
        _q(
            f"{sid}_q_update_latest_job_change",
            sid,
            "update",
            "回头想一下，我最近一次换工作去了哪家公司？",
            NOW_TS,
            {"type": "value"},
            [f["job"][2]],
            stale_atoms=[f["job"][0], f["job"][1]],
            gold_note="最近一次换工作的结果；旧公司不得排在前面",
        )
    )
    out.append(
        _q(
            f"{sid}_q_update_reversal",
            sid,
            "update",
            "我想确认下，我现在还喜欢香菜吗？",
            NOW_TS,
            {
                "type": "fact",
                "predicate": "喜欢",
                "object": REVERSAL_OBJECT,
                "polarity": 1,
                "stale_polarity": -1,
            },
            gold_note="偏好极性翻转：最新为 +1，旧的 -1 不得排在前面",
        )
    )

    # 偏好类 6 题：稳定偏好，逐条问「喜欢/讨厌/害怕」
    pref_specs = [
        ("喜欢", f["preferences"]["like1"]),
        ("讨厌", f["preferences"]["hate1"]),
        ("喜欢", f["preferences"]["hobby"]),
        ("害怕", f["preferences"]["fear"]),
        ("喜欢", f["preferences"]["like2"]),
        ("喜欢", f["preferences"]["hate2"]),
    ]
    for i, (pred, obj) in enumerate(pref_specs):
        polarity = 1
        if pred == "喜欢" and obj == f["preferences"]["hate2"]:
            polarity = -1
        out.append(
            _q(
                f"{sid}_q_preference_{i + 1}",
                sid,
                "preference",
                f"我{pred}{obj}吗？",
                NOW_TS,
                {"type": "fact", "predicate": pred, "object": obj, "polarity": polarity},
                gold_note=f"偏好事实：{pred} {obj}（polarity={polarity}）",
            )
        )

    # 多跳类 5 题：桥接实体 + 答案值；两条事实型题目要求两个值同时出现
    rm = f["roommate"]
    gf = f["girlfriend"]
    fr = f["friend"]
    pet = f["pet"]
    out.append(
        _q(
            f"{sid}_q_multihop_roommate_city",
            sid,
            "multi_hop",
            "我大学室友现在住在哪个城市？",
            NOW_TS,
            {"type": "multi_hop"},
            [rm["city"]],
            [rm["name"]],
            gold_note="我 —关系是→ 室友 —住在→ 城市（桥接实体+答案值都在 top-10 才算命中）",
        )
    )
    out.append(
        _q(
            f"{sid}_q_multihop_girlfriend_company",
            sid,
            "multi_hop",
            "我女朋友在哪家公司上班？",
            NOW_TS,
            {"type": "multi_hop"},
            [gf["company"]],
            [gf["name"]],
            gold_note="我 —关系是→ 女友 —工作在→ 公司",
        )
    )
    out.append(
        _q(
            f"{sid}_q_multihop_friend_company",
            sid,
            "multi_hop",
            "我朋友在哪家公司工作？",
            NOW_TS,
            {"type": "multi_hop"},
            [fr["company"]],
            [fr["name"]],
            gold_note="我 —认识→ 朋友 —工作在→ 公司",
        )
    )
    out.append(
        _q(
            f"{sid}_q_multihop_pet_name",
            sid,
            "multi_hop",
            f"我在{pet['city']}的时候养的那只{pet['species']}叫什么名字？",
            NOW_TS,
            {"type": "multi_hop"},
            [pet["name"]],
            [pet["species"], pet["city"]],
            gold_note="地点+物种桥接 → 宠物名",
        )
    )
    out.append(
        _q(
            f"{sid}_q_multihop_two_facts",
            sid,
            "multi_hop",
            "我大学室友搬去的城市，和我上次旅行去的城市，是同一个吗？",
            NOW_TS,
            {"type": "multi_hop"},
            [rm["city"], f["travel_place"]],
            gold_note="两条独立事实（室友城市 + 旅行城市）同时出现在 top-10 才算可作答",
        )
    )

    # 拒答类 6 题：与任何会话内容无关；正确答案是不召回
    picked = _pick(rng, ABSTAIN_POOL, 6)
    for i, text in enumerate(picked):
        out.append(
            _q(
                f"{sid}_q_abstain_{i + 1}",
                sid,
                "abstain",
                text,
                NOW_TS,
                {"type": "abstain"},
                gold_note="应拒答：题面与会话记忆无关",
            )
        )
    return out


# ── 落盘 ──────────────────────────────────────────────────────
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _dump_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")


def _dump_json(path: Path, obj) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(obj, fh, ensure_ascii=False, sort_keys=True, indent=2)
        fh.write("\n")


def generate() -> dict:
    """生成全部产物，返回 manifest。纯函数式：不读外部状态。"""
    sessions: dict[str, dict] = {}
    conversations: list[dict] = []
    questions: list[dict] = []
    for idx in range(N_SESSIONS):
        sess, convs, qs = build_session(idx)
        sessions[sess["session_id"]] = sess
        conversations.extend(convs)
        questions.extend(qs)
    conversations.sort(key=lambda c: (c["session_id"], c["t_event"], c["id"]))
    questions.sort(key=lambda q: (q["session_id"], q["category"], q["id"]))

    by_cat: dict[str, int] = {}
    for q in questions:
        by_cat[q["category"]] = by_cat.get(q["category"], 0) + 1

    _dump_jsonl(CONV_PATH, conversations)
    _dump_json(SESS_PATH, sessions)
    _dump_jsonl(QUEST_PATH, questions)

    manifest = {
        "name": "rosmontis-bench-synthetic-v1",
        "version": 1,
        "seed": SEED,
        "generator": "bench/dataset/gen_dataset.py",
        "sessions": N_SESSIONS,
        "dialogues": len(conversations),
        "turns": len(conversations) * 2,
        "user_turns": len(conversations),
        "assistant_turns": len(conversations),
        "questions": len(questions),
        "questions_by_category": dict(sorted(by_cat.items())),
        "horizon_days": HORIZON_DAYS,
        "base_ts": round(BASE_TS, 3),
        "now_ts": round(NOW_TS, 3),
        "files": {
            CONV_PATH.name: _sha256(CONV_PATH),
            SESS_PATH.name: _sha256(SESS_PATH),
            QUEST_PATH.name: _sha256(QUEST_PATH),
        },
        "commands": {
            "generate": "PYTHONIOENCODING=utf-8 python bench/dataset/gen_dataset.py",
            "eval": "PYTHONIOENCODING=utf-8 python bench/run_eval.py",
            "pareto": "PYTHONIOENCODING=utf-8 python bench/pareto.py",
        },
        "labels": "程序化金标：生成时由模板同步写出，非人工标注；抽取/检索判定口径见 bench/report.md",
    }
    _dump_json(MANIFEST_PATH, manifest)
    return manifest


def main() -> int:
    m = generate()
    print("=" * 64)
    print("rosmontis-bench 合成评测集已生成")
    print("=" * 64)
    print(f"种子           : {m['seed']}")
    print(f"会话 / 对话    : {m['sessions']} / {m['dialogues']} 段（{m['turns']} turn）")
    print(f"题目           : {m['questions']} 道")
    for cat, n in m["questions_by_category"].items():
        print(f"  - {cat:<12}: {n}")
    print(f"时间跨度       : {m['horizon_days']} 天（2026-01-05 起）")
    print(f"文件           : {', '.join(m['files'])}")
    print(f"manifest       : {MANIFEST_PATH.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

