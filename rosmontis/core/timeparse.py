"""中文相对时间解析。

解析不出来就返回 None——不猜。猜错的时间比没有时间更有害。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta

DAY = 86400.0

_CN_NUM = {"一": 1, "两": 2, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7,
           "八": 8, "九": 9, "十": 10, "十一": 11, "十二": 12, "半": 0.5}


@dataclass
class TimeSpan:
    start: float
    end: float
    granularity: str  # day / week / month / year / instant
    raw: str

    def contains(self, ts: float) -> bool:
        return self.start <= ts <= self.end


def _num(s: str) -> float | None:
    if s.isdigit():
        return float(s)
    if s in _CN_NUM:
        return float(_CN_NUM[s])
    return None


def parse(text: str, now: float) -> TimeSpan | None:
    """从一句话里找出时间指示。支持绝对日期与相对表达。"""
    if not text:
        return None
    t = text.strip()
    base = datetime.fromtimestamp(now)

    # ── 绝对日期：2024年5月3日 / 5月3日 / 2024-05-03 ──
    m = re.search(r"(\d{4})\s*[年\-/]\s*(\d{1,2})\s*[月\-/]\s*(\d{1,2})", t)
    if m:
        try:
            d = datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            return _day_span(d)
        except ValueError:
            pass
    m = re.search(r"(\d{1,2})\s*月\s*(\d{1,2})", t)
    if m:
        try:
            d = datetime(base.year, int(m.group(1)), int(m.group(2)))
            if d.timestamp() > now + 7 * DAY:  # 未来太远视为去年
                d = d.replace(year=base.year - 1)
            return _day_span(d)
        except ValueError:
            pass

    # ── 相对：N 天/周/月/年 前 ──
    m = re.search(r"([0-9一两二三四五六七八九十半]+)\s*(天|日|周|星期|个?月|年)\s*(?:以)?前", t)
    if m:
        n = _num(m.group(1))
        unit = m.group(2)
        if n is not None:
            if "天" in unit or "日" in unit:
                d = base - timedelta(days=n)
            elif "周" in unit or "星期" in unit:
                d = base - timedelta(weeks=n)
            elif "月" in unit:
                d = base - timedelta(days=30 * n)
            else:
                d = base - timedelta(days=365 * n)
            return _day_span(d)

    # ── 相对：N 天/周/月 后 ──
    m = re.search(r"([0-9一两二三四五六七八九十]+)\s*(天|日|周|个?月|年)\s*后", t)
    if m:
        n = _num(m.group(1))
        if n is not None:
            unit = m.group(2)
            days = n * (1 if ("天" in unit or "日" in unit) else 7 if ("周" in unit) else 30 if ("月" in unit) else 365)
            return _day_span(base + timedelta(days=days))

    # ── 简单词 ──
    if "前天" in t:
        return _day_span(base - timedelta(days=2))
    if "昨天" in t or "昨日" in t:
        return _day_span(base - timedelta(days=1))
    if "今天" in t or "今日" in t:
        return _day_span(base)
    if "明天" in t:
        return _day_span(base + timedelta(days=1))
    if "后天" in t:
        return _day_span(base + timedelta(days=2))

    # ── 上周三 / 这周一 ──
    m = re.search(r"(上|这|本|下)\s*(?:个)?\s*(?:周|星期)\s*([一二三四五六日天1-7])", t)
    if m:
        which, wd = m.group(1), m.group(2)
        idx = {"一": 0, "二": 1, "三": 2, "四": 3, "五": 4, "六": 5, "日": 6, "天": 6}
        target = idx.get(wd, int(wd) - 1 if wd.isdigit() else 0)
        monday = base - timedelta(days=base.weekday())
        if which == "上":
            monday -= timedelta(weeks=1)
        elif which == "下":
            monday += timedelta(weeks=1)
        return _day_span(monday + timedelta(days=target))

    # ── 上周 / 上个月 / 去年 ──
    if re.search(r"(上|这|本|下)\s*(?:个)?\s*(?:周|星期)", t) and not re.search(r"[一二三四五六日天]", t):
        monday = base - timedelta(days=base.weekday())
        if "上" in t:
            monday -= timedelta(weeks=1)
        elif "下" in t:
            monday += timedelta(weeks=1)
        return TimeSpan(monday.timestamp(), (monday + timedelta(days=7)).timestamp() - 1, "week", t)

    if re.search(r"(上|这|本|下)\s*(?:个)?\s*月", t):
        y, mo = base.year, base.month
        if "上" in t:
            mo -= 1
            if mo == 0:
                mo, y = 12, y - 1
        elif "下" in t:
            mo += 1
            if mo == 13:
                mo, y = 1, y + 1
        start = datetime(y, mo, 1)
        end = datetime(y + (mo // 12), (mo % 12) + 1, 1)
        return TimeSpan(start.timestamp(), end.timestamp() - 1, "month", t)

    if re.search(r"(去|今|明|前)年", t):
        y = base.year + (1 if "明" in t else -1 if "去" in t or "前" in t else 0)
        return TimeSpan(datetime(y, 1, 1).timestamp(), datetime(y + 1, 1, 1).timestamp() - 1, "year", t)

    # ── 最近 / 之前（模糊区间）──
    if re.search(r"最近|近期|这段时间|前几天", t):
        return TimeSpan((base - timedelta(days=14)).timestamp(), now, "week", t)

    return None


def _day_span(d: datetime) -> TimeSpan:
    start = datetime(d.year, d.month, d.day)
    return TimeSpan(start.timestamp(), start.timestamp() + DAY - 1, "day", d.strftime("%Y-%m-%d"))


def humanize(ts: float | None, now: float | None = None) -> str:
    if ts is None:
        return "—"
    if now is None:
        now = datetime.now().timestamp()
    delta = now - ts
    if delta < 60:
        return "刚刚"
    if delta < 3600:
        return f"{int(delta // 60)} 分钟前"
    if delta < DAY:
        return f"{int(delta // 3600)} 小时前"
    if delta < 30 * DAY:
        return f"{int(delta // DAY)} 天前"
    if delta < 365 * DAY:
        return f"{int(delta // (30 * DAY))} 个月前"
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
