"""命令行入口。

python -m rosmontis.cli <命令>
或者 python run.py <命令>
"""

from __future__ import annotations

import argparse
import ipaddress
import json
import os
import signal
import socket
import sys
import threading
import time

from . import __version__, api_version
from .config import Config
from .core import jobs
from .core import dream as DM
from .core.memory import Memory


def _fmt_ts(ts: float | None) -> str:
    if not ts:
        return "—"
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(ts))


def is_loopback_host(host: str) -> bool:
    """127.0.0.1 / ::1 / localhost / 其它回环地址。"""
    h = (host or "").strip().strip("[]").lower()
    if h in ("localhost", "::1", "0:0:0:0:0:0:0:1"):
        return True
    try:
        return ipaddress.ip_address(h).is_loopback
    except ValueError:
        return False


def bind_security_error(host: str, explicit_token: str | None) -> str | None:
    """非回环地址必须显式给 token，否则拒绝启动（P0-4）。返回错误文案或 None。"""
    if is_loopback_host(host):
        return None
    if (explicit_token or "").strip():
        return None
    return (
        f"拒绝启动：--host {host} 不是回环地址，且未提供 --token 或 ROSMONTIS_TOKEN。\n"
        "本机使用请加 --host 127.0.0.1；确实要对外监听请显式给 token：\n"
        "  rosmontis serve --host 0.0.0.0 --token <你的token>   # 或设置 ROSMONTIS_TOKEN"
    )


def cmd_serve(args) -> int:
    explicit_token = (getattr(args, "token", None) or os.environ.get("ROSMONTIS_TOKEN") or "").strip()
    bind_error = bind_security_error(getattr(args, "host", "127.0.0.1"), explicit_token)
    if bind_error:
        print(bind_error, file=sys.stderr)
        return 2
    try:
        import uvicorn
    except ImportError:
        print("缺少 uvicorn。装一下：pip install uvicorn fastapi")
        return 1

    # 启动即校验 embedding provider：文档承诺"配置未实现的通道即启动失败"，
    # 而 service.app 的 Memory 是懒加载的——不在这里拦，进程会好好活着，
    # 直到第一轮对话才 503，用户以为服务正常。构造一次 Embedder 就够（不联网、不开库）。
    try:
        _cfg = Config()
        from .core.embed import Embedder

        Embedder(
            provider=_cfg.embed_provider,
            model=_cfg.embed_model,
            dim=_cfg.embed_dim,
            ollama_url=_cfg.ollama_url,
            timeout=_cfg.embed_timeout,
            model_id=_cfg.embed_model_id,
            openai_base=_cfg.embed_openai_base,
            openai_key=_cfg.embed_openai_key,
            model_dir=_cfg.embed_model_dir,
        )
    except Exception as e:  # 含 EmbedProviderError（RosmontisError 子类）等启动期错误
        code = getattr(e, "code", "")
        msg = getattr(e, "message", None) or str(e)
        print(f"拒绝启动：{f'[{code}] ' if code else ''}{msg}", file=sys.stderr)
        return 1

    try:
        from .service.app import ensure_local_token, token_path
    except ImportError as e:  # 服务依赖缺失时给明确提示，不抛栈
        print(f"缺少服务依赖（需要 fastapi）：{e}")
        return 1

    if explicit_token:
        token = explicit_token
        os.environ["ROSMONTIS_TOKEN"] = token
        print("认证：使用显式 token（--token / ROSMONTIS_TOKEN，值不打印）", flush=True)
    else:
        try:
            token = ensure_local_token()
        except Exception as e:
            print(f"生成本地 token 失败：{type(e).__name__}: {e}", file=sys.stderr)
            return 2
        os.environ["ROSMONTIS_TOKEN"] = token
        print(f"本地访问 token（首次生成，已写入 {token_path()}）：{token}", flush=True)

    print(f"Rosmontis 启动中 → http://{args.host}:{args.port}", flush=True)
    print(f"接口文档 → http://{args.host}:{args.port}/docs", flush=True)
    uvicorn.run("rosmontis.service.app:app", host=args.host, port=args.port, reload=args.reload, log_level="info")
    return 0


# ── daemon：没人操作时让整理/做梦/结算/遗忘自己发生（M7b）─────
def _daemon_owner(explicit: str | None = None) -> str:
    return explicit or f"{socket.gethostname()}:{os.getpid()}"


def _env_float(key: str, default: float) -> float:
    try:
        raw = os.environ.get(key)
        return float(raw) if raw not in (None, "") else float(default)
    except (TypeError, ValueError):
        return float(default)


def _log_task(job: dict, status: str, duration_ms: float, error: str = "") -> None:
    """一行一任务日志（便于 grep / 审计）。"""
    jid = job.get("id") or job.get("job_id") or "?"
    line = (
        f"[daemon] job={jid} kind={job.get('kind')} target={job.get('target')} "
        f"status={status} ms={duration_ms:.1f}"
    )
    if error:
        line += f" error={error[:300]}"
    print(line, flush=True)


def _record_dream_check(mem, job: dict, result: dict, now: float) -> None:
    """D4b：dream 允许 skipped，但必须留 dream_check 审计（写 meta 表）。"""
    inner = (result or {}).get("result") or {}
    payload = {
        "at": now,
        "target": job.get("target"),
        "skipped": bool(inner.get("skipped", False)),
        "reason": str(inner.get("reason") or ""),
    }
    try:
        from .core import db as DB

        key = f"dream_check:{job.get('target')}:{int(now // 86400)}"
        with DB.tx(mem.cfg.db_path) as conn:
            conn.execute(
                "INSERT INTO meta(key,value) VALUES(?,?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, json.dumps(payload, ensure_ascii=False)),
            )
    except Exception as e:  # 审计失败不挡节拍
        print(f"[daemon] dream_check 写入失败：{type(e).__name__}: {e}", file=sys.stderr, flush=True)


def daemon_heartbeat(
    mem: Memory,
    *,
    now: float | None = None,
    idle_timeout: float | None = None,
    owner: str | None = None,
    limit: int = 20,
    discover: bool = True,
    forget_interval: float | None = None,
    max_session_age: float | None = None,
) -> dict:
    """一次心跳：收尸 → 发现到点任务 → 抢占并执行。

    可注入 `now`，测试不用真等 30 分钟。返回可直接打印/断言的摘要。
    """
    cfg = mem.cfg
    now = time.time() if now is None else float(now)
    owner = _daemon_owner(owner)
    idle_timeout = jobs.idle_timeout_from_env() if idle_timeout is None else float(idle_timeout)
    max_age = (
        _env_float("ROSMONTIS_SESSION_MAX_AGE", 86400.0)
        if max_session_age is None
        else float(max_session_age)
    )
    forget_interval = (
        _env_float("ROSMONTIS_FORGET_INTERVAL", jobs.DEFAULT_FORGET_INTERVAL)
        if forget_interval is None
        else float(forget_interval)
    )

    jobs.ensure_tables(cfg)
    out: dict = {
        "ok": True,
        "now": now,
        "owner": owner,
        "idle_timeout_s": idle_timeout,
        "requeued": {},
        "discovered": [],
        "claimed": 0,
        "done": [],
        "failed": [],
    }
    try:
        out["requeued"] = jobs.requeue_stale(cfg, now=now)
    except Exception as e:
        out["requeue_error"] = f"{type(e).__name__}: {e}"

    if discover:
        try:
            # ① 空闲会话 → settle（bucket=last_active，一个空闲期只结算一次）
            for item in mem.idle_sessions(idle_timeout, now=now, max_age_s=max_age):
                job = jobs.enqueue(
                    cfg, jobs.KIND_SETTLE, item["session_id"],
                    next_run=now, bucket=int(item["last_active"]), now=now,
                )
                if job.get("created"):
                    out["discovered"].append(
                        {"kind": jobs.KIND_SETTLE, "target": item["session_id"],
                         "job_id": job["job_id"], "idle_s": item["idle_s"]}
                    )

            # ② 最近会话的 review / dream 到点检查
            review_win = max(60.0, cfg.policy.review_interval_hours * 3600.0)
            dream_win = max(60.0, cfg.policy.dream_interval_hours * 3600.0)
            recent = [s for s in mem.sessions() if float(s.get("last_active") or 0) >= now - max_age]
            for s in recent:
                sid = s["id"]
                try:
                    if mem.inbox(sid).get("should_review"):
                        job = jobs.enqueue(
                            cfg, jobs.KIND_REVIEW, sid,
                            next_run=now, bucket=int(now // review_win), now=now,
                        )
                        if job.get("created"):
                            out["discovered"].append(
                                {"kind": jobs.KIND_REVIEW, "target": sid, "job_id": job["job_id"]}
                            )
                except Exception as e:
                    print(f"[daemon] review 发现失败 {sid}: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
                try:
                    ok, why = DM.should_dream(cfg, sid, now=now)
                    if ok:
                        job = jobs.enqueue(
                            cfg, jobs.KIND_DREAM, sid,
                            next_run=now, bucket=int(now // dream_win), now=now,
                        )
                        if job.get("created"):
                            out["discovered"].append(
                                {"kind": jobs.KIND_DREAM, "target": sid, "job_id": job["job_id"],
                                 "reason": why}
                            )
                except Exception as e:
                    print(f"[daemon] dream 发现失败 {sid}: {type(e).__name__}: {e}", file=sys.stderr, flush=True)

            # ③ 全局遗忘（按周期窗口去重；auto_forget=False 时跳过）
            if cfg.auto_forget and forget_interval > 0:
                job = jobs.enqueue(
                    cfg, jobs.KIND_FORGET, "*",
                    next_run=now, bucket=int(now // forget_interval), now=now,
                )
                if job.get("created"):
                    out["discovered"].append(
                        {"kind": jobs.KIND_FORGET, "target": "*", "job_id": job["job_id"]}
                    )
        except Exception as e:
            out["discovery_error"] = f"{type(e).__name__}: {e}"

    claimed = jobs.claim_due(cfg, limit=limit, owner=owner, now=now)
    out["claimed"] = len(claimed)
    for job in claimed:
        t0 = time.perf_counter()
        try:
            result = jobs.execute_job(mem, job, now=now)
        except Exception as e:
            jobs.fail(cfg, job["id"], e, now=now)
            ms = (time.perf_counter() - t0) * 1000
            err = f"{type(e).__name__}: {e}"
            out["failed"].append({"job_id": job["id"], "kind": job["kind"], "error": err})
            _log_task(job, "failed", ms, err)
            continue
        jobs.finish(cfg, job["id"], now=now)
        if job.get("kind") == jobs.KIND_DREAM:
            _record_dream_check(mem, job, result, now)
        ms = (time.perf_counter() - t0) * 1000
        brief = jobs.result_brief(result)
        brief["job_id"] = job["id"]
        out["done"].append(brief)
        _log_task(job, "done", ms)
    return out


def run_daemon(
    mem: Memory,
    *,
    once: bool = False,
    interval: float = 60.0,
    idle_timeout: float | None = None,
    owner: str | None = None,
    limit: int = 20,
    stop_event: threading.Event | None = None,
) -> dict:
    """单进程循环。`once=True` 只跑一拍；否则按 interval 心跳直到 stop_event。"""
    stop = stop_event or threading.Event()
    last: dict = {"ok": True, "iterations": 0, "done": [], "failed": []}
    while not stop.is_set():
        try:
            last = daemon_heartbeat(
                mem, idle_timeout=idle_timeout, owner=owner, limit=limit
            )
        except Exception as e:  # 单拍失败不能让 daemon 死
            last = {"ok": False, "error": f"{type(e).__name__}: {e}", "done": [], "failed": []}
            print(f"[daemon] heartbeat failed: {last['error']}", file=sys.stderr, flush=True)
        last["iterations"] = int(last.get("iterations", 0)) + 1
        if once:
            return last
        stop.wait(max(0.05, float(interval)))
    last["stopped"] = True
    return last


def install_stop_handlers(stop_event: threading.Event) -> None:
    """SIGINT/SIGTERM 优雅收尾；Windows 额外接管 SIGBREAK（CTRL_BREAK_EVENT）。"""

    def _handler(signum, _frame):
        print(f"[daemon] signal={signum}，当前任务执行完后退出", flush=True)
        stop_event.set()

    for name in ("SIGINT", "SIGTERM", "SIGBREAK", "SIGHUP"):
        sig = getattr(signal, name, None)
        if sig is None:
            continue
        try:
            signal.signal(sig, _handler)
        except (ValueError, OSError):
            continue


def cmd_daemon(args) -> int:
    cfg = Config()
    mem = Memory(cfg)
    idle_timeout = args.idle_timeout if args.idle_timeout is not None else jobs.idle_timeout_from_env()
    owner = _daemon_owner(args.owner)
    stop = threading.Event()
    install_stop_handlers(stop)
    print(
        f"[daemon] start owner={owner} db={cfg.db_path} "
        f"interval={args.interval}s idle_timeout={idle_timeout}s once={bool(args.once)}",
        flush=True,
    )
    try:
        summary = run_daemon(
            mem,
            once=bool(args.once),
            interval=float(args.interval),
            idle_timeout=idle_timeout,
            owner=owner,
            limit=int(args.limit),
            stop_event=stop,
        )
    except KeyboardInterrupt:  # pragma: no cover - 交互中断兜底
        stop.set()
        summary = {"ok": True, "stopped": True, "done": [], "failed": []}
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str), flush=True)
    if not args.once:
        print("[daemon] stopped", flush=True)
    return 0 if not summary.get("failed") else 1



def cmd_ingest(args) -> int:
    m = Memory(Config())
    r = m.ingest(args.text, session_id=args.session, role=args.role)
    print(json.dumps(r, ensure_ascii=False, indent=2))
    return 0


def cmd_recall(args) -> int:
    m = Memory(Config())
    r = m.recall(args.query, session_id=args.session, budget=args.budget)
    if args.raw:
        print(json.dumps(r, ensure_ascii=False, indent=2))
        return 0
    print(f"\n查询：{r['query']}")
    print(f"耗时 {r['latency_ms']}ms ｜ token {r['tokens_used']}/{r['budget']} ｜ 条目 {len(r['items'])}")
    if r["abstained"]:
        print(f"\n[拒答] {r['reason']}")
        return 0
    print("\n" + "-" * 56)
    print(r["prompt"])
    print("-" * 56)
    print("\n打分明细：")
    for i in r["items"]:
        print(f"  {i['score']:.3f}  L{i['level']}  {i['tokens']:>3}t  {i['kind']:<8} {i['text'][:60]}")
    if r["excluded"]:
        print("\n被排除：")
        for e in r["excluded"][:6]:
            print(f"  - {e['kind']} {e['id'][:14]} … {e['why']}")
    return 0


def cmd_stats(args) -> int:
    m = Memory(Config())
    print(json.dumps(m.stats(), ensure_ascii=False, indent=2))
    return 0


def cmd_models(args) -> int:
    """嵌入模型资产管理。模型是独立可下载资产，不进 wheel。"""
    from . import models as M

    cmd = getattr(args, "models_cmd", "list")
    name = getattr(args, "name", M.DEFAULT_MODEL)
    root = getattr(args, "dir", None)

    if cmd == "list":
        print(f"模型目录：{M.models_root(root)}")
        for r in M.describe():
            flag = "已安装" if r["installed"] else "未安装"
            note = f"  {r['note']}" if r.get("note") else ""
            print(f"  {r['name']:<22} {r['dim']:>4}d  ~{r['approx_mb']}MB  {flag}{note}")
        print(f"默认：{M.DEFAULT_MODEL}    安装：rosmontis models install {M.DEFAULT_MODEL}")
        return 0

    if cmd == "install":
        def _progress(got: int, total: int) -> None:
            if total:
                print(f"\r  下载中 {got / 1048576:.1f}/{total / 1048576:.1f} MB", end="")
            else:
                print(f"\r  下载中 {got / 1048576:.1f} MB", end="")

        try:
            M.install(
                name,
                root=root,
                mirror=getattr(args, "mirror", None),
                force=getattr(args, "force", False),
                progress=_progress,
            )
        except M.ModelError as e:
            print(f"\n安装失败：{e}")
            return 1
        print(f"\n已安装 {name} → {M.model_dir(name, root)}")
        return 0

    if cmd == "where":
        try:
            p = M.resolve(name, root)
        except M.ModelError as e:
            print(str(e))
            return 1
        print(f"模型：{p['name']}  维度：{p['dim']}")
        print(f"目录：{p['dir']}")
        print(f"onnx：{p['onnx']}")
        print(f"tokenizer：{p['tokenizer']}")
        return 0

    if cmd == "verify":
        ok, problems = M.verify(name, root)
        if ok:
            print(f"{name} 校验通过")
            return 0
        print(f"{name} 校验失败：" + "；".join(problems))
        return 1

    if cmd == "remove":
        print(f"已删除 {name}" if M.remove(name, root) else f"{name} 未安装")
        return 0

    print(f"未知子命令：{cmd}")
    return 2


def cmd_backup(args) -> int:
    """备份记忆库：走 SQLite 在线备份 API 并自检，打印行数与 sha256。

    为什么不是「checkpoint 后复制文件」：并发读者在场时 checkpoint 会 busy，
    WAL 里没合并的帧进不了主库文件，复制出来的 .bak 缺数据却报成功。
    恢复：停掉服务/daemon，把备份复制回目标路径（并删掉同目录的 -wal / -shm）即可。
    """
    from .core import db as DB

    db_path = args.db or os.environ.get("ROSMONTIS_DB") or "rosmontis.db"
    if not os.path.exists(db_path):
        print(f"库不存在：{db_path}")
        return 1
    size_mb = os.path.getsize(db_path) / 1048576
    try:
        info = DB.backup_db(db_path, args.out)
    except DB.BackupError as e:
        print(f"备份失败：{e}")
        return 1
    rows = ", ".join(f"{t}={n}" for t, n in info["tables"].items())
    print(f"已备份 {db_path}（{size_mb:.1f} MB）→ {info['path']}")
    print(f"行数：{rows}")
    print(f"sha256: {info['sha256']}")
    return 0


def cmd_audit(args) -> int:
    m = Memory(Config())
    d = m.audit(args.session)
    h = d["health"]
    print(f"\n健康度 {h['score']}/100（{h['label']}）  高危 {h['high']} 中危 {h['medium']}\n")
    print("规模：", json.dumps(d["counts"], ensure_ascii=False))
    if not d["issues"]:
        print("\n没有发现问题。")
        return 0
    print("\n问题清单：")
    for i in d["issues"]:
        print(f"  [{i['severity']:>6}] {i['type']}: {i['detail']}")
        print(f"           → {i['hint']}")
    return 0


def cmd_forget(args) -> int:
    m = Memory(Config())
    r = m.forget(args.session)
    print(json.dumps(r, ensure_ascii=False, indent=2))
    return 0


def cmd_entities(args) -> int:
    m = Memory(Config())
    rows = m.entities(args.session, args.type, args.limit)
    if not rows:
        print("没有实体。")
        return 0
    print(f"{'名字':<16}{'类型':<10}{'事实':<6}{'提及':<6}最近出现")
    for r in rows:
        print(f"{r['name'][:15]:<16}{r['type']:<10}{r['fact_count']:<6}{r['mentions']:<6}{_fmt_ts(r['last_seen'])}")
    return 0


def cmd_state(args) -> int:
    m = Memory(Config())
    d = m.state_snapshot(args.scope, args.as_of)
    print(f"\n{d['scope']} 的状态（截至 {_fmt_ts(d['as_of'])}）")
    if not d["slots"]:
        print("  没有记录")
    for s in d["slots"]:
        print(f"  {s['slot']:<14} = {s['value']}   （自 {_fmt_ts(s['since'])}）")
    return 0


def cmd_timeline(args) -> int:
    m = Memory(Config())
    d = m.entity_detail(args.entity_id)
    if not d:
        print("实体不存在")
        return 1
    print(f"\n{d['entity']['name']} 的事实时间线")
    for f in reversed(d["facts"]):
        mark = "✓" if f["status"] == "active" else "×"
        print(f"  {mark} [{_fmt_ts(f['valid_from'])}] {f['predicate']} {f['object']}")
        if f["invalid_at"]:
            print(f"      于 {_fmt_ts(f['invalid_at'])} 失效")
    return 0


def cmd_demo(args) -> int:
    from .demo_runner import run_demo

    return run_demo(args)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="rosmontis", description="Rosmontis — 长期记忆系统")
    p.add_argument("--db", default=None, help="数据库路径")
    p.add_argument(
        "--version", "-V",
        action="store_true",
        help="显示版本后退出",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("serve", help="启动 HTTP 服务")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8770)
    s.add_argument("--reload", action="store_true")
    s.add_argument(
        "--token", default=None,
        help="显式访问 token（非回环 --host 必须给；也可用 ROSMONTIS_TOKEN）",
    )
    s.set_defaults(func=cmd_serve)

    s = sub.add_parser(
        "daemon",
        help="后台守护：自动整理/做梦/结算/遗忘（--once 只跑一拍，供测试/验收）",
    )
    s.add_argument("--once", action="store_true", help="只跑一拍后退出（测试/验收用）")
    s.add_argument("--interval", type=float, default=60.0, help="心跳间隔秒数（默认 60）")
    s.add_argument("--idle-timeout", type=float, default=None,
                   help="空闲多少秒触发 settle（默认 ROSMONTIS_IDLE_TIMEOUT 或 1800）")
    s.add_argument("--limit", type=int, default=20, help="单拍最多执行多少 job")
    s.add_argument("--owner", default=None, help="租约持有者标识（默认 主机名:pid）")
    # 兼容 `rosmontis daemon --db X`（全局 --db 也必须可用；SUPPRESS 避免覆盖全局值）
    s.add_argument("--db", default=argparse.SUPPRESS, help="数据库路径（等价于全局 --db）")
    s.set_defaults(func=cmd_daemon)

    s = sub.add_parser("ingest", help="写入一句话")
    s.add_argument("text")
    s.add_argument("--session", default="default")
    s.add_argument("--role", default="user")
    s.set_defaults(func=cmd_ingest)

    s = sub.add_parser("recall", help="检索")
    s.add_argument("query")
    s.add_argument("--session", default=None)
    s.add_argument("--budget", type=int, default=None)
    s.add_argument("--raw", action="store_true", help="输出完整 JSON")
    s.set_defaults(func=cmd_recall)

    s = sub.add_parser("stats", help="库统计")
    s.set_defaults(func=cmd_stats)

    s = sub.add_parser("backup", help="备份记忆库（先 checkpoint WAL 再复制，运行中也安全）")
    s.add_argument("--db", default=None, help="库文件路径（默认 env ROSMONTIS_DB 或 rosmontis.db）")
    s.add_argument("--out", default=None, help="输出文件（默认 <库文件名>.<时间戳>.bak）")
    s.set_defaults(func=cmd_backup)

    s = sub.add_parser("models", help="嵌入模型资产：下载 / 校验 / 位置 / 删除")
    ms = s.add_subparsers(dest="models_cmd", required=True)

    m = ms.add_parser("list", help="列出可用模型与安装状态")
    m.add_argument("--dir", default=None, help="模型目录（默认 ROSMONTIS_MODEL_DIR 或 ~/.rosmontis/models）")
    m.set_defaults(func=cmd_models)

    m = ms.add_parser("install", help="下载并校验一个模型（支持断点续传与镜像）")
    m.add_argument("name", nargs="?", default="bge-small-zh-v1.5")
    m.add_argument("--dir", default=None)
    m.add_argument("--mirror", default=None, help="镜像根地址（默认 ROSMONTIS_MODEL_MIRROR 或 huggingface.co）")
    m.add_argument("--force", action="store_true", help="强制重新下载")
    m.set_defaults(func=cmd_models)

    m = ms.add_parser("where", help="打印模型目录与文件路径")
    m.add_argument("name", nargs="?", default="bge-small-zh-v1.5")
    m.add_argument("--dir", default=None)
    m.set_defaults(func=cmd_models)

    m = ms.add_parser("verify", help="校验已安装模型（SHA-256）")
    m.add_argument("name", nargs="?", default="bge-small-zh-v1.5")
    m.add_argument("--dir", default=None)
    m.set_defaults(func=cmd_models)

    m = ms.add_parser("remove", help="删除一个已安装模型")
    m.add_argument("name", nargs="?", default="bge-small-zh-v1.5")
    m.add_argument("--dir", default=None)
    m.set_defaults(func=cmd_models)

    s = sub.add_parser("audit", help="体检")
    s.add_argument("--session", default=None)
    s.set_defaults(func=cmd_audit)

    s = sub.add_parser("forget", help="跑一次遗忘周期")
    s.add_argument("--session", default=None)
    s.set_defaults(func=cmd_forget)

    s = sub.add_parser("entities", help="实体列表")
    s.add_argument("--session", default=None)
    s.add_argument("--type", default=None)
    s.add_argument("--limit", type=int, default=50)
    s.set_defaults(func=cmd_entities)

    s = sub.add_parser("state", help="某人当前状态")
    s.add_argument("scope")
    s.add_argument("--as-of", type=float, default=None)
    s.set_defaults(func=cmd_state)

    s = sub.add_parser("timeline", help="实体事实时间线")
    s.add_argument("entity_id")
    s.set_defaults(func=cmd_timeline)

    s = sub.add_parser("demo", help="跑三个月对话演示")
    s.add_argument("--db", default="demo.db", help="演示专用库，避免污染主库")
    s.add_argument("--reset", action="store_true")
    s.set_defaults(func=cmd_demo, db="demo.db")

    return p


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    # --version 必须在 argparse 的 required subparser 之前拦截，且输出缩进
    # 不被 HelpFormatter 折行（要求恰好两个空格）。
    if "--version" in argv or "-V" in argv:
        print(f"rosmontis {__version__}  (api v{api_version})")
        return 0
    args = build_parser().parse_args(argv)
    if getattr(args, "db", None):
        import os

        os.environ["ROSMONTIS_DB"] = args.db
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
