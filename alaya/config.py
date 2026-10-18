"""配置：环境变量优先，缺省值可直接跑。"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _load_dotenv() -> None:
    # 让 .env 里的配置在启动时生效，不用每次手动 export。
    # 只 setdefault：进程里已经设的环境变量优先。
    p = Path(__file__).resolve().parents[1] / ".env"
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


_load_dotenv()


def _env(key: str, default: str) -> str:
    v = os.environ.get(key)
    return v if v not in (None, "") else default


def _envf(key: str, default: float) -> float:
    try:
        return float(_env(key, str(default)))
    except ValueError:
        return default


def _envi(key: str, default: int) -> int:
    try:
        return int(_env(key, str(default)))
    except ValueError:
        return default


def _envb(key: str, default: bool) -> bool:
    return _env(key, "1" if default else "0").lower() in ("1", "true", "yes", "on")


@dataclass
class ScoringWeights:
    """检索打分权重。契约见 docs/03 §4.3，改动走 G2。"""

    rel: float = 0.34
    time: float = 0.14
    salience: float = 0.20
    graph: float = 0.14
    emotion: float = 0.18
    polarity_penalty: float = 0.35
    scope_penalty: float = 0.20
    access_boost: float = 0.08

    theta_high: float = 0.46
    theta_low: float = 0.24
    theta_abstain: float = 0.18


@dataclass
class RecallPolicy:
    """召回策略。参数取自长期陪伴场景的实测口径。"""

    # 三档注入的预算（互相独立，不可挪用）
    persona_budget_chars: int = 7600       # 人格预算：整条取舍，绝不切半句
    resident_budget_chars: int = 1500      # 常驻块：每轮注入
    resident_max_entries: int = 20
    resident_max_tokens: int = 420

    # 冷却：防止反复提同一件事
    cooldown_normal_days: float = 3.0
    cooldown_important_hours: float = 24.0

    # 反近因：陈旧的记忆要偶尔浮现（陪伴场景特有）
    stale_tau_days: float = 14.0
    w_fresh: float = 0.55                  # 刚想起过 → 权重压到 0.55
    w_stale: float = 1.00                  # 很久没想起 → 权重拉满

    # 召回地板
    s_min: float = 0.20                    # 普通记忆的语义地板
    s_min_important: float = 0.14          # 重要记忆的地板更低（更容易被想起）
    important_top_k: int = 2

    # 锦囊节律：收集与整理分离，整理必须到点发生，不能只等会话结束
    inbox_max_items: int = 30          # 积压超过这个数就强制整理
    inbox_max_age_hours: float = 24.0  # 最老一条超过这个年龄也强制整理
    review_interval_hours: float = 12.0  # 两次整理的最小间隔

    # 做梦节律
    dream_interval_hours: float = 6.0
    dream_min_items: int = 5

    # 反射深度：按问题复杂度选检索规模，不必每次都全量扫
    # 取值参考 NeuralMemory 的 chat-heavy preset
    reflex_default: str = "context"        # instant / context / habit / deep



@dataclass
class BudgetPlan:
    """注入预算。**这是约束，不是目标。**

    目标函数 = 记忆保真度（Σ 分数 × 保真度），约束 = 总 token ≤ 预算。
    优化顺序固定为：先保质量，再压表达，最后才砍内容。
    """

    state: float = 0.15
    fact: float = 0.40
    episode: float = 0.35
    theme: float = 0.10

    # 质量优先：先按分数选"要哪些"，再决定"怎么说得省"
    quality_first: bool = True
    # 性能地板：任何预算下都不可牺牲的核心条目数
    floor_facts: int = 6
    floor_states: bool = True  # 状态快照全保

    # 级别保真度：降级表达会丢多少信息
    fidelity_l3: float = 1.00
    fidelity_l2: float = 0.86
    fidelity_l1: float = 0.72
    fidelity_l0: float = 0.46

    # 候选池倍数：按分数取 K = 预算能容纳条数 × 此倍数
    pool_multiplier: float = 2.4

    # 阶段 B（表达降级）的触发点：总 token 超过预算的这个比例才降级
    compress_at: float = 1.0
    # 阶段 C（内容裁剪）只在降级到 L1 仍超预算时发生
    prune_at: float = 1.0

    redundancy_lambda: float = 0.28




@dataclass
class Config:
    # 存储
    db_path: str = field(default_factory=lambda: _env("ALAYA_DB", "alaya.db"))

    # Embedding
    embed_provider: str = field(default_factory=lambda: _env("ALAYA_EMBED", "ollama"))
    embed_model: str = field(default_factory=lambda: _env("ALAYA_EMBED_MODEL", "bge-m3:latest"))
    embed_dim: int = field(default_factory=lambda: _envi("ALAYA_EMBED_DIM", 1024))
    ollama_url: str = field(default_factory=lambda: _env("ALAYA_OLLAMA", "http://127.0.0.1:11434"))
    embed_timeout: float = field(default_factory=lambda: _envf("ALAYA_EMBED_TIMEOUT", 8.0))
    # >0 时按批调用，减少往返
    embed_batch: int = field(default_factory=lambda: _envi("ALAYA_EMBED_BATCH", 8))

    # LLM：hot=每条消息的抽取，deep=冲突消解/重要度等关键判断
    llm_hot: str = field(default_factory=lambda: _env("ALAYA_LLM_HOT", "ollama"))
    llm_deep: str = field(default_factory=lambda: _env("ALAYA_LLM_DEEP", "ollama"))
    llm_hot_model: str = field(default_factory=lambda: _env("ALAYA_LLM_HOT_MODEL", "minicpm5-2b:q8-64k"))
    llm_deep_model: str = field(default_factory=lambda: _env("ALAYA_LLM_DEEP_MODEL", "minicpm5-2b:q8-64k"))
    llm_timeout: float = field(default_factory=lambda: _envf("ALAYA_LLM_TIMEOUT", 45.0))
    llm_temperature: float = field(default_factory=lambda: _envf("ALAYA_LLM_TEMP", 0.1))

    # OpenAI 兼容通道（DeepSeek / 各家 relay 都走这套）
    openai_base: str = field(default_factory=lambda: _env("ALAYA_OPENAI_BASE", ""))
    openai_key: str = field(default_factory=lambda: _env("ALAYA_OPENAI_KEY", ""))
    openai_model: str = field(default_factory=lambda: _env("ALAYA_OPENAI_MODEL", "deepseek-chat"))

    # 检索
    recall_k_dense: int = field(default_factory=lambda: _envi("ALAYA_K_DENSE", 40))
    recall_k_sparse: int = field(default_factory=lambda: _envi("ALAYA_K_SPARSE", 40))
    recall_k_graph: int = field(default_factory=lambda: _envi("ALAYA_K_GRAPH", 30))
    budget_tokens: int = field(default_factory=lambda: _envi("ALAYA_BUDGET", 1400))
    mmr_lambda: float = field(default_factory=lambda: _envf("ALAYA_MMR", 0.72))
    graph_hops: int = field(default_factory=lambda: _envi("ALAYA_HOPS", 2))

    # 遗忘
    forget_archive_r: float = field(default_factory=lambda: _envf("ALAYA_ARCHIVE_R", 0.08))
    forget_fold_r: float = field(default_factory=lambda: _envf("ALAYA_FOLD_R", 0.25))
    forget_protect_salience: float = field(default_factory=lambda: _envf("ALAYA_PROTECT_sal", 0.8))
    fold_min_members: int = field(default_factory=lambda: _envi("ALAYA_FOLD_MIN", 4))
    decay_base_strength: float = field(default_factory=lambda: _envf("ALAYA_DECAY_BASE", 86400.0 * 21))

    # 行为
    weights: ScoringWeights = field(default_factory=ScoringWeights)
    budget: BudgetPlan = field(default_factory=BudgetPlan)
    policy: RecallPolicy = field(default_factory=RecallPolicy)
    use_llm_extract: bool = field(default_factory=lambda: _envb("ALAYA_LLM_EXTRACT", True))
    use_llm_resolve: bool = field(default_factory=lambda: _envb("ALAYA_LLM_RESOLVE", False))
    auto_forget: bool = field(default_factory=lambda: _envb("ALAYA_AUTO_FORGET", True))
    log_level: str = field(default_factory=lambda: _env("ALAYA_LOG", "INFO"))

    def as_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k not in ("weights", "budget", "policy", "openai_key")}
        d["weights"] = self.weights.__dict__
        d["budget"] = self.budget.__dict__
        d["policy"] = self.policy.__dict__
        return d
