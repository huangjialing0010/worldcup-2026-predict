"""
一键调参 + 回测 + 保存结果（向量化加速版）
ELO/泊松搜索从 O(n_iter × n_matches) Python 循环 → numpy 向量化
"""
import json, sys, io, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import numpy as np
from scipy.stats import poisson
from pathlib import Path

ROOT = Path(__file__).parent.parent
OUTPUT_DIR = ROOT / "output"
sys.path.insert(0, str(Path(__file__).parent))

from model_utils import load_matches, load_rankings, actual_result
from models import DixonColes

rng = np.random.default_rng(42)

# ============================================================
# 数据加载
# ============================================================
all_matches = load_matches()
rankings = load_rankings()
print(f"数据集: {len(all_matches)} 场")

n_tune = 800
indices = rng.permutation(len(all_matches))
tune_idx = set(indices[:n_tune])
tune_matches = [all_matches[i] for i in range(len(all_matches)) if i in tune_idx]
val_matches = [all_matches[i] for i in range(len(all_matches)) if i not in tune_idx]
print(f"调参: {len(tune_matches)} 场, 验证: {len(val_matches)} 场")

# ============================================================
# 预提取 numpy 数组（核心优化）
# ============================================================
tune_home_rk = np.array([rankings.get(h, 50) for h, a, _, _ in tune_matches], dtype=np.float64)
tune_away_rk = np.array([rankings.get(a, 50) for h, a, _, _ in tune_matches], dtype=np.float64)
tune_actual  = np.array([actual_result(hs, as_) for _, _, hs, as_ in tune_matches])

val_home_rk = np.array([rankings.get(h, 50) for h, a, _, _ in val_matches], dtype=np.float64)
val_away_rk = np.array([rankings.get(a, 50) for h, a, _, _ in val_matches], dtype=np.float64)
val_actual   = np.array([actual_result(hs, as_) for _, _, hs, as_ in val_matches])

MAX_G = 10
_g = np.arange(MAX_G + 1, dtype=np.float64)

# ============================================================
# 向量化评估函数
# ============================================================

def _vec_poisson_proba(lh, la):
    """向量化 H/D/A 概率 (n,) → (n,3)，用 CDF 避免 3D 数组"""
    pmf_h = poisson.pmf(_g[np.newaxis, :], lh[:, np.newaxis])   # (n, g+1)
    pmf_a = poisson.pmf(_g[np.newaxis, :], la[:, np.newaxis])   # (n, g+1)
    cdf_a = poisson.cdf(_g[np.newaxis, :] - 1, la[:, np.newaxis])  # P(j < i)
    cdf_a[:, 0] = 0.0  # cdf(-1) = 0

    p_h = np.sum(pmf_h * cdf_a, axis=1)
    p_d = np.sum(pmf_h * pmf_a, axis=1)
    p_a = 1.0 - p_h - p_d
    return p_h, p_d, p_a


def _proba_to_result(p_h, p_d, p_a):
    """最大概率 → 结果标签 (n,) → (n,)"""
    n = len(p_h)
    res = np.full(n, 'A', dtype='<U1')
    res[(p_h >= p_d) & (p_h >= p_a)] = 'H'
    res[(p_d > p_h) & (p_d >= p_a)] = 'D'
    return res


def eval_poisson_vec(avg_goals, scale, home_adv, home_rk, away_rk, actual):
    sh = np.exp(-home_rk / scale)
    sa = np.exp(-away_rk / scale)
    lh = avg_goals * (sh * home_adv) / (sh + sa)
    la = avg_goals * sa / (sh + sa)
    ph, pd, pa = _vec_poisson_proba(lh, la)
    preds = _proba_to_result(ph, pd, pa)
    return (preds == actual).mean() * 100


def eval_elo_vec(rw, hb, dc, home_rk, away_rk, actual):
    ed = (away_rk - home_rk) * rw + hb
    p_h = 1.0 / (1.0 + 10.0 ** (-ed / 400.0))
    p_d = dc * np.exp(-(np.abs(ed) / 400.0) ** 2)
    p_h = p_h - p_d / 2.0
    p_a = 1.0 - p_h - p_d
    preds = _proba_to_result(p_h, p_d, p_a)
    return (preds == actual).mean() * 100


# 快捷方式
def poisson_tune(ag, sc, ha):
    return eval_poisson_vec(ag, sc, ha, tune_home_rk, tune_away_rk, tune_actual)

def poisson_val(ag, sc, ha):
    return eval_poisson_vec(ag, sc, ha, val_home_rk, val_away_rk, val_actual)

def elo_tune(rw, hb, dc):
    return eval_elo_vec(rw, hb, dc, tune_home_rk, tune_away_rk, tune_actual)

def elo_val(rw, hb, dc):
    return eval_elo_vec(rw, hb, dc, val_home_rk, val_away_rk, val_actual)

# ============================================================
# ELO 随机搜索
# ============================================================
print("\n" + "=" * 60)
print("ELO 随机搜索 (2000 次)")
print("=" * 60)
t0 = time.time()
best_elo = {"acc": -1.0, "rw": 6, "hb": 50, "dc": 0.25}

for i in range(2000):
    rw = int(rng.integers(1, 21))
    hb = int(rng.integers(0, 121))
    dc = round(rng.uniform(0.05, 0.50), 2)
    acc = elo_tune(rw, hb, dc)
    if acc > best_elo["acc"]:
        best_elo = {"acc": acc, "rw": rw, "hb": hb, "dc": dc}

elo_val_acc = elo_val(best_elo["rw"], best_elo["hb"], best_elo["dc"])
print(f"ELO 最优: rw={best_elo['rw']} hb={best_elo['hb']} dc={best_elo['dc']}")
print(f"  调参集={best_elo['acc']:.1f}%  验证集={elo_val_acc:.1f}%  耗时={time.time()-t0:.1f}s")

# ============================================================
# 泊松随机搜索
# ============================================================
print("\n" + "=" * 60)
print("泊松随机搜索 (3000 次)")
print("=" * 60)
t0 = time.time()
best_poisson = {"acc": -1.0, "ag": 2.85, "sc": 100, "ha": 1.15}

for i in range(3000):
    ag = round(rng.uniform(1.6, 4.2), 2)
    sc = int(rng.integers(30, 500))
    ha = round(rng.uniform(0.90, 1.50), 2)
    acc = poisson_tune(ag, sc, ha)
    if acc > best_poisson["acc"]:
        best_poisson = {"acc": acc, "ag": ag, "sc": sc, "ha": ha}

poisson_val_acc = poisson_val(best_poisson["ag"], best_poisson["sc"], best_poisson["ha"])
print(f"泊松最优: avg_goals={best_poisson['ag']} scale={best_poisson['sc']} home_adv={best_poisson['ha']}")
print(f"  调参集={best_poisson['acc']:.1f}%  验证集={poisson_val_acc:.1f}%  耗时={time.time()-t0:.1f}s")

# ============================================================
# Dixon-Coles 拟合
# ============================================================
print("\n拟合 Dixon-Coles (全量数据)...")
t0 = time.time()
dc = DixonColes()
dc.fit(all_matches, rankings)
dc_val_acc = sum(1 for h, a, hs, as_ in val_matches
                 if dc.predict_result(h, a, rankings) == actual_result(hs, as_)) / len(val_matches) * 100
print(f"DC 参数: {dc.get_params()}")
print(f"DC 验证集准确率: {dc_val_acc:.1f}%  耗时={time.time()-t0:.1f}s")

# ============================================================
# 基准对比
# ============================================================
print("\n" + "=" * 60)
print("基准对比")
print("=" * 60)
orig_elo_acc = elo_val(6, 50, 0.25)
orig_poisson_acc = poisson_val(2.85, 100, 1.15)
print(f"ELO 原始 (rw=6, hb=50, dc=0.25):   {orig_elo_acc:.1f}%")
print(f"ELO 优化:                            {elo_val_acc:.1f}%")
print(f"泊松原始 (ag=2.85, sc=100, ha=1.15): {orig_poisson_acc:.1f}%")
print(f"泊松优化:                             {poisson_val_acc:.1f}%")
print(f"Dixon-Coles:                          {dc_val_acc:.1f}%")

# ============================================================
# 集成模型
# ============================================================
print("\n" + "=" * 60)
print("集成模型评估")
print("=" * 60)

accs = {"poisson": poisson_val_acc, "elo": elo_val_acc, "dixon_coles": dc_val_acc}
weights = {k: max(0, (v / 100) ** 2) for k, v in accs.items()}
w_sum = sum(weights.values())
if w_sum == 0:
    weights = {"poisson": 0.4, "elo": 0.4, "dixon_coles": 0.2}
else:
    weights = {k: v / w_sum for k, v in weights.items()}
print(f"集成权重: {weights}")

# 集成评估（向量化概率加权）
ph_ens = np.zeros(len(val_matches), dtype=np.float64)
pd_ens = np.zeros(len(val_matches), dtype=np.float64)
pa_ens = np.zeros(len(val_matches), dtype=np.float64)

# 泊松贡献
sh = np.exp(-val_home_rk / best_poisson["sc"])
sa = np.exp(-val_away_rk / best_poisson["sc"])
lh = best_poisson["ag"] * (sh * best_poisson["ha"]) / (sh + sa)
la = best_poisson["ag"] * sa / (sh + sa)
pp_h, pp_d, pp_a = _vec_poisson_proba(lh, la)
ph_ens += pp_h * weights["poisson"]
pd_ens += pp_d * weights["poisson"]
pa_ens += pp_a * weights["poisson"]

# ELO 贡献
ed = (val_away_rk - val_home_rk) * best_elo["rw"] + best_elo["hb"]
ep_h = 1.0 / (1.0 + 10.0 ** (-ed / 400.0))
ep_d = best_elo["dc"] * np.exp(-(np.abs(ed) / 400.0) ** 2)
ep_h = ep_h - ep_d / 2.0
ep_a = 1.0 - ep_h - ep_d
ph_ens += ep_h * weights["elo"]
pd_ens += ep_d * weights["elo"]
pa_ens += ep_a * weights["elo"]

# DC 贡献
dc_ph = np.zeros(len(val_matches), dtype=np.float64)
dc_pd = np.zeros(len(val_matches), dtype=np.float64)
dc_pa = np.zeros(len(val_matches), dtype=np.float64)
for i, (home, away, _, _) in enumerate(val_matches):
    dc_ph[i], dc_pd[i], dc_pa[i] = dc.predict_proba(home, away, rankings)
ph_ens += dc_ph * weights["dixon_coles"]
pd_ens += dc_pd * weights["dixon_coles"]
pa_ens += dc_pa * weights["dixon_coles"]

ens_preds = _proba_to_result(ph_ens, pd_ens, pa_ens)
ens_val_acc = (ens_preds == val_actual).mean() * 100
print(f"集成验证: {ens_val_acc:.1f}%")
print(f"最佳单模型 vs 集成: {max(poisson_val_acc, elo_val_acc, dc_val_acc):.1f}% vs {ens_val_acc:.1f}%")

# ============================================================
# 保存参数
# ============================================================
best_params = {
    "poisson": {
        "avg_goals": best_poisson["ag"],
        "scale": best_poisson["sc"],
        "home_advantage": best_poisson["ha"],
        "max_goals": 8,
    },
    "poisson_cv_accuracy": round(poisson_val_acc, 1),
    "poisson_original_accuracy": round(orig_poisson_acc, 1),
    "elo": {
        "rank_weight": best_elo["rw"],
        "home_bias": best_elo["hb"],
        "draw_coeff": best_elo["dc"],
    },
    "elo_cv_accuracy": round(elo_val_acc, 1),
    "elo_original_accuracy": round(orig_elo_acc, 1),
    "dixon_coles": dc.get_params(),
    "dixon_coles_cv_accuracy": round(dc_val_acc, 1),
    "ensemble_weights": {k: round(v, 4) for k, v in weights.items()},
    "ensemble_val_accuracy": round(ens_val_acc, 1),
    "n_tune": n_tune,
    "n_val": len(val_matches),
    "n_total": len(all_matches),
}

OUTPUT_DIR.mkdir(exist_ok=True)
out_path = OUTPUT_DIR / "best_params.json"
out_path.write_text(json.dumps(best_params, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\n参数已保存到 {out_path}")

# ============================================================
# 最终汇总
# ============================================================
print(f"\n{'='*60}")
print(f"  最终结果")
print(f"{'='*60}")
print(f"  数据: {len(all_matches)} 场 (调参 {n_tune} + 验证 {len(val_matches)})")
print(f"  随机基线: 33.3%")
print(f"  ELO:       {elo_val_acc:.1f}% (+{elo_val_acc-33.3:.1f})")
print(f"  泊松:      {poisson_val_acc:.1f}% (+{poisson_val_acc-33.3:.1f})")
print(f"  DC:        {dc_val_acc:.1f}% (+{dc_val_acc-33.3:.1f})")
print(f"  集成:      {ens_val_acc:.1f}% (+{ens_val_acc-33.3:.1f})")
print(f"  最佳提升:  {max(elo_val_acc, poisson_val_acc, dc_val_acc, ens_val_acc) - 33.3:.1f}pp")
