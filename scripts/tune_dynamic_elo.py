"""
用动态ELO评分重新调优泊松和ELO模型
时间切分：pre-2026 训练，2026 WC 测试
"""
import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import json

ROOT = Path(__file__).parent.parent
OUTPUT_DIR = ROOT / "output"

def actual_result(hg, ag):
    if hg > ag: return "H"
    elif ag > hg: return "A"
    return "D"

# ============================================================
# 数据加载
# ============================================================
def load_data():
    df = pd.read_csv(OUTPUT_DIR / "features.csv", encoding="utf-8-sig")
    df["date"] = pd.to_datetime(df["date"])
    # 时间切分
    train = df[df["date"] < "2026-01-01"]
    extra = df[(df["date"] >= "2026-01-01") & (df["round"] != "FIFA World Cup")]
    train = pd.concat([train, extra])
    test = df[(df["date"] >= "2026-06-01") & (df["round"] == "FIFA World Cup")]
    return train, test

# ============================================================
# ELO 模型（用动态ELO评分替代FIFA排名）
# ============================================================
def elo_predict_proba(elo_h, elo_a, rank_weight, home_bias, draw_coeff):
    """ELO 差值 → H/D/A 概率 (elo_h > elo_a → 主队更强)"""
    elo_diff = (elo_h - elo_a) * rank_weight + home_bias
    p_h = 1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0))
    p_d = draw_coeff * np.exp(-(np.abs(elo_diff) / 400.0) ** 2)
    p_h = p_h - p_d / 2.0
    p_a = 1.0 - p_h - p_d
    return p_h, p_d, p_a

def elo_evaluate(rw, hb, dc, data):
    elo_h = data["elo_h"].values
    elo_a = data["elo_a"].values
    actuals = np.array([actual_result(hg, ag) for hg, ag in zip(data["home_goals"], data["away_goals"])])

    ph, pd, pa = elo_predict_proba(elo_h, elo_a, rw, hb, dc)
    preds = np.full(len(ph), 'A', dtype='<U1')
    preds[(ph >= pd) & (ph >= pa)] = 'H'
    preds[(pd > ph) & (pd > pa)] = 'D'
    return (preds == actuals).mean() * 100

# ============================================================
# 泊松模型（用动态ELO评分计算攻防强度）
# ============================================================
def poisson_evaluate(avg_goals, scale, home_adv, data):
    elo_h = data["elo_h"].values
    elo_a = data["elo_a"].values
    actuals = np.array([actual_result(hg, ag) for hg, ag in zip(data["home_goals"], data["away_goals"])])

    # ELO → 强度 (ELO越高强度越大)
    sh = np.exp(elo_h / scale)
    sa = np.exp(elo_a / scale)
    lh = avg_goals * (sh * home_adv) / (sh + sa)
    la = avg_goals * sa / (sh + sa)

    MAX_G = 10
    g = np.arange(MAX_G + 1, dtype=np.float64)
    pmf_h = stats.poisson.pmf(g[np.newaxis, :], lh[:, np.newaxis])
    pmf_a = stats.poisson.pmf(g[np.newaxis, :], la[:, np.newaxis])
    cdf_a = np.cumsum(pmf_a, axis=1) - pmf_a  # P(j < i)

    p_h = np.sum(pmf_h * cdf_a, axis=1)
    p_d = np.sum(pmf_h * pmf_a, axis=1)
    p_a = 1.0 - p_h - p_d

    preds = np.full(len(p_h), 'A', dtype='<U1')
    preds[(p_h >= p_d) & (p_h >= p_a)] = 'H'
    preds[(p_d > p_h) & (p_d >= p_a)] = 'D'
    return (preds == actuals).mean() * 100

# ============================================================
# 主流程
# ============================================================
def main():
    train, test = load_data()
    print(f"训练集: {len(train)} 场")
    print(f"测试集: {len(test)} 场 (2026 World Cup)")

    rng = np.random.default_rng(42)

    # --- ELO 优化 ---
    print("\n" + "=" * 60)
    print("  ELO 模型优化 (动态ELO评分)")
    print("=" * 60)
    best_elo = {"acc": -1}
    for _ in range(3000):
        rw = round(rng.uniform(0.1, 3.0), 2)
        hb = int(rng.integers(0, 200))
        dc = round(rng.uniform(0.05, 0.60), 2)
        acc = elo_evaluate(rw, hb, dc, train)
        if acc > best_elo["acc"]:
            best_elo = {"acc": acc, "rw": rw, "hb": hb, "dc": dc}
    elo_train = best_elo["acc"]
    elo_test = elo_evaluate(best_elo["rw"], best_elo["hb"], best_elo["dc"], test)
    print(f"ELO最优: rw={best_elo['rw']} hb={best_elo['hb']} dc={best_elo['dc']}")
    print(f"  训练集: {elo_train:.1f}%  测试集: {elo_test:.1f}%")

    # --- 泊松优化 ---
    print("\n" + "=" * 60)
    print("  泊松模型优化 (动态ELO评分)")
    print("=" * 60)
    best_poisson = {"acc": -1}
    for _ in range(3000):
        ag = round(rng.uniform(1.5, 4.5), 2)
        sc = int(rng.integers(100, 800))
        ha = round(rng.uniform(1.00, 1.60), 2)
        acc = poisson_evaluate(ag, sc, ha, train)
        if acc > best_poisson["acc"]:
            best_poisson = {"acc": acc, "ag": ag, "sc": sc, "ha": ha}
    poisson_train = best_poisson["acc"]
    poisson_test = poisson_evaluate(best_poisson["ag"], best_poisson["sc"], best_poisson["ha"], test)
    print(f"泊松最优: avg_goals={best_poisson['ag']} scale={best_poisson['sc']} home_adv={best_poisson['ha']}")
    print(f"  训练集: {poisson_train:.1f}%  测试集: {poisson_test:.1f}%")

    # --- 基准：旧模型 (FIFA排名) ---
    print("\n" + "=" * 60)
    print("  基准对比 (旧FIFA排名模型)")
    print("=" * 60)
    from model_utils import load_rankings
    from models import create_optimized_poisson, create_optimized_elo, load_best_params
    rankings = load_rankings()
    old_params = load_best_params()
    old_poisson = create_optimized_poisson(old_params)
    old_elo = create_optimized_elo(old_params)

    old_poisson_correct = sum(
        1 for _, row in test.iterrows()
        if old_poisson.predict_result(row["home_team"], row["away_team"], rankings)
        == actual_result(row["home_goals"], row["away_goals"])
    )
    old_elo_correct = sum(
        1 for _, row in test.iterrows()
        if old_elo.predict_result(row["home_team"], row["away_team"], rankings)
        == actual_result(row["home_goals"], row["away_goals"])
    )
    n_test = len(test)
    print(f"旧泊松(FIFA): {old_poisson_correct}/{n_test} = {old_poisson_correct/n_test*100:.1f}%")
    print(f"旧ELO(FIFA):   {old_elo_correct}/{n_test} = {old_elo_correct/n_test*100:.1f}%")

    # --- 汇总 ---
    print(f"\n{'='*70}")
    print(f"  汇总")
    print(f"{'='*70}")
    print(f"{'模型':<25} {'训练集':>10} {'测试集(2026 WC)':>18}")
    print(f"{'-'*53}")
    print(f"{'旧ELO(FIFA排名)':<25} {'--':>10} {old_elo_correct/n_test*100:>17.1f}%")
    print(f"{'旧泊松(FIFA排名)':<25} {'--':>10} {old_poisson_correct/n_test*100:>17.1f}%")
    print(f"{'新ELO(动态ELO)':<25} {elo_train:>9.1f}% {elo_test:>17.1f}%")
    print(f"{'新泊松(动态ELO)':<25} {poisson_train:>9.1f}% {poisson_test:>17.1f}%")

    # --- 保存 ---
    new_params = {
        "elo": {
            "rank_weight": best_elo["rw"],
            "home_bias": best_elo["hb"],
            "draw_coeff": best_elo["dc"],
            "elo_source": "dynamic_elo_from_history",
        },
        "elo_train_accuracy": round(elo_train, 1),
        "elo_test_accuracy": round(elo_test, 1),
        "poisson": {
            "avg_goals": best_poisson["ag"],
            "scale": best_poisson["sc"],
            "home_advantage": best_poisson["ha"],
            "max_goals": 8,
            "strength_source": "dynamic_elo_from_history",
        },
        "poisson_train_accuracy": round(poisson_train, 1),
        "poisson_test_accuracy": round(poisson_test, 1),
        "n_train": len(train),
        "n_test": len(test),
        "old_elo_test": round(old_elo_correct / n_test * 100, 1),
        "old_poisson_test": round(old_poisson_correct / n_test * 100, 1),
    }
    out_path = OUTPUT_DIR / "dynamic_elo_params.json"
    out_path.write_text(json.dumps(new_params, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n参数保存到 {out_path}")


if __name__ == "__main__":
    main()
