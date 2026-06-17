"""
公平对比：动态ELO vs FIFA排名，同时间切分数据集
时间切分：pre-2026 训练 / 2026 全量测试
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
    train = df[df["date"] < "2026-01-01"]
    # 2026年所有非WC比赛也加入训练
    extra = df[(df["date"] >= "2026-01-01") & (df["round"] != "FIFA World Cup")]
    train = pd.concat([train, extra])
    # 测试 = 2026 WC 比赛（当前12场）
    test = df[(df["date"] >= "2026-06-01") & (df["round"] == "FIFA World Cup")]
    return train, test

# ============================================================
# ELO 模型概率计算
# ============================================================

def elo_proba_from_diff(elo_diff, rw, hb, dc):
    """elo_diff > 0 表示主队更强"""
    x = elo_diff * rw + hb
    p_h = 1.0 / (1.0 + 10.0 ** (-x / 400.0))
    p_d = dc * np.exp(-(np.abs(x) / 400.0) ** 2)
    p_h = p_h - p_d / 2.0
    p_a = 1.0 - p_h - p_d
    return p_h, p_d, p_a

def elo_result_from_proba(ph, pd, pa):
    n = len(ph)
    preds = np.full(n, 'A', dtype='<U1')
    preds[(ph >= pd) & (ph >= pa)] = 'H'
    preds[(pd > ph) & (pd >= pa)] = 'D'
    return preds

# ============================================================
# 泊松模型
# ============================================================
def poisson_proba(lh, la, max_g=10):
    g = np.arange(max_g + 1, dtype=np.float64)
    pmf_h = stats.poisson.pmf(g[np.newaxis, :], lh[:, np.newaxis])
    pmf_a = stats.poisson.pmf(g[np.newaxis, :], la[:, np.newaxis])
    cdf_a = np.cumsum(pmf_a, axis=1) - pmf_a
    p_h = np.sum(pmf_h * cdf_a, axis=1)
    p_d = np.sum(pmf_h * pmf_a, axis=1)
    p_a = 1.0 - p_h - p_d
    return p_h, p_d, p_a

# ============================================================
# 评估函数
# ============================================================

def eval_elo_dynamic(rw, hb, dc, data):
    ed = (data["elo_h"].values - data["elo_a"].values).astype(np.float64)
    ph, pd, pa = elo_proba_from_diff(ed, rw, hb, dc)
    preds = elo_result_from_proba(ph, pd, pa)
    actuals = np.array([actual_result(hg, ag) for hg, ag in zip(data["home_goals"], data["away_goals"])])
    return (preds == actuals).mean() * 100

def eval_elo_fifa(rw, hb, dc, data, rankings):
    """FIFA排名版 ELO: 排名差 * rw + hb"""
    home_rk = np.array([rankings.get(h, 50) for h in data["home_team"]], dtype=np.float64)
    away_rk = np.array([rankings.get(a, 50) for a in data["away_team"]], dtype=np.float64)
    ed = (away_rk - home_rk) * rw + hb  # lower rank = better
    ph, pd, pa = elo_proba_from_diff(ed, 1.0, 0.0, dc)  # rw already applied
    preds = elo_result_from_proba(ph, pd, pa)
    actuals = np.array([actual_result(hg, ag) for hg, ag in zip(data["home_goals"], data["away_goals"])])
    return (preds == actuals).mean() * 100

def eval_poisson_dynamic(ag, sc, ha, data):
    elo_h = data["elo_h"].values.astype(np.float64)
    elo_a = data["elo_a"].values.astype(np.float64)
    sh = np.exp(elo_h / sc)
    sa = np.exp(elo_a / sc)
    lh = ag * (sh * ha) / (sh + sa)
    la = ag * sa / (sh + sa)
    ph, pd, pa = poisson_proba(lh, la)
    preds = elo_result_from_proba(ph, pd, pa)
    actuals = np.array([actual_result(hg, ag) for hg, ag in zip(data["home_goals"], data["away_goals"])])
    return (preds == actuals).mean() * 100

def eval_poisson_fifa(ag, sc, ha, data, rankings):
    home_rk = np.array([rankings.get(h, 50) for h in data["home_team"]], dtype=np.float64)
    away_rk = np.array([rankings.get(a, 50) for a in data["away_team"]], dtype=np.float64)
    sh = np.exp(-home_rk / sc)  # lower rank = stronger
    sa = np.exp(-away_rk / sc)
    lh = ag * (sh * ha) / (sh + sa)
    la = ag * sa / (sh + sa)
    ph, pd, pa = poisson_proba(lh, la)
    preds = elo_result_from_proba(ph, pd, pa)
    actuals = np.array([actual_result(hg, ag) for hg, ag in zip(data["home_goals"], data["away_goals"])])
    return (preds == actuals).mean() * 100

# ============================================================
# 随机搜索
# ============================================================

def tune_elo_dynamic(train_data, n_iter=3000):
    rng = np.random.default_rng(42)
    best = {"acc": -1}
    for _ in range(n_iter):
        rw = round(rng.uniform(0.1, 3.0), 2)
        hb = int(rng.integers(0, 200))
        dc = round(rng.uniform(0.05, 0.60), 2)
        acc = eval_elo_dynamic(rw, hb, dc, train_data)
        if acc > best["acc"]:
            best = {"acc": acc, "rw": rw, "hb": hb, "dc": dc}
    return best

def tune_elo_fifa(train_data, rankings, n_iter=3000):
    rng = np.random.default_rng(42)
    best = {"acc": -1}
    for _ in range(n_iter):
        rw = int(rng.integers(1, 20))
        hb = int(rng.integers(0, 150))
        dc = round(rng.uniform(0.05, 0.60), 2)
        acc = eval_elo_fifa(rw, hb, dc, train_data, rankings)
        if acc > best["acc"]:
            best = {"acc": acc, "rw": rw, "hb": hb, "dc": dc}
    return best

def tune_poisson_dynamic(train_data, n_iter=3000):
    rng = np.random.default_rng(42)
    best = {"acc": -1}
    for _ in range(n_iter):
        ag = round(rng.uniform(1.5, 4.5), 2)
        sc = int(rng.integers(100, 800))
        ha = round(rng.uniform(1.00, 1.60), 2)
        acc = eval_poisson_dynamic(ag, sc, ha, train_data)
        if acc > best["acc"]:
            best = {"acc": acc, "ag": ag, "sc": sc, "ha": ha}
    return best

def tune_poisson_fifa(train_data, rankings, n_iter=3000):
    rng = np.random.default_rng(42)
    best = {"acc": -1}
    for _ in range(n_iter):
        ag = round(rng.uniform(1.5, 4.5), 2)
        sc = int(rng.integers(30, 500))
        ha = round(rng.uniform(1.00, 1.60), 2)
        acc = eval_poisson_fifa(ag, sc, ha, train_data, rankings)
        if acc > best["acc"]:
            best = {"acc": acc, "ag": ag, "sc": sc, "ha": ha}
    return best

# ============================================================
# 主流程
# ============================================================
def main():
    train, test = load_data()
    rankings = __import__("model_utils", fromlist=["load_rankings"]).load_rankings()

    print(f"训练集: {len(train)} 场")
    print(f"测试集: {len(test)} 场 (2026 World Cup)")
    print(f"FIFA排名覆盖: {len(rankings)} 队")

    # --- 调优 ---
    print("\n调优中...")

    print("  ELO(动态ELO)...")
    best_elo_dyn = tune_elo_dynamic(train)
    print("  ELO(FIFA排名)...")
    best_elo_fifa = tune_elo_fifa(train, rankings)
    print("  泊松(动态ELO)...")
    best_poi_dyn = tune_poisson_dynamic(train)
    print("  泊松(FIFA排名)...")
    best_poi_fifa = tune_poisson_fifa(train, rankings)

    # --- 评估 ---
    print("\n" + "=" * 70)
    print("  测试集评估 (2026 World Cup)")
    print("=" * 70)

    results = {}
    for name, fn, params in [
        ("ELO(动态ELO)", eval_elo_dynamic, (best_elo_dyn["rw"], best_elo_dyn["hb"], best_elo_dyn["dc"], test)),
        ("ELO(FIFA排名)", eval_elo_fifa, (best_elo_fifa["rw"], best_elo_fifa["hb"], best_elo_fifa["dc"], test, rankings)),
        ("泊松(动态ELO)", eval_poisson_dynamic, (best_poi_dyn["ag"], best_poi_dyn["sc"], best_poi_dyn["ha"], test)),
        ("泊松(FIFA排名)", eval_poisson_fifa, (best_poi_fifa["ag"], best_poi_fifa["sc"], best_poi_fifa["ha"], test, rankings)),
    ]:
        acc = fn(*params)
        results[name] = {"train_acc": None, "test_acc": acc}
        print(f"  {name}: {acc:.1f}%")

    # 训练集准确率
    results["ELO(动态ELO)"]["train_acc"] = best_elo_dyn["acc"]
    results["ELO(FIFA排名)"]["train_acc"] = best_elo_fifa["acc"]
    results["泊松(动态ELO)"]["train_acc"] = best_poi_dyn["acc"]
    results["泊松(FIFA排名)"]["train_acc"] = best_poi_fifa["acc"]

    # --- 汇总 ---
    print(f"\n{'='*70}")
    print(f"  公平对比汇总（同训练/测试切分）")
    print(f"{'='*70}")
    print(f"{'模型':<25} {'训练集':>10} {'测试集(WC)':>14} {'参数':>20}")
    print(f"{'-'*69}")
    print(f"{'ELO(动态ELO)':<25} {best_elo_dyn['acc']:>9.1f}% {results['ELO(动态ELO)']['test_acc']:>13.1f}%  rw={best_elo_dyn['rw']} hb={best_elo_dyn['hb']} dc={best_elo_dyn['dc']}")
    print(f"{'ELO(FIFA排名)':<25} {best_elo_fifa['acc']:>9.1f}% {results['ELO(FIFA排名)']['test_acc']:>13.1f}%  rw={best_elo_fifa['rw']} hb={best_elo_fifa['hb']} dc={best_elo_fifa['dc']}")
    print(f"{'泊松(动态ELO)':<25} {best_poi_dyn['acc']:>9.1f}% {results['泊松(动态ELO)']['test_acc']:>13.1f}%  ag={best_poi_dyn['ag']} sc={best_poi_dyn['sc']} ha={best_poi_dyn['ha']}")
    print(f"{'泊松(FIFA排名)':<25} {best_poi_fifa['acc']:>9.1f}% {results['泊松(FIFA排名)']['test_acc']:>13.1f}%  ag={best_poi_fifa['ag']} sc={best_poi_fifa['sc']} ha={best_poi_fifa['ha']}")

    # --- 详细预测 ---
    print(f"\n逐场预测对比:")
    for _, row in test.iterrows():
        h, a = row["home_team"], row["away_team"]
        hg, ag = int(row["home_goals"]), int(row["away_goals"])
        act = actual_result(hg, ag)
        print(f"  {h} vs {a} ({hg}:{ag} {act})")

        # ELO dynamic
        ed = row["elo_h"] - row["elo_a"]
        ph, pd, pa = elo_proba_from_diff(ed, best_elo_dyn["rw"], best_elo_dyn["hb"], best_elo_dyn["dc"])
        pred = "H" if ph >= max(pd, pa) else ("D" if pd >= pa else "A")
        print(f"    ELO(dyn): {pred} (H:{ph:.0%} D:{pd:.0%} A:{pa:.0%}) {'OK' if pred==act else 'XX'}")

        # ELO FIFA
        rk_h, rk_a = rankings.get(h, 50), rankings.get(a, 50)
        ed_fifa = (rk_a - rk_h) * best_elo_fifa["rw"] + best_elo_fifa["hb"]
        ph2, pd2, pa2 = elo_proba_from_diff(ed_fifa, 1.0, 0.0, best_elo_fifa["dc"])
        pred2 = "H" if ph2 >= max(pd2, pa2) else ("D" if pd2 >= pa2 else "A")
        print(f"    ELO(FIFA): {pred2} (H:{ph2:.0%} D:{pd2:.0%} A:{pa2:.0%}) {'OK' if pred2==act else 'XX'}")

    # 旧参数对比
    print(f"\n  旧参数(随机切分调优)在时间切分测试集:")
    from models import create_optimized_elo, create_optimized_poisson, load_best_params
    old_p = load_best_params()
    old_elo_m = create_optimized_elo(old_p)
    old_poi_m = create_optimized_poisson(old_p)
    n = len(test)
    old_elo_correct = sum(1 for _, row in test.iterrows()
                          if old_elo_m.predict_result(row["home_team"], row["away_team"], rankings)
                          == actual_result(row["home_goals"], row["away_goals"]))
    old_poi_correct = sum(1 for _, row in test.iterrows()
                          if old_poi_m.predict_result(row["home_team"], row["away_team"], rankings)
                          == actual_result(row["home_goals"], row["away_goals"]))
    print(f"  旧ELO(FIFA随机切分): {old_elo_correct}/{n} = {old_elo_correct/n*100:.1f}%")
    print(f"  旧泊松(FIFA随机切分): {old_poi_correct}/{n} = {old_poi_correct/n*100:.1f}%")

    # 保存
    summary = {
        "split_method": "time-based (pre-2026 train, 2026 WC test)",
        "n_train": len(train),
        "n_test": len(test),
        "models": {
            "elo_dynamic": {"params": best_elo_dyn, "train_acc": round(best_elo_dyn["acc"], 1), "test_acc": round(results["ELO(动态ELO)"]["test_acc"], 1)},
            "elo_fifa": {"params": best_elo_fifa, "train_acc": round(best_elo_fifa["acc"], 1), "test_acc": round(results["ELO(FIFA排名)"]["test_acc"], 1)},
            "poisson_dynamic": {"params": best_poi_dyn, "train_acc": round(best_poi_dyn["acc"], 1), "test_acc": round(results["泊松(动态ELO)"]["test_acc"], 1)},
            "poisson_fifa": {"params": best_poi_fifa, "train_acc": round(best_poi_fifa["acc"], 1), "test_acc": round(results["泊松(FIFA排名)"]["test_acc"], 1)},
        },
        "old_params_test": {
            "elo_fifa_random_split": round(old_elo_correct / n * 100, 1),
            "poisson_fifa_random_split": round(old_poi_correct / n * 100, 1),
        },
    }
    out_path = OUTPUT_DIR / "fair_comparison.json"
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果保存到 {out_path}")


if __name__ == "__main__":
    main()
