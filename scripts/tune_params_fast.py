"""
快速参数调优：随机采样 + 随机搜索，适用于大数据集
"""
import json
import numpy as np
from pathlib import Path
import sys

ROOT = Path(__file__).parent.parent
OUTPUT_DIR = ROOT / "output"
sys.path.insert(0, str(Path(__file__).parent))

from model_utils import load_matches, load_rankings, rank_to_strength, actual_result, poisson_probabilities


def poisson_predict(home, away, rankings, avg_goals, scale, home_advantage, max_goals):
    strength_h = rank_to_strength(rankings.get(home, 50), scale)
    strength_a = rank_to_strength(rankings.get(away, 50), scale)
    lambda_h = avg_goals * (strength_h * home_advantage) / (strength_h + strength_a)
    lambda_a = avg_goals * strength_a / (strength_h + strength_a)
    # 用聚合概率判结果
    p_h, p_d, p_a = poisson_probabilities(lambda_h, lambda_a)
    best = max(p_h, p_d, p_a)
    return "H" if best == p_h else ("D" if best == p_d else "A")
    return poisson_best_score(lambda_h, lambda_a, max_goals)


def elo_predict(home, away, rankings, rank_weight, home_bias, draw_coeff):
    elo_diff = (rankings.get(away, 50) - rankings.get(home, 50)) * rank_weight + home_bias
    p_h = 1.0 / (1 + 10 ** (-elo_diff / 400))
    p_draw = draw_coeff * np.exp(-(abs(elo_diff) / 400) ** 2)
    p_h -= p_draw / 2
    p_a = 1.0 - p_h - p_draw
    if p_h > max(p_draw, p_a):
        return "H"
    elif p_draw > max(p_h, p_a):
        return "D"
    else:
        return "A"


def evaluate_poisson(matches, rankings, avg_goals, scale, home_advantage, max_goals):
    correct = 0
    for home, away, h_score, a_score in matches:
        pred = poisson_predict(home, away, rankings, avg_goals, scale, home_advantage, max_goals)
        if pred == actual_result(h_score, a_score):
            correct += 1
    return correct / len(matches) * 100


def evaluate_elo(matches, rankings, rank_weight, home_bias, draw_coeff):
    correct = 0
    for home, away, h_score, a_score in matches:
        pred = elo_predict(home, away, rankings, rank_weight, home_bias, draw_coeff)
        if pred == actual_result(h_score, a_score):
            correct += 1
    return correct / len(matches) * 100


def main():
    all_matches = load_matches()
    rankings = load_rankings()
    rng = np.random.RandomState(42)

    print(f"全量数据: {len(all_matches)} 场")

    # 随机采样 600 场用于调参，其余用于验证
    n_tune = min(600, len(all_matches) // 2)
    indices = rng.permutation(len(all_matches))
    tune_idx = set(indices[:n_tune])
    tune_matches = [all_matches[i] for i in range(len(all_matches)) if i in tune_idx]
    val_matches = [all_matches[i] for i in range(len(all_matches)) if i not in tune_idx]

    print(f"调参集: {len(tune_matches)} 场, 验证集: {len(val_matches)} 场")

    # ---- ELO 随机搜索 ----
    print("\n" + "=" * 60)
    print("ELO 随机搜索 (600 次)")
    print("=" * 60)
    best_elo_acc = -1
    best_elo_params = {}
    for i in range(600):
        rw = int(rng.randint(2, 20))
        hb = int(rng.randint(0, 120))
        dc = round(rng.uniform(0.05, 0.50), 2)
        acc = evaluate_elo(tune_matches, rankings, rw, hb, dc)
        if acc > best_elo_acc:
            best_elo_acc = acc
            best_elo_params = {"rank_weight": rw, "home_bias": hb, "draw_coeff": dc}
            if i % 200 == 0 or best_elo_acc > 49:
                val_acc = evaluate_elo(val_matches, rankings, rw, hb, dc) if i > 500 else 0
                print(f"  [{i:4d}] 调参={best_elo_acc:.1f}%  rw={rw} hb={hb} dc={dc}")

    # 验证集评估
    elo_val_acc = evaluate_elo(val_matches, rankings, **best_elo_params)
    print(f"\nELO 最优: {best_elo_params}")
    print(f"  调参集: {best_elo_acc:.1f}%")
    print(f"  验证集: {elo_val_acc:.1f}%")

    # ---- 泊松随机搜索 ----
    print("\n" + "=" * 60)
    print("泊松随机搜索 (800 次)")
    print("=" * 60)
    best_poisson_acc = -1
    best_poisson_params = {}
    for i in range(800):
        avg_goals = round(rng.uniform(1.8, 4.0), 2)
        scale = int(rng.randint(40, 400))
        home_adv = round(rng.uniform(0.95, 1.45), 2)
        max_g = int(rng.choice([6, 7, 8, 9, 10]))
        acc = evaluate_poisson(tune_matches, rankings, avg_goals, scale, home_adv, max_g)
        if acc > best_poisson_acc:
            best_poisson_acc = acc
            best_poisson_params = {
                "avg_goals": avg_goals, "scale": scale,
                "home_advantage": home_adv, "max_goals": max_g,
            }
            if i % 300 == 0:
                print(f"  [{i:4d}] 调参={best_poisson_acc:.1f}%  {best_poisson_params}")

    poisson_val_acc = evaluate_poisson(val_matches, rankings, **best_poisson_params)
    print(f"\n泊松最优: {best_poisson_params}")
    print(f"  调参集: {best_poisson_acc:.1f}%")
    print(f"  验证集: {poisson_val_acc:.1f}%")

    # ---- 基准对比 ----
    print("\n" + "=" * 60)
    print("基准对比（原始参数）")
    print("=" * 60)
    orig_elo = evaluate_elo(val_matches, rankings, 6, 50, 0.25)
    orig_poisson = evaluate_poisson(val_matches, rankings, 2.85, 100, 1.15, 6)
    print(f"ELO 原始 (rw=6, hb=50, dc=0.25): {orig_elo:.1f}%")
    print(f"ELO 优化:                        {elo_val_acc:.1f}%")
    print(f"泊松原始 (ag=2.85, sc=100, ha=1.15): {orig_poisson:.1f}%")
    print(f"泊松优化:                            {poisson_val_acc:.1f}%")

    # 保存
    best = {
        "poisson": best_poisson_params,
        "poisson_tune_accuracy": round(best_poisson_acc, 1),
        "poisson_val_accuracy": round(poisson_val_acc, 1),
        "elo": best_elo_params,
        "elo_tune_accuracy": round(best_elo_acc, 1),
        "elo_val_accuracy": round(elo_val_acc, 1),
        "n_tune": len(tune_matches),
        "n_val": len(val_matches),
    }
    OUTPUT_DIR.mkdir(exist_ok=True)
    out_path = OUTPUT_DIR / "best_params.json"
    out_path.write_text(json.dumps(best, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n参数已保存到 {out_path}")

    # 汇报
    print(f"\n{'='*60}")
    print(f"  最终结果")
    print(f"{'='*60}")
    print(f"  数据集: {len(all_matches)} 场 (调参 {len(tune_matches)} + 验证 {len(val_matches)})")
    print(f"  最佳单个模型: ELO {elo_val_acc:.1f}%")
    print(f"  随机基线: 33.3%")
    print(f"  提升: +{elo_val_acc - 33.3:.1f}%")


if __name__ == "__main__":
    main()
