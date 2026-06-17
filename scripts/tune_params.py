"""
参数优化：随机搜索 + 留一法交叉验证
优化泊松和 ELO 模型的关键超参数
"""
import json
import numpy as np
from pathlib import Path
from itertools import product

from model_utils import (
    FIFA_RANKINGS,
    load_matches, load_rankings,
    rank_to_strength, actual_result, poisson_best_score,
)

ROOT = Path(__file__).parent.parent
OUTPUT_DIR = ROOT / "output"


def poisson_predict(home, away, rankings, avg_goals, scale, home_advantage, max_goals):
    strength_h = rank_to_strength(rankings.get(home, 50), scale)
    strength_a = rank_to_strength(rankings.get(away, 50), scale)
    lambda_h = avg_goals * (strength_h * home_advantage) / (strength_h + strength_a)
    lambda_a = avg_goals * strength_a / (strength_h + strength_a)
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


def loo_cv_poisson(matches, rankings, avg_goals, scale, home_advantage, max_goals):
    correct = 0
    for i in range(len(matches)):
        test_match = matches[i]
        home, away, h_score, a_score = test_match
        pred = poisson_predict(home, away, rankings, avg_goals, scale, home_advantage, max_goals)
        if pred[0] == actual_result(h_score, a_score):
            correct += 1
    return correct / len(matches) * 100


def loo_cv_elo(matches, rankings, rank_weight, home_bias, draw_coeff):
    correct = 0
    for i in range(len(matches)):
        test_match = matches[i]
        home, away, h_score, a_score = test_match
        pred = elo_predict(home, away, rankings, rank_weight, home_bias, draw_coeff)
        if pred == actual_result(h_score, a_score):
            correct += 1
    return correct / len(matches) * 100


def random_search_poisson(matches, rankings, n_iter=2000, seed=42):
    """随机搜索泊松参数"""
    rng = np.random.RandomState(seed)

    # 参数空间
    space = {
        "avg_goals": (2.0, 3.8),
        "scale": (50, 300),
        "home_advantage": (1.00, 1.40),
        "max_goals": [6, 7, 8, 9, 10],
    }

    print("泊松模型随机搜索（2000 次采样 × 留一法）...")
    print(f"参数范围: avg_goals=[{space['avg_goals'][0]:.1f}, {space['avg_goals'][1]:.1f}], "
          f"scale=[{space['scale'][0]}, {space['scale'][1]}], "
          f"home_adv=[{space['home_advantage'][0]:.2f}, {space['home_advantage'][1]:.2f}]")

    best_acc = -1
    best_params = {}

    for i in range(n_iter):
        avg_goals = round(rng.uniform(*space["avg_goals"]), 2)
        scale = int(rng.randint(*space["scale"]))
        home_adv = round(rng.uniform(*space["home_advantage"]), 2)
        max_g = int(rng.choice(space["max_goals"]))

        acc = loo_cv_poisson(matches, rankings, avg_goals, scale, home_adv, max_g)

        if acc > best_acc:
            best_acc = acc
            best_params = {
                "avg_goals": avg_goals,
                "scale": scale,
                "home_advantage": home_adv,
                "max_goals": max_g,
            }

        if (i + 1) % 400 == 0:
            print(f"  进度: {i+1}/{n_iter}  当前最佳: {best_acc:.1f}%  {best_params}")

    # 第二轮：在最优参数附近精细搜索
    print(f"\n  精细搜索（最优邻域 200 次采样）...")
    for i in range(200):
        avg_goals = round(best_params["avg_goals"] + rng.uniform(-0.15, 0.15), 2)
        avg_goals = max(1.8, min(4.0, avg_goals))
        scale = int(best_params["scale"] + rng.randint(-30, 30))
        scale = max(20, min(400, scale))
        home_adv = round(best_params["home_advantage"] + rng.uniform(-0.06, 0.06), 2)
        home_adv = max(0.90, min(1.50, home_adv))
        max_g = int(rng.choice(space["max_goals"]))

        acc = loo_cv_poisson(matches, rankings, avg_goals, scale, home_adv, max_g)
        if acc > best_acc:
            best_acc = acc
            best_params = {
                "avg_goals": avg_goals,
                "scale": scale,
                "home_advantage": home_adv,
                "max_goals": max_g,
            }

    print(f"\n泊松最优参数: {best_params}, LOO-CV 准确率: {best_acc:.1f}%")
    return best_params, best_acc


def grid_search_elo(matches, rankings):
    """ELO 参数空间小，直接网格搜索"""
    param_grid = {
        "rank_weight": range(3, 16),
        "home_bias": range(0, 121, 5),
        "draw_coeff": [round(x, 2) for x in np.arange(0.08, 0.46, 0.02)],
    }

    print("\nELO 模型网格搜索...")
    print(f"搜索空间: rank_weight=[3,15], home_bias=[0,120], draw_coeff=[0.08,0.44]")
    total = len(param_grid["rank_weight"]) * len(param_grid["home_bias"]) * len(param_grid["draw_coeff"])
    print(f"总组合数: {total:,}")

    best_acc = -1
    best_params = {}
    count = 0

    for rank_weight, home_bias, draw_coeff in product(
        param_grid["rank_weight"], param_grid["home_bias"], param_grid["draw_coeff"],
    ):
        acc = loo_cv_elo(matches, rankings, rank_weight, home_bias, draw_coeff)

        if acc > best_acc:
            best_acc = acc
            best_params = {
                "rank_weight": rank_weight,
                "home_bias": home_bias,
                "draw_coeff": draw_coeff,
            }

        count += 1
        if count % 1000 == 0:
            print(f"  进度: {count}/{total} ({count/total*100:.0f}%)  当前最佳: {best_acc:.1f}%")

    print(f"\nELO 最优参数: {best_params}, LOO-CV 准确率: {best_acc:.1f}%")
    return best_params, best_acc


def evaluate_original(matches, rankings):
    orig_poisson = loo_cv_poisson(matches, rankings, 2.85, 100, 1.15, 6)
    orig_elo = loo_cv_elo(matches, rankings, 6, 50, 0.25)
    return orig_poisson, orig_elo


def main():
    matches = load_matches()
    rankings = load_rankings()

    print(f"参数优化 — 留一法交叉验证")
    print(f"比赛场次: {len(matches)}")
    print(f"队伍排名: {len(rankings)} 队")
    print()

    # 原始参数基准
    orig_poisson, orig_elo = evaluate_original(matches, rankings)
    print(f"原始参数基准:")
    print(f"  泊松 (avg_goals=2.85, scale=100, home_adv=1.15, max_goals=6): {orig_poisson:.1f}%")
    print(f"  ELO  (rank_weight=6, home_bias=50, draw_coeff=0.25): {orig_elo:.1f}%")
    print()

    # 优化
    poisson_best, poisson_acc = random_search_poisson(matches, rankings, n_iter=2000)
    elo_best, elo_acc = grid_search_elo(matches, rankings)

    # 结果汇总
    print(f"\n{'='*60}")
    print(f"  优化结果汇总")
    print(f"{'='*60}")
    print(f"{'模型':<15} {'原始准确率':>12} {'优化准确率':>12} {'提升':>10}")
    print(f"{'-'*49}")
    print(f"{'泊松':<15} {orig_poisson:>10.1f}% {poisson_acc:>10.1f}% {poisson_acc-orig_poisson:>+9.1f}%")
    print(f"{'ELO':<15} {orig_elo:>10.1f}% {elo_acc:>10.1f}% {elo_acc-orig_elo:>+9.1f}%")

    # 保存
    OUTPUT_DIR.mkdir(exist_ok=True)
    best = {
        "poisson": poisson_best,
        "poisson_cv_accuracy": round(poisson_acc, 1),
        "poisson_original_accuracy": round(orig_poisson, 1),
        "elo": elo_best,
        "elo_cv_accuracy": round(elo_acc, 1),
        "elo_original_accuracy": round(orig_elo, 1),
        "n_matches": len(matches),
    }
    out_path = OUTPUT_DIR / "best_params.json"
    out_path.write_text(json.dumps(best, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n最优参数已保存到 {out_path}")


if __name__ == "__main__":
    main()
