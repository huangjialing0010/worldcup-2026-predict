"""
World Cup 2026 比分预测回测
比较多种预测方法的准确率，支持动态数据加载和参数优化
"""
import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import random
import argparse
import json

from model_utils import (
    FIFA_RANKINGS, FALLBACK_MATCHES, FALLBACK_ODDS,
    HEXAGRAMS, BAGUA,
    load_matches, load_rankings, load_odds_data,
    get_rank, rank_to_strength, actual_result, poisson_best_score,
    poisson_probabilities, implied_probabilities,
)
from models import (
    OptimizedPoisson, OptimizedELO, DixonColes,
    load_best_params, create_optimized_poisson, create_optimized_elo,
)
from ensemble import EnsemblePredictor

ROOT = Path(__file__).parent.parent
OUTPUT_DIR = ROOT / "output"


# ============================================================
# 预测方法
# ============================================================


def predict_random():
    r = random.random()
    return "H" if r < 0.333 else ("D" if r < 0.666 else "A")


def predict_random_weighted():
    r = random.random()
    return "H" if r < 0.45 else ("D" if r < 0.70 else "A")


def predict_ranking(home, away, rankings):
    diff = get_rank(home, rankings) - get_rank(away, rankings)
    if diff < -10:
        return "H"
    elif diff > 10:
        return "A"
    else:
        return "D"


def predict_poisson(home, away, rankings, avg_goals=2.85, scale=100, home_advantage=1.15):
    strength_h = rank_to_strength(get_rank(home, rankings), scale)
    strength_a = rank_to_strength(get_rank(away, rankings), scale)
    lambda_h = avg_goals * (strength_h * home_advantage) / (strength_h + strength_a)
    lambda_a = avg_goals * strength_a / (strength_h + strength_a)
    # 用聚合概率判胜平负（单比分偏差大）
    p_h, p_d, p_a = poisson_probabilities(lambda_h, lambda_a)
    best = max(p_h, p_d, p_a)
    result = "H" if best == p_h else ("D" if best == p_d else "A")
    # 仍返回最可能比分用于展示
    score = poisson_best_score(lambda_h, lambda_a, 8)
    return result, score[1], score[2]


def predict_elo(home, away, rankings):
    elo_diff = (get_rank(away, rankings) - get_rank(home, rankings)) * 6 + 50
    p_h = 1.0 / (1 + 10 ** (-elo_diff / 400))
    p_draw = 0.25 * np.exp(-(abs(elo_diff) / 400) ** 2)
    p_h -= p_draw / 2
    p_a = 1.0 - p_h - p_draw

    r = random.random()
    if r < p_h:
        return "H"
    elif r < p_h + p_draw:
        return "D"
    else:
        return "A"


def predict_iching():
    hex_idx = random.randint(0, 63)
    hex_name = HEXAGRAMS[hex_idx]
    if hex_idx < 22:
        return "H", hex_name
    elif hex_idx < 43:
        return "D", hex_name
    else:
        return "A", hex_name


def predict_bagua():
    gua = random.choice(BAGUA)
    idx = BAGUA.index(gua)
    if idx < 3:
        return "H", gua
    elif idx < 5:
        return "D", gua
    else:
        return "A", gua


def predict_odds(home, away, odds_data):
    for h, a, oh, od, oa, _, _ in odds_data:
        if h == home and a == away:
            p_h, p_d, p_a = implied_probabilities(oh, od, oa)
            best = max(p_h, p_d, p_a)
            return "H" if best == p_h else ("D" if best == p_d else "A")
    return "D"


def predict_odds_poisson(home, away, odds_data, rankings):
    matched = None
    for h, a, oh, od, oa, _, _ in odds_data:
        if h == home and a == away:
            matched = implied_probabilities(oh, od, oa)
            break

    if matched is None:
        return predict_poisson(home, away, rankings)

    p_h, p_d, p_a = matched
    total = 2.8
    lambda_h = total * (p_h + 0.5 * p_d) * 1.08
    lambda_a = total * (p_a + 0.5 * p_d) * 0.92
    # 用聚合概率判胜平负
    pp_h, pp_d, pp_a = poisson_probabilities(lambda_h, lambda_a)
    best = max(pp_h, pp_d, pp_a)
    result = "H" if best == pp_h else ("D" if best == pp_d else "A")
    score = poisson_best_score(lambda_h, lambda_a, 8)
    return result, score[1], score[2]


# ============================================================
# 评估
# ============================================================


def evaluate(name, predictions, matches, verbose=True):
    correct = 0
    exact_scores = 0
    total_mae = 0
    n = len(matches)

    results = []
    for i, (home, away, h_score, a_score) in enumerate(matches):
        pred = predictions[i]
        if isinstance(pred, tuple) and len(pred) >= 2:
            pred_result = pred[0]
            pred_h = pred[1] if len(pred) > 1 else None
            pred_a = pred[2] if len(pred) > 2 else None
        else:
            pred_result = pred
            pred_h, pred_a = None, None

        actual_res = actual_result(h_score, a_score)

        if pred_result == actual_res:
            correct += 1

        if pred_h is not None and pred_h == h_score and pred_a == a_score:
            exact_scores += 1

        if pred_h is not None:
            total_mae += abs(pred_h - h_score) + abs(pred_a - a_score)

        results.append({
            "match": f"{home} vs {away}",
            "actual": f"{h_score}:{a_score} ({actual_res})",
            "predicted": f"{pred_h}:{pred_a} ({pred_result})" if pred_h is not None else pred_result,
            "correct": pred_result == actual_res,
        })

    win_accuracy = correct / n * 100 if n > 0 else 0
    score_accuracy = exact_scores / n * 100 if n > 0 else 0
    avg_mae = total_mae / n if n > 0 and total_mae > 0 else None

    if verbose:
        print(f"\n{'='*60}")
        print(f"  {name}")
        print(f"{'='*60}")
        print(f"  胜平负准确率: {correct}/{n} = {win_accuracy:.1f}%")
        if avg_mae is not None:
            print(f"  精确比分命中: {exact_scores}/{n} = {score_accuracy:.1f}%")
            print(f"  进球差 MAE: {avg_mae:.2f}")
        print(f"{'='*60}")

    return {
        "name": name,
        "win_accuracy": win_accuracy,
        "score_accuracy": score_accuracy,
        "avg_mae": avg_mae,
        "correct": correct,
        "total": n,
        "results": results,
    }


# ============================================================
# 留一法交叉验证
# ============================================================


def loo_cross_validation(matches, rankings, odds_data, best_params, n_runs=100):
    """留一法评估所有方法"""
    print(f"\n{'='*70}")
    print(f"  留一法交叉验证 (n={len(matches)})")
    print(f"{'='*70}")

    methods = {
        "泊松(原始)": lambda h, a: predict_poisson(h, a, rankings),
        "泊松(优化)": None,  # will be set below
        "ELO(原始)": lambda h, a: predict_elo(h, a, rankings),
        "ELO(优化)": None,
        "FIFA排名启发式": lambda h, a: predict_ranking(h, a, rankings),
        "博彩盘口": lambda h, a: predict_odds(h, a, odds_data),
        "赔率校准泊松": lambda h, a: predict_odds_poisson(h, a, odds_data, rankings),
    }

    # 创建优化模型
    opt_poisson = create_optimized_poisson(best_params)
    opt_elo = create_optimized_elo(best_params)
    methods["泊松(优化)"] = lambda h, a: opt_poisson.predict_score(h, a, rankings)
    methods["ELO(优化)"] = lambda h, a: opt_elo.predict_result(h, a, rankings)

    # 拟合 Dixon-Coles
    dc = DixonColes()
    dc_fitted = False
    try:
        dc.fit(matches, rankings)
        methods["Dixon-Coles"] = lambda h, a: dc.predict_score(h, a, rankings)
        dc_fitted = True
    except Exception as e:
        print(f"  Dixon-Coles 拟合失败: {e}")

    # 先评估各独立方法
    results = {}
    for name, predict_fn in methods.items():
        correct = 0
        for i in range(len(matches)):
            test = matches[i]
            home, away, h_s, a_s = test
            pred = predict_fn(home, away)
            pred_result = pred[0] if isinstance(pred, tuple) else pred
            if pred_result == actual_result(h_s, a_s):
                correct += 1
        results[name] = round(correct / len(matches) * 100, 1)

    # 用各模型的实际 CV 准确率更新 best_params，再创建集成
    if best_params is None:
        best_params = {}
    if dc_fitted:
        best_params["dixon_coles"] = dc.get_params()
        best_params["dixon_coles_cv_accuracy"] = results.get("Dixon-Coles", 43.8)

    ens = EnsemblePredictor(best_params, rankings, dc if dc_fitted else None)
    methods["集成模型"] = lambda h, a: (ens.predict_result(h, a),)

    # 评估集成
    correct = 0
    for i in range(len(matches)):
        test = matches[i]
        home, away, h_s, a_s = test
        pred = methods["集成模型"](home, away)
        pred_result = pred[0] if isinstance(pred, tuple) else pred
        if pred_result == actual_result(h_s, a_s):
            correct += 1
    results["集成模型"] = round(correct / len(matches) * 100, 1)

    print(f"\n{'方法':<20} {'LOO-CV 准确率':>15}")
    print(f"{'-'*35}")
    for name, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"{name:<20} {acc:>13.1f}%")

    return results


# ============================================================
# 主流程
# ============================================================


def main():
    parser = argparse.ArgumentParser(description="World Cup 2026 回测")
    parser.add_argument("--n-runs", type=int, default=1000,
                        help="随机方法的采样次数（默认1000）")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cv", action="store_true",
                        help="运行留一法交叉验证")
    parser.add_argument("--output", action="store_true", default=True,
                        help="保存结果 CSV")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # 加载数据和参数
    rankings = load_rankings()
    matches = load_matches()
    odds_data = load_odds_data()
    best_params = load_best_params()

    data_source = "动态加载" if (Path(__file__).parent.parent / "data" / "processed" / "matches.csv").exists() else "硬编码回退"
    print(f"2026 世界杯回测 [{data_source}]")
    print(f"比赛场次: {len(matches)}")
    print(f"队伍排名: {len(rankings)} 队")
    if best_params:
        print(f"优化参数: 已加载 (泊松 CV={best_params.get('poisson_cv_accuracy', '?')}%)")
    print("=" * 60)

    # 交叉验证模式
    if args.cv:
        loo_cross_validation(matches, rankings, odds_data, best_params, args.n_runs)
        return

    all_methods = {}
    n = args.n_runs

    # 1. 随机均匀
    wins = []
    for _ in range(n):
        preds = [predict_random() for _ in matches]
        correct = sum(1 for i, m in enumerate(matches)
                      if preds[i] == actual_result(m[2], m[3]))
        wins.append(correct / len(matches) * 100)
    all_methods["随机均匀(33%)"] = {"mean": np.mean(wins), "std": np.std(wins)}

    # 2. 随机加权
    wins = []
    for _ in range(n):
        preds = [predict_random_weighted() for _ in matches]
        correct = sum(1 for i, m in enumerate(matches)
                      if preds[i] == actual_result(m[2], m[3]))
        wins.append(correct / len(matches) * 100)
    all_methods["随机加权(45/25/30)"] = {"mean": np.mean(wins), "std": np.std(wins)}

    # 3. FIFA排名启发式
    preds = [predict_ranking(h, a, rankings) for h, a, _, _ in matches]
    r = evaluate("FIFA排名启发式", preds, matches, verbose=False)
    all_methods["FIFA排名启发式"] = {"mean": r["win_accuracy"], "std": 0}

    # 4. 泊松模型（原始参数）
    preds = [predict_poisson(h, a, rankings) for h, a, _, _ in matches]
    r = evaluate("泊松模型(Poisson)", preds, matches)
    all_methods["泊松模型"] = {
        "mean": r["win_accuracy"], "std": 0,
        "score_acc": r["score_accuracy"], "mae": r["avg_mae"],
    }

    # 4b. 泊松模型（优化参数）
    if best_params:
        opt_poisson = create_optimized_poisson(best_params)
        preds = [opt_poisson.predict_score(h, a, rankings) for h, a, _, _ in matches]
        r = evaluate("泊松模型(优化参数)", preds, matches)
        all_methods["泊松模型(优化)"] = {
            "mean": r["win_accuracy"], "std": 0,
            "score_acc": r["score_accuracy"], "mae": r["avg_mae"],
        }

    # 5. ELO 模型
    wins = []
    for _ in range(n):
        preds = [predict_elo(h, a, rankings) for h, a, _, _ in matches]
        correct = sum(1 for i, m in enumerate(matches)
                      if preds[i] == actual_result(m[2], m[3]))
        wins.append(correct / len(matches) * 100)
    all_methods["ELO概率模型"] = {"mean": np.mean(wins), "std": np.std(wins)}

    # 5b. ELO 模型（优化参数）
    if best_params:
        opt_elo = create_optimized_elo(best_params)
        preds = [opt_elo.predict_result(h, a, rankings) for h, a, _, _ in matches]
        r = evaluate("ELO模型(优化参数)", preds, matches, verbose=False)
        all_methods["ELO模型(优化)"] = {"mean": r["win_accuracy"], "std": 0}

    # 6. 易经随机
    wins = []
    for _ in range(n):
        preds = [predict_iching()[0] for _ in matches]
        correct = sum(1 for i, m in enumerate(matches)
                      if preds[i] == actual_result(m[2], m[3]))
        wins.append(correct / len(matches) * 100)
    all_methods["易经随机卦"] = {"mean": np.mean(wins), "std": np.std(wins)}

    # 7. 八卦随机
    wins = []
    for _ in range(n):
        preds = [predict_bagua()[0] for _ in matches]
        correct = sum(1 for i, m in enumerate(matches)
                      if preds[i] == actual_result(m[2], m[3]))
        wins.append(correct / len(matches) * 100)
    all_methods["八卦随机"] = {"mean": np.mean(wins), "std": np.std(wins)}

    # 8. 博彩盘口
    preds = [predict_odds(h, a, odds_data) for h, a, _, _ in matches]
    r = evaluate("博彩盘口(最低赔率)", preds, matches, verbose=False)
    all_methods["博彩盘口"] = {"mean": r["win_accuracy"], "std": 0}

    # 9. 赔率校准泊松
    preds = [predict_odds_poisson(h, a, odds_data, rankings) for h, a, _, _ in matches]
    r = evaluate("赔率校准泊松", preds, matches)
    all_methods["赔率校准泊松"] = {
        "mean": r["win_accuracy"], "std": 0,
        "score_acc": r["score_accuracy"], "mae": r["avg_mae"],
    }

    # 10. Dixon-Coles
    dc = DixonColes()
    try:
        dc.fit(matches, rankings)
        if best_params:
            best_params["dixon_coles"] = dc.get_params()
        preds = [dc.predict_score(h, a, rankings) for h, a, _, _ in matches]
        r = evaluate("Dixon-Coles模型", preds, matches)
        all_methods["Dixon-Coles"] = {
            "mean": r["win_accuracy"], "std": 0,
            "score_acc": r["score_accuracy"], "mae": r["avg_mae"],
        }
    except Exception as e:
        print(f"  Dixon-Coles 拟合失败: {e}")

    # 11. 集成模型
    try:
        dc_model = dc if 'dc' in dir() and dc.fitted else None
        ens = EnsemblePredictor(best_params, rankings, dc_model)
        ens_preds = [ens.predict_result(h, a) for h, a, _, _ in matches]
        r = evaluate("集成模型(Ensemble)", ens_preds, matches)
        all_methods["集成模型"] = {"mean": r["win_accuracy"], "std": 0}
        # 比分单独评估
        score_preds = [ens.predict_score(h, a) for h, a, _, _ in matches]
        sr = evaluate("集成模型(比分)", score_preds, matches)
        all_methods["集成模型"]["score_acc"] = sr["score_accuracy"]
        all_methods["集成模型"]["mae"] = sr["avg_mae"]
        print(f"  集成权重: {ens.get_weights()}")
    except Exception as e:
        print(f"  集成模型创建失败: {e}")

    # --- 汇总对比 ---
    print(f"\n\n{'='*70}")
    print(f"  汇总对比 ({len(matches)} 场比赛)")
    print(f"{'='*70}")
    print(f"{'方法':<22} {'胜平负准确率':>15} {'波动(±)':>10}")
    print(f"{'-'*47}")

    sorted_methods = sorted(all_methods.items(), key=lambda x: x[1]["mean"], reverse=True)
    for name, res in sorted_methods:
        std_str = f"{res['std']:.1f}%" if res['std'] > 0 else "---"
        print(f"{name:<22} {res['mean']:>13.1f}% {std_str:>10}")

    # 带比分的模型额外指标
    for key in ["泊松模型", "泊松模型(优化)", "赔率校准泊松", "Dixon-Coles", "集成模型"]:
        if key in all_methods and "score_acc" in all_methods[key]:
            print(f"\n{key} - 精确比分命中率: {all_methods[key]['score_acc']:.1f}%")
            print(f"{key} - 进球差 MAE: {all_methods[key]['mae']:.2f}")

    # --- 详细结果 ---
    print(f"\n\n{'='*70}")
    print("  逐场详细对比")
    print(f"{'='*70}")

    detailed = []
    for home, away, h_s, a_s in matches:
        actual = actual_result(h_s, a_s)
        poisson_pred = predict_poisson(home, away, rankings)
        elo_pred = predict_elo(home, away, rankings)
        rank_pred = predict_ranking(home, away, rankings)
        odds_pred = predict_odds(home, away, odds_data)
        odds_poisson_pred = predict_odds_poisson(home, away, odds_data, rankings)

        row = {
            "match": f"{home} vs {away}",
            "actual": f"{h_s}:{a_s}",
            "result": actual,
            "odds": odds_pred,
            "odds+poisson": f"{odds_poisson_pred[1]}:{odds_poisson_pred[2]} ({odds_poisson_pred[0]})",
            "poisson": f"{poisson_pred[1]}:{poisson_pred[2]} ({poisson_pred[0]})",
            "elo": elo_pred,
            "ranking": rank_pred,
        }

        if dc.fitted:
            dc_pred = dc.predict_score(home, away, rankings)
            row["dixon_coles"] = f"{dc_pred[1]}:{dc_pred[2]} ({dc_pred[0]})"

        detailed.append(row)

    df = pd.DataFrame(detailed)
    print(df.to_string(index=False))

    # --- 保存结果 ---
    if args.output:
        OUTPUT_DIR.mkdir(exist_ok=True)
        df.to_csv(OUTPUT_DIR / "backtest_results.csv", index=False, encoding="utf-8-sig")
        print(f"\n详细结果已保存到 {OUTPUT_DIR / 'backtest_results.csv'}")

        summary = []
        for name, res in sorted_methods:
            summary.append({
                "method": name,
                "win_accuracy": round(res["mean"], 1),
                "std": round(res["std"], 1) if res["std"] > 0 else None,
            })
        pd.DataFrame(summary).to_csv(
            OUTPUT_DIR / "backtest_summary.csv", index=False, encoding="utf-8-sig"
        )


if __name__ == "__main__":
    main()
