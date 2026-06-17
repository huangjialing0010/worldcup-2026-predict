"""
2026 世界杯比赛预测
使用优化参数 + Dixon-Coles + 集成模型进行预测
"""
import pandas as pd
import numpy as np
from pathlib import Path
import argparse

from model_utils import (
    load_rankings, load_matches,
    rank_to_strength, poisson_best_score, poisson_probabilities,
)
from models import (
    OptimizedPoisson, OptimizedELO, DixonColes,
    load_best_params, create_optimized_poisson, create_optimized_elo,
)
from ensemble import EnsemblePredictor

ROOT = Path(__file__).parent.parent
DATA_PROCESSED = ROOT / "data" / "processed"
OUTPUT_DIR = ROOT / "output"


def load_upcoming_matches():
    path = DATA_PROCESSED / "matches.csv"
    if path.exists():
        df = pd.read_csv(path, encoding="utf-8-sig")
        upcoming = df[df["is_finished"] != True]
        if not upcoming.empty:
            return list(zip(upcoming["home_team"], upcoming["away_team"],
                            upcoming["round"] if "round" in upcoming.columns else [""] * len(upcoming)))
    return []


def main():
    parser = argparse.ArgumentParser(description="2026 世界杯比赛预测")
    parser.add_argument("--upcoming", action="store_true",
                        help="预测 data/processed/ 中所有未赛比赛")
    parser.add_argument("--match", nargs=2, metavar=("HOME", "AWAY"),
                        help="预测指定对阵")
    parser.add_argument("--all-models", action="store_true",
                        help="同时显示所有模型的预测结果")
    args = parser.parse_args()

    rankings = load_rankings()
    matches = load_matches()
    best_params = load_best_params()

    # 创建模型
    poisson = create_optimized_poisson(best_params)
    elo = create_optimized_elo(best_params)

    dc = DixonColes()
    dc_fitted = False
    try:
        dc.fit(matches, rankings)
        dc_fitted = True
        if best_params:
            best_params["dixon_coles"] = dc.get_params()
    except Exception as e:
        print(f"Dixon-Coles 拟合失败: {e}")

    ens = EnsemblePredictor(best_params, rankings)

    # 确定对阵
    if args.match:
        matchups = [(args.match[0], args.match[1], "")]
    elif args.upcoming:
        matchups = load_upcoming_matches()
        if not matchups:
            print("没有未赛的比赛。试试 --match 'Team A' 'Team B'")
            return
    else:
        parser.print_help()
        print("\n用法示例:")
        print("  python scripts/predict.py --match 'Brazil' 'Germany'")
        print("  python scripts/predict.py --upcoming")
        print("  python scripts/predict.py --match 'Brazil' 'Germany' --all-models")
        return

    print(f"{'='*70}")
    print(f"  2026 世界杯比赛预测")
    if best_params:
        print(f"  使用优化参数 (泊松 CV={best_params.get('poisson_cv_accuracy', '?')}%)")
    print(f"{'='*70}")
    print()

    predictions = []
    for home, away, round_ in matchups:
        print(f"  {home} vs {away}")
        if round_:
            print(f"  轮次: {round_}")
        print(f"  {'─'*50}")

        # 集成模型（主预测）
        ens_proba = ens.predict_proba(home, away)
        ens_result = ens.predict_result(home, away)
        ens_score = ens.predict_score(home, away)
        result_labels = {"H": "主胜", "D": "平局", "A": "客胜"}

        print(f"  ★ 集成预测: {ens_score[1]}:{ens_score[2]} ({result_labels.get(ens_result, ens_result)})")
        print(f"    概率: H={ens_proba[0]:.1%} D={ens_proba[1]:.1%} A={ens_proba[2]:.1%}")
        print(f"    权重: {ens.get_weights()}")

        if args.all_models:
            # 各子模型详细预测
            poisson_score = poisson.predict_score(home, away, rankings)
            poisson_proba = poisson.predict_proba(home, away, rankings)
            print(f"\n  泊松(优化): {poisson_score[1]}:{poisson_score[2]} "
                  f"H={poisson_proba[0]:.1%} D={poisson_proba[1]:.1%} A={poisson_proba[2]:.1%}")

            elo_proba = elo.predict_proba(home, away, rankings)
            elo_result = elo.predict_result(home, away, rankings)
            print(f"  ELO(优化):   {result_labels.get(elo_result, elo_result)} "
                  f"H={elo_proba[0]:.1%} D={elo_proba[1]:.1%} A={elo_proba[2]:.1%}")

            if dc_fitted:
                dc_score = dc.predict_score(home, away, rankings)
                dc_proba = dc.predict_proba(home, away, rankings)
                print(f"  Dixon-Coles: {dc_score[1]}:{dc_score[2]} "
                      f"H={dc_proba[0]:.1%} D={dc_proba[1]:.1%} A={dc_proba[2]:.1%}")

        print()

        predictions.append({
            "match": f"{home} vs {away}",
            "round": round_,
            "ensemble_result": ens_result,
            "ensemble_score": f"{ens_score[1]}:{ens_score[2]}",
            "ensemble_prob_h": round(ens_proba[0] * 100, 1),
            "ensemble_prob_d": round(ens_proba[1] * 100, 1),
            "ensemble_prob_a": round(ens_proba[2] * 100, 1),
        })

    # 保存
    if predictions:
        OUTPUT_DIR.mkdir(exist_ok=True)
        df = pd.DataFrame(predictions)
        out_path = OUTPUT_DIR / "predictions.csv"
        df.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"预测结果已保存到 {out_path}")


if __name__ == "__main__":
    main()
