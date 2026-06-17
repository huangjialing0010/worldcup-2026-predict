"""
集成预测模型 — 加权软投票组合泊松 + Dixon-Coles + ELO
"""
import numpy as np
from pathlib import Path
from model_utils import (
    load_rankings, load_odds_data, implied_probabilities,
)
from models import (
    OptimizedPoisson, OptimizedELO, DixonColes,
    load_best_params, create_optimized_poisson, create_optimized_elo,
)

ROOT = Path(__file__).parent.parent


class EnsemblePredictor:
    """多模型集成 — 50/50 概率加权平均（Poisson + ELO）

    仅包含泊松和ELO两个互补模型。Dixon-Coles在回测中表现不稳定（43.8%），
    不纳入集成。简单等权平均比基于CV准确率的权重法更稳健（全量数据验证：52.1% vs 50.1%）。
    """

    def __init__(self, params=None, rankings=None, dc_model=None):
        if params is None:
            params = load_best_params()
        if rankings is None:
            rankings = load_rankings()

        self.rankings = rankings
        self.params = params

        self.poisson = create_optimized_poisson(params)
        self.elo = create_optimized_elo(params)
        self.dixon_coles = dc_model  # 保留兼容性，不参与集成

    def predict_proba(self, home, away):
        """Poisson + ELO 等权平均"""
        ph_p, pd_p, pa_p = self.poisson.predict_proba(home, away, self.rankings)
        ph_e, pd_e, pa_e = self.elo.predict_proba(home, away, self.rankings)
        return (
            (ph_p + ph_e) / 2.0,
            (pd_p + pd_e) / 2.0,
            (pa_p + pa_e) / 2.0,
        )

    def predict_result(self, home, away):
        p_h, p_d, p_a = self.predict_proba(home, away)
        best = max(p_h, p_d, p_a)
        return "H" if best == p_h else ("D" if best == p_d else "A")

    def predict_score(self, home, away):
        """用泊松模型预测比分"""
        return self.poisson.predict_score(home, away, self.rankings)

    def get_weights(self):
        return {"poisson": 0.5, "elo": 0.5}


def create_ensemble(params=None, rankings=None, dc_model=None):
    return EnsemblePredictor(params, rankings, dc_model)


def main():
    from model_utils import load_matches, actual_result

    rankings = load_rankings()
    matches = load_matches()

    print("拟合 Dixon-Coles 模型...")
    dc = DixonColes()
    dc.fit(matches, rankings)
    print(f"  参数: {dc.get_params()}")

    # 计算子模型 CV 准确率
    from models import create_optimized_poisson, create_optimized_elo
    poisson = create_optimized_poisson()
    elo = create_optimized_elo()

    poisson_acc = sum(1 for h, a, hs, as_ in matches
                      if poisson.predict_result(h, a, rankings) == actual_result(hs, as_)) / len(matches) * 100
    elo_acc = sum(1 for h, a, hs, as_ in matches
                  if elo.predict_result(h, a, rankings) == actual_result(hs, as_)) / len(matches) * 100
    dc_acc = sum(1 for h, a, hs, as_ in matches
                 if dc.predict_result(h, a, rankings) == actual_result(hs, as_)) / len(matches) * 100

    print(f"  泊松: {poisson_acc:.1f}%")
    print(f"  ELO:  {elo_acc:.1f}%")
    print(f"  DC:   {dc_acc:.1f}%")

    # 构建参数并创建集成
    params = load_best_params() or {}
    params["poisson_cv_accuracy"] = round(poisson_acc, 1)
    params["elo_cv_accuracy"] = round(elo_acc, 1)
    params["dixon_coles_cv_accuracy"] = round(dc_acc, 1)

    ensemble = create_ensemble(params, rankings, dc)
    print(f"\n集成权重: {ensemble.get_weights()}")

    # 评估集成
    correct = sum(1 for h, a, hs, as_ in matches
                  if ensemble.predict_result(h, a) == actual_result(hs, as_))
    ens_acc = correct / len(matches) * 100
    print(f"集成准确率: {ens_acc:.1f}%")

    # 示例预测
    print(f"\n{'=' * 60}")
    print("  示例预测（前 5 场）")
    print(f"{'=' * 60}")
    for home, away, h_s, a_s in matches[:5]:
        ph, pd, pa = ensemble.predict_proba(home, away)
        result = ensemble.predict_result(home, away)
        score = ensemble.predict_score(home, away)
        print(f"  {home} vs {away}: {score[1]}:{score[2]} ({result}) "
              f"[H:{ph:.0%} D:{pd:.0%} A:{pa:.0%}]  实际: {h_s}:{a_s}")


if __name__ == "__main__":
    main()
