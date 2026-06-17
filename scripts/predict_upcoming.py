"""
2026 世界杯赛程预测
"""
import sys; sys.path.insert(0, 'scripts')
import numpy as np
from model_utils import load_rankings
from models import OptimizedPoisson, OptimizedELO, load_best_params

rankings = load_rankings()
params = load_best_params()

e_keys = {'rank_weight','home_bias','draw_coeff','draw_decay','draw_power','neutral_venue'}
elo_p = {k:v for k,v in params['elo'].items() if k in e_keys}
elo_n = {k:v for k,v in params['elo_neutral'].items() if k in e_keys}

poisson = OptimizedPoisson(**params['poisson'])
elo = OptimizedELO(**elo_p)
elo_neutral = OptimizedELO(**elo_n)

# Matchday 1: Groups I-L (June 16-17) + Matchday 2: Groups A-B (June 18)
fixtures = [
    # (home, away, date, group)
    ("France", "Senegal", "6/16", "I"),
    ("Iraq", "Norway", "6/16", "I"),
    ("Argentina", "Algeria", "6/16", "J"),
    ("Austria", "Jordan", "6/16", "J"),
    ("Portugal", "DR Congo", "6/17", "K"),
    ("England", "Croatia", "6/17", "L"),
    ("Ghana", "Panama", "6/17", "L"),
    ("Uzbekistan", "Colombia", "6/17", "K"),
    ("Czech Republic", "South Africa", "6/18", "A"),
    ("Switzerland", "Bosnia and Herzegovina", "6/18", "B"),
    ("Canada", "Qatar", "6/18", "B"),
    ("Mexico", "South Korea", "6/18", "A"),
]

print(f"{'='*75}")
print(f"  2026 世界杯比分预测 (参数优化 + 中立场感知)")
print(f"{'='*75}")
print(f"  {'比赛':<32} {'ELO':>8} {'泊松':>8} {'集成':>8}  | 概率(H/D/A)")
print(f"  {'─'*75}")

results = []
for home, away, date, grp in fixtures:
    rk_h = rankings.get(home, 50)
    rk_a = rankings.get(away, 50)

    # ELO (neutral)
    ph_e, pd_e, pa_e = elo_neutral.predict_proba(home, away, rankings)
    pred_e = elo_neutral.predict_result(home, away, rankings)

    # Poisson
    ph_p, pd_p, pa_p = poisson.predict_proba(home, away, rankings)
    pred_p = poisson.predict_result(home, away, rankings)

    # Ensemble
    ph_en = (ph_e + ph_p) / 2
    pd_en = (pd_e + pd_p) / 2
    pa_en = (pa_e + pa_p) / 2
    best = max(ph_en, pd_en, pa_en)
    pred_en = "H" if best == ph_en else ("D" if best == pd_en else "A")

    labels = {"H": "主胜", "D": "平局", "A": "客胜"}

    # 预期比分 (Poisson)
    from model_utils import poisson_best_score
    lh, la = poisson.get_lambdas(home, away, rankings)
    score = poisson_best_score(lh, la)

    print(f"  {home} vs {away} ({grp}组 {date})")
    print(f"    ELO: {labels.get(pred_e,pred_e)}  泊松: {labels.get(pred_p,pred_p)}  集成: {labels.get(pred_en,pred_en)}")
    print(f"    比分预测: {score[1]}:{score[2]}  |  H:{ph_en:.0%} D:{pd_en:.0%} A:{pa_en:.0%}")
    print()

    results.append({
        "match": f"{home} vs {away}",
        "group": grp,
        "date": date,
        "elo": labels.get(pred_e, pred_e),
        "poisson": labels.get(pred_p, pred_p),
        "ensemble": labels.get(pred_en, pred_en),
        "score": f"{score[1]}:{score[2]}",
        "prob_h": round(ph_en*100, 1),
        "prob_d": round(pd_en*100, 1),
        "prob_a": round(pa_en*100, 1),
    })

# 已结束比赛的结果（从搜索结果已知）
print(f"  {'='*75}")
print(f"  已知赛果 (6/16-17 已赛)")
print(f"  {'='*75}")
known = {
    ("France", "Senegal"): "France 3-1 Senegal",
    # 其他比赛需查结果
}
for (h, a), result in known.items():
    print(f"  {h} vs {a}: {result}")

# 保存
import pandas as pd
pd.DataFrame(results).to_csv('output/upcoming_predictions.csv', index=False, encoding='utf-8-sig')
print(f"\n预测保存到 output/upcoming_predictions.csv")
