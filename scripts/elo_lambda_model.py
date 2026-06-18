"""
Elo → Lambda 直接映射模型
log(lambda_h) = alpha + beta*(elo_h-elo_a) + gamma  (home advantage)
log(lambda_a) = alpha - beta*(elo_h-elo_a)

Elo差距400分 = lambda比率 exp(2*beta)，目标~3:1
+ Dixon-Coles tau for draw correlation
+ 直接用features.csv的Elo（通过比赛历史计算，已包含对手强度调整）
"""
import sys; sys.path.insert(0, 'scripts')
import numpy as np, pandas as pd
from scipy import stats
from scipy.optimize import minimize
from pathlib import Path
import json, warnings
warnings.filterwarnings('ignore')

ROOT = Path(__file__).parent.parent
OUTPUT = ROOT / "output"

# ============================================================
# Load
# ============================================================
features = pd.read_csv(OUTPUT / "features.csv", encoding="utf-8-sig")
features["date"] = pd.to_datetime(features["date"])
train = features[features["date"] >= "2022-01-01"].copy()
print(f"Training: {len(train)} matches (2022+)")

wc_df = pd.read_csv(ROOT / "data" / "raw" / "matches_2026.csv", encoding="utf-8-sig")

# Use FIFA rankings (stable, reliable) instead of dynamic Elo
from model_utils import load_rankings
rankings = load_rankings()
# For goal prediction, rank is a proxy for strength
# Convert: lower rank = stronger → strength = -rank (so higher = stronger)
# Or just use rank directly in the formula with appropriate sign

print(f"FIFA rank range: {min(rankings.values())} - {max(rankings.values())}")

# ============================================================
# Training arrays (use FIFA rank instead of Elo)
# ============================================================
# rank diff: (rank_away - rank_home) > 0 means home is stronger (lower rank = better)
train_rd = np.array([
    rankings.get(a, 50) - rankings.get(h, 50)
    for h, a in zip(train["home_team"], train["away_team"])
], dtype=np.float64)
train_hg = train["home_goals"].values.astype(int)
train_ag = train["away_goals"].values.astype(int)

print(f"Rank diff range: {train_rd.min():.0f} to {train_rd.max():.0f}")

# ============================================================
# DC neg log likelihood
# ============================================================
def neg_loglik(params):
    alpha, beta, gamma, rho = params

    if alpha <= -3 or alpha >= 2: return 1e10
    if beta <= 0 or beta >= 10: return 1e10
    if gamma <= -0.5 or gamma >= 1.0: return 1e10
    if abs(rho) >= 0.1: return 1e10

    rd = train_rd / 100.0  # normalized rank diff
    log_lh = alpha + beta * rd + gamma
    log_la = alpha - beta * rd
    lh = np.exp(np.clip(log_lh, -5, 5))
    la = np.exp(np.clip(log_la, -5, 5))

    ll = 0.0
    for i in range(len(train)):
        hg, ag = train_hg[i], train_ag[i]
        lh_i, la_i = lh[i], la[i]
        if lh_i <= 0 or la_i <= 0: return 1e10

        prob = stats.poisson.pmf(hg, lh_i) * stats.poisson.pmf(ag, la_i)

        # DC tau
        if hg == 0 and ag == 0:      prob *= (1 - lh_i * la_i * rho)
        elif hg == 0 and ag == 1:    prob *= (1 + lh_i * rho)
        elif hg == 1 and ag == 0:    prob *= (1 + la_i * rho)
        elif hg == 1 and ag == 1:    prob *= (1 - rho)

        if prob <= 0: return 1e10
        ll += np.log(prob)

    return -ll

# ============================================================
# Fit
# ============================================================
print("\nFitting Rank→Lambda model...")
result = minimize(neg_loglik, [0.3, 1.5, 0.1, 0.0],
                  method="L-BFGS-B",
                  bounds=[(-3, 2), (0.01, 10), (-0.5, 1.0), (-0.08, 0.08)],
                  options={"maxiter": 2000})

alpha, beta, gamma, rho = result.x
print(f"alpha={alpha:.4f}  beta={beta:.4f}  gamma={gamma:.4f}  rho={rho:.6f}")
print(f"Converged: {result.success}")

# Show what this means
rd_test = np.array([0, 20, 40, 60, 80])  # rank difference
print("\nExpected goals by rank gap:")
print(f"{'Rank gap':>10} {'λ_home':>8} {'λ_away':>8} {'ratio':>8}")
for gap in rd_test:
    lh = np.exp(alpha + beta * gap / 100 + gamma)
    la = np.exp(alpha - beta * gap / 100)
    print(f"{gap:>10.0f} {lh:>8.2f} {la:>8.2f} {lh/la:>8.2f}")

# ============================================================
# Predict function
# ============================================================
def predict(home, away, max_g=10):
    rk_h = rankings.get(home, 50)
    rk_a = rankings.get(away, 50)
    rd_raw = (rk_a - rk_h) / 100.0  # positive = home stronger
    rd = np.tanh(rd_raw * 3.0) / 3.0  # soft saturation

    lh = np.exp(alpha + beta * rd + gamma)
    la = np.exp(alpha - beta * rd)
    lh = np.clip(lh, 0.05, 15.0)
    la = np.clip(la, 0.05, 15.0)

    p_h, p_d, p_a = 0.0, 0.0, 0.0
    best_prob, best_h, best_a = -1, 0, 0

    for i in range(max_g + 1):
        for j in range(max_g + 1):
            prob = stats.poisson.pmf(i, lh) * stats.poisson.pmf(j, la)
            if i == 0 and j == 0:      prob *= (1 - lh * la * rho)
            elif i == 0 and j == 1:    prob *= (1 + lh * rho)
            elif i == 1 and j == 0:    prob *= (1 + la * rho)
            elif i == 1 and j == 1:    prob *= (1 - rho)

            if i > j: p_h += prob
            elif i == j: p_d += prob
            else: p_a += prob

            if prob > best_prob:
                best_prob, best_h, best_a = prob, i, j

    total = p_h + p_d + p_a
    if total > 0: p_h /= total; p_d /= total; p_a /= total

    result = "H" if p_h >= max(p_d, p_a) else ("D" if p_d >= max(p_h, p_a) else "A")
    return result, best_h, best_a, (p_h, p_d, p_a), (lh, la)


# ============================================================
# Backtest 2026 WC
# ============================================================
print(f"\n{'='*70}")
print("  2026 World Cup Backtest — Rank→Lambda DC Model")
print(f"{'='*70}")

rl = {'H': 'H', 'D': 'DRAW', 'A': 'A'}
correct = exact = mae = 0
draw_total = draw_correct = 0

for _, row in wc_df.iterrows():
    home, away = row["home_team"], row["away_team"]
    hg, ag = int(row["home_score"]), int(row["away_score"])
    act = "H" if hg > ag else ("D" if hg == ag else "A")

    result, ph, pa, probs, (lh, la) = predict(home, away)
    prob_h, prob_d, prob_a = probs

    ok = "OK" if result == act else "XX"
    if result == act: correct += 1
    if act == "D":
        draw_total += 1
        if result == "D": draw_correct += 1
    if ph == hg and pa == ag: exact += 1
    mae += abs(ph - hg) + abs(pa - ag)

    rk_h = rankings.get(home, 50)
    rk_a = rankings.get(away, 50)
    print(f"  {home:<15} vs {away:<15} {hg}:{ag} ({rl[act]:>4})  pred:{rl[result]:>5} {ph}:{pa} {ok}  [{prob_h:.0%}/{prob_d:.0%}/{prob_a:.0%}]  R:{rk_h}v{rk_a}  λ:{lh:.1f}v{la:.1f}")

n = len(wc_df)
print(f"\n  Result:  {correct}/{n} = {correct/n*100:.1f}%")
print(f"  Score:   {exact}/{n} = {exact/n*100:.1f}%")
print(f"  MAE:     {mae/n:.2f} goals/match")
print(f"  Draws:   {draw_correct}/{draw_total} = {draw_correct/draw_total*100:.1f}%")

# Compare with old ensemble
print(f"\n  vs Old Ensemble: 55.6% result, 1.50 MAE")
print(f"  vs Old ELO:      44.4% result, -- MAE")
print(f"  λ range: {np.exp(alpha + gamma + beta*0.8):.1f} - {np.exp(alpha + gamma + beta*(-0.6)):.1f}")

# Save
json.dump({
    "alpha": round(float(alpha), 4),
    "beta": round(float(beta), 4),
    "gamma": round(float(gamma), 4),
    "rho": round(float(rho), 6),
    "description": "FIFA rank → lambda model with DC tau correction",
}, open(OUTPUT / "rank_lambda_model.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("Saved to output/rank_lambda_model.json")
