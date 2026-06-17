"""
Dixon-Coles with DIRECT team-level scoring rates (no rank compression)
直接用历史数据算每队的进球率/失球率 + Bayesian shrinkage + DC tau
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
# Load data
# ============================================================
features = pd.read_csv(OUTPUT / "features.csv", encoding="utf-8-sig")
features["date"] = pd.to_datetime(features["date"])
# Use 2022+ for recency, weight World Cup higher
train = features[features["date"] >= "2022-01-01"].copy()
# Double-weight WC and continental championship matches
is_important = train["round"].isin({
    "FIFA World Cup", "UEFA Euro", "Copa América", "AFC Asian Cup",
    "African Cup of Nations", "Gold Cup"
})
# Duplicate important matches
train_imp = train[is_important]
train = pd.concat([train, train_imp])  # 2x weight for important matches
print(f"Training: {len(train)} matches (2022+, important matches 2x)")

# Get WC 2026 teams
wc_df = pd.read_csv(ROOT / "data" / "raw" / "matches_2026.csv", encoding="utf-8-sig")
wc_teams = sorted(set(wc_df["home_team"].unique()) | set(wc_df["away_team"].unique()))
print(f"WC teams: {len(wc_teams)}")

# ============================================================
# Direct team-level scoring rates
# ============================================================
# For each team: avg goals scored per match, avg goals conceded per match
# Shrink toward league average (Bayesian with beta prior equivalent)

all_teams = sorted(set(train["home_team"].unique()) | set(train["away_team"].unique()))

# League averages
all_goals = pd.concat([train["home_goals"], train["away_goals"]])
LEAGUE_AVG = all_goals.mean()
print(f"League avg goals: {LEAGUE_AVG:.2f}")

# Compute per-team rates
team_gf = {}  # goals for
team_ga = {}  # goals against
team_n = {}   # number of matches
for t in all_teams:
    home_m = train[train["home_team"] == t]
    away_m = train[train["away_team"] == t]
    gf = home_m["home_goals"].sum() + away_m["away_goals"].sum()
    ga = home_m["away_goals"].sum() + away_m["home_goals"].sum()
    n = len(home_m) + len(away_m)
    team_gf[t] = gf / n if n > 0 else LEAGUE_AVG
    team_ga[t] = ga / n if n > 0 else LEAGUE_AVG
    team_n[t] = n

# Bayesian shrinkage: shrink small-sample teams toward league average
# shrinkage weight = prior_strength / (prior_strength + n)
PRIOR_STRENGTH = 10  # equivalent to 10 matches of prior

for t in all_teams:
    n = team_n[t]
    w = PRIOR_STRENGTH / (PRIOR_STRENGTH + n)  # shrinkage weight
    team_gf[t] = w * LEAGUE_AVG + (1 - w) * team_gf[t]
    team_ga[t] = w * LEAGUE_AVG + (1 - w) * team_ga[t]

print(f"GF range: [{min(team_gf.values()):.2f}, {max(team_gf.values()):.2f}]")
print(f"GA range: [{min(team_ga.values()):.2f}, {max(team_ga.values()):.2f}]")

# ============================================================
# Dixon-Coles parameters to optimize
# ============================================================
# We optimize global gamma (home advantage) and rho (low-score correlation)
# Team lambdas: lh = gf_home * ga_away / LEAGUE_AVG * gamma
#               la = gf_away * ga_home / LEAGUE_AVG

# 为了进一步拉开差距，引入一个 power 参数
# lh = LEAGUE_AVG * (gf_home/LEAGUE_AVG)^power_att * (ga_away/LEAGUE_AVG)^power_def * gamma
# la = LEAGUE_AVG * (gf_away/LEAGUE_AVG)^power_att * (ga_home/LEAGUE_AVG)^power_def

def compute_lambdas(gf_h, ga_h, gf_a, ga_a, gamma, power_att, power_def):
    """Team strength to expected goals"""
    att_h = (gf_h / LEAGUE_AVG) ** power_att
    def_a = (ga_a / LEAGUE_AVG) ** power_def
    att_a = (gf_a / LEAGUE_AVG) ** power_att
    def_h = (ga_h / LEAGUE_AVG) ** power_def
    lh = LEAGUE_AVG * att_h * def_a * gamma
    la = LEAGUE_AVG * att_a * def_h
    return lh, la

# ============================================================
# Fit gamma, rho, power_att, power_def
# ============================================================
def neg_loglik(params):
    gamma, rho, p_att, p_def = params
    if gamma <= 0.7 or gamma >= 1.8: return 1e10
    if abs(rho) >= 0.08: return 1e10
    if p_att <= 0.2 or p_att >= 4.0: return 1e10
    if p_def <= 0.2 or p_def >= 4.0: return 1e10

    ll = 0.0
    for _, row in train.iterrows():
        h, a = row["home_team"], row["away_team"]
        hg, ag = int(row["home_goals"]), int(row["away_goals"])

        gfh = team_gf.get(h, LEAGUE_AVG)
        gah = team_ga.get(h, LEAGUE_AVG)
        gfa = team_gf.get(a, LEAGUE_AVG)
        gaa = team_ga.get(a, LEAGUE_AVG)

        lh, la = compute_lambdas(gfh, gah, gfa, gaa, gamma, p_att, p_def)
        lh = np.clip(lh, 0.02, 15.0)
        la = np.clip(la, 0.02, 15.0)

        prob = stats.poisson.pmf(hg, lh) * stats.poisson.pmf(ag, la)

        # DC tau
        if hg == 0 and ag == 0:
            prob *= (1 - lh * la * rho)
        elif hg == 0 and ag == 1:
            prob *= (1 + lh * rho)
        elif hg == 1 and ag == 0:
            prob *= (1 + la * rho)
        elif hg == 1 and ag == 1:
            prob *= (1 - rho)

        if prob <= 0: return 1e10
        ll += np.log(prob)

    return -ll

print("\nFitting gamma, rho, power_att, power_def...")
result = minimize(neg_loglik, [1.10, 0.0, 1.0, 1.0],
                  method="L-BFGS-B",
                  bounds=[(0.7, 1.8), (-0.06, 0.06), (0.2, 4.0), (0.2, 4.0)],
                  options={"maxiter": 2000})

gamma, rho, p_att, p_def = result.x
print(f"gamma={gamma:.4f}  rho={rho:.6f}  power_att={p_att:.3f}  power_def={p_def:.3f}")
print(f"Converged: {result.success}")

# ============================================================
# Predict function
# ============================================================
def predict_dc(home, away, max_g=10):
    gfh = team_gf.get(home, LEAGUE_AVG)
    gah = team_ga.get(home, LEAGUE_AVG)
    gfa = team_gf.get(away, LEAGUE_AVG)
    gaa = team_ga.get(away, LEAGUE_AVG)

    lh, la = compute_lambdas(gfh, gah, gfa, gaa, gamma, p_att, p_def)
    lh = np.clip(lh, 0.05, 15.0)
    la = np.clip(la, 0.05, 15.0)

    p_h, p_d, p_a = 0.0, 0.0, 0.0
    best_prob, best_h, best_a = -1, 0, 0

    for i in range(max_g + 1):
        for j in range(max_g + 1):
            prob = stats.poisson.pmf(i, lh) * stats.poisson.pmf(j, la)
            if i == 0 and j == 0:
                prob *= (1 - lh * la * rho)
            elif i == 0 and j == 1:
                prob *= (1 + lh * rho)
            elif i == 1 and j == 0:
                prob *= (1 + la * rho)
            elif i == 1 and j == 1:
                prob *= (1 - rho)

            if i > j: p_h += prob
            elif i == j: p_d += prob
            else: p_a += prob
            if prob > best_prob:
                best_prob, best_h, best_a = prob, i, j

    total = p_h + p_d + p_a
    if total > 0: p_h /= total; p_d /= total; p_a /= total

    best = max(p_h, p_d, p_a)
    result = "H" if best == p_h else ("D" if best == p_d else "A")
    return result, best_h, best_a, (p_h, p_d, p_a), (lh, la)


# ============================================================
# Backtest on 2026 World Cup
# ============================================================
print(f"\n{'='*70}")
print("  2026 World Cup Backtest — DC Direct Team Rates")
print(f"{'='*70}")

rl = {'H': 'H', 'D': 'DRAW', 'A': 'A'}

correct = exact = mae = 0
draw_total = draw_correct = 0
from model_utils import actual_result as act_fn

for _, row in wc_df.iterrows():
    home, away = row["home_team"], row["away_team"]
    hg, ag = int(row["home_score"]), int(row["away_score"])
    act = act_fn(hg, ag)

    result, ph, pa, probs, (lh, la) = predict_dc(home, away)
    prob_h, prob_d, prob_a = probs

    ok = "OK" if result == act else "XX"
    if result == act: correct += 1
    if act == "D":
        draw_total += 1
        if result == "D": draw_correct += 1
    if ph == hg and pa == ag: exact += 1
    mae += abs(ph - hg) + abs(pa - ag)

    gfh = team_gf.get(home, LEAGUE_AVG)
    gfa = team_gf.get(away, LEAGUE_AVG)

    print(f"  {home:<15} vs {away:<15} {hg}:{ag} ({rl[act]:>4})  pred: {rl[result]:>4} {ph}:{pa} {ok}  [{prob_h:.0%}/{prob_d:.0%}/{prob_a:.0%}]  gf:{gfh:.1f}v{gfa:.1f}")

n = len(wc_df)
print(f"\n  Result:  {correct}/{n} = {correct/n*100:.1f}%")
print(f"  Score:   {exact}/{n} = {exact/n*100:.1f}%")
print(f"  MAE:     {mae/n:.2f} goals/match")
print(f"  Draws:   {draw_correct}/{draw_total} = {draw_correct/draw_total*100:.1f}%")

# Top attacks
print(f"\nTop 10 attacks (goals/match):")
for t, v in sorted(team_gf.items(), key=lambda x: x[1], reverse=True)[:10]:
    n_matches = team_n.get(t, 0)
    print(f"  {t}: {v:.2f} ({n_matches} matches)")

# Save
json.dump({
    "gamma": round(float(gamma), 4),
    "rho": round(float(rho), 6),
    "power_att": round(float(p_att), 3),
    "power_def": round(float(p_def), 3),
    "league_avg": round(float(LEAGUE_AVG), 2),
    "teams": {t: {"gf": round(team_gf[t], 2), "ga": round(team_ga[t], 2), "n": team_n[t]}
              for t in wc_teams},
}, open(OUTPUT / "dc_team_model.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("Saved to output/dc_team_model.json")
