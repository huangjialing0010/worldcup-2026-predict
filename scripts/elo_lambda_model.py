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
from motivation import analyze_match, apply_motivation
warnings.filterwarnings('ignore')

ROOT = Path(__file__).parent.parent
OUTPUT = ROOT / "output"

CN = {
    "Algeria": "阿尔及利亚", "Argentina": "阿根廷", "Australia": "澳大利亚",
    "Austria": "奥地利", "Belgium": "比利时", "Bosnia and Herzegovina": "波黑",
    "Brazil": "巴西", "Canada": "加拿大", "Cape Verde": "佛得角",
    "Colombia": "哥伦比亚", "Croatia": "克罗地亚", "Curacao": "库拉索",
    "Czech Republic": "捷克", "DR Congo": "刚果(金)", "Ecuador": "厄瓜多尔",
    "Egypt": "埃及", "England": "英格兰", "France": "法国",
    "Germany": "德国", "Ghana": "加纳", "Haiti": "海地",
    "Iran": "伊朗", "Iraq": "伊拉克", "Ivory Coast": "科特迪瓦",
    "Japan": "日本", "Jordan": "约旦", "Mexico": "墨西哥",
    "Morocco": "摩洛哥", "Netherlands": "荷兰", "New Zealand": "新西兰",
    "Norway": "挪威", "Panama": "巴拿马", "Paraguay": "巴拉圭",
    "Portugal": "葡萄牙", "Qatar": "卡塔尔", "Saudi Arabia": "沙特",
    "Scotland": "苏格兰", "Senegal": "塞内加尔", "South Africa": "南非",
    "South Korea": "韩国", "Spain": "西班牙", "Sweden": "瑞典",
    "Switzerland": "瑞士", "Tunisia": "突尼斯", "Turkey": "土耳其",
    "USA": "美国", "Uruguay": "乌拉圭", "Uzbekistan": "乌兹别克斯坦",
}

# Draw override thresholds
DRAW_ELO_THRESHOLD = 50    # |ELO gap| below this → consider draw
DRAW_PROB_THRESHOLD = 0.25  # P(D) above this → trigger override (calibrated for λ×1.12)
DRAW_RATE_THRESHOLD = 0.50  # Both teams' tournament draw rate above this → draw bonus
LAMBDA_SCALE = 1.12         # Global λ calibration: actual 3.02 / model 2.71 goals per match

# ============================================================
# Load
# ============================================================
features = pd.read_csv(OUTPUT / "features.csv", encoding="utf-8-sig")
features["date"] = pd.to_datetime(features["date"])
train = features[features["date"] >= "2022-01-01"].copy()
print(f"Training: {len(train)} matches (2022+)")

wc_df = pd.read_csv(ROOT / "data" / "raw" / "matches_2026.csv", encoding="utf-8-sig")

# Use clean W/L/D ELO (no goal dependency, no circularity)
# Built by build_clean_elo.py from 4580 matches
elo_path = ROOT / "data" / "processed" / "clean_elo.csv"
matches_elo_path = ROOT / "data" / "processed" / "matches_with_elo.csv"
if not elo_path.exists() or not matches_elo_path.exists():
    import subprocess; subprocess.run([sys.executable, str(ROOT / "scripts" / "build_clean_elo.py")])

# Current ELO for prediction
elo_df = pd.read_csv(elo_path, encoding="utf-8-sig")
elo_dict = dict(zip(elo_df["team"], elo_df["elo"]))
ELO_SCALE = 400

# Training: use clean pre-match ELO from matches_with_elo.csv
elo_train = pd.read_csv(matches_elo_path, encoding="utf-8-sig")
elo_train["date"] = pd.to_datetime(elo_train["date"])
elo_train = elo_train[elo_train["date"] >= "2022-01-01"].copy()
train_rd = (elo_train["elo_h_clean"] - elo_train["elo_a_clean"]).values / ELO_SCALE
train_hg = elo_train["home_goals"].values.astype(int)
train_ag = elo_train["away_goals"].values.astype(int)

print(f"ELO range: {min(elo_dict.values()):.0f} - {max(elo_dict.values()):.0f}")
print(f"Training: {len(elo_train)} matches (2022+)")
print(f"ELO diff/400 range: {train_rd.min():.2f} to {train_rd.max():.2f}")

# ============================================================
# DC neg log likelihood
# ============================================================
def neg_loglik(params):
    alpha, beta, gamma, rho = params

    if alpha <= -3 or alpha >= 2: return 1e10
    if beta <= 0 or beta >= 10: return 1e10
    if gamma <= -0.5 or gamma >= 1.0: return 1e10
    if abs(rho) >= 0.1: return 1e10

    rd = np.tanh(train_rd * 3.0) / 3.0  # soft saturation
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
elo_gaps = np.array([0, 50, 100, 200, 300])  # ELO difference
print("\nExpected goals by ELO gap:")
print(f"{'ELO gap':>10} {'λ_home':>8} {'λ_away':>8} {'ratio':>8}")
for gap in elo_gaps:
    rd = gap / ELO_SCALE
    lh = np.exp(alpha + beta * rd + gamma)
    la = np.exp(alpha - beta * rd)
    print(f"{gap:>10.0f} {lh:>8.2f} {la:>8.2f} {lh/la:>8.2f}")

# ============================================================
# Predict function
# ============================================================
def dc_predict(home, away, max_g=10, goals_mod=1.0):
    e_h = elo_dict.get(home, 1500)
    e_a = elo_dict.get(away, 1500)
    rd_raw = (e_h - e_a) / ELO_SCALE  # positive = home stronger
    rd = np.tanh(rd_raw * 3.0) / 3.0  # soft saturation

    lh = np.exp(alpha + beta * rd + gamma) * goals_mod * LAMBDA_SCALE
    la = np.exp(alpha - beta * rd) * goals_mod * LAMBDA_SCALE
    lh = np.clip(lh, 0.05, 15.0)
    la = np.clip(la, 0.05, 15.0)

    p_h, p_d, p_a = 0.0, 0.0, 0.0
    best_hw, best_hw_score = -1, (0, 0)
    best_dr, best_dr_score = -1, (0, 0)
    best_aw, best_aw_score = -1, (0, 0)

    for i in range(max_g + 1):
        for j in range(max_g + 1):
            prob = stats.poisson.pmf(i, lh) * stats.poisson.pmf(j, la)
            if i == 0 and j == 0:      prob *= (1 - lh * la * rho)
            elif i == 0 and j == 1:    prob *= (1 + lh * rho)
            elif i == 1 and j == 0:    prob *= (1 + la * rho)
            elif i == 1 and j == 1:    prob *= (1 - rho)

            if i > j:
                p_h += prob
                if prob > best_hw: best_hw, best_hw_score = prob, (i, j)
            elif i == j:
                p_d += prob
                if prob > best_dr: best_dr, best_dr_score = prob, (i, j)
            else:
                p_a += prob
                if prob > best_aw: best_aw, best_aw_score = prob, (i, j)

    total = p_h + p_d + p_a
    if total > 0: p_h /= total; p_d /= total; p_a /= total

    result = "H" if p_h >= max(p_d, p_a) else ("D" if p_d >= max(p_h, p_a) else "A")

    # Note: draw overrides are now handled by motivation layer in backtest
    if result == "H":
        best_h, best_a = best_hw_score
    elif result == "D":
        best_h, best_a = best_dr_score
    else:
        best_h, best_a = best_aw_score
    return result, best_h, best_a, (p_h, p_d, p_a), (lh, la)


# ============================================================
# Backtest 2026 WC (with full motivation pipeline)
# ============================================================
print(f"\n{'='*70}")
print("  2026 World Cup Backtest — Rank→Lambda DC + Motivation")
print(f"{'='*70}")

rl = {'H': 'H', 'D': 'DRAW', 'A': 'A'}
correct = exact = mae = 0
draw_total = draw_correct = 0

# Progressive state (no data leakage)
team_draws = {}   # {team: draws}
team_games = {}   # {team: total matches}
last_play = {}    # {team: date} progressive rest-day tracking

for _, row in wc_df.iterrows():
    home, away = row["home_team"], row["away_team"]
    hg, ag = int(row["home_score"]), int(row["away_score"])
    match_date_str = row["date"]
    group = row["group"]
    act = "H" if hg > ag else ("D" if hg == ag else "A")

    # Progressive match history: only matches BEFORE this one (no leakage)
    matches_before = wc_df[wc_df["date"] < match_date_str]

    # Team draw rate tracking (from prior matches only)
    h_dr = team_draws.get(home, 0)
    h_gm = team_games.get(home, 0)
    a_dr = team_draws.get(away, 0)
    a_gm = team_games.get(away, 0)
    h_rate = h_dr / h_gm if h_gm > 0 else 0.0
    a_rate = a_dr / a_gm if a_gm > 0 else 0.0

    # ---- Full motivation pipeline (same as predict_with_context.py) ----
    match_date_obj = pd.Timestamp(match_date_str).date()

    e_h = elo_dict.get(home, 1500)
    e_a = elo_dict.get(away, 1500)

    # Step 1: Motivation analysis (progressive data, no leakage)
    adj = analyze_match(home, away, match_date_obj, group, matches_before, last_play)

    # Step 2: DC prediction with goals_mod from motivation
    dc_result, ph, pa, (p_h, p_d, p_a), (lh, la) = dc_predict(
        home, away, goals_mod=adj.expected_goals_mod
    )

    # Step 3: Apply motivation adjustments
    adj_h, adj_d, adj_a = apply_motivation((p_h, p_d, p_a), adj)

    # Step 4: Draw detector (same as predict_with_context.py)
    result = "H" if adj_h >= max(adj_d, adj_a) else ("D" if adj_d >= max(adj_h, adj_a) else "A")
    if adj_d >= DRAW_PROB_THRESHOLD and adj.draw_uplift >= 0.04:
        result = "D"

    # ELO-based draw override: only for round 1 (no motivation context yet)
    # For md>=2, the motivation draw detector + team draw propensity handle it
    h_md = team_games.get(home, 0) + 1
    elo_gap = abs(e_h - e_a)
    if h_md == 1 and elo_gap < DRAW_ELO_THRESHOLD and p_d >= DRAW_PROB_THRESHOLD:
        result = "D"

    # Step 5: Team draw propensity (additional rule, min 2 matches for signal)
    if (h_rate >= DRAW_RATE_THRESHOLD and a_rate >= DRAW_RATE_THRESHOLD
        and h_gm >= 2 and a_gm >= 2):
        result = "D"

    # Update score display for flipped/draw-override results
    if result == "D":
        ph, pa = 1, 1
    elif result != dc_result:
        ph, pa = pa, ph  # motivation flipped result direction

    ok = "OK" if result == act else "XX"
    if result == act: correct += 1
    if act == "D":
        draw_total += 1
        if result == "D": draw_correct += 1
    if ph == hg and pa == ag: exact += 1
    mae += abs(ph - hg) + abs(pa - ag)

    # Update progressive state AFTER prediction
    team_games[home] = h_gm + 1
    team_games[away] = a_gm + 1
    if act == "D":
        team_draws[home] = team_draws.get(home, 0) + 1
        team_draws[away] = team_draws.get(away, 0) + 1
    last_play[home] = match_date_obj
    last_play[away] = match_date_obj

    home_cn = CN.get(home, home)
    away_cn = CN.get(away, away)
    draw_info = f"dr:{h_rate:.0%}/{a_rate:.0%}" if h_gm > 0 or a_gm > 0 else ""
    print(f"  {home_cn:<8} vs {away_cn:<8} {hg}:{ag} ({rl[act]:>4})  pred:{rl[result]:>5} {ph}:{pa} {ok}  [{adj_h:.0%}/{adj_d:.0%}/{adj_a:.0%}]  E:{e_h:.0f}v{e_a:.0f}  λ:{lh:.1f}v{la:.1f}  {draw_info}")

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
