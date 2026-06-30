"""
predict_knockout.py — 淘汰赛单场预测（直接指定对阵）
"""
import sys; sys.path.insert(0, "scripts")
import numpy as np
from scipy import stats
from pathlib import Path
import pandas as pd
import json

ROOT = Path(".")

# Load model
params = json.load(open(ROOT / "output" / "rank_lambda_model.json", encoding="utf-8"))
alpha, beta, gamma, rho = params["alpha"], params["beta"], params["gamma"], params["rho"]

# Load ELO
ELO_DF = pd.read_csv(ROOT / "data" / "processed" / "clean_elo.csv", encoding="utf-8-sig")
ELO_DICT = dict(zip(ELO_DF["team"], ELO_DF["elo"]))
ELO_SCALE = 400
LAMBDA_SCALE = 1.12

# Load odds
ALL_ODDS = {}
for path in sorted((ROOT / "data" / "raw").glob("odds_*.csv")):
    df = pd.read_csv(path, encoding="utf-8-sig")
    for _, row in df.iterrows():
        ALL_ODDS[row["match"]] = (row["home_odds"], row["draw_odds"], row["away_odds"])

def odds_to_probs(h_odds, d_odds, a_odds):
    h_raw = 1 / h_odds; d_raw = 1 / d_odds; a_raw = 1 / a_odds
    total = h_raw + d_raw + a_raw
    return h_raw / total, d_raw / total, a_raw / total

# Load motivation
from motivation import analyze_match, apply_motivation, load_match_history
from model_utils import load_rankings
rankings = load_rankings()
last_play = load_match_history()
wc_df = pd.read_csv(ROOT / "data" / "raw" / "matches_2026.csv", encoding="utf-8-sig")

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

def dc_predict(home, away, max_g=10, goals_mod=1.0):
    e_h = ELO_DICT.get(home, 1500); e_a = ELO_DICT.get(away, 1500)
    rd_raw = (e_h - e_a) / ELO_SCALE
    rd = np.tanh(rd_raw * 3.0) / 3.0
    lh = np.exp(alpha + beta * rd + gamma) * goals_mod * LAMBDA_SCALE
    la = np.exp(alpha - beta * rd) * goals_mod * LAMBDA_SCALE
    lh = np.clip(lh, 0.05, 15.0); la = np.clip(la, 0.05, 15.0)
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
    result = "HOME" if p_h >= max(p_d, p_a) else ("DRAW" if p_d >= max(p_h, p_a) else "AWAY")
    if result == "HOME": best_h, best_a = best_hw_score
    elif result == "DRAW": best_h, best_a = best_dr_score
    else: best_h, best_a = best_aw_score
    return result, best_h, best_a, (p_h, p_d, p_a), (lh, la)

def result_label(result):
    return {"HOME": "主胜", "DRAW": "平局", "AWAY": "客胜"}[result]

matches_to_predict = [
    ("Brazil", "Japan", "2026-06-30", "R16"),
    ("Germany", "Paraguay", "2026-06-30", "R16"),
    ("Netherlands", "Morocco", "2026-06-30", "R16"),
]

print("=" * 95)
print("  2026 世界杯 1/16 决赛预测")
print("=" * 95)

for home, away, match_date, round_label in matches_to_predict:
    adj = analyze_match(home, away, match_date, "R16", wc_df, last_play)
    result, ph, pa, (p_h, p_d, p_a), (lh, la) = dc_predict(home, away, goals_mod=adj.expected_goals_mod)
    adj_h, adj_d, adj_a = apply_motivation((p_h, p_d, p_a), adj)

    match_key = f"{home} vs {away}"
    odds_data = ALL_ODDS.get(match_key)
    if odds_data:
        raw_h, raw_d, raw_a = odds_data
        o_h, o_d, o_a = odds_to_probs(*odds_data)
        if raw_d < 2.80: d_weight = 0.50
        elif raw_d < 3.50: d_weight = 0.40
        else: d_weight = 0.30
        final_h = adj_h * 0.7 + o_h * 0.3
        final_d = adj_d * (1 - d_weight) + o_d * d_weight
        final_a = adj_a * 0.7 + o_a * 0.3
        total = final_h + final_d + final_a
        final_h, final_d, final_a = final_h / total, final_d / total, final_a / total
    else:
        final_h, final_d, final_a = adj_h, adj_d, adj_a
        raw_h = raw_d = raw_a = 0.0

    adj_result_raw = "HOME" if final_h >= max(final_d, final_a) else ("DRAW" if final_d >= max(final_h, final_a) else "AWAY")
    lambda_ratio = max(lh, la) / min(lh, la) if min(lh, la) > 0 else 1.0
    if lambda_ratio >= 3.0: prob_th = 0.15
    elif lambda_ratio >= 1.5: prob_th = 0.22
    else: prob_th = 0.28
    adj_result = "DRAW" if (adj_d >= prob_th and adj.draw_uplift >= 0.04) else adj_result_raw
    draw_override = (adj_result == "DRAW" and adj_result_raw != "DRAW")

    risk = "!! HIGH" if final_d >= 0.30 else ("! MED" if final_d >= 0.25 else ("  LOW" if final_d >= 0.22 else ""))

    print(f"\n{'─'*95}")
    print(f"  {CN.get(home, home)} vs {CN.get(away, away)}  [{round_label} {match_date}]")
    print(f"  ELO: {ELO_DICT.get(home,0)} vs {ELO_DICT.get(away,0)}   λ {lh:.1f}:{la:.1f}  (λ比 {lambda_ratio:.1f})")
    print(f"  DC基础: {result_label(result)} {ph}:{pa}  H{p_h:.0%}/D{p_d:.0%}/A{p_a:.0%}")
    if odds_data:
        print(f"  赔率: H{raw_h:.2f} D{raw_d:.2f} A{raw_a:.2f}  ->  隐含概率 H{o_h:.0%} D{o_d:.0%} A{o_a:.0%}  (D权重{d_weight:.0%})")
    print(f"  最终预测: {result_label(adj_result)}  H{final_h:.0%}/D{final_d:.0%}/A{final_a:.0%}", end="")
    if draw_override: print("  [DRAW覆写]", end="")
    print(f"  {risk}")
    print(f"  比分预测: {ph}:{pa}  |  goals_mod={adj.expected_goals_mod:.2f}  |  draw_uplift={adj.draw_uplift:.3f}")

    if adj.risk_flags:
        for flag in adj.risk_flags[:5]:
            print(f"    [!] {flag}")
    if adj.notes:
        for note in adj.notes[:3]:
            print(f"    [i] {note}")

    # Show odds sources
    for path in sorted((ROOT / "data" / "raw").glob("odds_*.csv")):
        df = pd.read_csv(path, encoding="utf-8-sig")
        for _, row in df.iterrows():
            if row["match"] == match_key:
                print(f"    [$] {path.name}: H{row['home_odds']} D{row['draw_odds']} A{row['away_odds']}")

print(f"\n{'='*95}")
print("  注：淘汰赛预测按90分钟常规时间，不含加时/点球")
