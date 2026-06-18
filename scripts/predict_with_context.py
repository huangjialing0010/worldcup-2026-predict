"""
predict_with_context.py — 带动机与场外因素修正的预测脚本

= 基础 DC 泊松概率 (Rank→Lambda 模型) + 动机修正层 =
输出：基础预测 vs 修正后预测，含风险标注和场外备注
"""
import sys; sys.path.insert(0, 'scripts')
import numpy as np
from scipy import stats
from pathlib import Path
from datetime import date
import pandas as pd
import json

from model_utils import load_rankings
from motivation import (
    analyze_match, apply_motivation, MotivationAdjustment,
    load_match_history, KNOWN_MATCH_DATES, TEAM_GROUP,
    get_adjusted_rank, get_ranking_note
)

ROOT = Path(__file__).parent.parent


def get_remaining_matches(schedule_path: Path | None = None,
                          matches_path: Path | None = None) -> list[tuple[str, str, str, str, str]]:
    """从 schedule_2026.csv 减去已赛，返回未赛的 (home, away, date, group, round_label)"""
    if schedule_path is None:
        schedule_path = ROOT / "data" / "raw" / "schedule_2026.csv"
    if matches_path is None:
        matches_path = ROOT / "data" / "raw" / "matches_2026.csv"

    schedule = pd.read_csv(schedule_path, encoding="utf-8-sig")
    if matches_path.exists():
        played = pd.read_csv(matches_path, encoding="utf-8-sig")
        played_pairs = set(zip(played["home_team"], played["away_team"]))
    else:
        played_pairs = set()

    remaining = []
    for _, row in schedule.iterrows():
        pair = (row["home_team"], row["away_team"])
        if pair not in played_pairs:
            remaining.append((
                row["home_team"], row["away_team"],
                str(row["date"]), row["group"], row["round_label"]
            ))
    return remaining


# ============================================================
# 加载模型
# ============================================================
params = json.load(open(ROOT / "output" / "rank_lambda_model.json", encoding="utf-8"))
alpha = params["alpha"]
beta = params["beta"]
gamma = params["gamma"]
rho = params["rho"]

rankings = load_rankings()  # still used by motivation layer for path/display
last_play = load_match_history()
wc_df = pd.read_csv(ROOT / "data" / "raw" / "matches_2026.csv", encoding="utf-8-sig")

# Clean W/L/D ELO (no goal circularity)
ELO_DF = pd.read_csv(ROOT / "data" / "processed" / "clean_elo.csv", encoding="utf-8-sig")
ELO_DICT = dict(zip(ELO_DF["team"], ELO_DF["elo"]))
ELO_SCALE = 400

# 球队中文名映射
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

# ============================================================
# 赔率加载
# ============================================================
def load_all_odds():
    """从 odds_*.csv 加载所有赔率 → {match_name: (h, d, a)}（live 覆盖 round）"""
    odds = {}
    for path in sorted((ROOT / "data" / "raw").glob("odds_*.csv")):
        df = pd.read_csv(path, encoding="utf-8-sig")
        for _, row in df.iterrows():
            odds[row["match"]] = (row["home_odds"], row["draw_odds"], row["away_odds"])
    return odds

def odds_to_probs(h_odds, d_odds, a_odds):
    """Decimal odds → implied probabilities（去水）"""
    h_raw = 1 / h_odds
    d_raw = 1 / d_odds
    a_raw = 1 / a_odds
    total = h_raw + d_raw + a_raw
    return h_raw / total, d_raw / total, a_raw / total

ALL_ODDS = load_all_odds()


# ============================================================
# 平局检测器
# ============================================================
def apply_draw_detector(current_result, adj_d, draw_uplift):
    """
    当动机层发出强平局信号时，覆写泊松结果判平局。
    阈值：P(D)≥25% 且 平局上浮≥0.04（对应"双方均胜"/"双方均6分"等场景）
    """
    if adj_d >= 0.25 and draw_uplift >= 0.04:
        return "DRAW"
    return current_result


# ============================================================
# DC 预测函数（同 elo_lambda_model.py）
# ============================================================
def dc_predict(home, away, max_g=10, goals_mod=1.0):
    e_h = ELO_DICT.get(home, 1500)
    e_a = ELO_DICT.get(away, 1500)
    rd_raw = (e_h - e_a) / ELO_SCALE  # positive = home stronger
    rd = np.tanh(rd_raw * 3.0) / 3.0  # soft saturation

    lh = np.exp(alpha + beta * rd + gamma) * goals_mod
    la = np.exp(alpha - beta * rd) * goals_mod
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

    result = "HOME" if p_h >= max(p_d, p_a) else ("DRAW" if p_d >= max(p_h, p_a) else "AWAY")
    return result, best_h, best_a, (p_h, p_d, p_a), (lh, la)


def result_label(result):
    return {"HOME": "主胜", "DRAW": "平局", "AWAY": "客胜"}[result]


# ============================================================
# 剩余赛程
# ============================================================
REMAINING = get_remaining_matches()


# ============================================================
# 主程序
# ============================================================
if __name__ == "__main__":
    print("=" * 95)
    print("  2026 世界杯预测 — Rank→Lambda DC + 动机/场外因素修正")
    print("=" * 95)

    lines = []
    for home, away, match_date, group, round_label in REMAINING:
        # 动机分析（提前获取 goals_mod）
        adj = analyze_match(home, away, match_date, group, wc_df, last_play)

        # DC预测（goals_mod 直接作用于 λ）
        result, ph, pa, (p_h, p_d, p_a), (lh, la) = dc_predict(home, away, goals_mod=adj.expected_goals_mod)

        # 修正后概率
        adj_h, adj_d, adj_a = apply_motivation((p_h, p_d, p_a), adj)

        # 赔率融合（如有）
        match_key = f"{home} vs {away}"
        odds_data = ALL_ODDS.get(match_key)
        if odds_data:
            o_h, o_d, o_a = odds_to_probs(*odds_data)
            # 70% 模型 + 30% 市场
            final_h = adj_h * 0.7 + o_h * 0.3
            final_d = adj_d * 0.7 + o_d * 0.3
            final_a = adj_a * 0.7 + o_a * 0.3
            has_odds = True
        else:
            final_h, final_d, final_a = adj_h, adj_d, adj_a
            has_odds = False

        # 平局检测器覆写
        adj_result_raw = "HOME" if final_h >= max(final_d, final_a) else ("DRAW" if final_d >= max(final_h, final_a) else "AWAY")
        adj_result = apply_draw_detector(adj_result_raw, final_d, adj.draw_uplift)
        draw_override = (adj_result == "DRAW" and adj_result_raw != "DRAW")

        # 风险等级
        if final_d >= 0.30:
            risk_level = "!! HIGH"
        elif final_d >= 0.25:
            risk_level = "! MED"
        elif final_d >= 0.22:
            risk_level = "  LOW"
        else:
            risk_level = ""

        # 输出
        rk_h = get_adjusted_rank(home, rankings)
        rk_a = get_adjusted_rank(away, rankings)
        raw_rk_h = rankings.get(home, 50)
        raw_rk_a = rankings.get(away, 50)

        # 排名修正备注
        rank_note_h = get_ranking_note(home)
        rank_note_a = get_ranking_note(away)

        lines.append({
            "match": f"{home} vs {away}",
            "group": group,
            "date": match_date,
            "round": round_label,
            "ranks": f"#{rk_h}v#{rk_a}",
            "lambdas": f"λ{lh:.1f}:{la:.1f}",
            "lambda_h": lh,
            "lambda_a": la,
            "base_score": f"{ph}:{pa}",
            "base_result": result_label(result),
            "base_probs": f"H{p_h:.0%} D{p_d:.0%} A{p_a:.0%}",
            "motiv_probs": f"H{adj_h:.0%} D{adj_d:.0%} A{adj_a:.0%}",
            "final_probs": f"H{final_h:.0%} D{final_d:.0%} A{final_a:.0%}",
            "final_result": result_label(adj_result),
            "odds_blend": "Y" if has_odds else "N",
            "draw_override": "Y" if draw_override else "",
            "risk": risk_level,
            "risk_flags": " | ".join(adj.risk_flags[:3]) if adj.risk_flags else "",
            "notes": " | ".join(adj.notes[:2]) if adj.notes else "",
            "goals_mod": f"×{adj.expected_goals_mod:.2f}" if adj.expected_goals_mod != 1.0 else "",
        })

        rank_display = f"#{raw_rk_h}→#{rk_h} vs #{raw_rk_a}→#{rk_a}" if (raw_rk_h != rk_h or raw_rk_a != rk_a) else f"#{rk_h} vs #{rk_a}"
        print(f"\n{'─'*95}")
        print(f"  {home} vs {away}  [{group}组 {round_label}  {match_date}]  {rank_display}")
        print(f"  DC基础: {result_label(result)} {ph}:{pa}  λ{lh:.1f}:{la:.1f}  H{p_h:.0%}/D{p_d:.0%}/A{p_a:.0%}")
        motiv_str = f" 动机修正: {result_label(adj_result)}  H{adj_h:.0%}/D{adj_d:.0%}/A{adj_a:.0%}"
        if has_odds:
            motiv_str += f"  赔率融合: H{final_h:.0%}/D{final_d:.0%}/A{final_a:.0%}"
        if draw_override:
            motiv_str += "  [DRAW覆写]"
        motiv_str += f"  {risk_level}" + (f" 进球×{adj.expected_goals_mod:.2f}" if adj.expected_goals_mod != 1.0 else "")
        print(motiv_str)

        if rank_note_h:
            print(f"    [R] {rank_note_h}")
        if rank_note_a:
            print(f"    [R] {rank_note_a}")
        if adj.risk_flags:
            for flag in adj.risk_flags[:5]:
                print(f"    [!] {flag}")
        if adj.notes:
            for note in adj.notes[:3]:
                print(f"    [i] {note}")

    # CSV 输出
    out_df = pd.DataFrame(lines)
    out_path = ROOT / "output" / "predictions_with_context.csv"
    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")

    # 每日快照
    daily_dir = ROOT / "output" / "daily"
    daily_dir.mkdir(parents=True, exist_ok=True)
    today_str = str(date.today())
    daily_out = daily_dir / f"{today_str}.csv"
    # 中文列名 + 精简列
    daily_cols = {
        "match": "对阵", "group": "组", "date": "比赛日期", "round": "轮次",
        "base_score": "比分预测", "final_probs": "最终概率", "final_result": "最终预测",
        "draw_override": "平局覆写", "odds_blend": "赔率融合", "risk": "风险",
        "lambda_h": "_lh", "lambda_a": "_la",
    }
    daily_df = out_df[list(daily_cols.keys())].copy()
    daily_df.columns = [daily_cols[c] for c in daily_cols]
    daily_df.insert(0, "预测日期", today_str)
    # 期望进球（λ）跟在比分后面，方便理解
    daily_df.insert(3, "期望进球", "(" + daily_df["_lh"].apply(lambda x: f"{x:.1f}") + "/" + daily_df["_la"].apply(lambda x: f"{x:.1f}") + ")")
    daily_df.drop(columns=["_lh", "_la"], inplace=True)
    # 对阵名翻译成中文
    daily_df["对阵"] = daily_df["对阵"].apply(lambda m: " vs ".join(CN.get(t, t) for t in m.split(" vs ")))
    daily_df.to_csv(daily_out, index=False, encoding="utf-8-sig")

    print(f"\n{'='*95}")
    print(f"  已保存: {out_path}")
    print(f"  每日快照: {daily_out}")

    # 汇总
    print(f"\n  共 {len(REMAINING)} 场比赛预测")
    risk_count = sum(1 for l in lines if l["risk"] in ("!! HIGH", "! MED"))
    print(f"  其中 {risk_count} 场有平局风险标注")
