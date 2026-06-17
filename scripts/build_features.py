"""
构建预测特征：动态ELO + 近期状态 + 交手记录
从历史比赛数据逐场计算，避免未来信息泄露
"""
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).parent.parent
DATA_PROCESSED = ROOT / "data" / "processed"
OUTPUT_DIR = ROOT / "output"

# ELO 参数
ELO_INIT = 1500
ELO_HOME_ADV = 100
K_MAP = {
    "FIFA World Cup": 60,
    "FIFA World Cup qualification": 40,
    "UEFA Euro": 50,
    "Copa América": 50,
    "AFC Asian Cup": 50,
    "African Cup of Nations": 50,
    "Gold Cup": 40,
    "UEFA Euro qualification": 40,
    "African Cup of Nations qualification": 40,
    "AFC Asian Cup qualification": 40,
    "UEFA Nations League": 30,
    "CONCACAF Nations League": 30,
    "CONCACAF Nations League qualification": 30,
}
K_DEFAULT = 20

def get_k(round_name):
    for key, val in K_MAP.items():
        if key in str(round_name):
            return val
    return K_DEFAULT

def expected_score(elo_h, elo_a):
    return 1.0 / (1.0 + 10.0 ** (-(elo_h - elo_a) / 400.0))

def build_features():
    df = pd.read_csv(DATA_PROCESSED / "matches.csv", encoding="utf-8-sig")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    elo = defaultdict(lambda: ELO_INIT)
    # 近期比赛记录: team -> [(date, goals_for, goals_against, result)]
    recent = defaultdict(list)
    h2h = defaultdict(list)

    rows = []
    for _, match in df.iterrows():
        home = match["home_team"]
        away = match["away_team"]
        hg = int(match["home_goals"])
        ag = int(match["away_goals"])
        date = match["date"]
        round_name = match.get("round", "")

        # --- ELO ---
        elo_h = elo[home] + ELO_HOME_ADV
        elo_a = elo[away]

        # --- 近期状态 (赛前) ---
        def form_stats(team, date, n=10):
            recs = [r for r in recent[team] if r[0] < date]
            last_n = sorted(recs, key=lambda x: x[0], reverse=True)[:n]
            if not last_n:
                return 0.5, 0.0, 0.0, 0.0  # win%, gf/g, ga/g, gd/g
            w = sum(1 for r in last_n if r[3] == "W")
            gf = sum(r[1] for r in last_n) / len(last_n)
            ga = sum(r[2] for r in last_n) / len(last_n)
            return w / len(last_n), gf, ga, gf - ga

        h_win5, h_gf5, h_ga5, h_gd5 = form_stats(home, date, 5)
        a_win5, a_gf5, a_ga5, a_gd5 = form_stats(away, date, 5)
        h_win10, h_gf10, h_ga10, h_gd10 = form_stats(home, date, 10)
        a_win10, a_gf10, a_ga10, a_gd10 = form_stats(away, date, 10)

        # --- 交手记录 ---
        h2h_recs = [r for r in h2h.get((home, away), []) + h2h.get((away, home), [])
                    if r[0] < date]
        h2h_n = len(h2h_recs)
        h2h_h_wins = sum(1 for r in h2h_recs if r[1] == home and r[2] > r[3])
        h2h_a_wins = sum(1 for r in h2h_recs if r[1] == away and r[2] > r[3])
        h2h_draws = h2h_n - h2h_h_wins - h2h_a_wins

        # --- 构建特征行 ---
        rows.append({
            "date": date,
            "home_team": home,
            "away_team": away,
            "home_goals": hg,
            "away_goals": ag,
            "round": round_name,
            # ELO
            "elo_h": elo_h,
            "elo_a": elo_a,
            "elo_diff": elo_h - elo_a,
            # 近期5场
            "h_win5": round(h_win5, 4),
            "a_win5": round(a_win5, 4),
            "h_gf5": round(h_gf5, 2),
            "a_gf5": round(a_gf5, 2),
            "h_ga5": round(h_ga5, 2),
            "a_ga5": round(a_ga5, 2),
            "h_gd5": round(h_gd5, 2),
            "a_gd5": round(a_gd5, 2),
            # 近期10场
            "h_win10": round(h_win10, 4),
            "a_win10": round(a_win10, 4),
            "h_gf10": round(h_gf10, 2),
            "a_gf10": round(a_gf10, 2),
            "h_ga10": round(h_ga10, 2),
            "a_ga10": round(a_ga10, 2),
            "h_gd10": round(h_gd10, 2),
            "a_gd10": round(a_gd10, 2),
            # 交手
            "h2h_n": h2h_n,
            "h2h_h_winrate": round(h2h_h_wins / h2h_n, 4) if h2h_n else 0.0,
            "h2h_a_winrate": round(h2h_a_wins / h2h_n, 4) if h2h_n else 0.0,
        })

        # --- 赛后更新 ---
        # ELO 更新
        k = get_k(round_name)
        if hg > ag:
            s_h, s_a = 1.0, 0.0
        elif hg < ag:
            s_h, s_a = 0.0, 1.0
        else:
            s_h, s_a = 0.5, 0.5

        goal_diff = abs(hg - ag)
        goal_factor = 1.0 if goal_diff <= 1 else (1.5 if goal_diff == 2 else (1.75 + (goal_diff - 3) * 0.125))
        k_adj = k * goal_factor

        e_h = expected_score(elo_h, elo_a)
        elo[home] += k_adj * (s_h - e_h)
        elo[away] += k_adj * (s_a - (1 - e_h))

        # 近期状态更新
        result = "W" if hg > ag else ("L" if hg < ag else "D")
        recent[home].append((date, hg, ag, result))
        recent[away].append((date, ag, hg, "W" if result == "L" else ("L" if result == "W" else "D")))

        # 交手记录更新
        h2h[(home, away)].append((date, home, hg, ag))

    features_df = pd.DataFrame(rows)
    OUTPUT_DIR.mkdir(exist_ok=True)
    features_df.to_csv(OUTPUT_DIR / "features.csv", index=False, encoding="utf-8-sig")
    print(f"特征表: {len(features_df)} 行 × {len(features_df.columns)} 列")
    print(f"ELO 范围: {features_df['elo_h'].min():.0f} - {features_df['elo_h'].max():.0f}")
    print(f"保存到 {OUTPUT_DIR / 'features.csv'}")
    return features_df

if __name__ == "__main__":
    build_features()
