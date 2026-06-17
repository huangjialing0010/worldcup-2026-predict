"""
共享常量和工具函数 — 所有预测/回测脚本的单一数据源
"""
import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"

# ============================================================
# 硬编码回退数据（API 不可用时使用）
# ============================================================

FIFA_RANKINGS = {
    "Argentina": 1, "Spain": 2, "France": 3, "England": 4,
    "Portugal": 5, "Brazil": 6, "Morocco": 7, "Netherlands": 8,
    "Belgium": 9, "Germany": 10, "Croatia": 11, "Colombia": 13,
    "Mexico": 14, "Senegal": 15, "Uruguay": 16, "USA": 17,
    "Japan": 18, "Switzerland": 19, "Iran": 20, "Turkey": 22,
    "Ecuador": 23, "Austria": 24, "South Korea": 25, "Australia": 27,
    "Algeria": 28, "Egypt": 29, "Canada": 30, "Norway": 31,
    "Ivory Coast": 33, "Panama": 34, "Sweden": 38, "Czech Republic": 40,
    "Paraguay": 41, "Scotland": 42, "Tunisia": 45, "DR Congo": 46,
    "Uzbekistan": 50, "Qatar": 56, "Iraq": 57, "Saudi Arabia": 61,
    "Jordan": 63, "Bosnia and Herzegovina": 64, "South Africa": 60,
    "Cape Verde": 67, "Curacao": 82, "Haiti": 83, "New Zealand": 85,
    "Ghana": 73,
}

FALLBACK_MATCHES = [
    ("Mexico", "South Africa", 2, 0),
    ("South Korea", "Czech Republic", 2, 1),
    ("Canada", "Bosnia and Herzegovina", 1, 1),
    ("Qatar", "Switzerland", 1, 1),
    ("Brazil", "Morocco", 1, 1),
    ("Haiti", "Scotland", 0, 1),
    ("USA", "Paraguay", 4, 1),
    ("Australia", "Turkey", 2, 0),
    ("Germany", "Curacao", 7, 1),
    ("Ivory Coast", "Ecuador", 1, 0),
    ("Netherlands", "Japan", 2, 2),
    ("Sweden", "Tunisia", 5, 1),
    ("Belgium", "Egypt", 1, 1),
    ("Iran", "New Zealand", 2, 2),
    ("Spain", "Cape Verde", 0, 0),
    ("Saudi Arabia", "Uruguay", 1, 1),
]

FALLBACK_ODDS = [
    ("Mexico", "South Africa", 1.42, 4.40, 7.27, 2, 0),
    ("South Korea", "Czech Republic", 2.57, 2.95, 2.84, 2, 1),
    ("Canada", "Bosnia and Herzegovina", 1.76, 3.51, 4.62, 1, 1),
    ("Qatar", "Switzerland", 13.87, 6.22, 1.26, 1, 1),
    ("Brazil", "Morocco", 1.62, 3.77, 5.42, 1, 1),
    ("Haiti", "Scotland", 6.82, 4.60, 1.44, 0, 1),
    ("USA", "Paraguay", 1.93, 3.43, 3.82, 4, 1),
    ("Australia", "Turkey", 4.68, 3.68, 1.66, 2, 0),
    ("Germany", "Curacao", 1.03, 16.50, 42.00, 7, 1),
    ("Ivory Coast", "Ecuador", 3.53, 2.77, 2.36, 1, 0),
    ("Netherlands", "Japan", 1.91, 3.50, 3.78, 2, 2),
    ("Sweden", "Tunisia", 1.86, 3.30, 4.25, 5, 1),
    ("Belgium", "Egypt", 1.58, 3.98, 5.37, 1, 1),
    ("Iran", "New Zealand", 1.86, 3.57, 4.39, 2, 2),
    ("Spain", "Cape Verde", 1.09, 9.85, 29.50, 0, 0),
    ("Saudi Arabia", "Uruguay", 7.07, 4.35, 1.39, 1, 1),
]

HEXAGRAMS = [
    "乾为天", "坤为地", "水雷屯", "山水蒙", "水天需", "天水讼",
    "地水师", "水地比", "风天小畜", "天泽履", "地天泰", "天地否",
    "天火同人", "火天大有", "地山谦", "雷地豫", "泽雷随", "山风蛊",
    "地泽临", "风地观", "火雷噬嗑", "山火贲", "山地剥", "地雷复",
    "天雷无妄", "山天大畜", "山雷颐", "泽风大过", "坎为水", "离为火",
    "泽山咸", "雷风恒", "天山遁", "雷天大壮", "火地晋", "地火明夷",
    "风火家人", "火泽睽", "水山蹇", "雷水解", "山泽损", "风雷益",
    "泽天夬", "天风姤", "泽地萃", "地风升", "泽水困", "水风井",
    "泽火革", "火风鼎", "震为雷", "艮为山", "风山渐", "雷泽归妹",
    "雷火丰", "火山旅", "巽为风", "兑为泽", "风水涣", "水泽节",
    "风泽中孚", "雷山小过", "水火既济", "火水未济",
]

BAGUA = ["乾☰", "兑☱", "离☲", "震☳", "巽☴", "坎☵", "艮☶", "坤☷"]


# ============================================================
# 核心工具函数
# ============================================================


def get_rank(team, rankings=None, default=50):
    if rankings is None:
        rankings = FIFA_RANKINGS
    return rankings.get(team, default)


def rank_to_strength(rank, scale=100):
    return np.exp(-rank / scale)


def actual_result(h_score, a_score):
    if h_score > a_score:
        return "H"
    elif a_score > h_score:
        return "A"
    else:
        return "D"


def poisson_best_score(lambda_h, lambda_a, max_goals=8):
    """泊松模型：搜索最可能比分，返回 (result, h_goals, a_goals)"""
    best_prob = -1
    best_h, best_a = 0, 0
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            prob = stats.poisson.pmf(h, lambda_h) * stats.poisson.pmf(a, lambda_a)
            if prob > best_prob:
                best_prob = prob
                best_h, best_a = h, a

    if best_h > best_a:
        return "H", best_h, best_a
    elif best_a > best_h:
        return "A", best_h, best_a
    else:
        return "D", best_h, best_a


def poisson_probabilities(lambda_h, lambda_a, max_goals=10):
    """泊松模型：计算 H/D/A 全概率"""
    p_h = 0
    p_d = 0
    p_a = 0
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            prob = stats.poisson.pmf(i, lambda_h) * stats.poisson.pmf(j, lambda_a)
            if i > j:
                p_h += prob
            elif i == j:
                p_d += prob
            else:
                p_a += prob
    return p_h, p_d, p_a


def implied_probabilities(odds_h, odds_d, odds_a):
    imp_h = 1.0 / odds_h
    imp_d = 1.0 / odds_d
    imp_a = 1.0 / odds_a
    overround = imp_h + imp_d + imp_a
    return imp_h / overround, imp_d / overround, imp_a / overround


# ============================================================
# 数据加载（优先 processed > raw > fallback）
# ============================================================


def load_rankings():
    path = DATA_PROCESSED / "teams.csv"
    if path.exists():
        df = pd.read_csv(path, encoding="utf-8-sig")
        if "team_name" in df.columns and "fifa_rank" in df.columns:
            return dict(zip(df["team_name"], df["fifa_rank"].fillna(50).astype(int)))
    return dict(FIFA_RANKINGS)


def load_matches():
    path = DATA_PROCESSED / "matches.csv"
    if path.exists():
        df = pd.read_csv(path, encoding="utf-8-sig")
        finished = df[df["is_finished"] == True]
        if not finished.empty:
            matches = []
            for _, row in finished.iterrows():
                matches.append((
                    row["home_team"], row["away_team"],
                    int(row["home_goals"]), int(row["away_goals"]),
                ))
            return matches
    return list(FALLBACK_MATCHES)


def load_odds_data():
    # 优先从 CSV 加载
    csv_path = DATA_RAW / "odds_round1.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
        odds = []
        for _, row in df.iterrows():
            parts = row["match"].split(" vs ")
            if len(parts) == 2:
                odds.append((
                    parts[0], parts[1],
                    float(row.get("home_odds", row.get("home_odds", 1))),
                    float(row.get("draw_odds", row.get("draw_odds", 1))),
                    float(row.get("away_odds", row.get("away_odds", 1))),
                    int(row.get("actual_home", 0)),
                    int(row.get("actual_away", 0)),
                ))
        if odds:
            return odds
    return list(FALLBACK_ODDS)
