"""
motivation.py — 比赛动机与赛制因素修正模块

在 Rank→Lambda DC 泊松概率基础上叠加四层修正：
  A. 淘汰赛路径（小组排名→对阵路线→上下半区难度差）
  B. 体能/后勤（休息天数差、旅行负担）
  C. 积分形势（必须赢、接受平、可轮换、默契球场景）
  D. 场外地缘政治（战争、制裁、旅行禁令、FIFA暂停）
"""

import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from datetime import date, timedelta
import json
from model_utils import load_rankings as _load_fifa_rankings

ROOT = Path(__file__).parent.parent

# ============================================================
# 2026 世界杯分组
# ============================================================
GROUPS_2026 = {
    "A": ["Mexico", "South Korea", "Czech Republic", "South Africa"],
    "B": ["Canada", "Bosnia and Herzegovina", "Qatar", "Switzerland"],
    "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "D": ["USA", "Paraguay", "Australia", "Turkey"],
    "E": ["Germany", "Curacao", "Ivory Coast", "Ecuador"],
    "F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
    "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "H": ["Spain", "Cape Verde", "Saudi Arabia", "Uruguay"],
    "I": ["France", "Senegal", "Iraq", "Norway"],
    "J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "K": ["Portugal", "DR Congo", "Uzbekistan", "Colombia"],
    "L": ["England", "Croatia", "Ghana", "Panama"],
}

# 球队→组反向映射
TEAM_GROUP = {}
for g, teams in GROUPS_2026.items():
    for t in teams:
        TEAM_GROUP[t] = g

# ============================================================
# A. 淘汰赛路径
# ============================================================

# 各小组第一/第二对应的半区和对阵
# half: 'UPPER' → M101半决赛, 'LOWER' → M102半决赛
# opponent: '3rd'→打小组第三, '2X'→打X组第二, 'X1'→打X组第一
# opponent_from: 若打第三名，可能来自哪些组

KNOCKOUT_PATH = {
    "A": {"1st": {"half": "LOWER", "opponent": "3rd", "from": ["C","E","F","H","I"]},
          "2nd": {"half": "UPPER", "opponent": "2B"}},
    "B": {"1st": {"half": "LOWER", "opponent": "3rd", "from": ["E","F","G","I","J"]},
          "2nd": {"half": "UPPER", "opponent": "2A"}},
    "C": {"1st": {"half": "LOWER", "opponent": "2F"},
          "2nd": {"half": "UPPER", "opponent": "F1"}},
    "D": {"1st": {"half": "UPPER", "opponent": "3rd", "from": ["B","E","F","I","J"]},
          "2nd": {"half": "LOWER", "opponent": "2G"}},
    "E": {"1st": {"half": "UPPER", "opponent": "3rd", "from": ["A","B","C","D","F"]},
          "2nd": {"half": "LOWER", "opponent": "2I"}},
    "F": {"1st": {"half": "UPPER", "opponent": "2C"},
          "2nd": {"half": "LOWER", "opponent": "C1"}},
    "G": {"1st": {"half": "UPPER", "opponent": "3rd", "from": ["A","E","H","I","J"]},
          "2nd": {"half": "LOWER", "opponent": "2D"}},
    "H": {"1st": {"half": "UPPER", "opponent": "2J"},
          "2nd": {"half": "LOWER", "opponent": "J1"}},
    "I": {"1st": {"half": "UPPER", "opponent": "3rd", "from": ["C","D","F","G","H"]},
          "2nd": {"half": "LOWER", "opponent": "2E"}},
    "J": {"1st": {"half": "LOWER", "opponent": "2H"},
          "2nd": {"half": "UPPER", "opponent": "H1"}},
    "K": {"1st": {"half": "LOWER", "opponent": "3rd", "from": ["D","E","I","J","L"]},
          "2nd": {"half": "UPPER", "opponent": "2L"}},
    "L": {"1st": {"half": "LOWER", "opponent": "3rd", "from": ["E","H","I","J","K"]},
          "2nd": {"half": "UPPER", "opponent": "2K"}},
}

# 上半区强队 vs 下半区强队（基于FIFA排名和历史）
# 上半区: E1(德国#10), F1(荷兰#8), H1(西班牙#2), I1(法国#3), D1(USA#17), G1(比利时#9)
# 下半区: A1(墨西哥#14), B1(?), C1(巴西#6), J1(阿根廷#1), K1(葡萄牙#5), L1(英格兰#4)
# 下半区明显更强（阿根廷#1+巴西#6+葡萄牙#5+英格兰#4）
UPPER_HALF_STRONG = ["Germany", "Spain", "France", "Netherlands", "Belgium"]
LOWER_HALF_STRONG = ["Argentina", "Brazil", "Portugal", "England"]

def _half_strength(half):
    """评估半区整体强度：返回该半区潜在强队数量"""
    if half == "UPPER":
        return len(UPPER_HALF_STRONG)
    return len(LOWER_HALF_STRONG)

def knockout_path_desirability(group, position):
    """
    返回 (半区, 对手难度描述, 半区强度差)
    正数 = 该位置路径更轻松，负数 = 更艰难
    """
    path = KNOCKOUT_PATH.get(group, {}).get(position)
    if not path:
        return "?", "?", 0

    half = path["half"]
    opp = path["opponent"]

    # 对手难度
    if opp == "3rd":
        opp_desc = "小组第三（较弱）"
        opp_diff = +1  # 第三名相对弱
    elif opp.startswith("2"):
        opp_desc = f"对手组{opp[1]}第二"
        opp_diff = 0
    else:
        opp_desc = f"对手组{opp[0]}第一（强）"
        opp_diff = -1

    # 半区强度：下半区强队更多，去上半区更轻松
    if half == "UPPER":
        half_diff = +1  # 上半区相对轻松
    else:
        half_diff = -1  # 下半区更卷

    total = opp_diff + half_diff
    return half, opp_desc, total


# ============================================================
# B. 赛程 & 休息天数
# ============================================================

# 每队最后一战的日期 → 用于计算休息天数
# 从 matches_2026.csv 自动加载 + 本模块硬编码赛程
# 完整赛程：按 (date, group, home, away) 编码已知比赛

KNOWN_MATCH_DATES = {
    # Matchday 1
    ("Mexico", "South Africa"): date(2026,6,11),
    ("South Korea", "Czech Republic"): date(2026,6,11),
    ("Canada", "Bosnia and Herzegovina"): date(2026,6,12),
    ("USA", "Paraguay"): date(2026,6,12),
    ("Qatar", "Switzerland"): date(2026,6,13),
    ("Brazil", "Morocco"): date(2026,6,13),
    ("Haiti", "Scotland"): date(2026,6,13),
    ("Australia", "Turkey"): date(2026,6,13),
    ("Germany", "Curacao"): date(2026,6,14),
    ("Ivory Coast", "Ecuador"): date(2026,6,14),
    ("Netherlands", "Japan"): date(2026,6,14),
    ("Sweden", "Tunisia"): date(2026,6,14),
    ("Belgium", "Egypt"): date(2026,6,15),
    ("Iran", "New Zealand"): date(2026,6,15),
    ("Spain", "Cape Verde"): date(2026,6,15),
    ("Saudi Arabia", "Uruguay"): date(2026,6,15),
    ("France", "Senegal"): date(2026,6,16),
    ("Iraq", "Norway"): date(2026,6,16),
    ("Argentina", "Algeria"): date(2026,6,16),
    ("Austria", "Jordan"): date(2026,6,16),
    # Matchday 1 K+L (6/17)
    ("Portugal", "DR Congo"): date(2026,6,17),
    ("Uzbekistan", "Colombia"): date(2026,6,17),
    ("England", "Croatia"): date(2026,6,17),
    ("Ghana", "Panama"): date(2026,6,17),
    # Matchday 2 A+B (6/18)
    ("Czech Republic", "South Africa"): date(2026,6,18),
    ("Mexico", "South Korea"): date(2026,6,18),
    ("Switzerland", "Bosnia and Herzegovina"): date(2026,6,18),
    ("Canada", "Qatar"): date(2026,6,18),
}

def load_match_history():
    """从 matches_2026.csv + schedule_2026.csv → {team: last_play_date}"""
    import pandas as pd
    last_play = {}

    # 已赛
    csv_path = ROOT / "data" / "raw" / "matches_2026.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
        for _, row in df.iterrows():
            d = pd.Timestamp(row["date"]).date()
            for team in [row["home_team"], row["away_team"]]:
                if team not in last_play or d > last_play[team]:
                    last_play[team] = d

    # 赛程中的未来比赛也纳入，用于计算休息日
    schedule_path = ROOT / "data" / "raw" / "schedule_2026.csv"
    if schedule_path.exists():
        sched = pd.read_csv(schedule_path, encoding="utf-8-sig")
        for _, row in sched.iterrows():
            d = pd.Timestamp(row["date"]).date()
            for team in [row["home_team"], row["away_team"]]:
                if team not in last_play or d > last_play[team]:
                    last_play[team] = d
    return last_play


def get_rest_days(team, match_date, last_play=None):
    """返回某队到 match_date 的休息天数。未打过比赛 → 返回 99（无限休息）"""
    if last_play is None:
        last_play = load_match_history()

    if isinstance(match_date, str):
        match_date = date.fromisoformat(match_date)

    prev = last_play.get(team)
    if prev is None:
        # 还没打过比赛（首轮）→ 无休息天数问题
        return 99
    return (match_date - prev).days


def rest_advantage(home, away, match_date, last_play=None):
    """
    返回 (home_rest, away_rest, advantage)
    advantage > 0 → 主队休息更充分
    """
    hr = get_rest_days(home, match_date, last_play)
    ar = get_rest_days(away, match_date, last_play)
    return hr, ar, hr - ar


# ============================================================
# C. 积分榜
# ============================================================

def get_group_standings(matches_df):
    """从比赛DataFrame计算小组积分榜 → {group: {team: {pts, gf, ga, gd, mp}}}"""
    standings = {}
    for g in GROUPS_2026:
        standings[g] = {}
        for t in GROUPS_2026[g]:
            standings[g][t] = {"pts": 0, "gf": 0, "ga": 0, "gd": 0, "mp": 0}

    if matches_df is None or len(matches_df) == 0:
        return standings

    for _, row in matches_df.iterrows():
        g = row.get("group")
        if g not in standings:
            continue
        h, a = row["home_team"], row["away_team"]
        hg, ag = int(row["home_score"]), int(row["away_score"])

        for team, gf, ga in [(h, hg, ag), (a, ag, hg)]:
            if team in standings[g]:
                s = standings[g][team]
                s["gf"] += gf
                s["ga"] += ga
                s["gd"] = s["gf"] - s["ga"]
                s["mp"] += 1

        if hg > ag:
            standings[g][h]["pts"] += 3
        elif ag > hg:
            standings[g][a]["pts"] += 3
        else:
            standings[g][h]["pts"] += 1
            standings[g][a]["pts"] += 1

    return standings


def get_team_standings(team, standings):
    """获取某队当前积分数据"""
    g = TEAM_GROUP.get(team)
    if not g or g not in standings:
        return None
    return standings[g].get(team)


def matchday_number(team, standings):
    """判断这是球队的第几轮比赛（已赛场次+1）"""
    s = get_team_standings(team, standings)
    if s is None:
        return 1
    return s["mp"] + 1


# ============================================================
# D. 排名修正 — FIFA 排名因球员实力、年龄结构等明显失真
# ============================================================

# 负值 = 真实实力比排名更强（排名数字应减小）
# 正值 = 真实实力比排名更弱（排名数字应增大）
RANKING_OVERRIDE = {
    # 明显低估 — 五大联赛球星集中但国家队排名低
    "Norway": -12,        # #31 → ~#19: Haaland(曼城)+Odegaard(阿森纳)+Sorloth(马竞)
    "Ghana": -20,         # #73 → ~#53: Partey(阿森纳)+Kudus(西汉姆)+Lamptey(布莱顿)
    "Canada": -5,         # #30 → ~#25: Davies(拜仁)+David(里尔)
    # 明显高估 — 黄金一代老化，真实实力下滑
    "Croatia": +7,        # #11 → ~#18: Modric 39岁, Perisic 37岁
    "Belgium": +5,         # #9 → ~#14: 黄金一代全面老化
    "Panama": +8,         # #34 → ~#42: 球员多在MLS/中北美联赛
}

def get_adjusted_rank(team, rankings):
    """返回修正后的排名，未在覆盖列表的返回原始排名"""
    raw = rankings.get(team, 50)
    override = RANKING_OVERRIDE.get(team, 0)
    return max(1, min(100, raw + override))

def get_ranking_note(team):
    """如果有排名修正，返回说明文字"""
    override = RANKING_OVERRIDE.get(team, 0)
    if override == 0:
        return None
    direction = "↓低估" if override < 0 else "↑高估"
    return f"RANK_ADJ: {team} FIFA排名{direction}（修正{abs(override)}位）"

# ============================================================
# E. 地缘政治档案
# ============================================================

@dataclass
class GeoProfile:
    level: str        # "SEVERE" | "MODERATE" | "MILD" | "NONE"
    win_penalty: float  # 胜率百分点扣除
    draw_uplift: float  # 平局概率上浮
    labels: list        # 风险标签

GEOPOLITICAL = {
    "Iran": GeoProfile(
        level="SEVERE",
        win_penalty=-0.12,   # 战争状态：胜率 -12pp
        draw_uplift=+0.05,   # 不确定性增加
        labels=[
            "WAR: 美伊处于战争状态（6/14和平协议刚签署）",
            "BASE: 训练营在墨西哥蒂华纳，不能在美国停留超48h",
            "FANS: 球迷票被美方取消，佩戴'168'金别针纪念空袭遇难者",
            "TRAVEL: 球员赛前10天才获签证，14名官员被拒签",
            "FORM: 已2-2平新西兰（排名85），表现明显低于纸面实力",
        ]
    ),
    "Haiti": GeoProfile(
        level="MILD",
        win_penalty=-0.03,
        draw_uplift=+0.01,
        labels=["US_TRAVEL_BAN: 在美旅行禁令名单"],
    ),
    "Ivory Coast": GeoProfile(
        level="MILD",
        win_penalty=-0.03,
        draw_uplift=+0.01,
        labels=["US_TRAVEL_BAN: 在美旅行禁令名单"],
    ),
    "Senegal": GeoProfile(
        level="MILD",
        win_penalty=-0.03,
        draw_uplift=+0.01,
        labels=["US_TRAVEL_BAN: 在美旅行禁令名单"],
    ),
    "DR Congo": GeoProfile(
        level="MILD",
        win_penalty=0.02,
        draw_uplift=+0.01,
        labels=["FIFA_SUSPENSION: 2月被FIFA暂停，5月恢复——逆境凝聚团队，反而提升战斗力"],
    ),
}

def get_geopolitical_impact(team):
    """返回 (win_penalty, draw_uplift, labels) 或 (0,0,[])"""
    gp = GEOPOLITICAL.get(team)
    if gp is None:
        return 0.0, 0.0, []
    return gp.win_penalty, gp.draw_uplift, gp.labels


# ============================================================
# 综合分析
# ============================================================

@dataclass
class MotivationAdjustment:
    draw_uplift: float = 0.0
    home_boost: float = 0.0
    away_boost: float = 0.0
    expected_goals_mod: float = 1.0
    risk_flags: list = field(default_factory=list)
    notes: list = field(default_factory=list)


def analyze_match(home, away, match_date, group, matches_df, last_play=None):
    """
    综合分析一场比赛的动机因素。
    match_date: date 对象或 'YYYY-MM-DD' 字符串
    """
    adj = MotivationAdjustment()
    if isinstance(match_date, str):
        match_date = date.fromisoformat(match_date)

    standings = get_group_standings(matches_df)
    home_s = get_team_standings(home, standings)
    away_s = get_team_standings(away, standings)
    home_md = matchday_number(home, standings)
    away_md = matchday_number(away, standings)
    md = max(home_md, away_md)

    # --- A. 赛制路径 + 已赛信息（晚踢的队看着前面结果踢） ---
    if group:
        _, _, h1_diff = knockout_path_desirability(group, "1st")
        _, _, h2_diff = knockout_path_desirability(group, "2nd")
        path_gap = h2_diff - h1_diff  # 正值 = 第二路径更好

        if md == 1:
            raw_rk = _load_fifa_rankings()
            adj_h = get_adjusted_rank(home, raw_rk)
            adj_a = get_adjusted_rank(away, raw_rk)
            rank_diff = abs(adj_h - adj_a)

            # A2. R32对手可见度：小组第一的潜在对手现在是什么水平
            matches_played = len(matches_df) if matches_df is not None else 0
            if matches_played >= 16:
                if abs(path_gap) <= 1:
                    w_path = KNOCKOUT_PATH.get(group, {}).get("1st", {})
                    opponent_from = w_path.get("from", [])
                    weak_opps = 0; total_opps = 0
                    for og in opponent_from:
                        for team in GROUPS_2026.get(og, []):
                            s = standings.get(og, {}).get(team, {})
                            if s:
                                total_opps += 1
                                if s.get("pts", 99) <= 1:
                                    weak_opps += 1
                    if total_opps >= 6:
                        weak_ratio = weak_opps / total_opps
                        if weak_ratio >= 0.6:
                            adj.expected_goals_mod *= 1.10
                            adj.home_boost += 0.02
                            adj.notes.append(f"R32: 小组第一潜在对手{weak_opps}/{total_opps}仅0-1分→路径很好，有动力")

            # A3. 强队路径选择：排名差大时调整发力
            if rank_diff >= 30:
                if path_gap <= -2:
                    adj.expected_goals_mod *= 1.15
                    adj.notes.append(f"PATH: 小组第一路径更优（gap={path_gap}），强队刷净胜球")
                elif path_gap >= 2:
                    adj.expected_goals_mod *= 0.85
                    adj.notes.append(f"PATH: 小组第一路径明显更差（gap={path_gap}），强队领先后收力")
                # gap在-1到+1之间：路径差异不大，看A2的R32分析

        elif md >= 3:
            # 第三轮：出线形势明朗，路径选择策略生效（权重减半，只在有明确动机时）
            if path_gap >= 2:
                adj.draw_uplift += 0.02
                adj.notes.append(f"PATH: 小组第一去{'下半区' if KNOCKOUT_PATH[group]['1st']['half']=='LOWER' else '上半区'}（艰难），第二更轻松（+{path_gap}）")
            elif path_gap >= 1:
                adj.draw_uplift += 0.01
                adj.notes.append(f"PATH: 小组第一路径略差于第二（差{path_gap}）")

    # --- B. 休息天数 ---
    if last_play is None:
        last_play = load_match_history()
    hr, ar, rest_diff = rest_advantage(home, away, match_date, last_play)

    if abs(rest_diff) >= 4:
        if rest_diff > 0:
            adj.home_boost += 0.06
            adj.notes.append(f"REST: {home}休{hr}天 vs {away}休{ar}天 → {home}优势 +6%")
        else:
            adj.away_boost += 0.06
            adj.notes.append(f"REST: {away}休{ar}天 vs {home}休{hr}天 → {away}优势 +6%")
    elif abs(rest_diff) >= 2:
        if rest_diff > 0:
            adj.home_boost += 0.03
            adj.notes.append(f"REST: {home}休{hr}天 vs {away}休{ar}天 → {home}优势 +3%")
        else:
            adj.away_boost += 0.03
            adj.notes.append(f"REST: {away}休{ar}天 vs {home}休{hr}天 → {away}优势 +3%")

    # --- C. 积分形势 ---
    if md == 2:
        # 第二轮：形势开始分化
        home_pts = home_s["pts"] if home_s else 0
        away_pts = away_s["pts"] if away_s else 0

        if home_pts == 3 and away_pts == 3:
            # 双方都赢了第一场 → 接受平局，各拿4分基本出线
            adj.draw_uplift += 0.03
            adj.risk_flags.append("双方首轮均胜→平局可接受")
        elif home_pts == 0 and away_pts == 0:
            # 双方都输了第一场 → 必须赢，更开放
            adj.expected_goals_mod = 1.15
            adj.draw_uplift -= 0.03
            adj.risk_flags.append("双方首轮均败→必争胜，比赛更开放")
        elif home_pts == 3 and away_pts == 0:
            adj.home_boost += 0.05
            adj.risk_flags.append(f"{home}首胜(3分) vs {away}首败(0分)→{home}可接受平局")
        elif home_pts == 0 and away_pts == 3:
            adj.away_boost += 0.05
            adj.risk_flags.append(f"{away}首胜(3分) vs {home}首败(0分)→{away}可接受平局")
        elif home_pts == 1 and away_pts == 1:
            adj.draw_uplift += 0.03
            adj.risk_flags.append("双方首轮均平→赢球可掌握主动权")

    elif md == 3:
        # 第三轮：出线生死战 — 完整场景分析
        home_pts = home_s["pts"] if home_s else 0
        away_pts = away_s["pts"] if away_s else 0
        home_gd = home_s["gd"] if home_s else 0
        away_gd = away_s["gd"] if away_s else 0
        home_gf = home_s["gf"] if home_s else 0
        away_gf = away_s["gf"] if away_s else 0

        # 同组其他两队的分差（用于判断是否需要看另一场结果）
        other_teams = [t for t in GROUPS_2026.get(group, []) if t not in (home, away)]
        other_standings = []
        for ot in other_teams:
            os = standings.get(group, {}).get(ot, {})
            other_standings.append((ot, os.get("pts", 0), os.get("gd", 0)))

        # 计算小组排名（当前）
        all_teams = [(home, home_pts, home_gd, home_gf),
                      (away, away_pts, away_gd, away_gf)]
        for ot_name, ot_pts, ot_gd in other_standings:
            ot_s = standings.get(group, {}).get(ot_name, {})
            all_teams.append((ot_name, ot_pts, ot_gd, ot_s.get("gf", 0)))
        all_teams.sort(key=lambda x: (-x[1], -x[2], -x[3]))

        # 判断每个队的当前排名
        current_rank = {}
        for i, (tname, _, _, _) in enumerate(all_teams):
            current_rank[tname] = i + 1

        home_rank = current_rank.get(home, 4)
        away_rank = current_rank.get(away, 4)

        def classify_team(pts, gd, gf, rank, other_pts_list):
            """判断一支球队在第三轮前的出线形势"""
            # 已确定出线
            if pts >= 6 and rank <= 2:
                return "QUALIFIED", "已出线→可能轮换主力"
            # 4分+好净胜球→平局基本出线
            if pts >= 4 and gd >= 3:
                return "NEARLY_THROUGH", f"{pts}分 GD{gd:+d}→平局即出线"
            # 4分+一般净胜球→需要拿分
            if pts >= 4:
                return "FIGHTING_EDGE", f"{pts}分 GD{gd:+d}→不败即出线"
            # 3分→赢球即出线，平局看另一场
            if pts == 3:
                return "FIGHTING", f"3分→必须拿分争夺出线权"
            # 1-2分→必须赢，但第三名也有机会
            if pts >= 1:
                return "MUST_WIN", f"{pts}分→必须赢球，争取第三名晋级"
            # 0分→基本出局但荣誉之战
            return "PRIDE", "0分→荣誉之战"

        home_status, home_note = classify_team(home_pts, home_gd, home_gf, home_rank,
                                                [p for _, p, _ in other_standings])
        away_status, away_note = classify_team(away_pts, away_gd, away_gf, away_rank,
                                                [p for _, p, _ in other_standings])

        # 状态→动机修正映射
        # (draw_uplift, goals_mod, home_boost, away_boost, risk_flag)
        STATUS_ADJUSTMENTS = {
            ("QUALIFIED", "QUALIFIED"): (0.15, 0.70, 0, 0,
                "** 双方均已出线→大面积轮换，低强度默契球"),
            ("QUALIFIED", "NEARLY_THROUGH"): (0.08, 0.80, -0.03, 0.03,
                "一方已出线轮换，另一方不败即出线"),
            ("QUALIFIED", "FIGHTING_EDGE"): (0.02, 0.90, -0.05, 0.08,
                "一方已出线轮换，另一方需拿分"),
            ("QUALIFIED", "FIGHTING"): (-0.03, 1.05, -0.06, 0.10,
                "一方已出线轮换，另一方必须抢分"),
            ("QUALIFIED", "MUST_WIN"): (-0.06, 1.15, -0.08, 0.12,
                "一方已出线轮换，另一方生死战"),
            ("QUALIFIED", "PRIDE"): (0.00, 1.00, -0.04, 0.06,
                "一方已出线，另一方为荣誉而战"),
            ("NEARLY_THROUGH", "NEARLY_THROUGH"): (0.12, 0.75, 0, 0,
                "** 双方不败即双双出线→大概率平局"),
            ("NEARLY_THROUGH", "FIGHTING_EDGE"): (0.06, 0.90, 0.02, 0.02,
                "双方均有出线主动权，平局对排名有利"),
            ("NEARLY_THROUGH", "FIGHTING"): (0.04, 1.00, 0.04, 0.00,
                "一方平局即出线，另一方需争胜"),
            ("NEARLY_THROUGH", "MUST_WIN"): (-0.02, 1.10, 0.06, 0.04,
                "一方轻松，另一方背水一战"),
            ("NEARLY_THROUGH", "PRIDE"): (0.00, 0.95, 0.06, 0.04,
                "一方接近出线，另一方荣誉之战"),
            ("FIGHTING_EDGE", "FIGHTING_EDGE"): (0.10, 0.85, 0, 0,
                "双方不败即出线→平局是双赢结果"),
            ("FIGHTING_EDGE", "FIGHTING"): (0.05, 1.00, 0.03, 0.03,
                "一方不败即出线，另一方需赢球"),
            ("FIGHTING_EDGE", "MUST_WIN"): (0.00, 1.10, 0.05, 0.05,
                "一方有主动权，另一方生死战"),
            ("FIGHTING_EDGE", "PRIDE"): (0.00, 1.05, 0.08, 0.02,
                "一方争出线，另一方荣誉之战"),
            ("FIGHTING", "FIGHTING"): (0.04, 1.08, 0, 0,
                "双方均需抢分→激烈对攻但平局可接受"),
            ("FIGHTING", "MUST_WIN"): (-0.03, 1.15, 0.03, 0.07,
                "一方有退路，另一方无退路"),
            ("FIGHTING", "PRIDE"): (-0.02, 1.10, 0.08, 0.02,
                "一方争出线，另一方为荣誉"),
            ("MUST_WIN", "MUST_WIN"): (-0.08, 1.25, 0, 0,
                "** 双方均须赢球→极度开放对攻"),
            ("MUST_WIN", "PRIDE"): (-0.04, 1.15, 0.06, 0.04,
                "一方生死战，另一方荣誉之战"),
            ("PRIDE", "PRIDE"): (0.02, 1.10, 0, 0,
                "双方均已出局→荣誉之战，开放但无压力"),
        }

        # 标准化状态对（QUALIFIED > NEARLY_THROUGH > FIGHTING_EDGE > FIGHTING > MUST_WIN > PRIDE）
        # 确保无论主客队都能匹配
        status_pair = (home_status, away_status)
        if status_pair in STATUS_ADJUSTMENTS:
            du, gm, hb, ab, flag = STATUS_ADJUSTMENTS[status_pair]
        else:
            # 反向匹配
            rev_pair = (away_status, home_status)
            if rev_pair in STATUS_ADJUSTMENTS:
                du, gm, ab, hb, flag = STATUS_ADJUSTMENTS[rev_pair]
            else:
                du, gm, hb, ab, flag = 0.0, 1.0, 0.0, 0.0, ""

        adj.draw_uplift += du
        adj.expected_goals_mod *= gm
        adj.home_boost += hb
        adj.away_boost += ab
        if flag:
            adj.risk_flags.append(flag)
        adj.notes.append(f"[R3] {home}({home_pts}pt GD{home_gd:+d}): {home_note}")
        adj.notes.append(f"[R3] {away}({away_pts}pt GD{away_gd:+d}): {away_note}")

    # --- D. 地缘政治 ---
    for team, is_home in [(home, True), (away, False)]:
        wp, du, labels = get_geopolitical_impact(team)
        if wp != 0:
            if is_home:
                adj.home_boost += wp  # wp 为负值 = 削弱
            else:
                adj.away_boost += wp
            adj.draw_uplift += du
            adj.risk_flags.extend(labels)

    return adj


def apply_motivation(base_probs, adj):
    """
    将 MotivationAdjustment 应用到基础概率上。
    base_probs: (p_h, p_d, p_a) 来自 DC 泊松模型
    返回: (p_h', p_d', p_a') 修正后概率
    """
    p_h, p_d, p_a = base_probs

    # 1. 应用主客队动机修正（胜率偏移）
    p_h += adj.home_boost
    p_a += adj.away_boost

    # 2. 应用平局上浮（从主客双方各取一半）
    p_h -= adj.draw_uplift * 0.5
    p_a -= adj.draw_uplift * 0.5
    p_d += adj.draw_uplift

    # 3. Clamp & renormalize
    p_h = max(0.01, p_h)
    p_d = max(0.01, p_d)
    p_a = max(0.01, p_a)
    total = p_h + p_d + p_a
    return p_h / total, p_d / total, p_a / total


# ============================================================
# 测试
# ============================================================
if __name__ == "__main__":
    import pandas as pd

    wc_path = ROOT / "data" / "raw" / "matches_2026.csv"
    df = pd.read_csv(wc_path, encoding="utf-8-sig") if wc_path.exists() else None

    print("=" * 70)
    print("  Motivation Module — 赛制 & 场外因素分析")
    print("=" * 70)

    # 积分榜
    standings = get_group_standings(df)
    for g in ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]:
        print(f"\n  Group {g}:")
        for t, s in sorted(standings[g].items(), key=lambda x: (-x[1]["pts"], -x[1]["gd"], -x[1]["gf"])):
            print(f"    {t:<20} {s['mp']}MP  {s['pts']}pts  GF{s['gf']} GA{s['ga']} GD{s['gd']:+d}")

    # 地缘政治影响
    print(f"\n  Geopolitical Impact:")
    for team in ["Iran", "Haiti", "Ivory Coast", "Senegal", "DR Congo"]:
        wp, du, labels = get_geopolitical_impact(team)
        if labels:
            print(f"    {team}: win{wp:+.0%} draw{du:+.0%}")
            for lb in labels:
                print(f"      → {lb}")

    # 休息天数示例
    print(f"\n  Rest Day Analysis (for 6/18 matches):")
    last_play = load_match_history()
    md = date(2026, 6, 18)
    for t in ["Mexico", "South Korea", "Czech Republic", "South Africa",
              "Canada", "Bosnia and Herzegovina", "Qatar", "Switzerland"]:
        rd = get_rest_days(t, md, last_play)
        print(f"    {t}: {rd}d rest before 6/18")
