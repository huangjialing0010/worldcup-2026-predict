"""
清洗原始 API 数据 → 标准化 CSV 放入 data/processed/
输出: matches.csv (统一格式的比赛数据), teams.csv (队伍+排名)
"""
import json
import pandas as pd
from pathlib import Path
from datetime import datetime

from model_utils import FIFA_RANKINGS

ROOT = Path(__file__).parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"


def load_raw_fixtures(season=2026):
    """加载原始比赛 JSON"""
    path = DATA_RAW / f"fixtures_{season}.json"
    if not path.exists():
        print(f"未找到 {path}，请先运行 fetch_data.py")
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def load_raw_teams(season=2026):
    """加载原始队伍 JSON"""
    path = DATA_RAW / f"teams_{season}.json"
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_rapidapi_fixture(f):
    """解析 RapidAPI 格式的比赛数据"""
    fixture = f.get("fixture", {})
    league = f.get("league", {})
    teams = f.get("teams", {})
    goals = f.get("goals", {})
    score = f.get("score", {})

    status = fixture.get("status", {}).get("short", "")
    home = teams.get("home", {}).get("name", "")
    away = teams.get("away", {}).get("name", "")
    home_goals = goals.get("home")
    away_goals = goals.get("away")

    return {
        "fixture_id": fixture.get("id"),
        "date": fixture.get("date", "")[:10],
        "round": league.get("round", ""),
        "home_team": home,
        "away_team": away,
        "home_goals": home_goals,
        "away_goals": away_goals,
        "status": status,
        "venue": fixture.get("venue", {}).get("name", ""),
        "source": "rapidapi",
    }


def _parse_footballdata_fixture(f):
    """解析 football-data.org 格式的比赛数据"""
    home = f.get("homeTeam", {}).get("name", "")
    away = f.get("awayTeam", {}).get("name", "")
    score = f.get("score", {}).get("fullTime", {})
    status = f.get("status", "")

    return {
        "fixture_id": f.get("id"),
        "date": f.get("utcDate", "")[:10],
        "round": f.get("matchday", ""),
        "home_team": home,
        "away_team": away,
        "home_goals": score.get("home"),
        "away_goals": score.get("away"),
        "status": status,
        "venue": f.get("venue", ""),
        "source": "footballdata",
    }


def process_fixtures(fixtures):
    """标准化比赛数据 → DataFrame"""
    rows = []
    for f in fixtures:
        if isinstance(f, dict):
            if "fixture" in f:
                rows.append(_parse_rapidapi_fixture(f))
            elif "homeTeam" in f:
                rows.append(_parse_footballdata_fixture(f))

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values("date").reset_index(drop=True)

    # 标记是否已完成
    df["is_finished"] = df["status"].isin(("FT", "AET", "PEN", "FINISHED"))

    return df


def process_teams(raw_teams):
    """标准化队伍数据，附带 FIFA 排名"""
    rows = []
    for t in raw_teams:
        if isinstance(t, dict):
            if "team" in t:  # RapidAPI 格式
                info = t.get("team", {})
                name = info.get("name", "")
            elif "name" in t:  # football-data.org 格式
                info = t
                name = info.get("name", "")
            else:
                continue

            rows.append({
                "team_name": name,
                "fifa_rank": FIFA_RANKINGS.get(name, None),
                "country": info.get("country", ""),
                "team_id": info.get("id", ""),
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("fifa_rank").reset_index(drop=True)
    return df


def generate_elo_ratings(df_teams):
    """
    基于 FIFA 排名生成近似 ELO 评分
    ELO ≈ 2400 - (FIFA rank - 1) * 6
    """
    df = df_teams.copy()
    df["elo"] = 2400 - (df["fifa_rank"].fillna(50) - 1) * 6
    return df


def generate_fallback_matches():
    """从硬编码数据生成标准化的比赛 DataFrame"""
    FALLBACK_MATCHES = [
        ("Mexico", "South Africa", 2, 0, "2026-06-11", "Group Stage - 1"),
        ("South Korea", "Czech Republic", 2, 1, "2026-06-11", "Group Stage - 1"),
        ("Canada", "Bosnia and Herzegovina", 1, 1, "2026-06-12", "Group Stage - 1"),
        ("Qatar", "Switzerland", 1, 1, "2026-06-12", "Group Stage - 1"),
        ("Brazil", "Morocco", 1, 1, "2026-06-12", "Group Stage - 1"),
        ("Haiti", "Scotland", 0, 1, "2026-06-13", "Group Stage - 1"),
        ("USA", "Paraguay", 4, 1, "2026-06-13", "Group Stage - 1"),
        ("Australia", "Turkey", 2, 0, "2026-06-13", "Group Stage - 1"),
        ("Germany", "Curacao", 7, 1, "2026-06-14", "Group Stage - 1"),
        ("Ivory Coast", "Ecuador", 1, 0, "2026-06-14", "Group Stage - 1"),
        ("Netherlands", "Japan", 2, 2, "2026-06-14", "Group Stage - 1"),
        ("Sweden", "Tunisia", 5, 1, "2026-06-14", "Group Stage - 1"),
        ("Belgium", "Egypt", 1, 1, "2026-06-15", "Group Stage - 1"),
        ("Iran", "New Zealand", 2, 2, "2026-06-15", "Group Stage - 1"),
        ("Spain", "Cape Verde", 0, 0, "2026-06-15", "Group Stage - 1"),
        ("Saudi Arabia", "Uruguay", 1, 1, "2026-06-15", "Group Stage - 1"),
    ]
    rows = []
    for h, a, hg, ag, date, round_ in FALLBACK_MATCHES:
        rows.append({
            "fixture_id": None,
            "date": date,
            "round": round_,
            "home_team": h,
            "away_team": a,
            "home_goals": hg,
            "away_goals": ag,
            "status": "FT",
            "venue": "",
            "source": "fallback",
            "is_finished": True,
        })
    return pd.DataFrame(rows)


def generate_fallback_teams():
    """从硬编码 FIFA 排名生成队伍 DataFrame"""
    rows = [{"team_name": k, "fifa_rank": v, "country": "", "team_id": ""}
            for k, v in FIFA_RANKINGS.items()]
    return pd.DataFrame(rows)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="处理原始世界杯数据")
    parser.add_argument("--fallback", action="store_true",
                        help="API 数据不可用时用硬编码回退数据")
    args = parser.parse_args()

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    fixtures = load_raw_fixtures()
    if fixtures:
        df_matches = process_fixtures(fixtures)
    elif args.fallback:
        print("API 数据不可用，使用硬编码回退数据")
        df_matches = generate_fallback_matches()
    else:
        print("无原始比赛数据，跳过。请先运行 fetch_data.py 或加 --fallback")
        return

    raw_teams = load_raw_teams()

    # 处理比赛数据
    matches_path = DATA_PROCESSED / "matches.csv"
    df_matches.to_csv(matches_path, index=False, encoding="utf-8-sig")
    print(f"比赛数据: {len(df_matches)} 场 → {matches_path}")

    # 处理队伍数据
    if raw_teams:
        df_teams = process_teams(raw_teams)
    else:
        df_teams = generate_fallback_teams()
    df_teams = generate_elo_ratings(df_teams)
    teams_path = DATA_PROCESSED / "teams.csv"
    df_teams.to_csv(teams_path, index=False, encoding="utf-8-sig")
    print(f"队伍数据: {len(df_teams)} 队 → {teams_path}")

    # 打印概要
    finished = df_matches[df_matches["is_finished"]] if not df_matches.empty else pd.DataFrame()
    upcoming = df_matches[~df_matches["is_finished"]] if not df_matches.empty else pd.DataFrame()

    print(f"\n概览:")
    print(f"  已完成: {len(finished)} 场")
    print(f"  未赛: {len(upcoming)} 场")

    if not finished.empty:
        total_goals = finished["home_goals"].sum() + finished["away_goals"].sum()
        games = len(finished)
        print(f"  场均进球: {total_goals / games:.2f}" if games > 0 else "")

        home_wins = (finished["home_goals"] > finished["away_goals"]).sum()
        draws = (finished["home_goals"] == finished["away_goals"]).sum()
        away_wins = (finished["home_goals"] < finished["away_goals"]).sum()
        print(f"  主胜/平/客胜: {home_wins}/{draws}/{away_wins}")


if __name__ == "__main__":
    main()
