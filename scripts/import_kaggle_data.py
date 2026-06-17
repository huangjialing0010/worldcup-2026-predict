"""
从 Kaggle 国际比赛数据集导入历史数据
数据集: martj42/international-football-results-from-1872-to-2017
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys

ROOT = Path(__file__).parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"

KAGGLE_DIR = Path.home() / ".cache" / "kagglehub" / "datasets" / \
    "martj42" / "international-football-results-from-1872-to-2017" / "versions" / "123"

# 团队名映射：Kaggle → 我们的名称
TEAM_NAME_MAP = {
    "Curaçao": "Curacao",
    "United States": "USA",
    "Korea Republic": "South Korea",
}

# WC 2026 参赛队（我们排名表中有的队伍）
OUR_TEAMS = {
    "Argentina", "Spain", "France", "England", "Portugal", "Brazil", "Morocco",
    "Netherlands", "Belgium", "Germany", "Croatia", "Colombia", "Mexico", "Senegal",
    "Uruguay", "USA", "Japan", "Switzerland", "Iran", "Turkey", "Ecuador", "Austria",
    "South Korea", "Australia", "Algeria", "Egypt", "Canada", "Norway", "Ivory Coast",
    "Panama", "Sweden", "Czech Republic", "Paraguay", "Scotland", "Tunisia", "DR Congo",
    "Uzbekistan", "Qatar", "Iraq", "Saudi Arabia", "Jordan", "Bosnia and Herzegovina",
    "South Africa", "Cape Verde", "Curacao", "Haiti", "New Zealand", "Ghana",
}


def map_team(name):
    return TEAM_NAME_MAP.get(name, name)


def main():
    results_path = KAGGLE_DIR / "results.csv"
    if not results_path.exists():
        print("Kaggle 数据未找到，请先运行: python -c \"import kagglehub; kagglehub.dataset_download('martj42/international-football-results-from-1872-to-2017')\"")
        return 1

    df = pd.read_csv(results_path)
    print(f"原始数据: {len(df)} 场比赛 ({df['date'].min()} ~ {df['date'].max()})")

    # 过滤有效比分
    df = df.dropna(subset=["home_score", "away_score"])
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    print(f"有效比分: {len(df)} 场")

    # 映射团队名
    df["home_team"] = df["home_team"].map(map_team)
    df["away_team"] = df["away_team"].map(map_team)
    df["date"] = pd.to_datetime(df["date"])

    # ---- 策略 ----
    # 1. 所有世界杯正赛（含历年）
    # 2. 2018+ 国际友谊赛、洲际杯、预选赛（涉及 WC 2026 参赛队）

    wc = df[df["tournament"] == "FIFA World Cup"].copy()
    print(f"\n世界杯正赛: {len(wc)} 场")

    # 世界杯按年份分布
    wc["year"] = wc["date"].dt.year
    wc_years = wc["year"].value_counts().sort_index()
    print("世界杯年份分布:")
    for y, c in wc_years.items():
        print(f"  {y}: {c} 场")

    # 其他赛事: 2018 年后 + 涉及 WC 2026 参赛队
    other = df[
        (df["tournament"] != "FIFA World Cup") &
        (df["date"] >= "2018-01-01") &
        (df["home_team"].isin(OUR_TEAMS) | df["away_team"].isin(OUR_TEAMS))
    ].copy()
    print(f"\n2018+ 其他赛事 (涉及 WC2026 队): {len(other)} 场")
    print(f"  赛事类型: {other['tournament'].value_counts().head(10).to_dict()}")

    # 合并
    all_matches = pd.concat([wc, other], ignore_index=True)
    all_matches = all_matches.drop_duplicates(
        subset=["date", "home_team", "away_team"]
    ).sort_values("date")

    print(f"\n合并去重后: {len(all_matches)} 场")

    # 保存原始数据
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    out_path = DATA_RAW / "historical_matches.csv"
    all_matches.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"原始数据已保存: {out_path}")

    # 构建 processed 格式
    processed = all_matches.copy()
    processed = processed.rename(columns={
        "home_score": "home_goals",
        "away_score": "away_goals",
    })
    processed["round"] = processed["tournament"]
    processed["is_finished"] = True
    processed["source"] = "kaggle"
    processed["fixture_id"] = ""
    processed["status"] = "FT"
    processed["venue"] = ""

    cols = ["fixture_id", "date", "round", "home_team", "away_team",
            "home_goals", "away_goals", "status", "venue", "source", "is_finished"]
    processed = processed[cols]

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    proc_path = DATA_PROCESSED / "matches.csv"
    processed.to_csv(proc_path, index=False, encoding="utf-8-sig")
    print(f"处理后数据已保存: {proc_path}")
    print(f"  → {len(processed)} 场比赛可供回测")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
