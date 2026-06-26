"""
odds_scraper_sporttery.py — 从中国体彩竞彩网抓取赔率
API: webapi.sporttery.cn (无需代理)
保存到 data/raw/odds_sporttery.csv，格式与 odds_live.csv 一致
"""
import requests, json, pandas as pd
from pathlib import Path
from datetime import date, datetime

ROOT = Path(__file__).parent.parent
OUTPUT = ROOT / "data" / "raw" / "odds_sporttery.csv"

URL = "https://webapi.sporttery.cn/gateway/uniform/football/getMatchListV1.qry?clientCode=3001"

# 中文队名 → 英文队名（从 JSON 加载，避免源文件编码问题）
import json as _json
with open(ROOT / "data" / "processed" / "team_name_map.json", "r", encoding="utf-8") as _f:
    CN_TO_EN = _json.load(_f)


def fetch():
    """从体彩API抓取赔率，返回 [(home_en, away_en, h_odds, d_odds, a_odds, match_date), ...]"""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Referer": "https://www.sporttery.cn/",
        "Accept": "application/json",
    }
    try:
        r = requests.get(URL, headers=headers, timeout=15)
        r.raise_for_status()
    except Exception as e:
        print(f"[sporttery] HTTP error: {e}")
        return []

    data = r.json()
    if not data.get("success"):
        print(f"[sporttery] API error: {data.get('errorMessage', 'unknown')}")
        return []

    results = []
    for match in _find_matches(data["value"]):
        home_cn = match.get("homeTeamAllName", "")
        away_cn = match.get("awayTeamAllName", "")
        match_date = match.get("matchDate", "")
        match_time = match.get("matchTime", "00:00")
        league = match.get("leagueAllName", "")

        # 只取世界杯
        if "世界杯" not in league:
            continue

        # 跳过非竞彩场次
        if match.get("matchStatus") != "Selling":
            continue

        home_en = CN_TO_EN.get(home_cn, home_cn)
        away_en = CN_TO_EN.get(away_cn, away_cn)

        # 提取 HAD 赔率
        had_odds = None
        for pool in match.get("oddsList", []):
            if pool.get("poolCode") == "HAD":
                try:
                    h = float(pool["h"])
                    d = float(pool["d"])
                    a = float(pool["a"])
                    if h >= 1.01 and d >= 1.5 and a >= 1.01:
                        had_odds = (h, d, a)
                except (ValueError, KeyError):
                    pass
                break

        if had_odds is None:
            continue

        # 构造 match_date + match_time
        try:
            dt = datetime.fromisoformat(f"{match_date}T{match_time}:00")
            date_str = dt.strftime("%Y-%m-%d")
        except ValueError:
            date_str = match_date

        results.append((home_en, away_en, had_odds[0], had_odds[1], had_odds[2], date_str))

    return results


def _find_matches(data):
    """递归遍历 JSON 树，找到所有 match 对象"""
    results = []
    if isinstance(data, list):
        for item in data:
            results.extend(_find_matches(item))
    elif isinstance(data, dict):
        if "homeTeamAllName" in data and "awayTeamAllName" in data:
            results.append(data)
        for v in data.values():
            results.extend(_find_matches(v))
    return results


if __name__ == "__main__":
    print(f"[sporttery] {date.today()} Fetching from 中国体彩竞彩网...")
    data = fetch()
    print(f"[sporttery] Found {len(data)} upcoming World Cup matches with HAD odds")

    if data:
        df = pd.DataFrame({
            "match": [f"{h} vs {a}" for h, a, _, _, _, _ in data],
            "home_odds": [ho for _, _, ho, _, _, _ in data],
            "draw_odds": [do for _, _, _, do, _, _ in data],
            "away_odds": [ao for _, _, _, _, ao, _ in data],
            "match_date": [md for _, _, _, _, _, md in data],
            "actual_home": [None] * len(data),
            "actual_away": [None] * len(data),
        })
        df.to_csv(OUTPUT, index=False, encoding="utf-8-sig")
        print(f"[sporttery] Saved to {OUTPUT}")
        for _, row in df.iterrows():
            print(f"  {row['match']:40s} {row['match_date']}  H:{row['home_odds']:.2f} D:{row['draw_odds']:.2f} A:{row['away_odds']:.2f}")
    else:
        print("[sporttery] No World Cup odds found")
