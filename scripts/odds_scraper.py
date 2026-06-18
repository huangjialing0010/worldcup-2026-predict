"""
odds_scraper.py — 从 BetExplorer 抓取剩余比赛赔率
保存到 data/raw/odds_live.csv，供 predict_with_context.py 赔率融合使用
失败时不影响主流程
"""
import sys, os, re, requests, pandas as pd
from pathlib import Path
from datetime import date

ROOT = Path(__file__).parent.parent
OUTPUT = ROOT / "data" / "raw" / "odds_live.csv"

# 球队名映射：BetExplorer → 我们的 schedule
NAME_MAP = {
    "Bosnia & Herzegovina": "Bosnia and Herzegovina",
    "D.R. Congo": "DR Congo",
    "Cape Verde Islands": "Cape Verde",
    "Czech Republic": "Czech Republic",
    "South Korea": "South Korea",
    "South Africa": "South Africa",
    "Saudi Arabia": "Saudi Arabia",
    "Ivory Coast": "Ivory Coast",
    "New Zealand": "New Zealand",
    "Costa Rica": "Costa Rica",
    "United States": "USA",
    "Bosnia": "Bosnia and Herzegovina",
}

URL = "https://www.betexplorer.com/football/world/world-championship-2026/"

def fetch():
    """抓取赔率，返回 [(home, away, h_odds, d_odds, a_odds), ...]"""
    try:
        r = requests.get(URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
        r.raise_for_status()
    except Exception as e:
        print(f"[odds_scraper] HTTP error: {e}")
        return []

    rows = re.findall(r"<tr[^>]*>(.*?)</tr>", r.text, re.DOTALL)
    results = []

    for row in rows:
        if "data-odd" not in row:
            continue
        # 跳过已赛（含比分）
        if re.search(r">\d+:\d+<", row):
            continue

        # 去掉 HTML 标签
        clean = re.sub(r"<[^>]+>", " ", row)
        clean = re.sub(r"&nbsp;", " ", clean)
        clean = re.sub(r"\s+", " ", clean).strip()

        # 提取队名 "Team1 - Team2"
        m = re.search(
            r"([A-Z][A-Za-z]+(?:\s+(?:[A-Z][a-z]+|&[\w#]+;|\.R\.|\.\s*[A-Z][a-z]+|de\s+[A-Z][a-z]+|da\s+[A-Z][a-z]+|and\s+[A-Z][a-z]+|of\s+[A-Z][a-z]+))*)\s+-\s+"
            r"([A-Z][A-Za-z]+(?:\s+(?:[A-Z][a-z]+|&[\w#]+;|\.R\.|\.\s*[A-Z][a-z]+|de\s+[A-Z][a-z]+|da\s+[A-Z][a-z]+|and\s+[A-Z][a-z]+|of\s+[A-Z][a-z]+|[A-Z]\.\s*[A-Z][a-z]+))*)",
            clean,
        )
        if not m:
            continue

        home_raw = m.group(1).strip()
        away_raw = m.group(2).strip()

        # 去掉时间标记（Today / Tomorrow 等）
        home_raw = home_raw.replace(" Today", "").replace(" Tomorrow", "").strip()
        away_raw = away_raw.replace(" Today", "").replace(" Tomorrow", "").strip()

        # 标准化队名
        home = NAME_MAP.get(home_raw, home_raw)
        away = NAME_MAP.get(away_raw, away_raw)

        odds = re.findall(r'data-odd="([^"]+)"', row)
        if len(odds) < 3:
            continue

        try:
            h_odds = float(odds[0])
            d_odds = float(odds[1])
            a_odds = float(odds[2])
        except ValueError:
            continue

        # 过滤异常值
        if h_odds < 1.01 or d_odds < 1.5 or a_odds < 1.01:
            continue

        results.append((home, away, h_odds, d_odds, a_odds))

    return results


if __name__ == "__main__":
    print(f"[odds_scraper] {date.today()} Fetching from BetExplorer...")
    data = fetch()
    print(f"[odds_scraper] Found {len(data)} upcoming matches with odds")

    if data:
        df = pd.DataFrame({
            "match": [f"{h} vs {a}" for h, a, _, _, _ in data],
            "home_odds": [ho for _, _, ho, _, _ in data],
            "draw_odds": [do for _, _, _, do, _ in data],
            "away_odds": [ao for _, _, _, _, ao in data],
            "actual_home": [None] * len(data),
            "actual_away": [None] * len(data),
        })
        df.to_csv(OUTPUT, index=False, encoding="utf-8-sig")
        print(f"[odds_scraper] Saved to {OUTPUT}")
        for _, row in df.iterrows():
            print(f"  {row['match']:45s} H:{row['home_odds']:.2f} D:{row['draw_odds']:.2f} A:{row['away_odds']:.2f}")
    else:
        print("[odds_scraper] No odds found (page structure may have changed)")
