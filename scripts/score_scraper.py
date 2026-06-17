"""
score_scraper.py — Wikipedia 比分爬虫
从 Wikipedia 小组子页面的 raw wikitext 解析比赛结果，追加到 matches_2026.csv
"""
import sys; sys.path.insert(0, 'scripts')
import re
import time
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).parent.parent

FIFA_CODE_TO_TEAM = {
    "MEX": "Mexico", "KOR": "South Korea", "CZE": "Czech Republic", "RSA": "South Africa",
    "CAN": "Canada", "BIH": "Bosnia and Herzegovina", "QAT": "Qatar", "SUI": "Switzerland",
    "BRA": "Brazil", "MAR": "Morocco", "HAI": "Haiti", "SCO": "Scotland",
    "USA": "USA", "PAR": "Paraguay", "AUS": "Australia", "TUR": "Turkey",
    "GER": "Germany", "CUW": "Curacao", "CIV": "Ivory Coast", "ECU": "Ecuador",
    "NED": "Netherlands", "JPN": "Japan", "SWE": "Sweden", "TUN": "Tunisia",
    "BEL": "Belgium", "EGY": "Egypt", "IRN": "Iran", "NZL": "New Zealand",
    "ESP": "Spain", "CPV": "Cape Verde", "KSA": "Saudi Arabia", "URU": "Uruguay",
    "FRA": "France", "SEN": "Senegal", "IRQ": "Iraq", "NOR": "Norway",
    "ARG": "Argentina", "ALG": "Algeria", "AUT": "Austria", "JOR": "Jordan",
    "POR": "Portugal", "COD": "DR Congo", "UZB": "Uzbekistan", "COL": "Colombia",
    "ENG": "England", "CRO": "Croatia", "GHA": "Ghana", "PAN": "Panama",
}

WIKI_RAW_URL = "https://en.wikipedia.org/w/index.php?title=2026_FIFA_World_Cup_Group_{group}&action=raw"
GROUPS = ["A","B","C","D","E","F","G","H","I","J","K","L"]
REQUEST_TIMEOUT = 30


def fetch_group_wikitext(group: str) -> str:
    url = WIKI_RAW_URL.format(group=group)
    try:
        resp = requests.get(url, timeout=REQUEST_TIMEOUT, headers={
            "User-Agent": "WorldCup2026-Predict-Bot/1.0"
        })
        resp.raise_for_status()
        return resp.text
    except requests.RequestException as e:
        print(f"  [WARN] Group {group} fetch failed: {e}")
        return ""


def extract_football_boxes(wikitext: str) -> list[str]:
    """用括号深度计数提取 football box 块，正确处理嵌套模板"""
    blocks = []
    lines = wikitext.split('\n')
    in_block = False
    current = []
    depth = 0

    for line in lines:
        if '{{#invoke:football box' in line:
            in_block = True
            current = [line]
            depth = line.count('{{') - line.count('}}')
        elif in_block:
            current.append(line)
            depth += line.count('{{') - line.count('}}')
            if depth <= 0:
                blocks.append('\n'.join(current))
                in_block = False
                current = []
    return blocks


def parse_date(box_text: str) -> str | None:
    m = re.search(r'\{\{Start date\|(\d{4})\|(\d{1,2})\|(\d{1,2})\}\}', box_text)
    if m:
        return f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
    return None


def parse_score(box_text: str) -> tuple[int, int] | None:
    """解析 |score= 参数：已完成返回 (h,a)，未赛(Match N)返回 None"""
    m = re.search(r'\|\s*score\s*=\s*\{\{score link\|[^|]*\|([^|}]+)\}\}', box_text)
    if not m:
        return None
    score_str = m.group(1).strip()
    if re.match(r'^Match\s+\d+', score_str):
        return None
    for sep in ['–', '-', '—']:
        if sep in score_str:
            parts = score_str.split(sep)
            try:
                return (int(parts[0].strip()), int(parts[1].strip()))
            except ValueError:
                return None
    return None


def parse_teams(box_text: str) -> tuple[str | None, str | None]:
    """从 football box 提取主客队名。主队取第一个 flag 调用，客队取第二个"""
    codes = re.findall(r'\{\{#invoke:flag\|(?:fb-rt|fb)\|([A-Z]+)\}\}', box_text)
    if len(codes) >= 2:
        home = FIFA_CODE_TO_TEAM.get(codes[0])
        away = FIFA_CODE_TO_TEAM.get(codes[1])
        return home, away
    return None, None


def load_schedule(schedule_path: Path) -> dict[tuple[str, str], str]:
    """从 schedule_2026.csv 加载 (home,away) → group 映射"""
    if not schedule_path.exists():
        return {}
    df = pd.read_csv(schedule_path, encoding="utf-8-sig")
    return dict(zip(zip(df["home_team"], df["away_team"]), df["group"]))


def load_existing_matches(csv_path: Path) -> set[tuple[str, str]]:
    if not csv_path.exists():
        return set()
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    return set(zip(df["home_team"], df["away_team"]))


def scrape_all() -> dict:
    csv_path = ROOT / "data" / "raw" / "matches_2026.csv"
    schedule_path = ROOT / "data" / "raw" / "schedule_2026.csv"
    log_path = ROOT / "data" / "raw" / "scraping_log.txt"

    existing = load_existing_matches(csv_path)
    schedule_map = load_schedule(schedule_path)
    all_new = []
    groups_failed = []
    groups_succeeded = []

    for group in GROUPS:
        print(f"  Fetching Group {group}...")
        wikitext = fetch_group_wikitext(group)
        if not wikitext:
            groups_failed.append(group)
            continue

        boxes = extract_football_boxes(wikitext)
        group_matches = 0
        for box in boxes:
            score = parse_score(box)
            if score is None:
                continue
            home, away = parse_teams(box)
            if not home or not away:
                continue
            match_date = parse_date(box) or "unknown"

            # 查 schedule 获取 group（Wikipedia 小组子页面可能包含跨组比赛）
            match_group = schedule_map.get((home, away), group)

            all_new.append({
                "date": match_date,
                "group": match_group,
                "home_team": home,
                "away_team": away,
                "home_score": score[0],
                "away_score": score[1],
            })
            group_matches += 1

        print(f"    {group_matches} completed matches")
        groups_succeeded.append(group)
        time.sleep(1)

    # 去重追加
    count = 0
    for m in all_new:
        pair = (m["home_team"], m["away_team"])
        if pair not in existing:
            row = f"{m['date']},{m['group']},{m['home_team']},{m['away_team']},{m['home_score']},{m['away_score']}\n"
            with open(csv_path, "a", encoding="utf-8-sig") as f:
                f.write(row)
            existing.add(pair)
            count += 1

    # 状态
    if not groups_succeeded:
        status = "error"
    elif groups_failed:
        status = "partial"
    elif count == 0:
        status = "no_new"
    else:
        status = "ok"

    total = len(existing)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"{timestamp} | {status.upper()} | {len(groups_succeeded)}/{len(GROUPS)} groups, {count} new (total: {total})"
    if groups_failed:
        log_entry += f" | failed: {','.join(groups_failed)}"

    print(f"\n  {log_entry}")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(log_entry + "\n")

    return {
        "status": status,
        "groups_succeeded": groups_succeeded,
        "groups_failed": groups_failed,
        "new_matches": count,
        "total_matches": total,
        "log": log_entry,
    }


if __name__ == "__main__":
    print("=" * 60)
    print("  Score Scraper — Wikipedia 2026 World Cup")
    print("=" * 60)
    result = scrape_all()
    sys.exit(0 if result["status"] != "error" else 1)
