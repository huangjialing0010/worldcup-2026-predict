"""
从 API 数据源拉取 2026 世界杯数据
支持 RapidAPI (API-Football) 和 football-data.org
"""
import os
import json
import time
import hashlib
from pathlib import Path
from datetime import datetime

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).parent.parent
DATA_RAW = ROOT / "data" / "raw"
CACHE_DIR = ROOT / "data" / ".cache"

load_dotenv(ROOT / ".env")

RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY", "")
FOOTBALL_DATA_KEY = os.getenv("FOOTBALL_DATA_API_KEY", "")
DATA_SOURCE = os.getenv("DATA_SOURCE", "rapidapi")

# WC 2026: 联赛 ID 在各平台的映射
COMPETITION_IDS = {
    "rapidapi": 1,  # API-Football 中世界杯 ID = 1
    "footballdata": "WC",  # football-data.org 中世界杯 code = WC
}

RAPIDAPI_HOST = "api-football-v1.p.rapidapi.com"
FOOTBALLDATA_HOST = "https://api.football-data.org/v4"


def cache_key(url, params=None):
    raw = url + json.dumps(params or {}, sort_keys=True)
    return hashlib.md5(raw.encode()).hexdigest()


def cache_get(key):
    path = CACHE_DIR / key
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        if time.time() - data["ts"] < 3600:  # 1 小时缓存
            return data["payload"]
    return None


def cache_set(key, payload):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    (CACHE_DIR / key).write_text(
        json.dumps({"ts": time.time(), "payload": payload}, ensure_ascii=False),
        encoding="utf-8",
    )


def _rapidapi(endpoint, params=None):
    url = f"https://{RAPIDAPI_HOST}/v3/{endpoint}"
    headers = {
        "X-RapidAPI-Key": RAPIDAPI_KEY,
        "X-RapidAPI-Host": RAPIDAPI_HOST,
    }
    resp = requests.get(url, headers=headers, params=params or {}, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if data.get("errors"):
        raise RuntimeError(f"API error: {data['errors']}")
    return data


def _footballdata(endpoint, params=None):
    url = f"{FOOTBALLDATA_HOST}/{endpoint}"
    headers = {"X-Auth-Token": FOOTBALL_DATA_KEY}
    resp = requests.get(url, headers=headers, params=params or {}, timeout=30)
    resp.raise_for_status()
    return resp.json()


def _fetch(endpoint, params=None):
    """统一入口：根据 DATA_SOURCE 路由，带缓存"""
    ck = cache_key(f"{DATA_SOURCE}:{endpoint}", params)
    cached = cache_get(ck)
    if cached is not None:
        return cached

    if DATA_SOURCE == "footballdata" and FOOTBALL_DATA_KEY:
        payload = _footballdata(endpoint, params)
    elif DATA_SOURCE == "rapidapi" and RAPIDAPI_KEY:
        payload = _rapidapi(endpoint, params)
    else:
        raise RuntimeError(
            f"数据源 {DATA_SOURCE} 未配置 API key。请设置 .env 中的对应 key。"
        )

    cache_set(ck, payload)
    time.sleep(1.5)  # rate limit
    return payload


# --- 比赛数据 ---


def fetch_fixtures(season=2026):
    """
    拉取 2026 世界杯所有比赛（含比分）
    RapidAPI: fixtures?league=1&season=2026
    FootballData: competitions/WC/matches?season=2026
    """
    if DATA_SOURCE == "footballdata":
        data = _fetch(f"competitions/WC/matches", {"season": season})
        matches = data.get("matches", [])
    else:
        data = _fetch("fixtures", {
            "league": COMPETITION_IDS["rapidapi"],
            "season": season,
        })
        matches = data.get("response", [])

    DATA_RAW.mkdir(parents=True, exist_ok=True)
    out_path = DATA_RAW / f"fixtures_{season}.json"
    out_path.write_text(
        json.dumps(matches, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"已拉取 {len(matches)} 场比赛 → {out_path}")
    return matches


def fetch_teams(season=2026):
    """
    拉取世界杯参赛队伍
    RapidAPI: teams?league=1&season=2026
    FootballData: competitions/WC/teams?season=2026
    """
    if DATA_SOURCE == "footballdata":
        data = _fetch(f"competitions/WC/teams", {"season": season})
        teams = data.get("teams", [])
    else:
        data = _fetch("teams", {
            "league": COMPETITION_IDS["rapidapi"],
            "season": season,
        })
        teams = data.get("response", [])

    DATA_RAW.mkdir(parents=True, exist_ok=True)
    out_path = DATA_RAW / f"teams_{season}.json"
    out_path.write_text(
        json.dumps(teams, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"已拉取 {len(teams)} 支队伍 → {out_path}")
    return teams


def fetch_odds(fixture_id):
    """拉取单场比赛的赔率（RapidAPI 专属）"""
    if DATA_SOURCE != "rapidapi":
        print("赔率数据仅 RapidAPI 支持")
        return None

    data = _fetch("odds", {"fixture": fixture_id})
    odds = data.get("response", [])

    path = DATA_RAW / f"odds_{fixture_id}.json"
    path.write_text(json.dumps(odds, ensure_ascii=False, indent=2), encoding="utf-8")
    return odds


def main():
    print(f"数据源: {DATA_SOURCE}")
    print(f"RapidAPI key: {'已配置' if RAPIDAPI_KEY else '未配置'}")
    print(f"FootballData key: {'已配置' if FOOTBALL_DATA_KEY else '未配置'}")
    print()

    try:
        teams = fetch_teams(2026)
        fixtures = fetch_fixtures(2026)
    except RuntimeError as e:
        print(f"\n错误: {e}")
        print("请在 .env 中配置至少一个数据源的 API key。")
        print("  cp .env.example .env")
        print("  然后编辑 .env 填入你的 API key")
        return 1

    # 汇总信息
    finished = [f for f in fixtures
                if _get_status(f) in ("FT", "AET", "PEN", "FINISHED")]
    scheduled = [f for f in fixtures
                 if _get_status(f) not in ("FT", "AET", "PEN", "FINISHED")]

    print(f"\n汇总: {len(teams)} 队, {len(fixtures)} 场比赛")
    print(f"  已完成: {len(finished)}")
    print(f"  未赛: {len(scheduled)}")
    return 0


def _get_status(fixture):
    """兼容不同数据源的字段名"""
    if isinstance(fixture, dict):
        f = fixture.get("fixture", fixture)
        return f.get("status", {}).get("short", "") if isinstance(f.get("status"), dict) else f.get("status", "")
    return ""


if __name__ == "__main__":
    raise SystemExit(main())
