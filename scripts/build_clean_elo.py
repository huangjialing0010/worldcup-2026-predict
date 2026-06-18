"""
build_clean_elo.py — 纯 W/L/D ELO 评级（不依赖进球差分）
基于 4580 场历史比赛，只有赢/平/输传递信号，无循环依赖
"""
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent.parent
K = 24  # ELO K-factor
START = 1500
SCALE = 400

# Load all historical matches
matches = pd.read_csv(ROOT / "data" / "processed" / "matches.csv", encoding="utf-8-sig")
matches["date"] = pd.to_datetime(matches["date"])
matches = matches.sort_values("date").reset_index(drop=True)
print(f"Processing {len(matches)} matches from {matches['date'].min().date()} to {matches['date'].max().date()}")

# ELO computation — save pre-match ELO for every match
elo = {}
elo_h_list, elo_a_list = [], []
for _, row in matches.iterrows():
    h, a = row["home_team"], row["away_team"]
    hg, ag = int(row["home_goals"]), int(row["away_goals"])

    elo_h = elo.get(h, START)
    elo_a = elo.get(a, START)
    elo_h_list.append(elo_h)
    elo_a_list.append(elo_a)

    # Expected score
    e_h = 1 / (1 + 10 ** ((elo_a - elo_h) / SCALE))
    e_a = 1 - e_h

    # Actual score (W/L/D only, no goal margin)
    if hg > ag:
        s_h, s_a = 1.0, 0.0
    elif ag > hg:
        s_h, s_a = 0.0, 1.0
    else:
        s_h, s_a = 0.5, 0.5

    elo[h] = elo_h + K * (s_h - e_h)
    elo[a] = elo_a + K * (s_a - e_a)

# Save pre-match ELO for training
matches["elo_h_clean"] = elo_h_list
matches["elo_a_clean"] = elo_a_list
matches.to_csv(ROOT / "data" / "processed" / "matches_with_elo.csv", index=False, encoding="utf-8-sig")

# Save latest ELO for each team
pd.DataFrame({"team": list(elo.keys()), "elo": [round(v) for v in elo.values()]}).to_csv(
    ROOT / "data" / "processed" / "clean_elo.csv", index=False, encoding="utf-8-sig")

print(f"Final ELO range: {min(elo.values()):.0f} - {max(elo.values()):.0f}")
print(f"Median ELO: {np.median(list(elo.values())):.0f}")
print(f"Teams: {len(elo)}")
print(f"Saved clean_elo.csv and matches_with_elo.csv")

# Show 2026 WC teams
WC_TEAMS = {
    "Algeria", "Argentina", "Australia", "Austria", "Belgium",
    "Bosnia and Herzegovina", "Brazil", "Canada", "Cape Verde", "Colombia",
    "Croatia", "Curacao", "Czech Republic", "DR Congo", "Ecuador",
    "Egypt", "England", "France", "Germany", "Ghana",
    "Haiti", "Iran", "Iraq", "Ivory Coast", "Japan",
    "Jordan", "Mexico", "Morocco", "Netherlands", "New Zealand",
    "Norway", "Panama", "Paraguay", "Portugal", "Qatar",
    "Saudi Arabia", "Scotland", "Senegal", "South Africa", "South Korea",
    "Spain", "Sweden", "Switzerland", "Tunisia", "Turkey",
    "USA", "Uruguay", "Uzbekistan",
}
print(f"\n2026 World Cup teams (ELO):")
for t in sorted(WC_TEAMS, key=lambda x: elo.get(x, START), reverse=True):
    print(f"  {t:25s} {elo.get(t, START):.0f}")
