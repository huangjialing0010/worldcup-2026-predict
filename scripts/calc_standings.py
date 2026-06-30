"""
Calculate group standings and knockout bracket for 2026 World Cup.
"""
import pandas as pd
import numpy as np
from io import StringIO

df = pd.read_csv('data/raw/matches_2026.csv', encoding='utf-8-sig')

# Calculate standings
standings = {}
for group in sorted(df['group'].unique()):
    gdf = df[df['group'] == group]
    teams = {}
    for _, row in gdf.iterrows():
        home, away = row['home_team'], row['away_team']
        hg, ag = row['home_score'], row['away_score']

        for team in [home, away]:
            if team not in teams:
                teams[team] = {'Pts': 0, 'GF': 0, 'GA': 0, 'GD': 0}

        # Goals
        teams[home]['GF'] += hg
        teams[home]['GA'] += ag
        teams[away]['GF'] += ag
        teams[away]['GA'] += hg

        # Points
        if hg > ag:
            teams[home]['Pts'] += 3
        elif hg < ag:
            teams[away]['Pts'] += 3
        else:
            teams[home]['Pts'] += 1
            teams[away]['Pts'] += 1

    for t in teams:
        teams[t]['GD'] = teams[t]['GF'] - teams[t]['GA']

    sorted_teams = sorted(teams.items(), key=lambda x: (-x[1]['Pts'], -x[1]['GD'], -x[1]['GF']))
    standings[group] = sorted_teams

    print(f"\n=== Group {group} ===")
    for i, (team, stats) in enumerate(sorted_teams):
        pos = f"{'1st' if i==0 else '2nd' if i==1 else '3rd' if i==2 else '4th'}"
        print(f"  {pos} {team:<25} {stats['Pts']}pts  GF{stats['GF']} GA{stats['GA']} GD{stats['GD']:+d}")

# Determine 3rd place ranking
print("\n=== 3rd Place Teams ===")
third_place = []
for group in sorted(standings.keys()):
    team, stats = standings[group][2]
    third_place.append((team, group, stats['Pts'], stats['GD'], stats['GF']))

third_place.sort(key=lambda x: (-x[2], -x[3], -x[4]))
for i, (team, group, pts, gd, gf) in enumerate(third_place):
    qual = "IN" if i < 8 else "OUT"
    print(f"  {qual} {team:<25} (Grp {group}) {pts}pts GD{gd:+d} GF{gf}")

# Now determine the knockout bracket
# 2026 World Cup Round of 32 matchups (pre-determined by FIFA):
# Based on the official match schedule pattern for 2026
print("\n=== Group Winners & Runners-up ===")
for group in sorted(standings.keys()):
    w = standings[group][0]
    r = standings[group][1]
    print(f"  Group {group}: 1st={w[0]:<22} 2nd={r[0]:<22}")

# The 2026 bracket follows this pattern:
# Match 73: 1A vs 3rd (best from CDEFIJKL - one of the 8 best)
# Match 74: 1B vs 3rd
# etc.

# Actually, FIFA pre-announced the bracket pairs. Without the official bracket,
# let me use the most common 2026 format:
# The 48-team format has pre-set paths. Let me check if Wikipedia has the bracket info.

# For now, let's print what we can figure out from the odds data
print("\n=== Confirmed from odds data ===")
print("Brazil vs Japan - Brazil (1st Grp C?) vs Japan (2nd Grp F?)")
print("Germany vs Paraguay - Germany (1st Grp E) vs Paraguay (2nd Grp D?)")
print("Netherlands vs Morocco - Netherlands (1st/2nd Grp F?) vs Morocco (1st/2nd Grp C?)")
print("Ivory Coast vs Norway")
print("France vs Sweden")
print("Mexico vs Ecuador")
print("England vs DR Congo")
print("Belgium vs Senegal")
print("USA vs Bosnia and Herzegovina")
