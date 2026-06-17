"""
时间切分版：中立场 + 平局优化
训练=pre-2026, 测试=2026 WC
"""
import sys; sys.path.insert(0, 'scripts')
import numpy as np
import pandas as pd
from model_utils import load_rankings, actual_result

rankings = load_rankings()
features = pd.read_csv('output/features.csv', encoding='utf-8-sig')
features['date'] = pd.to_datetime(features['date'])

NEUTRAL = {"FIFA World Cup", "UEFA Euro", "Copa América", "AFC Asian Cup",
           "African Cup of Nations", "Gold Cup", "OFC Nations Cup"}

# 时间切分
train = features[features['date'] < '2026-01-01']
extra = features[(features['date'] >= '2026-01-01') & (features['round'] != 'FIFA World Cup')]
train = pd.concat([train, extra])
test = features[(features['date'] >= '2026-06-01') & (features['round'] == 'FIFA World Cup')]

print(f"训练: {len(train)} 场  测试: {len(test)} 场")

def eval_on(data, rw, hb, hb_n, dc, decay, power):
    home_rk = np.array([rankings.get(h, 50) for h in data['home_team']], dtype=np.float64)
    away_rk = np.array([rankings.get(a, 50) for a in data['away_team']], dtype=np.float64)
    is_n = np.array([r in NEUTRAL for r in data['round']], dtype=bool)
    actuals = np.array([actual_result(int(hg), int(ag))
                        for hg, ag in zip(data['home_goals'], data['away_goals'])])

    bias = np.where(is_n, hb_n, hb).astype(np.float64)
    ed = (away_rk - home_rk) * rw + bias
    p_h = 1.0 / (1.0 + 10.0 ** (-ed / 400.0))
    p_d = dc * np.exp(-(np.abs(ed) / decay) ** power)
    p_h -= p_d / 2.0
    p_a = 1.0 - p_h - p_d

    n = len(p_h)
    preds = np.full(n, 'A', dtype='<U1')
    preds[(p_h >= p_d) & (p_h >= p_a)] = 'H'
    preds[(p_d > p_h) & (p_d >= p_a)] = 'D'
    acc = (preds == actuals).mean() * 100
    draw_rate = (preds == 'D').sum() / n
    return acc, draw_rate

# 在训练集上搜索
rng = np.random.default_rng(42)
best = {"acc": -1}
for _ in range(5000):
    rw = round(rng.uniform(1.0, 8.0), 2)
    hb = int(rng.integers(0, 120))
    hb_n = int(rng.integers(-30, 40))
    dc = round(rng.uniform(0.22, 0.50), 2)
    decay = int(rng.integers(300, 800))
    power = round(rng.uniform(0.5, 2.5), 1)

    acc, dr = eval_on(train, rw, hb, hb_n, dc, decay, power)
    if dr < 0.10:  # 至少10%平局预测
        continue
    if acc > best["acc"]:
        best = {"acc": acc, "rw": rw, "hb": hb, "hb_n": hb_n,
                "dc": dc, "decay": decay, "power": power, "draw_rate": dr}

train_acc, train_dr = eval_on(train, best["rw"], best["hb"], best["hb_n"],
                               best["dc"], best["decay"], best["power"])
test_acc, test_dr = eval_on(test, best["rw"], best["hb"], best["hb_n"],
                             best["dc"], best["decay"], best["power"])

# 旧模型对比
old_train_acc, old_train_dr = eval_on(train, 3, 75, 75, 0.38, 400, 2.0)
old_test_acc, old_test_dr = eval_on(test, 3, 75, 75, 0.38, 400, 2.0)

# 直接用旧 best_params 文件（含数据泄露）
from models import create_optimized_elo, load_best_params
old_elo = create_optimized_elo(load_best_params())
old_leak_correct = sum(1 for _, row in test.iterrows()
                       if old_elo.predict_result(row['home_team'], row['away_team'], rankings)
                       == actual_result(row['home_goals'], row['away_goals']))
old_leak_acc = old_leak_correct / len(test) * 100

print(f"\n{'='*60}")
print(f"  时间切分验证 (训练={len(train)}场, 测试={len(test)}场)")
print(f"{'='*60}")
print(f"")
print(f"{'模型':<25} {'训练集':>10} {'测试集':>10} {'平局率':>8}")
print(f"{'-'*53}")
print(f"{'旧ELO(同参数全量拟合)':<25} {old_train_acc:>9.1f}% {old_test_acc:>9.1f}% {old_train_dr:>7.1%}")
print(f"{'新ELO(时间切分调优)':<25} {train_acc:>9.1f}% {test_acc:>9.1f}% {train_dr:>7.1%}")
print(f"{'旧ELO(随机切分调优=泄露)':<25} {'--':>10} {old_leak_acc:>9.1f}% {'--':>8}")
print(f"\n  新参数: rw={best['rw']} hb={best['hb']} hb_neutral={best['hb_n']}")
print(f"          dc={best['dc']} decay={best['decay']} power={best['power']}")

# 逐场
print(f"\n  逐场预测:")
for _, row in test.iterrows():
    h, a = row['home_team'], row['away_team']
    hg, ag = int(row['home_goals']), int(row['away_goals'])
    act = actual_result(hg, ag)

    # New
    rk_h, rk_a = rankings.get(h,50), rankings.get(a,50)
    n = row['round'] in NEUTRAL
    bias = best['hb_n'] if n else best['hb']
    ed = (rk_a - rk_h) * best['rw'] + bias
    ph = 1.0/(1.0+10.0**(-ed/400.0))
    pd = best['dc']*np.exp(-(np.abs(ed)/best['decay'])**best['power'])
    ph -= pd/2.0; pa = 1.0-ph-pd
    new_pred = 'H' if ph>=max(pd,pa) else ('D' if pd>=pa else 'A')

    # Old
    ed_o = (rk_a - rk_h)*3 + 75
    ph_o = 1.0/(1.0+10.0**(-ed_o/400.0))
    pd_o = 0.38*np.exp(-(np.abs(ed_o)/400.0)**2.0)
    ph_o -= pd_o/2.0; pa_o = 1.0-ph_o-pd_o
    old_pred = 'H' if ph_o>=max(pd_o,pa_o) else ('D' if pd_o>=pa_o else 'A')

    # Leaked
    leak_pred = old_elo.predict_result(h, a, rankings)

    print(f"  {h} vs {a} ({hg}:{ag} {act}) | 旧:{old_pred} 新:{new_pred} 泄露:{leak_pred}")
