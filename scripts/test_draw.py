"""测试平局检测阈值"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from model_utils import load_matches, load_rankings, actual_result
from models import load_best_params, DrawAwarePoisson
import random

random.seed(42)
matches = load_matches()
rankings = load_rankings()
params = load_best_params()
p = params["poisson"]

# 原始泊松准确率
model = DrawAwarePoisson(**p, draw_goal_margin=0, draw_rank_closeness=0)
correct = sum(1 for h, a, hs, as_ in matches
              if model.predict_result(h, a, rankings) == actual_result(hs, as_))
print(f"原始泊松 (无平局检测): {correct}/16 = {correct/16*100:.1f}%")

# 分析每场比赛
print(f"\n{'比赛':<35} {'实际':>5} {'λ_h':>6} {'λ_a':>6} {'λ差':>6} {'排差':>5}")
print("-" * 70)
for home, away, hs, as_ in matches:
    lh, la = model.get_lambdas(home, away, rankings)
    gd = abs(lh - la)
    rd = abs(rankings.get(home, 50) - rankings.get(away, 50))
    result = actual_result(hs, as_)
    print(f"{home} vs {away:<20} {hs}:{as_} {result}  {lh:>5.2f} {la:>5.2f} {gd:>5.2f} {rd:>4}")

# 随机搜索最佳阈值
print(f"\n随机搜索最佳平局阈值...")
best_acc = correct
best_t = (0, 0)
for _ in range(3000):
    gm = round(random.uniform(0.1, 1.5), 2)
    rc = random.randint(5, 60)
    model = DrawAwarePoisson(**p, draw_goal_margin=gm, draw_rank_closeness=rc)
    correct = sum(1 for h, a, hs, as_ in matches
                  if model.predict_result(h, a, rankings) == actual_result(hs, as_))
    if correct > best_acc:
        best_acc = correct
        best_t = (gm, rc)
        print(f"  新最佳: {correct}/16 = {correct/16*100:.1f}% (goal_margin={gm:.2f}, rank_close={rc})")

print(f"\n最佳阈值: goal_margin={best_t[0]:.2f}, rank_closeness={best_t[1]}, 准确率={best_acc/16*100:.1f}%")
