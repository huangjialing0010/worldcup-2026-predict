# 2026 世界杯比分预测

Rank→Lambda Dixon-Coles 泊松模型 + 四层动机修正（赛制路径 / 体能 / 积分形势 / 地缘政治）。

**当前准确率：60.0%（12/20），非平局 100%（12/12）。**

## 快速开始

```bash
pip install -r requirements.txt
python scripts/elo_lambda_model.py        # 回测
python scripts/predict_with_context.py     # 预测
```

## 自动化

GitHub Actions 每天 4 次自动：爬 Wikipedia 比分 → 回测 → 预测 → 推送。无需手动操作。

## 核心文件

| 文件 | 说明 |
|------|------|
| `scripts/elo_lambda_model.py` | Rank→Lambda DC 模型，当前最佳 |
| `scripts/motivation.py` | 动机修正模块（赛制/体能/积分/地缘） |
| `scripts/predict_with_context.py` | 带上下文预测，动态赛程 |
| `scripts/score_scraper.py` | Wikipedia 比分爬虫 |
| `data/raw/schedule_2026.csv` | 完整 72 场小组赛赛程 |
| `data/raw/matches_2026.csv` | 2026 世界杯赛果（自动更新） |
| `output/rank_lambda_model.json` | 模型参数 |

## 模型演化

| 版本 | 准确率 | 说明 |
|------|--------|------|
| Rank→Lambda DC | 60.0% | FIFA 排名 → λ，Dixon-Coles τ 修正 |
| + motivation 层 | 待验证 | 赛制/体能/积分/地缘四层修正 |
