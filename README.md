# 2026 世界杯比分预测

Clean ELO→Lambda Dixon-Coles 泊松模型 + 四层动机修正 + 平局检测器 + 赔率融合。

**当前准确率：70.5%（31/44），非平局 87.1%（27/31），平局 30.8%（4/13）。**

## 快速开始

```bash
pip install -r requirements.txt
python scripts/elo_lambda_model.py              # 回测
python scripts/odds_scraper.py                   # 爬赔率（预测前必须）
python scripts/predict_with_context.py           # 预测（含赔率融合）
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
| `scripts/odds_scraper.py` | BetExplorer 赔率爬虫 |
| `data/raw/schedule_2026.csv` | 完整 72 场小组赛赛程 |
| `data/raw/matches_2026.csv` | 2026 世界杯赛果（自动更新） |
| `output/rank_lambda_model.json` | 模型参数 |

## 模型演化

| 版本 | 准确率 | 说明 |
|------|--------|------|
| Clean ELO→Lambda DC + 动机 | 70.5% | 44场: 纯W/L/D ELO + 动机校准 + 平局检测 + 赔率融合 + λ×1.12 |
