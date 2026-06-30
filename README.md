# 2026 世界杯比分预测

Clean ELO→Lambda Dixon-Coles 泊松模型 + 四层动机修正 + 分级平局检测器 + 双赔率源融合。

**淘汰赛进行中，76 场准确率 68.4%（52/76），非平局 75.9%（41/54），平局 50.0%（11/22）。**

## 快速开始

```bash
pip install -r requirements.txt
python scripts/elo_lambda_model.py              # 回测
python scripts/odds_scraper_sporttery.py         # 爬体彩赔率（覆盖面最大）
python scripts/odds_scraper.py                   # 爬BetExplorer赔率（补充）
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
| `scripts/predict_knockout.py` | 淘汰赛单场预测（指定对阵） |
| `scripts/score_scraper.py` | Wikipedia 比分爬虫 |
| `scripts/odds_scraper.py` | BetExplorer 赔率爬虫 |
| `scripts/odds_scraper_sporttery.py` | 中国体彩竞彩网赔率爬虫 |
| `data/raw/schedule_2026.csv` | 完整赛程（72场小组赛+16场淘汰赛） |
| `data/raw/matches_2026.csv` | 2026 世界杯赛果 |
| `output/rank_lambda_model.json` | 模型参数 |

## 模型演化

| 版本 | 准确率 | 说明 |
|------|--------|------|
| Clean ELO→Lambda DC + 动机 + 分级平局 | 70.0% | 60场: 纯W/L/D ELO + 动机校准 + 分级平局阈值 + 双赔率源 + λ×1.12 |
| → 小组赛收官 | 69.4% | 72场: 最后12场平局4/4全中，平局整体11/20(55.0%) |
| → 淘汰赛R32+KO约束 | 68.4% | 76场: 淘汰赛4场2/4，KO平局≥25%+胜率≤68%约束上线 |
