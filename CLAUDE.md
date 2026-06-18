# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

2026 世界杯比分预测与回测分析。活跃追踪中——每天更新赛果、追踪预测准确率。

## 当前状态 (2026-06-18)

**已赛 22 场，预测准确率 59.1%（13/22），非平局比赛 100%（13/13），平局 0/9。**

当前最佳模型：**Rank→Lambda DC** (`scripts/elo_lambda_model.py`)
- 用 FIFA 排名计算期望进球 λ，不经过排名压缩（λ 范围 0.4–4.1）
- Dixon-Coles τ 修正低比分相关性
- 参数: alpha=0.0689, beta=1.277, gamma=0.3158, rho=-0.05185
- 参数文件: `output/rank_lambda_model.json`

### 动机修正层（NEW: 2026-06-17）

新增 `scripts/motivation.py` + `scripts/predict_with_context.py`，在 DC 泊松概率上叠加四层修正：

| 层级 | 因子 | 示例 |
|------|------|------|
| A. 赛制路径 | 小组第一/第二去不同半区，下半区（阿根廷/巴西/英格兰）明显强于上半区 | 墨西哥vs韩国：A1去下半区更差 → 平局上浮8pp |
| B. 体能 | 休息天数差≥2天 → 优势方+3~6% | 第二轮自动计算 |
| C. 积分形势 | 双方均胜→可接受平局；双方均败→必争胜 | 0分队对战进球×1.15 |
| D. 地缘政治 | 伊朗-美国战争、旅行禁令、FIFA暂停 | 伊朗胜率-12% |

**关键地缘背景（当前世界杯）：**
- **伊朗**: 2/28美以空袭伊朗，6/14刚签和平协议。训练营迁至墨西哥蒂华纳，不能在美国停留超48h，球迷票被取消。已影响表现：2-2平新西兰（排名85）
- **海地/科特迪瓦/塞内加尔**: 在美旅行禁令名单
- **刚果(金)**: 2月被FIFA暂停资格，5月恢复

### 模型演化历程

| 版本 | 准确率 | 核心问题 |
|------|--------|---------|
| 旧ELO (FIFA排名, 随机切分) | 56.2%→75% | 数据泄露（调参包含了测试集） |
| 旧集成 (Poisson+ELO+DC加权) | 43.8% | DC拖后腿，权重公式错误 |
| 修复集成 (Poisson+ELO 50/50) | 52.1% | 全量数据，比分全是1:1/2:1 |
| **Rank→Lambda DC (当前)** | **59.1%** | tanh λ 饱和 + 地缘因子修正 |

### 平局风险标注系统

- 风险 LOW/NONE → 模型判胜负可信度 100%（当前13/13）
- 风险 MED/HIGH → 平局概率约29%，胜负仅供参考
- 9场平局中：2场被预警、3场有微弱信号、4场完全无法预测

### 已知局限

1. **平局预测 0%** — 泊松框架下平局永远不是概率最大值，这是理论极限
2. **比分 MAE 1.95** — 极端比分（7:1）无法预测
3. **λ 饱和修正（NEW 6/18）** — tanh 软饱和防止大排名差过度外推（90%+→~71%），但平局仍无法预测
4. **地缘因子方向修正（NEW 6/18）** — 刚果(金) FIFA 停赛恢复从负面转为正面（+2pp boost）
5. **淘汰赛预测力未知** — 小组赛快结束，淘汰赛需要升级

## 目录结构

```
data/raw/               # 原始数据
  matches_2026.csv      # 2026世界杯赛果（手动+自动更新，22场）
  schedule_2026.csv     # ★ 完整赛程（72场小组赛+32场淘汰赛框架）
  odds_round1.csv       # 第一轮赔率（16场）
  odds_round2.csv       # 第二轮赔率（A+B组，4场）
  scraping_log.txt      # 爬虫运行日志
  historical_matches.csv # 历史比赛
data/processed/         # 清洗后数据
  matches.csv           # 4580场历史比赛
  teams.csv             # 球队排名
scripts/                # 核心脚本
  model_utils.py        # 共享工具函数、数据加载
  models.py             # OptimizedPoisson, OptimizedELO, DixonColes
  ensemble.py           # EnsemblePredictor (Poisson+ELO 50/50)
  elo_lambda_model.py   # ★ 当前最佳: Rank→Lambda DC 模型
  motivation.py         # ★ 动机修正: 赛制路径/体能/积分/地缘政治
  predict_with_context.py # ★ 带动机修正的预测（动态赛程）
  score_scraper.py      # ★ Wikipedia 比分爬虫（GitHub Actions 自动运行）
  backtest.py           # 全量回测脚本
  predict.py            # 命令行预测工具
  build_features.py     # 动态ELO + 近期状态特征
  fair_compare.py       # 时间切分公平对比
  optimize_venue_draw.py # 中立场+平局参数优化
  dc_team_model.py      # Dixon-Coles 球队级参数（实验性）
  train_model.py        # 泊松回归模型（效果不如简单模型）
  predict_upcoming.py   # 旧赛程预测（已过时，用predict_with_context.py替代）
.github/workflows/      # ★ GitHub Actions 自动化
  daily-update.yml      # 每天4次: 爬比分→回测→预测→推送
output/                 # 输出文件
  rank_lambda_model.json # ★ 当前模型参数
  best_params.json       # 旧模型参数（有数据泄露）
  features.csv           # 特征表（4580场，含ELO+状态）
  predictions.csv        # 预测结果
  backtest_19.txt        # 19场回测详细
  final_backtest.txt     # 最终回测汇总
  predictions_with_risk.txt # 带平局风险的预测
  upcoming_predictions.csv  # 剩余赛程预测（旧）
  predictions_with_context.csv # ★ 带动机修正的预测
```

## 常用命令

```bash
pip install -r requirements.txt

# 回测所有已赛比赛（更新准确率）
python scripts/elo_lambda_model.py

# 带动机修正的预测（动态从 schedule_2026.csv 计算剩余场次）
python scripts/predict_with_context.py

# 手动爬比分（通常由 GitHub Actions 自动运行）
python scripts/score_scraper.py

# 动机模块独立测试（积分榜+地缘+休息日）
python scripts/motivation.py

# 重新拟合模型（有新数据时）
python scripts/elo_lambda_model.py
```

## 自动化 (GitHub Actions)

`.github/workflows/daily-update.yml` 每天 4 次 (UTC 3/12/18/21) 自动执行:
1. `score_scraper.py` — 从 Wikipedia 抓取比分
2. `elo_lambda_model.py` — 回测更新准确率
3. `predict_with_context.py` — 生成新预测
4. 自动 commit + push

手动触发: `gh workflow run daily-update.yml`
验证: 看 GitHub commit 历史是否有 `auto: daily update` 提交

## 赛后更新流程

**自动（推荐）：** GitHub Actions 自动爬 Wikipedia → 回测 → 预测 → 推送，无需手动。

**手动：**
1. 更新 `data/raw/matches_2026.csv`（或运行 `python scripts/score_scraper.py` 自动爬）
2. 运行 `python scripts/elo_lambda_model.py` 更新回测
3. 运行 `python scripts/predict_with_context.py` 更新预测

## 技术约定

- Python 3.11+, scipy, numpy, pandas
- 所有脚本从项目根目录 `D:\世界杯` 运行
- CSV 用 UTF-8 with BOM
- 中立场赛事: FIFA World Cup, UEFA Euro, Copa America, AFC Asian Cup, African Cup of Nations, Gold Cup, OFC Nations Cup

## 下一步

1. 等 6/17 K+L 组赛果（Wikipedia 自动抓取）
2. 6/18 A+B 组第二轮 — 验证 motivation 层的积分形势逻辑
3. 累计 30+ 场后校准动机修正权重
4. 淘汰赛建模：休息天数差、加时→点球、单场淘汰心理因素
5. 淘汰赛对阵动态生成（schedule_2026.csv 目前只有小组赛）
