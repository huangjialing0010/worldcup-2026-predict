# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

2026 世界杯比分预测与回测分析。活跃追踪中——每天更新赛果、追踪预测准确率。

## 当前状态 (2026-06-26)

**已赛 60 场，预测准确率 70.0%（42/60），非平局 79.5%（35/44），平局 43.8%（7/16）。**

当前最佳模型：**Clean ELO→Lambda DC** (`scripts/elo_lambda_model.py`)
- 用纯 W/L/D ELO 计算期望进球 λ（无进球循环依赖）
- Dixon-Coles τ + tanh 软饱和
- 参数: alpha=0.0536, beta=1.9714, gamma=0.2611, rho=-0.067428
- ELO 范围: 1362 (库拉索) - 1826 (阿根廷)
- 参数文件: `output/rank_lambda_model.json`

### 动机修正层

`scripts/motivation.py` + `scripts/predict_with_context.py`，在 DC 泊松概率上叠加四层修正：

| 层级 | 因子 | 示例 |
|------|------|------|
| A. 赛制路径 | 小组第一/第二去不同半区，下半区（阿根廷/巴西/英格兰）明显强于上半区 | 墨西哥vs韩国：A1去下半区更差 → 平局上浮 |
| B. 体能 | 休息天数差≥2天 → 优势方+3~6% | 第二轮自动计算 |
| C. 积分形势 | 双方均胜→可接受平局；双方均败→必争胜 | 0分队对战进球×1.15 |
| D. 地缘政治 | 伊朗-美国战争、旅行禁令、FIFA暂停 | 伊朗胜率-12% |
| **λ 饱和** (6/18) | tanh 软饱和防止大排名差过度外推 | 90%+主胜→~71%，λ比上限~3:1 |
| **平局检测器** (6/18→6/26) | 分级阈值：λ比>3→P(D)≥15%，λ比1.5-3→≥22%，λ比<1.5→≥28%，uplift≥0.04 | 第三轮-0.03，16场平局抓到7场(43.8%) |
| **赔率融合** (6/21→6/26) | 模型 70% + 市场赔率 30%，D<2.80→D权重50%，D<3.50→40% | 市场平局信号加权，实战价值大于回测 |
| **λ 全局校准** (6/21) | 实际 3.02 vs 模型 2.71 gpg → λ×1.12 | 系统性低估修正 |
| **第三轮资格模型** (6/21) | 出线/淘汰形势驱动动机修正 | 3分队争胜、0分队背水、默契球 |
| **球队平局倾向** (6/21) | ≥2场平局记录的队触发平局上浮 | 埃及、比利时等平局专业户 |
| **动机校准** (6/23) | 砍 md=1 全局 draw_rate、路径分析限第三轮+权重减半、首轮均胜 0.06→0.03 | 回测验证：动机层在裸模型上净增值为零时做减法 |

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
| **Clean ELO→Lambda DC + 动机修正** | **70.5%** | 44场: 纯W/L/D ELO + tanh饱和 + 动机校准 + 平局检测 + 赔率融合 + λ×1.12 + 第三轮资格模型 |
| **→ 第三轮小组赛** | **66.7%** | 60场: 第三轮新增16场9/16=56.2%，平局密集（16场中5场正确），整体下滑3.8pp |
| **→ 分级平局阈值** | **70.0%** | 60场: prob按λ比分档+uplift统一0.04，平局5→7(+43.8%)，无假阳性回退，+3.3pp |

### 平局风险标注系统

- 风险 LOW/NONE → 模型判胜负可信度 ~80%（60场中非平局 35/44 正确）
- 风险 MED/HIGH → 平局概率约29%，胜负仅供参考
- 16场平局中：7场正确预测（含平局覆写）、4场被预警、5场完全无法预测

### 已知局限

1. **平局预测** — 分级阈值后抓到 7/16（43.8%），+12.6pp。大差距平局仍有 9 场漏网（base P(D)太低，uplift 信号不够强）。首轮平局仍无法预测
2. **比分 MAE 1.85** — 泊松众数预测比分，极端比分（7:1）无法预测；60场后比分精确率 13.3%（8/60）
3. **地缘权重待校准** — 伊朗-12pp、刚果(金)+2pp 均为经验估计，60场回测可开始反推最优参数
4. **淘汰赛预测力未知** — 小组赛即将结束，淘汰赛需升级（加时点球、单场淘汰心理）

## 目录结构

```
data/raw/               # 原始数据
  matches_2026.csv      # 2026世界杯赛果（手动+自动更新，60场）
  schedule_2026.csv     # ★ 完整赛程（72场小组赛+32场淘汰赛框架）
  odds_round1.csv       # 第一轮赔率（16场）
  odds_round2.csv       # 第二轮赔率
  odds_live.csv         # BetExplorer实时赔率
  odds_sporttery.csv    # 中国体彩竞彩赔率（覆盖面更大，推荐）
  scraping_log.txt      # 爬虫运行日志
  historical_matches.csv # 历史比赛
data/processed/         # 清洗后数据
  matches.csv           # 4580场历史比赛
  matches_with_elo.csv  # 含clean ELO的历史比赛
  clean_elo.csv         # 纯W/L/D ELO评级（48队）
  teams.csv             # 球队排名
  team_name_map.json    # 中文→英文队名映射（体彩爬虫用）
scripts/                # 核心脚本
  model_utils.py        # 共享工具函数、数据加载
  models.py             # OptimizedPoisson, OptimizedELO, DixonColes
  ensemble.py           # EnsemblePredictor (Poisson+ELO 50/50)
  build_clean_elo.py    # ★ 纯W/L/D ELO构建（无循环依赖）
  elo_lambda_model.py   # ★ 当前最佳: Clean ELO→Lambda DC 模型
  motivation.py         # ★ 动机修正: 赛制路径/体能/积分/地缘政治
  predict_with_context.py # ★ 带动机修正的预测（动态赛程）
  score_scraper.py      # ★ Wikipedia 比分爬虫（GitHub Actions 自动运行）
  odds_scraper.py       # ★ BetExplorer 赔率爬虫（预测前必须运行）
  odds_scraper_sporttery.py # ★ 中国体彩竞彩网赔率爬虫（覆盖面更大）
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

# 带动机修正的预测（预测前必须先爬赔率）
python scripts/odds_scraper.py && python scripts/predict_with_context.py

# 手动爬比分（通常由 GitHub Actions 自动运行）
python scripts/score_scraper.py

# 手动爬赔率（预测前必须执行，确保融入市场信息）
python scripts/odds_scraper.py

# 动机模块独立测试（积分榜+地缘+休息日）
python scripts/motivation.py

# 重新拟合模型（有新数据时）
python scripts/elo_lambda_model.py
```

## 自动化 (GitHub Actions)

`.github/workflows/daily-update.yml` 每天 4 次 (UTC 7/12/16/21) 自动执行:
1. `score_scraper.py` — 从 Wikipedia 抓取比分
2. `elo_lambda_model.py` — 回测更新准确率
3. `odds_scraper.py` — 爬最新赔率
4. `predict_with_context.py` — 生成新预测（含赔率融合）
5. 自动 commit + push

手动触发: `gh workflow run daily-update.yml`
验证: 看 GitHub commit 历史是否有 `auto: daily update` 提交

## 赛后更新流程

**自动（推荐）：** GitHub Actions 自动爬 Wikipedia → 回测 → 预测 → 推送，无需手动。

**手动：**
1. 更新 `data/raw/matches_2026.csv`（或运行 `python scripts/score_scraper.py` 自动爬）
2. 运行 `python scripts/elo_lambda_model.py` 更新回测
3. 运行 `python scripts/odds_scraper.py` 爬最新赔率
4. 运行 `python scripts/predict_with_context.py` 更新预测

## 预测流程（必须遵守）

**每次预测前必须先爬赔率。** 赔率是市场信息的核心载体，70%模型+30%市场赔率融合才能给出最终预测。

两个赔率源，互补使用：
```bash
python scripts/odds_scraper.py            # BetExplorer（欧洲盘口）
python scripts/odds_scraper_sporttery.py  # 中国体彩竞彩网（覆盖面更大，推荐优先）
python scripts/predict_with_context.py    # 含赔率融合的预测（自动加载所有 odds_*.csv）
```

体彩赔率覆盖面通常优于 BetExplorer（8/12 vs 5/12），且 D<2.80 时触发 50% 平局权重加权。

## 技术约定

- Python 3.11+, scipy, numpy, pandas
- 所有脚本从项目根目录 `D:\世界杯` 运行
- CSV 用 UTF-8 with BOM
- 中立场赛事: FIFA World Cup, UEFA Euro, Copa America, AFC Asian Cup, African Cup of Nations, Gold Cup, OFC Nations Cup
- **预测必须含赔率对比** — 每次预测时自动拉市场赔率，输出模型 vs 赔率对比表。赔率存在 `data/raw/odds_round*.csv`

## 下一步

1. 小组赛剩余比赛等 Wikipedia 更新 → 回测最终小组赛准确率
2. 淘汰赛 32 强对阵确认 → 赔率 + 预测
3. **淘汰赛建模升级**：加时→点球、单场淘汰心理、休息天数差权重放大
4. 地缘政治权重网格搜索（伊朗-12pp、刚果(金)+2pp 等参数反推最优值）
5. 第三轮单独动机模型（行为模式与前两轮差异大，16场仅56.2%准确率）
