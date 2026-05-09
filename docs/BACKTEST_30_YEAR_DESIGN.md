# 30 年 backtest 環境 設計 (Session #84)

> TFJV 90 年保有 data から 30 年抽出し、 V15 → V22 RL までの戦略を 一貫評価。
> 作成: 2026-05-09 (Session #84)

---

## 1. 背景

| 項目 | 値 |
|------|----|
| TFJV 既保有 | 90 年分 (45,000+ files / 6 GB) |
| 抽出範囲 | 1995-2024 (30 年) |
| 目的 | V15 → V22 RL 5 戦略 を walk-forward で 一貫評価 |
| 後段 | Session #82 hybrid 判定、 Session #83 V22 RL PoC、 Session #79 V20 検証 |

---

## 2. 期間設計

### 2.1 30 年 (1995-2024) を採用する理由
- TFJV UM_DATA = 1936-2025 (90 年) 利用可能
- TFJV W5_DATA = 2011-2026 (15 年) 限定 → 30 年中 12 年で W5 features 利用不可
- TFJV WD_DATA / TM_DATA は 1995+ で 利用可能
- → **1995-2024 の 30 年が data 揃う最大範囲**

### 2.2 datatype 別 利用可能性

| datatype | 範囲 | 30 年 backtest 内 利用可能性 |
|----------|------|----------------------------|
| UM_DATA (馬基本) | 1936-2025 | 全 30 年 OK |
| RA_DATA (race) | 1986-2025 | 全 30 年 OK |
| SE_DATA (出走) | 1986-2025 | 全 30 年 OK |
| HR_DATA (払戻) | 1986-2025 | 全 30 年 OK |
| WD_DATA (調教木) | 1995-2025 | 全 30 年 OK |
| TM_DATA (タイム) | 1995-2025 | 全 30 年 OK |
| O1-O6_DATA (オッズ) | 2003-2025 | 1995-2002 欠損 ★ |
| W5_DATA (調教坂路) | 2011-2026 | 2011 以降のみ ★ |
| TF_DATA (調整) | 1986-2025 | 全 30 年 OK |
| H1_DATA (票数) | 2003-2025 | 1995-2002 欠損 ★ |

→ **1995-2002 の期間** は Pattern B (オッズ込み) features 一部 NA、 Pattern A (リークフリー) は完全。

---

## 3. Data 量

### 3.1 R 数 試算

| 期間 | 年数 | R/年 | 累計 R |
|------|------|------|--------|
| 1995-2024 | 30 年 | ~6,500 R | **約 195,000 R** |

### 3.2 horse-run 数

| 期間 | R 数 | 平均 馬数/R | horse-runs |
|------|------|----------|-----------|
| 1995-2024 | 195,000 R | 11 頭 | **約 215 万 horse-runs** |

### 3.3 features 数

| layer | features 数 | 備考 |
|-------|-----------|------|
| V15 (current) | 145 | Pattern A 124 + 当日 8 + 環境 13 |
| V20 統合 | 200+ | V15 + sib_*_exp + KKA + interaction |
| V21 動画 | 220+ | V20 + 動画 5 件 |
| V22 RL | 220+ | V21 + RL state |

### 3.4 storage

| 形式 | size | 備考 |
|------|------|------|
| TFJV raw (現状) | 6 GB | 圧縮済 binary |
| 30 年 抽出 raw | ~3 GB | 60% (1995-2024) |
| 30 年 features parquet | **約 50-100 GB** | 200 features × 215 万 rows |
| model 集合 | ~5 GB | LGB+XGB+FT+IR × 5 strategies |

→ **disk 使用量 100 GB 想定** (現 keiba-ai/ + 70 GB)

---

## 4. 学習 data 分割

### 4.1 基本分割 (single-fold)

| split | 期間 | 年数 | R 数 |
|-------|------|------|------|
| train | 1995-2018 | 24 年 | ~156,000 R |
| valid | 2019-2021 | 3 年 | ~19,500 R |
| test | 2022-2024 | 3 年 | ~19,500 R |

→ test は **out-of-sample**、 5/16 V18 trial 〜 7/1 V20 の真の forward-test data に **接続可能**。

### 4.2 forward-test 連続性
- backtest test = 2022-2024 (~20 万 R)
- 真の forward = 2025-01 〜 2026-05 production (~7,000 R)
- → test → forward で **滑らかに接続** 可能

---

## 5. Walk-forward validation

### 5.1 5-fold rolling window

| fold | train | valid | test (forward) |
|------|-------|-------|---------------|
| 1 | 1995-2000 (6 年) | 2001 | — |
| 2 | 2001-2006 (6 年) | 2007 | — |
| 3 | 2007-2012 (6 年) | 2013 | — |
| 4 | 2013-2018 (6 年) | 2019-2021 | — |
| 5 | 2019-2021 (3 年) | 2022 | 2023-2024 |

### 5.2 KPI per fold
- AUC (mean、 each year)
- ROI (案B改 strict / 戦略⑦ 適用後)
- Sharpe ratio (annualized)
- max DD
- 的中率 (top1 / top3)
- bootstrap CI95

### 5.3 採用閾値
- 全 fold AUC >= 0.85
- 各 fold ROI >= 110%
- bootstrap CI95 lower >= 100%

---

## 6. 期待効果

### 6.1 各戦略 期待 ROI (30 年 backtest)

| 戦略 | 期待 AUC | 期待 ROI | sample size |
|------|---------|---------|-------------|
| 案B改 V15 (current) | 0.8939 | 110-130% | 195,000 R |
| 案B改 V18 (5/16 trial) | 0.890+ | 115-135% | 195,000 R |
| 案B改 V20 (7/1 投入) | 0.880-0.895 | 130-150% | 195,000 R |
| hybrid (Session #82) | 0.880-0.895 | 140-160% | 195,000 R |
| V22 RL (Session #83) | 0.880-0.895 | 150-180% | 195,000 R |

### 6.2 30 年 backtest が解決する疑問
- ★ surface / 頭数別 differential が 1995-2024 で robust か (hybrid 評価)
- ★ V20 sib_*_exp 効果が 30 年で 一貫するか (LEAK 確認)
- ★ V22 RL が pre-2010 でも有効か (汎化性能)
- ★ 経年 model drift (年代別 AUC 安定性)

---

## 7. 関連 doc

- `docs/BACKTEST_DATA_PIPELINE.md` — pipeline 実装設計
- `docs/BACKTEST_STRATEGY_COMPARISON.md` — 5 戦略 比較 plan
- `docs/BACKTEST_BUILD_TIMELINE.md` — 構築 schedule
- `docs/STRATEGY_HYBRID_DESIGN.md` (Session #82)
- `docs/PHASE_3_4_5_INTEGRATED_ROADMAP_v3.md` (Session #44)

---

**結論**: 30 年 (1995-2024) で 195,000 R / 215 万 horse-runs / features 200+ の backtest 環境を構築。 5 fold walk-forward で V15 → V22 RL を一貫評価。 storage 約 100 GB、 学習時間 multiprocessing で削減見込。
