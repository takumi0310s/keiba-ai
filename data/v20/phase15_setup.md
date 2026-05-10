# Phase 15 — V20 学習環境構築 (2026-05-10)

## GPU 環境

| 項目 | 値 |
|------|----|
| GPU | NVIDIA GeForce RTX 4070 Ti SUPER 16GB |
| Driver | 591.86 |
| CUDA | 13.1 |
| torch | 2.11.0+cu126 |
| torch.cuda.is_available() | True |
| memory baseline | 794MB / 16,376MB |

## library

| pkg | version | GPU 利用 |
|-----|---------|---------|
| lightgbm | (既存) | `device='gpu'` 指定済 (fallback CPU) |
| xgboost | (既存) | `device='cuda'` + `tree_method='hist'` |
| torch | 2.11.0+cu126 | FT-Transformer + IntraRace |

## 既存 V15 model

| file | 概要 |
|------|------|
| `keiba_model_v15_central.pkl.gz` | LGB+XGB ensemble、 145 features |
| `keiba_model_v15_central_live.pkl.gz` | live 用 (Pattern B) |

## V15 cache (V20 学習 base)

| file | size | 内容 |
|------|------|------|
| `data/_v15_optuna_df_cache.pkl.gz` | 104MB | df 527,280 × 232 cols + features list (145) |

| 列 | 用途 |
|----|------|
| `target` | 複勝圏 binary (V15 学習目標) |
| `is_top3` | target alias |
| `finish` | 着順 (1-18+) |
| `year` | 15-25 (2015-2025) |
| `date_num` | YYYYMMDD |
| `race_id` | course_kai_nichi_R encoded |
| `race_id_unique` | year_date_course_R |

年別 行数 (近年):
| year | rows |
|------|------|
| 2022 | 46,841 |
| 2023 | 47,274 |
| 2024 | 46,752 |
| 2025 | 47,497 |

## V20 features 設計 (202 候補)

| group | count | source | 状態 |
|-------|-------|--------|------|
| V15 base | 145 | 既存 cache | 実 signal あり |
| Phase 11 (JRDB candidate) | 15 | tools/predict_core_v18.py | constant default |
| Phase 12 (JV-Link candidate) | 17 | tools/predict_core_v18.py | constant default |
| Phase 13 (netkeiba master) | 25 | tools/netkeiba_master_scraper.py | constant default |
| **計** | **202** | — | V15 145 のみ実 signal |

★ 重要: Phase 11/12/13 = 全 row 同値の constant default fill。 LGB/XGB は constant column を自動 drop するので、 V20 候補 = **実質 V15 の retrain (145 features)** ★

## 5/10 actual V15 結果 (比較 baseline)

| metric | 値 |
|--------|----|
| 予測 R 数 | 35 |
| 結果確定 R | 34 |
| trio hit | 11 (32.4%) |
| 投資 | (full betting=¥24,500) |
| 配当 | ¥27,790 |
| 利益 | +¥3,290 |
| ROI | 113.8% |
| top1 1 着率 | 26.5% (9/34) |
