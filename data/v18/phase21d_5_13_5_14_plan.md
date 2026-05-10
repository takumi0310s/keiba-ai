# Phase 21D - 5/13-5/14 (火水) 詳細 plan

> 作成: 2026-05-11 (Phase 21D)
> 投資保護: V15 production 触らず、 5/14 が ★PC フル活用★ day
> 5/13 平日: parser + backfill 整備
> 5/14 平日: V18 学習 + V21 PoC + V20 ensemble + paper test

## 5/13 (火、 平日) 4 並行 task

### A. JV-Link parser 仕上げ (90-120 min)

**目的**: Session #44 で実装済の tfjv_parser.py を JV-Link 用に extend

| 範囲 | 内容 |
|------|------|
| 入力 | C:\TFJV (実体) + JV-Link DLL (32-bit Python venv) |
| 出力 | tools/jvlink_parser_v2.py |
| 対象 | RACE / HR / O1 / TCOV / WOOD / BLOD (6 datatype) |
| 検証 | 1 ヶ月 backfill (5/3 PoC 実績 29 ファイル) で format 一致確認 |

**完了条件**: 6 datatype parser 動作 + sample 100 ファイル parse PASS

### B. 過去 1 年 JV-Link backfill (180-240 min、 background)

**目的**: 2025-05-13 〜 2026-05-12 (1 年) bulk fetch + parse + CSV 出力

| 範囲 | 内容 |
|------|------|
| 期間 | 約 365 日 |
| 想定 file 数 | 約 13,000 ファイル (1 日約 35 file) |
| 出力 | data/jvlink_backfill_v1/ (RACE/HR/O1/TCOV/WOOD/BLOD 各 CSV) |
| 並行 | task A の parser を使う (依存) → A 完了後 開始 |

**完了条件**: 365 日 ALL fetch + CSV 行数チェック (HR ≥ 10K) + Discord 完了通知

### C. V18 学習 準備 (60-90 min)

**目的**: 5/14 の 学習 run に向けた pipeline 整備

| 内容 | 詳細 |
|------|------|
| dataset | data/v18_dataset_v1.parquet (5/12 Terminal D 成果) |
| feature spec | V20 構造ベース、 SKB 完全除外 (Session #38 教訓) |
| WF split | 6-fold (2020-2025) |
| 学習 script | train/train_v18_full.py (整備) |

**完了条件**: dry-run 1 fold が 30 分以内に PASS (LGB single)

### D. V21 動画 dataset 調査 (90-120 min)

**目的**: Phase 4 動画 PoC (Session #41) の data 蓄積 状況確認 + 5/14 PoC 実行 spec 確定

| 内容 | 詳細 |
|------|------|
| 入力 | C:\video_cache\ + JRA-VAN ネクスト 録画 |
| 確認 | YOLOv8 + DLC SuperAnimal の zero-shot 動作 |
| 出力 | data/v18/phase21d_v21_video_spec_5_13.md |

**完了条件**: 50 レース 1,500 動画 spec 確定 + 5/14 PoC 開始 OK

## 5/14 (水、 平日) ★PC フル活用★ 4 並行 task

### A. V18 4-model ensemble 学習 (4-6 hours、 GPU + CPU フル)

**目的**: V18 (sib_*_exp 修正版) を 6-fold WF 学習 → backtest AUC 算出

| 内容 | 詳細 |
|------|------|
| 学習 spec | LGB + XGB + FT-Transformer + IntraRace Attention |
| dataset | data/v18_dataset_v1.parquet |
| target | finish ≤ 3 (binary) |
| 出力 | keiba_model_v18_central.pkl.gz + data/v18/v18_wf_results.json |
| 期待 | WF AUC ≥ 0.880 / sib_w5 corr 0.20 維持 |

**実行**:
```
python train/train_v18_full.py --output keiba_model_v18_central.pkl.gz --device gpu --threads 16
```

**完了条件**: 6-fold ALL 完了 + AUC PASS + LEAK audit PASS + 完了通知 Discord

### B. V21 動画 PoC fine-tune (3-5 hours、 GPU)

**目的**: YOLOv8 馬体検出 + DLC HORSE-10 fine-tune 動作確認

| 内容 | 詳細 |
|------|------|
| 入力 | 5/13 D で確定した spec |
| dataset | 50 レース 1,500 動画 |
| GPU usage | RTX 4070 Ti SUPER 16GB |
| 出力 | data/v18/phase21d_v21_poc_5_14.md |

**完了条件**: stride / gait / posture features 抽出 PASS + 1 レース 動作 demo

### C. V20 ensemble 重み 再 grid search (2-3 hours、 CPU)

**目的**: V20 (Session #44) の 4-model grid 重み 再最適化

| 内容 | 詳細 |
|------|------|
| 入力 | V20 v1 学習結果 |
| 探索 | LGB / XGB / FT / IR の grid (各 0.05 step) |
| 出力 | data/v18/phase21d_v20_grid_5_14.md |

**完了条件**: optimal 重み 確定 + delta AUC 報告

### D. V18 paper trade test (60 min、 D は最後に B 完了後)

**目的**: V18 学習結果を 5/10 retro で paper trade 動作確認

| 内容 | 詳細 |
|------|------|
| 入力 | V18 model + 5/10 race data |
| 検証 | winner_top1 / shift / ROI 算出 |
| 出力 | data/v18/phase21d_v18_paper_5_14.md |

**完了条件**: V18 paper ROI ≥ 110% / winner_top1 ≥ 30% / shift ≤ 12x → 5/15 GO 判定 input

## 投資保護 (絶対遵守)

- 🔴 predict_core.py / V15 production model: 5/14 学習中も NEVER 触る
- 🔴 V15 production の予測 / 投票 flow は 5/14 中も独立稼働 (干渉ゼロ)
- 🟢 5/14 学習は別 venv + 別 model file → V15 と完全分離

## 5/13-5/14 完了条件 (8 件 ALL PASS)

5/13:
1. JV-Link parser v2 動作
2. 1 年 backfill 完了
3. V18 学習 dry-run PASS
4. V21 video spec 確定

5/14:
5. V18 ensemble 学習 + WF AUC ≥ 0.880
6. V21 PoC features 抽出 PASS
7. V20 grid search 重み 確定
8. V18 paper ROI ≥ 110%

## 失敗時 fallback

- V18 学習失敗 → 5/15 dry run で 切り戻し、 5/17 投入は V15 案 B 改 単独継続
- 動画 PoC NG → V21 を 9/1 投入から 10/1 に押し下げ (Phase 4 v3)
- backfill NG → V20 6/29 GO 判定 を 7/15 に push、 影響軽微

## 次 step

- 5/15-5/16: data/v18/phase21d_5_15_5_16_plan.md (V18 dry run + V21 YOLO + V22 RL + 5/17 GO 判定)
