# Phase 21D - 5/15-5/16 (木金) 詳細 plan

> 作成: 2026-05-11 (Phase 21D)
> 投資保護: V15 production 触らず、 5/16 中央なし (NAR のみ) → V15 影響ゼロ
> 5/15 平日: V18 dry run + V21 YOLO + V22 500K + 5/17 GO 判定 input
> 5/16 平日: 全 system 確認 + V22 1M + 5/17 final 確定

## 5/15 (木、 平日) 4 並行 task

### A. V18 dry run on 5/3-5/10 retro (90-120 min)

**目的**: 5/14 で学習した V18 を 過去 1 週間 (5/3-5/10) で dry run 検証

| 内容 | 詳細 |
|------|------|
| 入力 | V18 model + 5/3-5/10 race data |
| 検証 | winner_top1 ≥ 30% / shift ≤ 12x / ROI ≥ 110% |
| 出力 | data/v18/phase21d_v18_dryrun_5_15.md |

**完了条件**: ALL PASS なら V18 5/17 投入候補 (週末上限 5,000円/日) → 5/16 final で確定

### B. V21 YOLO + DLC fine-tune 続行 (3-5 hours、 GPU)

**目的**: 5/14 PoC の features を 全 50 レース で完全抽出 + AUC 評価

| 内容 | 詳細 |
|------|------|
| 入力 | 5/14 PoC + 全 50 レース 動画 |
| GPU usage | RTX 4070 Ti SUPER 16GB |
| 出力 | data/v18/phase21d_v21_features_full_5_15.md |
| 期待 | V20 + 動画 features で WF AUC ≥ V20 + 0.005 |

**完了条件**: features 50 レース 抽出 + WF AUC delta 算出 + 9/1 投入候補 確定

### C. V22 RL 学習 (500K steps、 4-6 hours、 CPU + GPU)

**目的**: Session #84 設計の V22 RL 投資 policy を 500K step で 1st PoC

| 内容 | 詳細 |
|------|------|
| 入力 | data/v18_dataset_v1.parquet + V18 score |
| RL spec | PPO (stable-baselines3) / action = bet 額 0-1万 / reward = ROI |
| GPU/CPU | both フル使用 |
| 出力 | keiba_rl_v22_500k.zip + data/v18/phase21d_v22_500k_5_15.md |

**完了条件**: 学習完了 + paper test ROI ≥ V15 (119.2%) → 5/16 1M step に拡張

### D. 5/17 GO 判定 logic 整備 (60 min)

**目的**: 5/15 までの 結果から V18 5/17 投入 GO/no-go 判定 logic を整備

| 判定 axis | 閾値 | source |
|-----------|------|--------|
| WF AUC | ≥ 0.880 | 5/14 学習 |
| LIVE retro winner_top1 | ≥ 30% | 5/15 dry run |
| shift | ≤ 12x | 5/15 dry run |
| paper ROI | ≥ 110% | 5/14 paper |
| LEAK audit | PASS | Session #38 SKB 教訓 |

**完了条件**: 5/16 朝に GO/no-go 判定 input 揃う

## 5/16 (金、 平日) 4 並行 task

### A. 全 system 確認 (60-90 min)

**目的**: 5/17 朝の 全自動 production flow を完全 dry run

| 内容 | 詳細 |
|------|------|
| 確認対象 | daily_premium_scrape (03:00) / daily_predict (08:00) / race_auto_notify (08:45) / morning_weight_check (09:30) / daily_results (18:00) |
| race_test | python tools/dryrun_weekend_full.py |
| nightly_sanity | python tools/nightly_sanity_check.py --target 20260517 |
| 出力 | data/v18/phase21d_5_17_dryrun.md |

**完了条件**: 5 タスク ALL OK + Discord 通知 PASS + nightly sanity ALL GREEN

### B. V22 RL 1M step 拡張 (6-8 hours、 background)

**目的**: 5/15 500K の続き、 1M step まで拡張 + paper test

| 内容 | 詳細 |
|------|------|
| 入力 | keiba_rl_v22_500k.zip (warm start) |
| target | 1,000,000 step |
| 並行 | 5/16 22:00 まで継続、 5/17 朝には間に合う |
| 出力 | keiba_rl_v22_1m.zip + data/v18/phase21d_v22_1m_5_16.md |

**完了条件**: 1M step 完了 + paper test ROI 算出 (5/17 production には 含めない、 評価のみ)

### C. 5/17 final schedule 確定 (60 min)

**目的**: 5/15 GO 判定 input + 5/16 全 system 確認結果から、 5/17 の 全 schedule 確定

| 内容 | 詳細 |
|------|------|
| 入力 | 5/15 GO/no-go input + 5/16 dryrun 結果 |
| 出力 | data/v18/phase21d_5_17_final.md (内容 確定) |
| 確定 axis | V15 戦略 (絶対) / V18 paper trade (条件 PASS なら 並行) / 投票 上限 |

**完了条件**: schedule 確定 + Discord 通知 + 全 task scheduler 状態確認 PASS

### D. 失敗 fallback drill (60 min)

**目的**: 過去事故 (4/19 SCRAPER-GUARD 誤停止 等) の再発防止 drill

| 内容 | 詳細 |
|------|------|
| 確認 | OPERATIONAL_CALLERS ホワイトリスト / Mon 早朝特例 / process_watchdog v2 / Ctrl+C 対策 |
| 検証 | python tools/verify_scraper_guard_sunday.py + python tools/nightly_sanity_check.py |
| 出力 | data/v18/phase21d_fallback_drill_5_16.md |

**完了条件**: 全 verification PASS + 万が一 5/17 朝事故時の 手順 書面化

## 投資保護 (絶対遵守)

- 🔴 predict_core.py / V15 production model: 5/15-5/16 中も NEVER 触る
- 🔴 V18 / V21 / V22 学習 → 全部 別 venv + 別 model file → V15 と完全分離
- 🟢 5/16 中央なし → V15 production 影響ゼロ、 NAR は通常運用 OK

## 5/15-5/16 完了条件 (8 件 ALL PASS)

5/15:
1. V18 dry run ALL PASS
2. V21 features 50 レース + AUC delta 算出
3. V22 500K step + paper test ≥ V15
4. 5/17 GO 判定 logic input 揃う

5/16:
5. 全 system dry run ALL OK
6. V22 1M step 完了 + paper 算出
7. 5/17 final schedule 確定
8. fallback drill 全 PASS

## 失敗時 fallback

- V18 dry run NG → 5/17 投入 NO-GO、 V15 案 B 改 単独継続 (絶対 安全)
- 全 system dry run NG → 5/16 22:00 までに 修正 + Discord 通知 + 翌朝 手動確認
- nightly sanity 誤検知 → 4/19 教訓 通り 即 patch + commit

## 次 step

- 5/17 (土): data/v18/phase21d_5_17_final.md (final schedule 03:00-23:00)
