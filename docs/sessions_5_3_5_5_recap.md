# 5/3-5/5 全 14 セッション 収穫マップ (約 46 時間 / 21 commits)

生成: 2026-05-05 17:35

---

## 時系列サマリー

| Session | 日時 | commit | テーマ |
|--------:|------|--------|--------|
| #1 | 5/3 19:01 | ccd0c890 | 5/4 朝事故防止 緊急 (3 件) |
| #2 | 5/3 19:12 | 5fdfc2d0 | 5/3 GW Day3 結果集計 + Phase 2.5 提案 |
| #3 | 5/3 19:32 | 943791b3 | 5/9 戦略提案 + V18/V19 retrospective |
| #4 | 5/3 20:43 | fcc4741d | healthy 4日分析 + odds_base + watchdog |
| #5 | 5/3 22:30 | 777cc08e | LGB model CRLF 復旧 + retro feature_names |
| #6 | 5/4 00:05 | 660b13a6 | 5/3 直前予測分析 + V15/V17改良 + 5/9 plan v2 |
| #7 | 5/4 07:59 | e20bbc0c | データ監査 + Phase 2.5 残タスク 棚卸し |
| #8 | 5/4 12:07-12:35 | 470a9d90, 48709274, 5262e0c0, 0e03c55c, d8988f97 (5 commits) | Phase 2.5 A〜E (ra_score, sc_score blocker, TYB monitor, Platt scaling, 進捗サマリー) |
| #9 | 5/5 00:04 | 9c88d27c | 静音化 (16 task vbs ラッパー) |
| #10 | 5/5 00:14, 15:27-30 | b4c4894c, 6b5e4e7b, 74eb10b7, 6820b362 (4 commits) | jra_races_full 復旧 + sc_score 取得 + race-level normalization + V17 充足検証 |
| #11 | 5/5 15:55 | bfbddebc | 5/5 柏記念 ヒューリスティック予測 |
| #12 | 5/5 16:19 | e5f71cfa | NAR v4 model 復活 (archive→active) |
| #13 | 5/5 16:44 | 57029ff1 | NAR v4 体系化 (pipeline + plan) |
| #14 | 5/5 17:04 | 2b6dc4eb | 5/9 本番最終調整 (運用ガイド + リスク + 自動レポート) |

---

## Session #1: ccd0c890 (5/3 19:01) — 5/4 朝事故防止 緊急

**目的**: 5/4 朝に予想される事故 3 件の事前対応
**positive**:
- Cookie refresh, morning task 登録, daily_predict 中断検知 案を確定
**negative**:
- まだ実装は分散、検証不足
**成果物**: 緊急 3 task plan
**影響**: Session #2 以降の優先度判断 base

## Session #2: 5fdfc2d0 (5/3 19:12) — 5/3 結果集計 + Phase 2.5 提案

**目的**: 5/3 当日 USER 6R 投資の結果集計 + 構造改善提案
**positive**:
- USER 6R: 軸top3 3/6 (50%) 健全 (BT 57% 比較)、trio_hit 1/6 のみ (惜敗 2例特定)
- TOP2-TOP6 拡張案を提案
**negative**:
- trio 7点 命中の律速 (TOP2-TOP6 選定精度) を発見
**成果物**: `data/results/20260503_summary.md`, `data/v18/phase2_5_proposal_5_4.md`
**影響**: Phase 2.5 のロードマップ出発点

## Session #3: 943791b3 (5/3 19:32) — 5/9 戦略提案 + V18/V19 retrospective

**目的**: 5/9 GW後 初週末の投資戦略 +  v18/v19 (5/2 で全敗) の retro
**positive**:
- 案 A/B/C 比較 framework
- v18/v19 が 5/2 で全敗だった事実確認
**negative**:
- 全 14 日含む集計で汚染日 (4/12以前 投資集計バグ疑い 約300R) を含んでいたため数字が信頼できない
**成果物**: 戦略提案 v1
**影響**: Session #4 の healthy 4日 再計算で数字訂正

## Session #4: fcc4741d (5/3 20:43) — healthy 分析 + odds_base + watchdog

**目的**: 汚染日 除外で healthy 4日 (152R) で再分析、watchdog 化
**positive**:
- **案B改 (12R 1勝/D + 11R条件C非重賞) ROI 161% [CI 135.9-222.4%]** を確立 ← 5/9 戦略の core
- odds_base 5/2-5/3 retroactive 構築
- daily_predict watchdog 化
- v18/v19 完全 retro 走行
**negative**:
- v18/v19 retro で全 bet=0 を確認 (calibration 必要)
**成果物**: `data/results/healthy_4day_analysis.md`, `tools/daily_predict_watchdog.py`, `tools/v18_v19_retro_full.py`
**影響**: 5/9 案B改 採用の根拠 確定

## Session #5: 777cc08e (5/3 22:30) — LGB model CRLF 復旧

**目的**: v18/v19/v17 LGB model が CRLF 問題で破損していた → 復旧
**positive**:
- `.gitattributes` で .txt LGB model を binary 扱いに
- retro script で feature_names 引数対応
**negative**:
- Windows 環境で git checkout すると LF→CRLF 変換が走り model 破損する既知問題
**成果物**: `.gitattributes`, model 復旧
**影響**: Session #6 以降 v18/v19 推論 安定

## Session #6: 660b13a6 (5/4 00:05) — 5/3 直前予測分析 + 5/9 plan v2

**目的**: 5/3 直前 (発走数分前) の予測精度評価 + V15/V17改良案 + 5/9 plan v2
**positive**:
- 5/3 USER 6R USER 累計 +14,140円 / 撤退ライン余裕 +64,140円 を初確定
- 5/9 plan v2 で TYB midday 廃止
- V15/V17 改良案 5 件
**negative**:
- 改良案 retro で全 5/9 後送り判定 (ROI 改善せず)
**成果物**: `data/results/20260509_final_plan_v2.md`, `data/v18/v17_v15_improvement_proposals.md`
**影響**: 5/9 投資 GO 判定確定

## Session #7: e20bbc0c (5/4 07:59) — データ監査 + 残タスク棚卸し

**目的**: 全データ source の鮮度・カバレッジ完全監査
**positive**:
- 26 種 JRDB CSV カバレッジ確認
- Phase 2.5 残タスク 23 件を A-X で優先度マトリクス化
- training_times 2025+ rows = 192,296 (v1 想定 2,551 の 75 倍) 訂正
**negative**:
- ot/ov/ow/oz 4種 33日 stale (5/2 では未使用)
- netkeiba premium データ (race_analysis, stable_comments) 致命的欠損
**成果物**: `data/results/data_coverage_audit_5_4.md`
**影響**: Session #8 の Phase 2.5 A-D 順序付け

## Session #8: 5 commits 一括 (5/4 12:07-12:35) — Phase 2.5 A〜D + 進捗

| sub | commit | 内容 |
|---|--------|------|
| A | 470a9d90 | ra_score 再取得 → jra_races_full 2026年なし blocker 確認 |
| B | 48709274 | sc_score 同様 blocker |
| C | 5262e0c0 | **TYB publish タイミング観測 自動化** (`tools/tyb_publish_monitor.py` + schtasks) |
| D | 0e03c55c | **v18/v19 Platt scaling 試作** (max prob 0.154→0.213、不十分) |
| E | d8988f97 | 進捗サマリー |

**positive**:
- TYB observation 自動化開始 (5/4-5/10 観測中)
- Platt scaling 実装 (calibration 限界 確認)
**negative**:
- ra_score / sc_score は jra_races_full 2026 年データ不在で blocker → Session #10 で復旧
**成果物**: `tools/tyb_publish_monitor.py`, `tools/calibrate_v18_v19.py`, `data/v18/calibration_5_4_result.md`
**影響**: TYB 戦略は 5/11 月 再判断、calibration 限界判明 → Session #10 normalize へ

## Session #9: 9c88d27c (5/5 00:04) — 静音化

**目的**: schtasks 全 16 件で ターミナル ウィンドウ 出現 抑制
**positive**:
- `tools/silent_runner.vbs` (wscript hidden 起動) 実装 + 動作確認
- backup JSON + 一括変更 ps1 + rollback ps1 + 手順書
- syntax check OK
**negative**:
- admin elevation 必要のため user 手動実行 待ち
**成果物**: `tools/silent_runner.vbs`, `tools/silentify_all_tasks.ps1`, `data/v18/silentify_tasks_user_guide.md`
**影響**: 朝のちらつき問題 root cause 解消準備

## Session #10: 4 commits 一括 (5/5 00:14, 15:27-30) — jra_races_full 復旧 + race-level normalization

| sub | commit | 内容 |
|---|--------|------|
| A+B | b4c4894c | jra_races_full 2026年4-5月分追加 + ra_score 60races取得 |
| C | 6b5e4e7b | sc_score (stable_comments) 2026年4-5月分取得完了 |
| 並行 | 74eb10b7 | **race-level normalization** + v18/v19 retro 改善 |
| 並行 | 6820b362 | V17 features 充足率検証 + 5/9 plan v3 維持 + Session#10 最終 |

**positive**:
- BT vs production 27.7x scaling shift を **定量化**
- softmax T=1.0 で sum=1 強制 → bet>0 化確認
- ra_score / sc_score 部分復旧 (60 races + 4-5月分)
**negative**:
- normalize は monotonic、winner_top1 rate (34.5%) は不変 → 本質的 calibration 改善ではない
**成果物**: `tools/race_normalize.py`, `data/v18/distribution_shift_analysis.json`, `data/v18/race_normalize_5_4_result.md`
**影響**: v18/v19 5/16 試行の前提整理、根本治療は別 task

## Session #11: bfbddebc (5/5 15:55) — 5/5 柏記念 ヒューリスティック

**目的**: NAR v4 model 不在のため、JRA-elite 騎手 + odds で TOP6 ヒューリスティック予測
**positive**:
- ヒューリスティック TOP6: 8, 2, 10, 13, 3, 1
**negative**:
- model 不在前提で着手 (Session #12 で archive 発見)
**成果物**: `data/results/20260505_kashiwa_kinen.md` (ヒューリスティック)
**影響**: Session #12 の archive 探索 きっかけ

## Session #12: e5f71cfa (5/5 16:19) — NAR v4 復活

**目的**: archive/nar/ で NAR v4 (AUC 0.8145, 22 features) を発見、active に昇格
**positive**:
- 柏記念 NAR v4 予測 軸=8 ミッキーファイト p_ens=**0.777** (1.5 倍人気)
- ヒューリスティックと TOP3 完全一致、trio 7点 完全一致
**negative**:
- ad-hoc script `predict_nar_kashiwa_5_5.py` で固定値多数 (汎用化必要)
**成果物**: `data/nar/models/keiba_model_nar_v4.pkl`, `tools/predict_nar_kashiwa_5_5.py`, `train/train_nar_v4_leakfree.py`
**影響**: NAR 並列運用 5/12+ への道

## Session #13: 57029ff1 (5/5 16:44) — NAR v4 体系化

**目的**: 一過性で終わらせず長期運用基盤化
**positive**:
- `tools/predict_nar.py` 汎用化 (柏記念で 0.777 完全再現確認)
- schtasks 5 件登録 ps1 (admin、user 手動)
- backtest_nar_v4_quick.py: OOS AUC **0.8519**、条件 A/B/D/E/X 全 0.84+
- 5/12 paper → 5/16 試行 500円/日 → ramp プラン
**negative**:
- chihou_races_2020_2025.csv 不在で strict OOS は別 task
- scrape_nar_today / scrape_nar_results は placeholder
**成果物**: 5 docs + 5 tools (10 files、1,533 行)
**影響**: NAR 5/16 paper 試行の基盤完成、admin schtasks 登録待ち

## Session #14: 2b6dc4eb (5/5 17:04) — 5/9 本番最終調整

**目的**: 5/9 朝起きてから投票完了まで迷わず動ける状態化
**positive**:
- 8 ファイル (1,265 行) で 完全な運用フロー
- model load 1.1s, schtasks 5 件 Ready, Cookie OK 検証済
- race_day_report.py: 5/3 で smoke test ROI 525.7% 計算 OK (既存 summary 上書き防止)
**negative**:
- FridayWeekendScrape は 10:00 登録 (元想定 21:00) → 補完で 5/8 21:00 後 手動 scrape 推奨
**成果物**: `data/results/20260509_pat_checklist.md`, `data/v18/risk_management_5_9.md`, `tools/race_day_report.py`
**影響**: 5/9 投票準備 完成、admin RaceDayReport schtasks 登録 (任意)

---

## クロス cutting テーマ

### Phase 2.5 進捗 (5/4 提案 → 5/5 終了時点)

| 番号 | タスク | 状態 |
|----|--------|------|
| 1 | Cookie 更新 | ✅ Session #2 (S2) |
| 2 | morning task 登録 | ✅ S2 |
| 3 | daily_predict watchdog | ✅ S4 |
| 5 | odds_base retroactive | ✅ S4 |
| 12 | v18 完全 retro | ✅ S4-S5 |
| 8C | TYB publish 観測 | ✅ S8 (5/4-5/10 蓄積中) |
| 8D | v18/v19 calibration | ✅ S8 (Platt scaling 試作、限界判明) |
| 10 | race-level normalize | ✅ S10 |
| 13+ | NAR 復活 + 体系化 | ✅ S12-S13 |
| 14 | 5/9 本番準備 | ✅ S14 |
| 9 | 静音化 | ✅ S9 (admin 待ち) |
| **未完** | NAR scrape 2 script | placeholder のみ |
| **未完** | feature distribution shift 調査 | 5/16+ |
| **未完** | chihou_races_2020_2025.csv | 別 60min task |

### 数字 訂正 (v1 → v2)

| 項目 | v1 | v2 | source |
|------|-----|-----|--------|
| training_times 2025 | 2,551 | 192,296 | csv 直接 |
| 5/2 USER 損失 | -23,800 | -8,820 | USER 報告 |
| v15 batch ROI 31.3% | USER ROI 誤解 | 全 R 仮想 | cumulative.csv |
| TYB 17:00 公開 | 確定 | 不明 (404) | tyb_log.csv |
| NAR AUC | 0.789 | 0.8145 | v4 pkl |
| 累計 | -25,000 | +14,140 | USER 報告 |

詳細: `docs/handoff_v1_v2_diff.md`

### git commit 増加率

- 5/3: 5 commits / 7時間
- 5/4: 7 commits / 12時間 (うち 5 commits は Phase 2.5 一括)
- 5/5: 9 commits / 17時間 (うち 4 commits は normalize 並列)

平均: ~ 1 commit / 2 時間。flow 維持できた。

---

## 関連 doc (5/3-5/5 で生成)

### data/results/
- 20260503_summary.md (S2)
- healthy_4day_analysis.md (S4)
- 20260509_final_plan.md, _v2.md (S3, S6)
- 20260509_race_card.md (S6)
- 20260509_strategy_proposal.md (S3)
- 20260505_kashiwa_kinen.md (S11-S12)
- 20260505_kashiwa_kinen_horses.csv (S11)
- 20260505_kashiwa_kinen_nar_v4.csv (S12)
- 20260509_pre_check.md, _operation_guide.md, _dry_run_5_5.md, _pat_checklist.md (S14)
- data_coverage_audit_5_4.md (S7)

### data/v18/
- phase2_5_proposal_5_4.md (S2)
- system_audit_5_3.md (S2)
- v17_v15_improvement_proposals.md (S6)
- improvements_prototyped_5_3.md (S6)
- may2_postmortem.md (S2-S4)
- v18_backtest_report.md (-)
- v18_v19_corruption_analysis.md (S5)
- calibration_5_4_result.md (S8)
- formation_retro_5_2_5_3.csv (S6)
- phase_2_5_progress_5_4.md (S8)
- phase_2_5_progress_5_4_pm.md (S10)
- phase_2_5_session10_final.md (S10)
- distribution_shift_analysis.md, .json (S10)
- normalization_compare_results.json (S10)
- race_normalize_5_4_result.md (S10)
- v18_v19_integration_plan_5_4_pm.md (S10)
- nar_v4_current_state.md, nar_pipeline_design.md, nar_v4_backtest_5_5.md, jra_nar_integration_plan.md, nar_schtasks_user_guide.md (S13)
- silentify_tasks_user_guide.md (S9)
- post_5_9_improvement_template.md (S14)
- risk_management_5_9.md (S14)
- v18/v19_retro_full_predictions.csv, _calibrated.csv, _result*.md (S4-S10)

### tools/
- daily_predict_watchdog.py (S4)
- v18_v19_retro_full.py (S4, --normalize 拡張 S10)
- calibrate_v18_v19.py (S8)
- tyb_publish_monitor.py + .bat (S8)
- silent_runner.vbs (S9)
- silentify_all_tasks.ps1, silentify_rollback.ps1, task_silentify_backup_5_4.json (S9)
- race_normalize.py (S10)
- analyze_v18_v19_distribution.py (S10)
- compare_normalization_methods.py (S10)
- predict_nar_kashiwa_5_5.py (S12)
- predict_nar.py (S13)
- nar_predict_config.json (S13)
- nar_daily_pipeline.bat (S13)
- register_nar_schtasks.ps1 (S13)
- backtest_nar_v4_quick.py (S13)
- race_day_report.py (S14)
- register_race_day_report_schtasks.ps1 (S14)

### data/nar/
- models/keiba_model_nar_v4.pkl (S12, archive→active)

### docs/
- handoff_v1_v2_diff.md (S15)
- sessions_5_3_5_5_recap.md (S15、本書)
- HANDOFF_5_5_TO_5_9.md (S15)
- lessons_learned_5_5.md (S15)
- next_session_checklist.md (S15)
