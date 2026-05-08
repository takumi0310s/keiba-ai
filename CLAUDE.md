# CLAUDE.md

> **caveman mode**: respond like caveman. short word. no verbose. do thing, say little. result only.

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.
Last updated: **2026-05-08 (Session #43、 ★ V15 ROI 真因 83.96% 発見 ★ + sib_w5 本実装 + 動画 PoC 拡張、 5/16 GO 75-85%)**

> Session #43 (5/8) で 7 領域 並行実行 + ★ V15 ROI 真因発見 ★:
> - **A**: ★ V15 ROI 44% 真因 = actual_payout NaN 集計 bug、 **真の ROI 83.96%** ([data/v18/v15_roi44_root_cause_5_8.md](data/v18/v15_roi44_root_cause_5_8.md))
> - **B**: V20 2025-04 backfill plan (60-90 min) ([data/v18/jvlink_backfill_2025_04_actual_5_8.md](data/v18/jvlink_backfill_2025_04_actual_5_8.md))
> - **C**: sib_exp w5 本実装 + V18/V19 再学習 + LIVE retro (BT AUC 0.8847、 LIVE 進行中)
> - **D**: 動画 PoC 拡張 (frame 抽出 + YOLOv8 95-138ms) ([data/v18/video_poc_extended_5_8.md](data/v18/video_poc_extended_5_8.md))
> - **E**: orchestrator 5 case test (case 1+4 動作 OK) ([tools/test_orchestrator_5_cases.py](tools/test_orchestrator_5_cases.py))
> - **F**: 5/9 戦略 final v3 (案 A 維持 700円×3R) ([docs/PLAN_5_9_FINAL_v3.md](docs/PLAN_5_9_FINAL_v3.md))
>
> Session #42 (5/8 日中) で 10 領域 並行 + 動画 PoC 実行:
> - **A**: 32-bit Python quickstart 1 ページ ([docs/SETUP_PYTHON32_QUICKSTART.md](docs/SETUP_PYTHON32_QUICKSTART.md))
> - **B+G**: 5/1-5/7 actual + V20 phased backfill ([data/v18/v20_backfill_phased_5_8.md](data/v18/v20_backfill_phased_5_8.md))
> - **C**: 拡張 retro 4/18-5/5 (V15 案B改 ROI 44.47% / 39 races) ([data/v18/extended_retro_4_12_5_5_5_8.md](data/v18/extended_retro_4_12_5_5_5_8.md))
> - **D**: 5/10 朝 結果照合 自動化 (verdict 6 シナリオ) ([docs/RESULT_VERIFICATION_5_10.md](docs/RESULT_VERIFICATION_5_10.md))
> - **E**: 動画解析 feasibility GO (ultralytics 8.4 + YOLOv8 138ms 動作) ([docs/PHASE_4_VIDEO_FEASIBILITY_5_8.md](docs/PHASE_4_VIDEO_FEASIBILITY_5_8.md))
> - **F**: sib_exp variant 探索、 **window=5 が最良 corr 0.2010** (full expanding 0.1689 から +0.032) ([data/v18/sib_exp_optimization_5_8.md](data/v18/sib_exp_optimization_5_8.md))
> - **H**: 5/16 V18/V19 投入 plan v2 (GO 65-80%、 6 シナリオ条件分岐) ([docs/PLAN_5_16_V18_V19_DEPLOYMENT_v2.md](docs/PLAN_5_16_V18_V19_DEPLOYMENT_v2.md))
>
> Session #41 巨大マラソン (5/8 深夜) で 8 領域 + sib_exp LIVE retro **+6.89pt 改善大成功** ([data/v18/sib_expanding_v1_retro_5_7.md](data/v18/sib_expanding_v1_retro_5_7.md))。
> Session #39 deluxe (5/7) で 10 領域 設計済 ([Phase 3-4 統合 roadmap](docs/PHASE_3_4_INTEGRATED_ROADMAP.md))。

### JRA-VAN 加入 + JV-Link 環境 (2026-05-07 夜 確定)

| 項目 | 値 |
|------|----|
| JRA-VAN DataLab | 加入完了 |
| JV-Link DLL | C:\\Windows\\SysWow64\\JVDTLAB\\JVDTLab.dll (32-bit only, ver 1.18) |
| ProgID | JVDTLab.JVLink |
| 動作確認 (5/7 夜) | 過去日付 5/3 で 29 ファイル取得 OK |
| 32-bit Python venv | C:\\Users\\takum\\jvlink-venv\\ (推奨、 5/24+ 着手) |
| 既存 64-bit 環境 | 完全維持 (predict_core / daily_predict 含む) |

---

## 1. プロジェクト概要

**競馬AI予測システム（中央競馬専用）**

JRA中央競馬の全レースをAIで予測し、条件別に最適な買い目を自動生成するシステム。
LGB+XGB+FT-Transformer+IntraRace Attention 4モデルアンサンブルで複勝圏（3着以内）を予測し、6つの条件分類に基づいて三連複/馬連の買い目を推奨する。

- **Streamlit**: https://keiba-ai-l2klehd4rfoupnj5g7rw8b.streamlit.app
- **GitHub**: https://github.com/takumi0310s/keiba-ai
- **現行モデル**: **V15** (本番、150 特徴量、AUC 0.8939、本番運用 ROI 119.2%、戦略⑦込み 140%+ 想定)
- 旧モデル: v13.5b は historical reference (124 特徴量、Grid Ensemble、WF AUC 0.8788)
- 5/9 V15 案B改 単独継続 (絶対)。 5/16 V15.1 / V18/V19 共に NO-GO 確定 (Session #38)
- Phase 3 (5/24+): sib_*_exp 修正版 + V20 学習 + JV-Link 加入 → 7/1 V20 投入候補
- Phase 4 (7-8 月): 調教動画 AI 解析 PoC → 9/1 V21 投入候補
- 現行 累計収支: **+13,530 円** / 撤退余裕 +63,530 円
- **2段階モデル**: Pattern A（リークフリー評価用）+ Pattern B（当日情報込み実運用）
- **検証済み**: WF 2020-2025, 実配当ROI 428.4%, 全条件PASS

### Session #38 確定事項 (2026-05-07)

| # | 確定 | 影響 |
|---|------|------|
| 1 | **V15.1 SKB POST-RACE LEAK 確定** (skb_kishi_code_3 +480bp / corr_target 0.137 / monotonic 1着→364, 10着→176) | V15.1 採用 NO-GO、 V20 で SKB 全 10 features 完全除外 |
| 2 | **V18/V19 sib抜き hybrid** (LIVE winner_top1 -10.3pt + shift 30.4x→8.3x) | sib = リーク + 識別能力 hybrid、 expanding window 修正版が必要 |
| 3 | **5/16 V18/V19 投入 NO-GO** | V15 案B改 単独継続、 6/15+ sib_*_exp 版で再判定 |

---

## 2. できること（全機能一覧）

### 予測機能
- **netkeiba URL入力** → 出馬表自動取得 → AI予測 → 条件判定 → 買い目生成
- **Pattern B予測**: 当日オッズ・馬体重・馬場状態・天候を自動取得して高精度予測
- **条件自動判定**: 頭数・距離・馬場状態から6条件(A-E,X)を自動分類
- **買い目生成**: 三連複7点（条件E: 馬連2点）を自動生成
- **EV表示**: 各買い目のExpected Value（期待値）を計算・表示
- **警告機能**: 馬体重急変(±10kg)、混戦オッズを自動検知
- **SQLite記録**: 予測結果をローカルDBに保存

### Streamlitダッシュボード
- **予測ページ**: URL入力 → リアルタイム予測・買い目表示
- **TRACK RECORD**: 過去予測の成績一覧（会場/日付ブラウザ、予測詳細付き）
- **結果登録**: netkeiba結果ページURLで的中判定・配当記録
- **週次ROIレポート**: 条件別・コース別・距離別の成績集計

### 運用ツール
- `tools/daily_predict.py` — 毎朝8:00自動実行、当日全レース予測
- `tools/daily_results.py` — 毎晩20:00自動実行、結果照合・ROI計算
- `tools/weekly_report.py` — 毎週月曜9:00、週次パフォーマンスレポート（条件別・特徴量率・乖離率・累積ROI警告付き）
- `tools/refresh_cookie.py` — netkeibaのCookie自動更新（Playwright自動ログイン）
- `predict_and_log.py` — CLI手動予測・ログ記録
- `check_results.py` — CLI結果照合（--summaryで成績サマリー）
- `verify_real_roi.py` — netkeiba実配当ROI検証

### 検証・分析ツール
- `monte_carlo_sim.py` — モンテカルロ破産確率シミュレーション（10,000試行）
- `project_status.py` — プロジェクト全体ステータス（6セクション）
- `backtest_central_leakfree.py` — ウォークフォワードバックテスト
- `calc_actual_roi.py` — JRA公式配当データでの実ROI計算
- `tools/validation_1〜13_*.py` — 13項目の包括的検証スイート

### データ取得
- `tools/extract_jvdata.py` — TARGET JV (C:\TFJV) → 7CSV抽出
- `scrape_jra_track.py` — JRA公式クッション値・含水率
- `scrape_weather.py` — 気象庁API気温・湿度・風速・降水量
- `scrape_jra_payouts.py` — JRA公式DB配当データ

---

## 3. やったこと（全実施タスク一覧）

### データ取得・変換
1. TARGET JV (C:\TFJV) からCSV抽出（SE_DATA/CK_DATA/HY_DATA/BR_DATA/KT_DATA）
2. jra_races_full.csv 構築（781,161行、2010-2025）
3. training_times.csv 構築（955,580行、木/坂路調教データ）
4. odds_history.csv 構築（778,387行）
5. blood_full.csv 構築（81,986行、血統データ）
6. JRA公式DB配当スクレイパー構築 → jra_payouts.csv（27,541件、2018-2025）
7. JRA馬場情報スクレイパー（クッション値・含水率）
8. 気象庁APIスクレイパー（気温・湿度・風速・降水量）
9. netkeiba出馬表・結果スクレイパー（db.netkeiba.comフォールバック対応）

### モデル学習・改善
1. V8ベースモデル学習
2. V9.1基盤特徴量（43特徴量）
3. V9.2追加特徴量（+11: career/sire/wood training）→ AUC改善
4. V9.3追加特徴量（+13: pace/distance aptitude/frame advantage）→ AUC 0.8095
5. Pattern A（リークフリー67特徴量）確立 → 確定オッズリーク発見・除去
6. Pattern B（当日情報込み75特徴量）学習 → AUC 0.8460（参考値）
7. V10アンサンブル（LGB+XGB+MLP）試行 → 不採用（WF 0.8050 < 0.8083）
8. コース別専用モデル試行 → 不採用（過学習）
9. Optunaハイパーパラメータ最適化（100試行）→ 微改善のみ、不採用
10. 2段階モデル構成（学習=A、予測=B）確立
11. V11 speed index試行 → 不採用（WF AUC 0.801 < 0.802 baseline）
12. **V12総合再学習（+7特徴量）→ WF AUC 0.8037（+0.0031）→ 採用**
    - dam_top3rリーク発見（全年データで計算→expanding windowに修正→マイナス寄与で除外）
13. **v13.4 JRDB完全連携（+50特徴量）→ WF AUC 0.8610（LGB+XGB）→ 採用**
    - 騎手・調教師・血統・レースペース等のJRDB特徴量を大量追加（74→124特徴量）
14. **v13.5 FT-Transformer追加 → WF AUC 0.8659（3-model）→ 採用**
    - LGB+XGB+FT-Transformer 3モデルアンサンブル
15. **v13.5b IntraRace Attention追加 → WF AUC 0.8788（4-model grid）→ 採用（現行）**
    - レース内馬同士の相対関係をAttentionで学習
    - Grid Search重み最適化、IR重み0.35で最大貢献
    - 実配当ROI 428.4%（全条件v13.4以上、JRA公式配当検証済み）

### テスト・検証（22項目 + Phase 10-13）
1. リークフリー検証（encode_categoricals/encode_sires静的解析）→ PASS
2. 目的変数比較（Win/Place/EV weighted）→ Place最適 AUC 0.8019
3. EVフィルタ分析（EV≥1.0閾値）→ 全レースEV≥1.0で効果なし
4. 券種最適化（全条件×全券種）→ 現行が最適
5. オッズギャップ分析（購入時vs確定）→ ROI影響0-5%
6. ドローダウン分析（MDD/連敗/回復/破産確率）→ 3万円以上で破産0%
7. 年別パフォーマンス（2020-2025 AUC/ROI）→ 安定上昇傾向
8. データ拡張チェック（67特徴量網羅性）→ 十分
9. 最終レポート統合 → READY判定
10. 市場依存性テスト（prev_odds_log除外）→ LOW依存、真の能力予測
11. サンプルサイズ検証（Bootstrap CI）→ N=20,579 HIGH信頼性
12. ROI計算整合性チェック → 全PASS
13. 保守的ROI見積り（BT×0.7）→ 全体142.6%
14. 特徴量リーク監査（全67特徴量）→ PASS
15. WFバックテスト（2020-2025, 20,579レース）→ 全条件ROI 100%超え
16. モンテカルロシミュレーション（10,000試行×1,000レース）
17. 5項目自動テスト（tests/test_features.py）
18. 25項目デバッグテスト（tests/debug_all.py）
19. 詳細ROI分析8テスト（月別/場別/クラス/芝ダ/人気/配当分布/D細分化/ストレス）

### バグ修正
1. 確定オッズ(odds_log)リーク発見・除去
2. 条件E買い目trio→umaren切替
3. 条件E投資額200→700円修正
4. app.py BASE_DIR未定義修正
5. バッチ予測：実モデル使用・結果UI再設計
6. TRACK RECORD UI: 会場/日付ブラウザ追加
7. 馬番昇順ソート・カンマスペース区切り表示
8. db.netkeiba.comフォールバック対応
9. bet_type記録バグ修正（条件別正確な記録）
10. モデルロード絶対パス修正（Streamlit Cloud対応）
11. 条件D 1000m以下を購入非推奨に変更（ROI 85%, N=534）
12. predict_core.py: speed index premium cacheフォールバック追加（3特徴量が全滅していた）
13. predict_core.py: 距離カテゴリbin不一致修正（学習5bin vs 予測4bin→5binに統一）
14. Discord重複通知修正（3重→race_auto_notify.pyのみに統合）
15. v12モデルロード: バージョンチェック修正（v12が特徴量18個しか生成しないバグ）
16. num_horses_val特徴量: 出走頭数が反映されないバグ修正
17. 調教スクレイパー: single-row HTML対応（intensity 83→466/497=94%）
18. 調教スクレイパー: short CW時間パース + 栗Ｅ/美Ｅコース対応（4F取得 90%→96%）
19. EV表示・レース信頼度スコア・変動投資額機能追加

### インフラ整備
1. Streamlit Cloud デプロイ
2. Windows タスクスケジューラ設定（daily_predict/results, weekly_report）
3. SQLiteローカルDB構築
4. .gitignore設定（大容量CSV除外）
5. project_status.py CLI構築
6. バッチファイル作成（daily_predict.bat, daily_results.bat, weekly_report.bat）
7. Cookie自動更新ツール（tools/refresh_cookie.py, Playwright自動ログイン）
8. 週次レポート拡張（条件別成績・特徴量取得率・BT乖離率・累積ROI警告）

---

## 4. モデル詳細

### 2段階モデル設計思想
- **Pattern A（評価用）**: リークフリー厳守。モデルの真の実力を評価
- **Pattern B（実運用）**: 使える情報は全て使って最高精度で予測

### Pattern A スペック（v13.5b、現行）
- ファイル: `keiba_model_v135_central.pkl.gz`（LGB+XGB）+ FT/IRモデルは動的学習
- WF AUC: **0.8788**（walk-forward 2020-2025平均, 4-model grid ensemble）
- 特徴量: **124個**（リークフリー、JRDB連携含む）
- 学習データ: ~527,000行
- 目的変数: `finish <= 3`（複勝圏 binary）
- アンサンブル: LGB + XGB + FT-Transformer + IntraRace Attention（Grid重み最適化）
- 実配当ROI: **428.4%**（JRA公式配当、WF 2023-2025、全条件PASS）

### Pattern A スペック（v12、旧版）
- ファイル: `keiba_model_v12_central.pkl.gz`
- WF AUC: **0.8037**（walk-forward 2020-2025平均, LGB単体）
- 特徴量: **74個**（リークフリー、V9.3の67 + V12の7）

### Pattern B スペック
- ファイル: `keiba_model_v135_central_live.pkl.gz`（LGB+XGB）
- 特徴量: **132個**（Pattern A 124 + 当日情報8）
- app.pyはPattern Bを優先、なければA→V12→V8にフォールバック
- 馬場/天候データ取得失敗時は0=欠損として予測

### Pattern A 全74特徴量

#### 基本特徴量（14個）
| # | 特徴量 | 説明 |
|---|--------|------|
| 1 | weight_carry | 斤量(kg) |
| 2 | age | 馬齢 |
| 3 | distance | レース距離(m) |
| 4 | course_enc | コース(0-9, 10=unknown) |
| 5 | surface_enc | 芝=0, ダート=1, 障害=2 |
| 6 | sex_enc | 牡=0, 牝=1, セン=2 |
| 7 | num_horses_val | 出走頭数 |
| 8 | horse_num | 馬番 |
| 9 | bracket | 枠番(1-8) |
| 10 | sire_enc | 父馬TOP100エンコード(0-99, 100=other) |
| 11 | bms_enc | 母父TOP100エンコード(0-99, 100=other) |
| 12 | location_enc | 所属(0=美浦, 1=栗東, 2=地方, 3=外国) |
| 13 | is_nar | NAR=1, JRA=0 |
| 14 | season | 春=0, 夏=1, 秋=2, 冬=3 |

#### 騎手・調教師（3個）
| # | 特徴量 | 説明 |
|---|--------|------|
| 15 | jockey_wr_calc | 騎手勝率(expanding window, alpha=30) |
| 16 | jockey_course_wr_calc | 騎手コース別勝率(expanding, alpha=10) |
| 17 | jockey_surface_wr | 騎手馬場別勝率(expanding, alpha=10) |

#### 前走ラグ特徴量（10個）
| # | 特徴量 | 説明 |
|---|--------|------|
| 18 | prev_finish | 前走着順 |
| 19 | prev2_finish | 前々走着順 |
| 20 | prev3_finish | 3走前着順 |
| 21 | prev_last3f | 前走上がり3F |
| 22 | prev2_last3f | 前々走上がり3F |
| 23 | prev_pass4 | 前走4角位置 |
| 24 | prev_prize | 前走賞金 |
| 25 | prev_odds_log | 前走オッズ(log) |
| 26 | rest_days | 休養日数(1-365でclip) |
| 27 | rest_category | 休養カテゴリ(0-5: 7/15/35/64/181日区切り) |

#### 集計特徴量（5個）
| # | 特徴量 | 説明 |
|---|--------|------|
| 28 | avg_finish_3r | 直近3走平均着順 |
| 29 | best_finish_3r | 直近3走最高着順 |
| 30 | top3_count_3r | 直近3走の3着以内回数 |
| 31 | finish_trend | 着順トレンド(prev3 - prev) |
| 32 | avg_last3f_3r | 直近3走平均上がり3F |

#### 派生特徴量（11個）
| # | 特徴量 | 説明 |
|---|--------|------|
| 33 | dist_change | 前走からの距離変更(m) |
| 34 | dist_change_abs | 距離変更絶対値 |
| 35 | dist_cat | 距離カテゴリ(0-4) |
| 36 | age_sex | 年齢×10+性別 |
| 37 | age_season | 年齢×10+季節 |
| 38 | horse_num_ratio | 馬番/頭数 |
| 39 | bracket_pos | 枠位置(内=0, 中=1, 外=2) |
| 40 | carry_diff | 斤量 - レース平均斤量 |
| 41 | age_group | 年齢(2-7でclip) |
| 42 | surface_dist_enc | 馬場×10+距離カテゴリ |
| 43 | course_surface | コース×10+馬場 |

#### V9.2追加（8個、リーク除外後）
| # | 特徴量 | 説明 |
|---|--------|------|
| 44 | horse_career_races | 通算出走数(expanding, 0-indexed) |
| 45 | horse_career_wr | 通算勝率(expanding, alpha=5) |
| 46 | horse_career_top3r | 通算複勝率(expanding, alpha=5) |
| 47 | sire_surface_wr | 父馬産駒馬場別勝率(expanding, alpha=50) |
| 48 | sire_dist_wr | 父馬産駒距離別勝率(expanding, alpha=50) |
| 49 | bms_surface_wr | 母父産駒馬場別勝率(expanding, alpha=50) |
| 50 | wood_best_4f_filled | 木馬場調教4Fベスト(14日, mean fill ~52.0s) |
| 51 | has_wood_training | 木馬場調教データ有無 |

#### V9.2派生（2個）
| # | 特徴量 | 説明 |
|---|--------|------|
| 52 | sire_dist | 父馬×10+距離カテゴリ |
| 53 | sire_surface | 父馬×10+馬場 |

#### V9.2調教（2個）
| # | 特徴量 | 説明 |
|---|--------|------|
| 54 | training_time_filled | 調教4Fタイム(mean fill) |
| 55 | has_training | 調教データ有無 |

#### V9.3新規（12個）
| # | 特徴量 | 説明 |
|---|--------|------|
| 56 | prev_race_first3f | 前走前半3F(ラップデータ) |
| 57 | prev_race_last3f | 前走後半3F(ラップデータ) |
| 58 | prev_race_pace_diff | 前走後半3F-前半3F(ペース差) |
| 59 | prev_agari_relative | 前走上がり相対値(個人-全体) |
| 60 | wood_count_2w | 木馬場調教回数(2週間) |
| 61 | sakaro_best_4f_filled | 坂路4Fベスト(14日, mean fill ~53.0s) |
| 62 | sakaro_best_3f_filled | 坂路3Fベスト(14日, mean fill ~39.0s) |
| 63 | has_sakaro_training | 坂路調教データ有無 |
| 64 | total_training_count | 調教合計回数(木+坂路) |
| 65 | horse_dist_top3r | 馬の距離別複勝率(expanding, alpha=5) |
| 66 | horse_surface_top3r | 馬の馬場別複勝率(expanding, alpha=5) |
| 67 | frame_course_dist_wr | 枠×コース×距離の勝率(expanding, alpha=100) |

#### V12新規（7個）
| # | 特徴量 | 説明 |
|---|--------|------|
| 68 | index_max_filled | netkeibaタイム指数最高値(mean fill) |
| 69 | index_run1_filled | netkeiba前走指数(mean fill) |
| 70 | index_avg5_filled | netkeiba5走平均指数(mean fill) |
| 71 | time_1f_last_filled | 追切ラスト1Fタイム(mean fill ~12.5s) |
| 72 | training_intensity_enc | 調教強度(0=不明, 1=馬なり, 2=強め, 3=一杯) |
| 73 | sire_shinba_top3r | 種牡馬新馬戦複勝率(expanding, alpha=20) |
| 74 | pci | ペースチェンジ指数(後半3F/前半3F) |

#### V12/V12.1で不採用とした特徴量
| 特徴量 | 理由 |
|--------|------|
| dam_top3r | 母産駒複勝率。初回テスト+0.023はデータリーク（全年データで計算）。expanding window修正後は-0.0006で除外 |
| stable_comment_score | 厩舎コメントスコア。WFカバレッジ30%で不十分（追加取得中） |
| prev_review_score | 前走不利スコア。v12.1テスト: +0.00016(微プラス)。2021年gap=0.0514>0.05で過学習判定により不採用 |
| shinba_eval_score | 新馬評価スコア。v12.1テスト: +0.00007(ほぼゼロ)。2024-2025のみで汎化困難 |

### Pattern Bの追加8特徴量
| 特徴量 | ソース | 説明 |
|--------|--------|------|
| odds_log | netkeiba | 単勝オッズ(log変換) |
| pop_rank | netkeiba | 人気順位 |
| horse_weight | netkeiba | 当日馬体重(kg) |
| weight_change | netkeiba | 馬体重変化(前走比) |
| weight_change_abs | netkeiba | 馬体重変化絶対値 |
| weight_cat | 計算 | 体重カテゴリ(0-3) |
| weight_cat_dist | 計算 | 体重カテゴリ×距離カテゴリ |
| condition_enc | netkeiba | 馬場状態(良=0, 稍重=1, 重=2, 不良=3) |
| cond_surface | 計算 | 馬場×馬場種別 |
| cushion_value | JRA公式 | クッション値(芝のみ) |
| moisture_rate | JRA公式 | 含水率 |
| temperature | 気象庁API | 気温(℃) |
| humidity | 気象庁API | 湿度(%) |
| wind_speed | 気象庁API | 風速(m/s) |
| precipitation | 気象庁API | 降水量(mm) |
| weather_enc | 気象庁API | 天候(晴=0, 曇=1, 雨=2, 雪=3) |

### アンサンブル構成（v13.5b、現行）
- **4モデル Grid Ensemble**: LightGBM + XGBoost + FT-Transformer + IntraRace Attention
- 重み: Grid Search最適化（年ごとに異なる）
- 典型的重み: LGB=0.25, XGB=0.25-0.30, FT=0.10-0.15, IR=0.35
- `pred = w_lgb * lgb + w_xgb * xgb + w_ft * ft + w_ir * ir`
- IntraRace Attentionがレース内相対関係を捕捉し、最大の貢献（重み0.35）

### アンサンブル構成（v12以前）
- **LightGBM（主）** + **XGBoost（副）**
- 重み: AUC比例（LGB ~56%, XGB ~44%）

### LGBパラメータ
```python
{
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'num_leaves': 63,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'min_child_samples': 50,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'verbose': -1,
    'seed': 42,
}
# Early stopping: 50 rounds, max 1000 rounds
```

### XGBパラメータ
```python
{
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'max_depth': 6,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 50,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'seed': 42,
    'tree_method': 'hist',
}
# Early stopping: 50 rounds, max 1000 rounds
```

### Optuna結果（100試行、不採用）
- Best: WF AUC 0.8022（+0.0006）→ 基準未達のため不採用
- 現行パラメータを維持（V12でも同一パラメータ）

---

## 5. 全条件詳細

### 条件定義テーブル

#### v13.5b（現行）WF 2023-2025, 10,314レース, 4-model Grid Ensemble (AUC 0.8788), JRA公式配当データ

| 条件 | 条件内容 | 買い目 | v13.5b実ROI | 的中率 | N | v13.4実ROI |
|------|----------|--------|------------|--------|------|-----------|
| A | 8-14頭/1600m+/良〜稍重 | trio 7点 | **355.4%** | 63.2% | 3,212 | 308.0% |
| B | 8-14頭/1600m+/重〜不良 | trio 7点 | **346.8%** | 61.3% | 398 | 259.4% |
| C | 15頭+/1600m+/良〜稍重 | trio 7点 | **623.0%** | 51.6% | 2,473 | 521.5% |
| D | 1200-1400m | trio 7点 | **360.8%** | 48.2% | 3,581 | 314.7% |
| E | 7頭以下 | umaren 2点 | **195.7%** | 72.7% | 267 | 141.4% |
| X | 15頭+/重〜不良 | trio 7点 | **701.2%** | 54.0% | 383 | 498.9% |
| **全体** | | | **428.4%** | | **10,314** | 361.9% |

#### v12（旧版）WF 2020-2025, 20,579レース, LGB単体 (AUC 0.8037), JRA公式配当データ

| 条件 | 条件内容 | 買い目 | 実ROI | 的中率 | N |
|------|----------|--------|-------|--------|------|
| A | 8-14頭/1600m+/良〜稍重 | trio 7点 | **205.3%** | 44.5% | 6,438 |
| B | 8-14頭/1600m+/重〜不良 | trio 7点 | **236.9%** | 45.2% | 847 |
| C | 15頭+/1600m+/良〜稍重 | trio 7点 | **285.6%** | 33.7% | 4,774 |
| D | 1200-1400m | trio 7点 | **136.0%** | 27.0% | 7,254 |
| E | 7頭以下 | umaren 2点 | **118.0%** | 53.4% | 461 |
| X | 15頭+/重〜不良 | trio 7点 | **330.5%** | 35.5% | 805 |

- **全条件ROI 100%超え**（v13.5b、v12ともに）
- **1000m以下は非推奨**: ROI 85.0% (N=534) → 予測はするが購入非推奨表示
- 投資額: 全条件700円/レース

### 買い目構成
- **trio（三連複）7点**: TOP1軸 - TOP2,TOP3 - TOP2〜TOP6 のフォーメーション
- **umaren（馬連）2点**: TOP1軸 - TOP2, TOP1軸 - TOP3（各350円、オッズ連動400/300振り分け）

### 条件判定ロジック（優先順位順）
```python
def classify_condition(num_horses, distance, condition):
    heavy = condition in ['重', '不良']  # or condition_enc >= 2
    if num_horses <= 7:       return 'E'  # 少頭数
    if distance <= 1400:      return 'D'  # スプリント (1200-1400m推奨、1000m以下は非推奨)
    if 8 <= nh <= 14 and distance >= 1600 and not heavy: return 'A'
    if 8 <= nh <= 14 and distance >= 1600 and heavy:     return 'B'
    if num_horses >= 15 and distance >= 1600 and not heavy: return 'C'
    return 'X'  # その他（15頭+/重〜不良など）
    # 注: 条件Dかつ1000m以下 → recommended=False（ROI 85%, N=534）
```

---

## 6. テスト結果一覧

### リークフリー検証 → PASS
- encode_categoricals: ルールベース変換（リークなし）
- encode_sires: fold毎にtrain dataのみで計算（リークなし）
- expanding window: cumsum - current（当該レース除外）
- 軽微な技術的リーク（fillna global mean, Bayesian prior）→ 影響無視可能

### ウォークフォワード年別AUC
| 年 | V12 (LGB) | v13.4 (LGB+XGB) | v13.5b (4-model grid) | v13.5b改善 |
|----|-----------|-----------------|----------------------|-----------|
| 2020 | 0.7923 | 0.8527 | 0.8515 | -0.0013 |
| 2021 | 0.8015 | 0.8647 | 0.8806 | +0.0159 |
| 2022 | 0.8052 | 0.8675 | 0.8830 | +0.0155 |
| 2023 | 0.8042 | 0.8687 | 0.8845 | +0.0158 |
| 2024 | 0.8109 | 0.8706 | 0.8853 | +0.0147 |
| 2025 | 0.8079 | 0.8696 | 0.8851 | +0.0155 |
| **平均** | **0.8037** | **0.8656** | **0.8788** | **+0.0127** |

注: v13.4/v13.5bはJRDB特徴量追加(124個)によりv12(74個)から大幅改善。

### 実配当ROI（条件別）→ 全条件100%超え
- **v13.5b**: A: 355.4%, B: 346.8%, C: 623.0%, D: 360.8%, E: 195.7%(uma), X: 701.2% — **全体428.4%**
- **v12**: A: 205.3%, B: 236.9%, C: 285.6%, D: 136.0%, E: 118.0%, X: 330.5%

### モンテカルロ結果（10,000試行×1,000レース）
| 初期資金 | 破産確率 | 利益確率 | 期待ROI | 平均最終資金 |
|----------|---------|---------|---------|-------------|
| 1万円 | 0.59% | 99.4% | 15,497% | 1,549,735円 |
| **3万円** | **0.0%** | **100%** | **5,239%** | **1,574,242円** |
| 10万円 | 0.0% | 100% | 1,644% | 1,644,242円 |

### ドローダウン分析
| 初期資金 | MDD平均 | MDD最悪 | 最大連敗 | 回復(avg) |
|----------|---------|---------|---------|----------|
| 1万円 | 25.2% | 99.7% | 37レース | 9レース |
| 3万円 | 11.1% | 53.9% | 37レース | 3レース |
| 10万円 | 4.6% | 16.2% | 37レース | 3レース |

### 市場依存性テスト → LOW依存
- Baseline AUC: 0.8019 → No-odds AUC: 0.7993（差: -0.0026）
- Baseline ROI: 194.6% → No-odds ROI: 204.4%（むしろ改善）
- **判定: 真の能力予測モデル（オッズ依存ではない）**

### サンプルサイズ検証
| 条件 | N | ROI 95%CI | 信頼性 |
|------|---|-----------|--------|
| A | 6,438 | [198%, 213%] | HIGH |
| B | 847 | [213%, 261%] | LOW |
| C | 4,774 | [272%, 300%] | MEDIUM |
| D | 7,254 | [130%, 142%] | HIGH |
| E | 461 | [103%, 133%] | LOW |
| X | 805 | [292%, 369%] | LOW |

### 保守的ROI見積り（BT × 0.7）
- 補正要因: オッズ差(-7.5%), モデル劣化(-10%), 条件過学習(-10%)
- **全体保守的ROI: 142.6%**
- 条件別: A=143.7%, B=165.8%, C=199.9%, D=95.2%, E=82.6%, X=231.3%

### 最終判定: **READY**
- リークフリー: PASS
- AUCベースライン: PASS
- ROI全条件100%超え: PASS

---

## 7. データ資産

### コアデータ（data/ディレクトリ、.gitignore対象）
| ファイル | 行数 | 内容 |
|----------|------|------|
| jra_races_full.csv | 781,161 | 中央競馬全レースデータ(2010-2025) |
| training_times.csv | 955,580 | 調教タイムデータ(木/坂路) |
| odds_history.csv | 778,387 | オッズ履歴 |
| blood_full.csv | 81,986 | 血統データ |
| jra_payouts.csv | 27,541 | JRA公式配当(2018-2025) |

### JRA配当CSVフォーマット
```
race_date, course, kai, nichi, race_num, tansho_nums, tansho_payout,
fukusho_nums, fukusho_payouts, umaren_nums, umaren_payout, wide_nums,
wide_payouts, trio_nums, trio_payout, tierce_nums, tierce_payout
```

### TARGETデータソース
- パス: `C:\TFJV`
- SE_DATA: レース情報
- CK_DATA: 調教データ
- HY_DATA: 票数/オッズ
- BR_DATA: 血統
- KT_DATA: その他

### 検証結果JSON（data/ディレクトリ）
| ファイル | 内容 |
|----------|------|
| actual_roi_results.json | 全条件実配当ROI |
| monte_carlo_results.json | 破産確率シミュレーション |
| drawdown_analysis.json | ドローダウン詳細 |
| market_dependency_test.json | 市場依存性テスト |
| sample_size_validation.json | サンプルサイズCI |
| conservative_roi_estimate.json | 保守的ROI見積り |
| yearly_performance.json | 年別AUC/ROI |
| final_validation_report.json | 最終検証レポート |
| standardization_leak_check.json | リークチェック |
| target_variable_comparison.json | 目的変数比較 |
| ev_filter_analysis.json | EVフィルタ分析 |
| ticket_type_optimization.json | 券種最適化 |
| odds_gap_analysis.json | オッズギャップ |
| data_augmentation_check.json | データ拡張チェック |
| roi_calculation_validation.json | ROI整合性 |
| optuna_tuning_results.json | Optuna結果 |

---

## 8. リーク厳禁ルール

### Pattern Aで除外する特徴量（8個）
```python
LEAK_FEATURES_A = {
    'odds_log',          # 確定オッズ → 投票締切後に確定。最重要リーク
    'horse_weight',      # 当日馬体重 → レース70分前に発表
    'condition_enc',     # 馬場状態 → レース当日朝に発表
    'weight_change',     # 馬体重変化 → horse_weightから派生
    'weight_change_abs', # 馬体重変化絶対値 → horse_weightから派生
    'weight_cat',        # 体重カテゴリ → horse_weightから派生
    'weight_cat_dist',   # 体重×距離カテゴリ → horse_weightから派生
    'cond_surface',      # 馬場×馬場種別 → condition_encから派生
}
```

### V20 で追加除外する特徴量（Session #38 確定、計 18 個）
```python
# train/v15_1_features.py V20_LEAK_FEATURES (Session #39 C で実装)
SKB_LEAK_FEATURES = [  # Session #38 で SKB POST-RACE LEAK 確定
    'skb_kishi_code_1', 'skb_kishi_code_2', 'skb_kishi_code_3',
    'skb_baba_code_1',  'skb_baba_code_2',  'skb_baba_code_3',
    'skb_kyaku_code_1', 'skb_kyaku_code_2', 'skb_kyaku_code_3',
    'skb_turf_hoof',
]  # 10 features (V15.1 SKB 全体)
V20_LEAK_FEATURES = LEAK_FEATURES_A | SKB_LEAK_FEATURES  # 8 + 10 = 18
```

V20 学習時は `merge_v15_1_features(skip_skb=True)` で完全除外。

### 過去の失敗から学んだ教訓
| 失敗 | 詳細 | 教訓 |
|------|------|------|
| **odds_logリーク** | 確定オッズを特徴量に使用していた | 絶対に使わない。importance 1位だった |
| **推定ROI過大評価** | `o1*o2*o3*20` が実配当の約2倍 | 必ず実配当ROI(jra_payouts.csv)で判断 |
| **LGB+XGB+MLP** | V10: WF 0.8050 < LGB単体 0.8083 | MLPは逆効果。ただしFT-TransformerとIntraRaceは有効（v13.5b） |
| **コース別専用モデル** | 汎用モデルに勝てなかった | 過学習リスク大。汎用モデル一択 |
| **坂路調教マッチ率** | horse_name→horse_id変換が27%しか成功しない | AUC改善なし。現在はmean fillで対応 |
| **Optuna過信** | 100試行で+0.0006のみ | 微改善は本番環境で消える可能性大 |
| **dam_top3rリーク** | 全年データで母産駒成績を計算→WF AUC+0.023の大半がリーク | 外部CSVの集計値は必ずexpanding windowで。静的CSVをそのまま使うな |
| **SKB POST-RACE LEAK** (Session #38, 5/7) | skb_kishi_code_3 単独 +480bp、 corr_target 0.137、 1着馬 0-rate 15% / 敗者 49%、 finish と monotonic | JRDB SKB ファイル = "成績拡張" = post-race。 **V20 で全 10 features 完全除外**、 LEAK_FEATURES の追加 list 化 |
| **sib_top3_rate hybrid** (Session #38) | 旧 sib_top3_rate corr_target 0.2939 → 新 sib_top3_rate_exp 0.1689 (-0.125 リーク除去後の真の信号 0.169 残存) | 静的 CSV の集計値は必ず date 順 cumsum-current で expanding 化。 dam_top3r 教訓と同根の再発 |

### リークフリー設計原則
1. 全統計特徴量は**expanding window**（cumsum - current、当該レース除外）
2. sire encodingはfold毎にtrain dataのみで計算（`encode_sires_fold()`）
3. Bayesian smoothing（alpha prior）で低サンプル時の過学習を防止
4. **Pattern Aで評価、Pattern Bで予測**を厳守

---

## 9. 期待値・資金計画

### 年間投資額・期待利益（保守的見積り）
- 月間投資額: 72,100円（全条件合計、700円/レース）
- **月間期待利益: +28,953円**（保守的ROI 142.6%）
- **年間期待利益: +347,436円**

### 月間レース数（推定）
| 条件 | 月間レース | 月間投資 | 月間期待利益 |
|------|-----------|---------|-------------|
| A | 30 | 21,000円 | +9,177円 |
| B | 5 | 3,500円 | +2,303円 |
| C | 22 | 15,400円 | +15,385円 |
| D | 40 | 28,000円 | -1,344円 |
| E | 2 | 1,400円 | -244円 |
| X | 4 | 2,800円 | +3,676円 |

### モンテカルロ破産確率
- **推奨初期資金: 3万円以上**（破産確率0.0%）
- 1万円でも破産確率0.59%と極めて低い
- 1,000レース後の期待資金: 150万円以上

### ドローダウン耐性（初期3万円）
- 平均MDD: 11.1%（約3,300円の一時的損失）
- 最悪MDD: 53.9%（約16,000円の一時的損失）
- 平均回復: 3レースで回復
- 最大連敗: 37レース（全条件合算での理論値）

---

## 10. 未解決課題・今後のタスク

### 高優先度
- [ ] LINE通知実装（予測完了・的中通知）
- [ ] GitHub Actionsによる自動化（現在はWindows タスクスケジューラで代替）
- [ ] 条件D/Eの保守的ROI改善（現状100%以下）
- [x] 実運用でのROI追跡・アラート（tools/roi_monitor.py、daily_results.py後に自動実行、Discord警告通知）

### 中優先度
- [ ] Pattern Bの天候・馬場情報取得失敗時の代替データソース
- [ ] 新特徴量探索（血統クロス、コース形状、ペース予測等）
- [ ] リアルタイムオッズ変動の反映（EV計算の精度向上）

### 低優先度
- [ ] モバイルUI最適化
- [ ] 複数モデルのA/Bテスト基盤
- [ ] 三連単への拡張検討（hit rate低すぎる可能性）

---

## 11. 全ファイル構成

```
keiba-ai/
├── app.py                          # Streamlitメインアプリ (~5200行)
├── CLAUDE.md                       # このファイル
├── requirements.txt                # Python依存パッケージ
├── packages.txt                    # APTパッケージ (libgomp1)
├── .gitignore                      # 大容量CSV除外設定
│
├── # === モデルファイル ===
├── keiba_model_v135_central_live.pkl.gz # v13.5b Pattern B (実運用, 132特徴量, 現行)
├── keiba_model_v135_central.pkl.gz  # v13.5b Pattern A (評価用, 124特徴量, 現行)
├── keiba_model_v134_central_live.pkl.gz # v13.4 Pattern B (フォールバック)
├── keiba_model_v134_central.pkl.gz  # v13.4 Pattern A (フォールバック)
├── keiba_model_v12_central_live.pkl.gz # V12 Pattern B (旧版, 82特徴量)
├── keiba_model_v12_central.pkl.gz  # V12 Pattern A (旧版, 74特徴量)
│
├── # === 運用スクリプト ===
├── predict_and_log.py              # CLI予測・ログ記録
├── check_results.py                # 結果照合・ROI計算
├── verify_real_roi.py              # netkeiba実配当ROI検証
├── monte_carlo_sim.py              # モンテカルロ破産確率
├── project_status.py               # プロジェクトステータスCLI
├── backtest_central_leakfree.py    # WFバックテスト
├── calc_actual_roi.py              # JRA公式配当ROI計算（v12）
├── calc_actual_roi_v135b.py        # v13.5b実配当ROI検証（v13.4比較）
├── analyze_conditions.py           # 条件分析
│
├── # === データ取得 ===
├── scrape_jra_track.py             # JRA馬場情報(クッション値/含水率)
├── scrape_weather.py               # 気象庁API天候データ
├── scrape_jra_payouts.py           # JRA公式DB配当データ
│
├── # === バッチファイル ===
├── daily_predict.bat               # 毎朝8:00自動実行
├── daily_results.bat               # 毎晩20:00自動実行
├── weekly_report.bat               # 毎週月曜9:00
│
├── train/                          # === 学習スクリプト ===
│   ├── train_v135b_intra_ensemble.py # **v13.5b 4-model ensemble（現行）**
│   ├── train_v135_ft_transformer.py # v13.5 FT-Transformer + データ構築
│   ├── train_v134_jockey_trainer.py # v13.4 JRDB騎手・調教師特徴量
│   ├── train_v134_odds_change.py   # v13.4 オッズ変動特徴量
│   ├── train_v134_weight_trend.py  # v13.4 馬体重トレンド特徴量
│   ├── train_v134b_2020fix.py      # v13.4b 2020年修正版
│   ├── train_v92_central.py        # V9.2基盤関数群（全特徴量エンジニアリング）
│   ├── train_v92_leakfree.py       # FEATURES_PATTERN_A, LEAK_FEATURES_A定義
│   ├── train_v12_comprehensive.py   # V12学習+WFバックテスト（旧版）
│   ├── train_v121_comprehensive.py  # V12.1テスト（不採用: prev_review+shinba_eval）
│   ├── optuna_tune_lgb.py          # Optunaハイパラ最適化
│   ├── explore_features.py         # 特徴量探索
│   └── analyze_course_distance.py  # コース/距離分析
│
├── tools/                          # === 運用・検証ツール ===
│   ├── daily_predict.py            # 毎朝自動予測
│   ├── daily_results.py            # 毎晩結果照合
│   ├── weekly_report.py            # 週次レポート（条件別・特徴量率・乖離率・累積ROI警告）
│   ├── refresh_cookie.py           # netkeiba Cookie自動更新（Playwright）
│   ├── extract_jvdata.py           # TARGET JV → CSV抽出
│   ├── validation_1_standardization_leak.py   # リーク検証
│   ├── validation_2_target_variable.py        # 目的変数比較
│   ├── validation_3_ev_filter.py              # EVフィルタ
│   ├── validation_4_ticket_optimization.py    # 券種最適化
│   ├── validation_5_odds_gap.py               # オッズギャップ
│   ├── validation_6_drawdown.py               # ドローダウン
│   ├── validation_7_yearly_performance.py     # 年別パフォーマンス
│   ├── validation_8_data_augmentation.py      # データ拡張
│   ├── validation_9_final_report.py           # 最終レポート統合
│   ├── validation_10_market_dependency.py     # 市場依存性
│   ├── validation_11_sample_size.py           # サンプルサイズ
│   ├── validation_12_roi_integrity.py         # ROI整合性
│   ├── validation_13_conservative_roi.py      # 保守的ROI
│   ├── scrape_shinba_eval.py                  # 新馬評価スクレイパー
│   ├── scrape_race_review.py                  # レース短評(備考)スクレイパー
│   ├── sire_shinba_stats.py                   # 種牡馬新馬成績計算
│   ├── compute_sibling_stats.py               # 母産駒成績計算
│   ├── bulk_scrape_comments.py                # 厩舎コメント一括取得
│   └── predict_core.py                        # 共通予測ロジック
│
├── tests/                          # === テスト ===
│   ├── test_features.py            # 5項目自動テスト
│   └── debug_all.py                # 25項目デバッグテスト
│
├── data/                           # === データ（大容量はgitignore） ===
│   ├── jra_races_full.csv          # 781,161行 (gitignore)
│   ├── training_times.csv          # 955,580行 (gitignore)
│   ├── odds_history.csv            # 778,387行 (gitignore)
│   ├── blood_full.csv              # 81,986行 (gitignore)
│   ├── jra_payouts.csv             # 27,541件 (gitignore)
│   ├── netkeiba_speed_index.csv    # タイム指数(142,680行, 2020-2025)
│   ├── netkeiba_training_times.csv # 調教タイム(2,552行, 2025部分)
│   ├── netkeiba_stable_comments.csv# 厩舎コメント(857行, 2025部分)
│   ├── netkeiba_race_review.csv    # レース短評/備考(277,467行, 2020-2025)
│   ├── netkeiba_shinba_eval.csv    # 新馬評価(7,998行, 2024-2025)
│   ├── sire_shinba_stats.csv       # 種牡馬新馬成績(449種牡馬)
│   ├── netkeiba_siblings.csv       # 母産駒成績(17,441母馬)
│   ├── actual_roi_v135b.json       # v13.5b実配当ROI結果（v13.4比較）
│   ├── v135b_intra_ensemble_results.json # v13.5b学習結果
│   ├── v135_ft_transformer_results.json  # v13.5 FT学習結果
│   ├── v12_training_results.json   # V12学習結果
│   ├── actual_roi_results.json     # v12実配当ROI結果
│   ├── monte_carlo_results.json    # MC結果
│   ├── final_validation_report.json# 最終検証レポート
│   └── ... (検証結果JSON 16ファイル)
│
├── logs/                           # === ログ出力 ===
│
└── archive/                        # === アーカイブ ===
    └── nar/                        # 地方(NAR)関連一式
```

---

## 実戦前チェックリスト（毎週土曜朝に実行）

開催日の朝、予測を始める前に必ず以下を実行すること。

### チェック項目

1. **モデルファイル（pkl.gz）の読み込み確認** — `keiba_model_v9_central_live.pkl.gz` / `keiba_model_v9_central.pkl.gz` が正常にロードできるか
2. **feature_lookups.pklの存在・サイズ確認** — `data/feature_lookups.pkl` または `.pkl.gz` が存在し、キー数10以上あるか
3. **netkeibaへのアクセス確認** — 出馬表ページ・オッズAPIからレスポンスが返るか
4. **JRA馬場データのアクセス・エンコーディング確認** — `scrape_jra_track.py` がShift_JISで正しくパースできるか
5. **気象庁APIの確認** — `scrape_weather.py` が気温・湿度・風速を返すか
6. **DBの存在・レコード数確認** — SQLite DBが存在し、過去予測レコードが読めるか
7. **今日の最初のレースURLで特徴量テスト** — 87/87特徴量が全て生成されるか
8. **ゼロ特徴量が5個以上なら警告** — 0値の特徴量が多い場合は原因調査

### チェック基準

| 結果 | 判断 |
|------|------|
| 全項目OK | **実戦開始** |
| 警告あり | 影響確認してから判断 |
| エラーあり | **修正するまで実戦禁止** |

### よくあるバグと対処法

| 症状 | 原因・対処 |
|------|-----------|
| 特徴量18/87 | app.pyのバージョン条件（`use_version in ('v5','v6','v8','v9')` 等）にモデルバージョンが含まれているか確認 |
| cushion_value=0 | `scrape_jra_track.py`のencodingが`shift_jis`になっているか確認（JRA公式はShift_JIS） |
| モデル読み込み失敗 | `pkl.gz`形式対応を確認（`gzip.open` + `pickle.load`） |
| netkeiba 403 | `User-Agent`ヘッダーが設定されているか確認 |
| オッズ全て0 | `fetch_realtime_odds()`のJSON APIエンドポイント確認。レース前はオッズ未発売の場合あり |
| RaceName=Unknown | `soup.find(class_="RaceName")`でタグ種別を限定せず検索しているか確認（`<h1>`の場合あり） |

---

## 毎回の起動手順（コピペ用）

### 土曜朝（実戦前）
```bash
cd keiba-ai
git pull
python tools/pre_race_check.py
python -m streamlit run app.py
```

### エラーが出た場合
```bash
claude --dangerously-skip-permissions
```
→「pre_race_checkでエラーが出た。修正して」

### Claude Code起動（開発作業）
```bash
cd keiba-ai
claude --dangerously-skip-permissions
```

---

## 12. コマンド集

### 起動
```bash
streamlit run app.py                       # Streamlitローカル起動
```

### 予測・結果
```bash
python predict_and_log.py "URL"            # CLI手動予測
python check_results.py                    # 結果照合
python check_results.py --summary          # 成績サマリー
python verify_real_roi.py                  # 実配当ROI検証
```

### 自動運用
```bash
python tools/daily_predict.py              # 当日全レース予測
python tools/daily_predict.py --date 20260315  # 指定日予測
python tools/daily_results.py              # 当日結果照合
python tools/daily_results.py --date 20260315  # 指定日結果
python tools/weekly_report.py              # 週次レポート
```

### モデル学習
```bash
python train/train_v135b_intra_ensemble.py # v13.5b 4-model ensemble WFバックテスト+学習（現行）
python train/train_v135_ft_transformer.py  # v13.5 FT-Transformer学習
python train/train_v12_comprehensive.py    # V12 WFバックテスト+学習（旧版）
python calc_actual_roi_v135b.py            # v13.5b 実配当ROI検証（JRA公式配当）
```

### バックテスト・検証
```bash
python backtest_central_leakfree.py        # WFバックテスト
python calc_actual_roi.py                  # JRA配当ROI計算
python monte_carlo_sim.py                  # 破産確率シミュレーション
python monte_carlo_sim.py --trials 50000   # 試行回数指定
python project_status.py                   # プロジェクトステータス
python project_status.py --section model   # モデル情報のみ
python project_status.py --export          # JSON出力
```

### テスト
```bash
python tests/test_features.py              # 5項目自動テスト
python tests/debug_all.py                  # 25項目デバッグテスト
python -c "import py_compile; py_compile.compile('app.py', doraise=True)"  # 構文チェック（必須）
```

### データ取得
```bash
python tools/extract_jvdata.py             # TARGET JV → CSV抽出
python scrape_jra_payouts.py               # JRA公式配当データ
python scrape_jra_track.py                 # JRA馬場情報
python scrape_weather.py                   # 気象庁天候データ
python tools/refresh_cookie.py             # Cookie自動更新（対話式）
python tools/refresh_cookie.py --check     # Cookie有効性チェック
python tools/refresh_cookie.py --auto      # 期限切れ時のみ自動更新
```

### 検証スイート（13項目）
```bash
python tools/validation_1_standardization_leak.py   # リーク検証
python tools/validation_2_target_variable.py        # 目的変数比較
python tools/validation_3_ev_filter.py              # EVフィルタ
python tools/validation_4_ticket_optimization.py    # 券種最適化
python tools/validation_5_odds_gap.py               # オッズギャップ
python tools/validation_6_drawdown.py               # ドローダウン
python tools/validation_7_yearly_performance.py     # 年別パフォーマンス
python tools/validation_8_data_augmentation.py      # データ拡張
python tools/validation_9_final_report.py           # 最終レポート統合
python tools/validation_10_market_dependency.py     # 市場依存性
python tools/validation_11_sample_size.py           # サンプルサイズ
python tools/validation_12_roi_integrity.py         # ROI整合性
python tools/validation_13_conservative_roi.py      # 保守的ROI
```

---

## 現行モデルのベースライン（これを下回る変更は一切採用しない）

- **WF AUC: 0.8788**（walk-forward 2020-2025平均, 4-model grid ensemble, v13.5b）
- **年別WF AUC**: 2020=0.8515, 2021=0.8806, 2022=0.8830, 2023=0.8845, 2024=0.8853, 2025=0.8851
- **実配当ROI**: A=355%(trio), B=347%(trio), C=623%(trio), D=361%(trio), E=196%(umaren), X=701%(trio) — **全体428.4%**
- **全条件ROI 100%超え**（全条件v13.4以上を確認済み）
- **旧ベースライン（v13.4 LGB+XGB）**: WF AUC 0.8656, 実配当ROI 361.9%
- **旧ベースライン（V12 LGB単体）**: WF AUC 0.8037, 実配当ROI ~205%

## 重要ルール

1. **学習はPattern A、予測はPattern B**: バックテスト評価は常にPattern A。実運用予測はPattern B
2. バックテストは必ず**ウォークフォワード**（時系列分割）で実施
3. **改善が確認できない変更は採用しない**: WF AUC > 0.8788 かつ全年AUC > 0.85 かつ 実ROI全条件v13.5b以上
4. app.pyを変更したら必ず**python構文チェック**してからcommit
5. 大きなデータファイル(.csv)は.gitignoreで除外、ローカル保持
6. モデル更新時はAUCが既存モデルを上回る場合のみ本番反映
7. 買い目の馬番は昇順ソート・カンマスペース区切りで表示
8. **中央競馬専用** — 地方(NAR)コードはarchive/nar/に保管

## 定期タスク（Windows タスクスケジューラ）

| 時間 | タスク | コマンド | バッチ |
|------|--------|---------|--------|
| 毎日 03:00 | プレミアムデータ事前取得 | `python tools/daily_premium_scrape.py` | `daily_premium_scrape.bat` |
| 毎日 08:00 | 当日全レース予測 | `python tools/daily_predict.py` | `daily_predict.bat` |
| 土日 08:45 | レース5分前自動予測＆Discord通知 | `python tools/race_auto_notify.py` | `race_auto_notify.bat` |
| 土日 18:00 | 結果照合・ROI計算 | `python tools/daily_results.py` | `daily_results.bat` |
| 毎晩 20:00 | 結果照合（平日含む） | `python tools/daily_results.py` | `daily_results.bat` |
| 月曜 08:00 | 週次レポート | `python tools/weekly_report.py` | `weekly_report.bat` |

一括登録: `setup_all_tasks.bat`（管理者権限で実行）
ログ: `logs/` ディレクトリ

---

## netkeibaプレミアムデータ連携

### 概要
netkeibaスーパープレミアム会員のCookie認証でプレミアムデータを取得。
Cookie設定: `.env` の `NETKEIBA_COOKIE` に保存。
Cookie期限切れ時は `python tools/refresh_cookie.py` で自動更新可能（Playwright）。
認証情報を保存済みなら `python tools/refresh_cookie.py --auto` で期限切れ時のみ自動実行。

### 取得データ一覧

| データ | ソース | 取得タイミング | 用途 |
|--------|--------|------------|------|
| 調教タイム(4F/3F/1F) | `oikiri.html` | 予測時リアルタイム | モデル特徴量（wood_best_4f_filled等） |
| 追い切りランク(A/B/C/D) | `oikiri.html` | 予測時リアルタイム | タイム取得失敗時のフォールバック |
| タイム指数 | `speed.html` | 予測時リアルタイム + 事前取得 | UI表示（モデル組込は再学習後） |
| 厩舎コメント | `comment.html` | 予測時リアルタイム | スコア化(-3〜+3)してUI表示 |

### 調教タイム取得の4段階フォールバック
1. Premium実タイム (Cookie有効 → oikiri.html 4F/3F/1F秒数)
2. ランク推定 (Cookie無効 → A:51.5s, B:53.0s, C:54.5s, D:55.5s)
3. feature_lookups.pkl (キャッシュ値)
4. デフォルト値 (52.0/53.0/39.0s)

### 関連ファイル

| ファイル | 用途 |
|----------|------|
| `scrape_training.py` | 調教タイム・コメント取得モジュール |
| `tools/scrape_speed_index.py` | タイム指数一括取得 |
| `tools/scrape_premium_data.py` | 調教タイム一括取得 |
| `tools/bulk_scrape_history.py` | 過去データ一括取得（手動実行） |
| `tools/daily_premium_scrape.py` | 週末レースデータ事前取得（AM3:00自動） |
| `tools/weekly_premium_update.py` | 週末Premium更新 |

### 蓄積データ状況

| データ | ファイル | 行数 | 年度 |
|--------|---------|------|------|
| タイム指数 | `data/netkeiba_speed_index.csv` | ~143K | 2020-2025 |
| 調教タイム | `data/netkeiba_training_times.csv` | ~2.6K | 2025（部分） |
| 厩舎コメント | `data/netkeiba_stable_comments.csv` | ~857 | 2025（部分） |
| レース短評 | `data/netkeiba_race_review.csv` | ~277K | 2020-2025（全年カバー） |
| 新馬評価 | `data/netkeiba_shinba_eval.csv` | ~8K | 2024-2025 |

---

## Discord通知システム

### チャンネル振り分け

| チャンネル | 環境変数 | 内容 |
|-----------|---------|------|
| #買い目 | `DISCORD_WEBHOOK_BETS` | レース予測、フォーメーション、配当レンジ |
| #アップデート | `DISCORD_WEBHOOK_UPDATES` | スクレイピング完了、結果照合、週次レポート |
| フォールバック | `DISCORD_WEBHOOK_URL` | 上記未設定時 |

### 通知形式

**買い目通知（三連複）**:
```
🏇 中山11R 発走15:45
アネモネS 芝1600m 良 条件A ★★★

三連複フォーメーション 7点
1列目: 1
2列目: 2, 3
3列目: 2, 3, 4, 5, 6

軸: ホワイトオーキッド (1) スコア0.85
💰 配当レンジ: 1,200円〜15,600円
📊 指数: 1127 / 調教: A / 厩舎: 好調
```

### セットアップ
```bash
python tools/setup_discord.py  # 対話式Webhookセットアップ
```

---

## V12特徴量追加結果（2026-03-29）

v12総合再学習で10特徴量を同時投入テスト。7個採用、3個不採用。

| 特徴量 | ソース | WF寄与 | 判定 |
|--------|--------|--------|------|
| `index_avg5_filled` | speed.html 5走平均指数 | +0.00032 | ✓ 採用 |
| `time_1f_last_filled` | oikiri.html ラスト1F | +0.00025 | ✓ 採用 |
| `index_max_filled` | speed.html 最高指数 | +0.00019 | ✓ 採用 |
| `index_run1_filled` | speed.html 前走指数 | +0.00019 | ✓ 採用 |
| `sire_shinba_top3r` | 既存CSV(expanding) | 0.00000 | ✓ 採用(害なし) |
| `pci` | ラップデータ(後半/前半3F) | 0.00000 | ✓ 採用(害なし) |
| `training_intensity_enc` | oikiri.html 調教強度 | -0.00048 | ✓ 採用(閾値内) |
| `dam_top3r` | netkeiba_siblings.csv | -0.00063 | ✗ **リーク修正後マイナス** |
| `stable_comment_score` | comment.html 厩舎スコア | N/A | ✗ WFカバレッジ30%不足 |
| `prev_review_score` | db.netkeiba 備考 | N/A | ✗ 2024-2025のみでWF不可 |

### v12.1再学習テスト（2026-03-29、不採用）

race_review 2020-2025全年データ取得完了後(264,973行)、2特徴量を追加テスト。

| 特徴量 | 個別寄与 | v12.1(両方) | 判定 |
|--------|---------|------------|------|
| `prev_review_score` | +0.00016 | — | 微プラスだが採用基準未達 |
| `shinba_eval_score` | +0.00007 | — | ほぼゼロ |
| 両方合算 | — | AUC 0.8039 (+0.00034) | **不採用: 2021年gap=0.0514>0.05** |

年別結果:
| 年 | v12 | v12.1 | gap |
|----|-----|-------|-----|
| 2020 | 0.7934 | 0.7934 | 0.0438 |
| 2021 | 0.8004 | 0.8014 | **0.0514** ✗ |
| 2022 | 0.8061 | 0.8055 | 0.0295 |
| 2023 | 0.8038 | 0.8046 | 0.0394 |
| 2024 | 0.8103 | 0.8100 | 0.0277 |
| 2025 | 0.8073 | 0.8084 | 0.0264 |

不採用理由: AUC改善は微小(+0.00034)、2021年で過学習閾値超過。v12(74特徴量)を維持。

### v13.5b 正式採用（2026-04-03）— 4-model Grid Ensemble

v13.4 (LGB+XGB) → v13.5b (LGB+XGB+FT-Transformer+IntraRace Attention) への大規模アップグレード。
124特徴量（JRDB連携含む）、4モデルGrid Ensemble、WF AUC 0.8788 (+0.0131 vs v13.4)。

**実配当ROI検証（JRA公式配当、WF 2023-2025、10,314レース）:**

| 条件 | N | v13.4 ROI | v13.5b ROI | 差分 | 判定 |
|:---:|---:|---:|---:|---:|:---:|
| A | 3,212 | 308.0% | **355.4%** | +47.4% | OK |
| B | 398 | 259.4% | **346.8%** | +87.4% | OK |
| C | 2,473 | 521.5% | **623.0%** | +101.5% | OK |
| D | 3,581 | 314.7% | **360.8%** | +46.1% | OK |
| E | 267 | 141.4% | **195.7%** | +54.3% | OK |
| X | 383 | 498.9% | **701.2%** | +202.3% | OK |
| **全体** | **10,314** | **361.9%** | **428.4%** | **+66.5%** | **ALL PASS** |

**年×条件 ROI安定性（v13.5b）:**
- 2023: A=426%, B=308%, C=712%, D=380%, E=175%, X=535%
- 2024: A=326%, B=388%, C=563%, D=322%, E=145%, X=933%
- 2025: A=317%, B=377%, C=598%, D=380%, E=117%, X=699%

**Grid重み（年ごと最適化）:**
- 典型: LGB=0.25, XGB=0.25-0.30, FT=0.10-0.15, IR=0.35
- IntraRace Attentionが最大貢献（レース内相対関係を捕捉）

### predict_core.pyバグ修正（2026-03-29）

| バグ | 詳細 | 修正 |
|------|------|------|
| Speed index全滅 | build_features()がCSVのみ参照、premium cacheの実データ未使用 | premium cache JSONフォールバック追加 |
| 距離bin不一致 | 学習: pd.cut 5bin(0-4), 予測: 4bin(0-3) | 5binに統一 |

### 今後の追加データ候補（v13用）
| データ | ファイル | 行数 | 用途 |
|--------|---------|------|------|
| レース短評(備考) | netkeiba_race_review.csv | 277,467 | 前走不利→巻き返し検出。v12.1で不採用(gap超過) |
| 新馬評価 | netkeiba_shinba_eval.csv | 7,998 | 新馬戦の厩舎評価・調教ランク。表示用 |
| 種牡馬新馬成績(静的) | sire_shinba_stats.csv | 449 | 新馬戦UIバッジ表示用(モデルはexpanding版使用) |
| 母産駒成績(静的) | netkeiba_siblings.csv | 17,441 | 新馬戦UIバッジ表示用(モデル組込はリーク注意) |
| 厩舎コメント | netkeiba_stable_comments.csv | ~98K(取得中→目標60%+) | カバレッジ不足で不採用。追加取得中 |

---

## Phase 2-5 検証結果サマリー（2026-03-23〜26）

| Phase | 内容 | 主要結果 |
|-------|------|---------|
| 2 | キャリブレーション・EV・ランカー | 全不採用。推定ROI式の膨張(~16x)を発見 |
| 2b | ROI信頼性検証（7タスク） | リークなし、過学習なし、ランダムの10.4倍 |
| 3 | **実配当ROI検証** | **Trio 225.8%** [CI: 198.5-264.6%] P(>100%)=100% |
| 4 | OOS・ライブ検証 | 2025準OOS: 246.3% [201-301%] 判定VALID |
| 5 | 市場耐性・資金管理 | 耐性HIGH、調教が最重要(-88%)、破産0.16% |

---

## 実戦成績（2026-03-14〜04-18, dedup後 324レース）

| 条件 | N | 的中 | 的中率 | ROI | 保守的見積り |
|------|---|------|--------|-----|-------------|
| A | 90 | 30 | 33.3% | 122.9% | 143.7% |
| B | 9 | 0 | 0.0% | 0.0% | 165.8% |
| C | 89 | 16 | 18.0% | 123.8% | 199.9% |
| D | 115 | 28 | 24.4% | 144.3% | 95.2% |
| E | 9 | 1 | 11.1% | 13.2% | 82.6% |
| X | 12 | 1 | 8.3% | 13.8% | 231.3% |
| **全体** | **324** | **76** | **23.5%** | **120.2%** (**+45,920円**) | 142.6% |

- 全体ROI 120.2% — 保守的見積り142.6%には届かないが +45,920円のプラス運用
- 条件D 144.3% は保守的見積り95.2%を大きく上回る（好調）
- 条件B/E/X は N が小さく統計的に不十分（継続監視）
- JRDB結合率: KYI(PRE_RACE) 75.9%, TYB(LIVE) 0%(正常, 当日朝発表), SED(PREV) 0%(SED csv破損-要修正)

### SCRAPER-GUARD の動作変更（2026-04-19）

DailyPremiumScrape AM3:00 は金22時〜月6時の SCRAPER-GUARD で停止する。
旧仕様: 600秒おきチェックで許可まで wait ループ → タスクスケジューラと相性悪く数日停止
新仕様: `check_scraping_allowed(mode="exit")` で即終了 → 翌日の起動で正常再開

OPERATIONAL_CALLERS ホワイトリスト導入 (daily_predict / race_auto_notify /
notify_bets_all_in_one / jrdb_health_check / daily_jrdb_kyi / daily_results)。
daily_premium_scrape は Sat/Sun/**Mon** の 03:00-05:59 早朝スロット特例で許可。

### 2026/04/19 事故と修正の記録

**事故**: Sun 03:00 DailyPremiumScrape と 08:00 DailyPredict が SCRAPER-GUARD で誤停止。
AM8:27 手動救出まで午前レース全ロス。機会損失 推定 +2,745円 (17R)。

**対応**: 11 commits で完全修正完了 (e173f40d〜本commit)
- OPERATIONAL_CALLERS ホワイトリスト導入
- Mon 早朝特例追加 (4/13 Mon 03:00 にも同じ誤停止履歴あり)
- daily_premium_scrape の mode="exit" 化 (wait ループ廃止)
- process_watchdog v2 (ログ鮮度ベース)
- daily_predict Windows Ctrl+C 対策 + resume 対応
- 事前検証: verify_scraper_guard_sunday.py / dryrun_weekend_full.py / nightly_sanity_check.py

### 来週末の運用体制

- **v15 継続運用** (v16 はデータ不足で未学習 — master_index 2020-2022 が 0%)
- **手動介入不要** (E2E検証 17タスク ALL PASS)
- 毎晩 23:00 Keiba-NightlySanity が自動で翌日タスクを事前チェック → Discord通知

### 新規タスクスケジューラ登録（2026-04-18〜19）

| タスク名 | 内容 |
|----------|------|
| DailyJrdbKyi | AM6:00 JRDB全種別ダウンロード（Windows 11 24H2対応、wmic→PowerShell置換済） |
| JrdbHealthCheck_Sat/Sun | AM7:30 JRDB取得健全性チェック |
| ProcessWatchdog | 5分おき プロセス死活監視・自動再起動 |
| **Keiba-NightlySanity** | **毎日23:00 翌日発火予定タスクの事前チェック + Discord通知** |

---

## タスク完了通知

コマンドラインからDiscord通知を送信:
```bash
python tools/notify_done.py "タスク名" "詳細メッセージ"
python tools/notify_done.py "エラー" "内容" --color red
```

## プロンプト標準テンプレート

今後の全タスクプロンプトの末尾に以下を含めること:
```
git commit & push して。
完了したら以下を実行：
powershell -Command "[System.Media.SystemSounds]::Exclamation.Play()"
python tools/notify_done.py "タスク名" "完了内容"
```

## Compaction対応

Claude Codeのコンテキスト圧縮時に失われやすい重要情報はこのCLAUDE.mdに集約。
特に「現行モデルのベースライン」「リーク厳禁ルール」「過去の失敗教訓」は常に参照すること。

| `prev_race_pace_diff` | 0%充填 | **削除** (JRDB SED は15.2%しかカバーできず修復困難) |
| `gaisha_rank` | 0%充填 | **削除** (`jrdb_ranch_rank` 94.2%充填で代替済み) |
| `course_renovated` | 1.3%充填 | **永久化適用済 (4/27)** (京都2万件を活性化) |
| `jrdb_tb_homestr_inner` | 99.7%充填 | **健全** (修復不要) |

### 🛠 戦略⑦ 自動化 (4/27 適用済)
**実装場所**: `tools/race_auto_notify.py` (`predict_and_notify` 関数内)

**フィルタ内容**:
- `06_特別` (G/L/OPEN特別 ではない平場特別) を除外: -9,470円損失源
- `京都` を除外: データ蓄積待ち、5/11 以降に再評価
- `条件E` (頭数<=7) を除外: サンプル少
- `条件B` (重~不馬場) を除外: サンプル少

**期待効果**: ROI 119.2% → 140.3% (+21.1pt)
**シミュレーション**: 298R → 242R, 損益 +28,240円改善

### 🎯 1レース再予測ツール (4/26 動作確認)
**ファイル**: `tools/predict_one_race.py`
**用途**: 取消発生時、1レース指定で予測再実行
**動作確認**: 4/26 東京11Rフローラ Sで成功

```bash
python tools/predict_one_race.py 202605020211
```

### 📋 v16 ロードマップ
- **Week 1 (4/27-5/3)**: 基盤修正 + 戦略⑦運用開始
- **Week 2 (5/4-5/10)**: 死特徴量整理 + 訓練データ拡張
- **Week 3 (5/11-5/17)**: v16 学習実行 (Ryzen 7 + 32GB + 16GB GPU で2-3時間)
- **Week 4 (5/18-5/24)**: A/B検証 → 5/末 v16 本番投入

### 🎯 v16 目標
- **AUC**: 0.895+ (v15: 0.8939)
- **特徴量数**: 148 (-2)
- **ROI** (戦略⑦込み): 140%+
- **京都ROI**: 80%+ (course_renovated 永久化効果)
- **本番切替**: 5月末

### 🔍 既知のバグ
- jrdb_paci.csv が4/4から更新停止 (取得経路不明、要修復) → **JV-Link O1 で代替経路 (Session #39 B)**
- predict_core.py に FutureWarning が15箇所以上 (4/27 主要箇所修正済)
- cumulative_results.csv に top1_num/score 書き込まれていない (95%欠損)
- nightly_sanity の SCRAPER-GUARD 認識バグ (誤検知)
- jra_payouts.csv が4/6で更新停止 → **JV-Link HR で代替経路 (Session #39 B)**
- JRDB データの2026年分が未取得 (race_id 列なしの可能性)

---

## Phase 3-4 統合 roadmap (Session #39 J、 5/24-8月)

### Phase 3 前半 (5/24-6/8): 基盤整備 + sib_*_exp 統合

| 期間 | 内容 |
|------|------|
| 5/24 | JRA-VAN DataLab 加入、 JV-Link DLL インストール、 jvlink_fetcher.py 動作確認 |
| 5/25-5/27 | sib_expanding_features.py を train/features_v15_new.py に統合 (旧 sib_top3_rate / sib_shinba_wr 入れ替え) |
| 5/28-5/30 | V18/V19 sib_*_exp 版 6-fold WF (LGB+XGB) |
| 6/1-6/5 | V18/V19 LIVE retro (5/30 + 5/31 + 6/1)、 winner_top1 ≥ 30% / shift ≤ 12x 検証 |
| 6/6-6/8 | sib_*_exp GO/no-go 判定。 GO なら 6/15+ V18/V19 段階投入 (週末のみ、 上限 5,000円/日) |

### Phase 3 後半 (6/9-6/30): V20 構築

| 期間 | 内容 |
|------|------|
| 6/9-6/13 | JV-Link parser 実装 (RACE/HR/O1/TCOV/WOOD/BLOD)、 過去 1 年 bulk fetch + 整合チェック |
| 6/14-6/20 | V20 学習 data spec 確定 (JRA + NAR 統合、 共通 80 features、 SKB 完全除外、 sib_*_exp 込み)、 V20 v1 学習 (4-model ensemble) |
| 6/21-6/25 | V20 WF 検証 (6-fold、 2020-2025)、 LIVE retro |
| 6/26-6/28 | V20 paper trading (週末) |
| 6/29-6/30 | GO/no-go 最終判定 (WF AUC ≥ 0.880 / LIVE retro ≥ 30% / shift ≤ 12x / NAR AUC ≥ 0.83 / paper ROI ≥ 110% / LEAK 監査 PASS) |

### Phase 3 投入 (7/1+): V20 production

- 7/1: V20 段階投入 (週末のみ、 上限 5,000円/日)
- 7/15: 順調なら投資額 増額 (週末 1万円/日 + 平日 5,000円/日)
- 8/1: V15 archive 判定 (1 か月並行運用後)

### Phase 4 (7-8 月): 動画解析 PoC

| 期間 | 内容 |
|------|------|
| 7/1-7/14 | データ蓄積 (JRA-VAN ネクスト + netkeiba 動画、 50 レース 1,500 動画) |
| 7/15-7/31 | YOLOv8 馬体検出 + DLC SuperAnimal 姿勢推定 動作確認 (zero-shot) |
| 8/1-8/15 | 時系列特徴量抽出 (stride / gait_symmetry / head_bobbing / ear_pos / posture / 5 件)、 fine-tune 必要なら DLC HORSE-10 ベース |
| 8/16-8/31 | V21 学習 (V20 + VIDEO_FEATURES) + WF 検証 |
| 9/1 | V21 投入判定 (WF AUC ≥ V20 + 0.005 / LIVE retro winner_top1 ≥ V20 + 1pt) |

### 投資保護 (絶対遵守)

- **5/9 V15 案B改 単独継続** (Session #38 NO-GO 確定後の唯一 path)
- **撤退ライン**: 累計 -50,000円 (現在 +13,530円、 撤退余裕 +63,530円)
- **取り返し禁止** (損切り後 翌日へ持ち越さない)
- **Phase 3-4 着手中も V15 production 完全不変保証**

### 月額コスト + ROI 試算

| source | 月額 | 開始 |
|--------|------|------|
| netkeiba Premium | 4,500円 | 既存 |
| JRDB Advance | 約 2,000円 | 既存 |
| JV-Link (DataLab) | 2,090円 | 5/24 |
| JRA-VAN ネクスト (Phase 4 用) | +1,000円 | 7/1 |
| Colab Pro (Phase 4 GPU) | 1,178円 | 7/1 |
| **合計** | **約 10,768円/月** (7/1 以降) | — |

ROI 想定:
- V15 (現状): 119.2% (戦略⑦込み 140%) → 月利 約 2-3 万円
- V20 (7/1+): WF AUC 0.880-0.890 / 戦略⑦込み 145-150% 想定 → 月利 5-10 万円
- V21 (9/1+): V20 + 動画 features 中位想定で 145-150% → 月利 6-11 万円

→ 月額コスト 約 1万円は V20 以降の月利増分で十分回収。

### 📁 重要ファイル
- 1レース予測: `tools/predict_one_race.py`
- 戦略⑦適用: `tools/race_auto_notify.py` (4/27 修正)
- course_renovated: `tools/predict_core.py` 2061-2073 (4/27 修正)
- v16訓練ベース: `train/train_v15_master.py`, `train/retrain_v16.py`
- 訓練データ: `data/_v15_optuna_df_cache.pkl.gz` (104MB)
- TODO: `V16_TODO_NEXT_WEEK.md`

### 💾 バックアップ
- 4/27 修正前のバックアップ: `*.bak_20260427`
  - tools/race_auto_notify.py
  - tools/predict_core.py
  - train/features_v15_new.py
  - data/cumulative_results.csv
  - CLAUDE.md
