# keiba-ai システム アップデート棚卸し総合レポート

**作成**: 2026-05-05 深夜 (Opus xhigh、6 領域並列調査)
**対象期間**: 5/5 PM 〜 Phase 3 移行 (5/24+)
**ベース commit**: `f408d93d` (5/5 柏記念予測)
**ユーザー絶対方針**: 取り返し禁止 / 累計 +14,140円 死守 / 撤退ライン -50,000円

---

## 0. エグゼクティブサマリー (朝1分で読む)

### 5/9 (土) 投資セッション まで「だけ」やる必要があること
| # | やること | 工数 | 備考 |
|---|----------|------|------|
| 1 | **5/8 (金) 21:00 後 1度 race_name 確認** | 5min | `data/results/20260509_pre_check.md` の python snippet 1 個実行 |
| 2 | **Cookie 有効性 check** (`refresh_cookie.py --check`) | 1min | .env mtime 5/3 = 直近更新済、念のため |
| 3 | **5/9 朝 `data/results/20260509_pat_checklist.md` を順番通り** | (人手) | 06:30/08:00 自動発火後、checklist だけ追従 |
| 4 | **5/9 20:30 `post_5_9_improvement_template.md` 埋め** | 30min | 5/16 戦略決定の input |

→ **これ以外は 5/10 以降 でよい**。 V15 model 関連は 5/9 で何も触らない (現状維持で投資完遂)。

> **5/6 Session #27 訂正**: 累計 +14,140 円 (USER 申告) は生データで再検証すると **+13,530 円** (`data/v18/may_2_3_truth_audit_5_6.md`)。 ±610 円差は要 USER 確認 (5/4 月曜の何か)。 5/9 投資判断 (撤退余裕 +63,530 円) は変化なし。

### 🔴緊急 (5/9 当日リスク) - **3 件 → 全対応済 (5/5 夜 Session #24)**
1. **`ProcessWatchdog v2` schtasks 未登録** ✅ → ps1 + 手順書作成完了 (`tools/register_process_watchdog_v2.ps1` + `data/v18/process_watchdog_v2_setup.md`)。 admin 権限での実行のみユーザー手動。
2. **fire_check 系 caller 引数監査** ✅ → 4 種すべて健全と確認 (`pre_fire_check.py` L47 で `caller="daily_premium_scrape"` 渡し済)。 dry-run で発見した cp932 バグ + am8 平日誤判定 を修正済 (`data/v18/fire_check_audit_5_5.md`)。
3. **`chihou_races_2020_2025.csv` 不在** ❌→🟢低 → **誤情報と判明**。 `archive/nar/train_nar_v4.py` 解析で `nar_all_races.csv` (54,160 行) のみ実使用、`chihou_races_2020_2025.csv` は変数定義のみで未読込。 5/12 NAR paper 開始の blocker ではない (`data/v18/chihou_races_recovery_5_5.md`)。

### 🟠高 (5/12 - 5/16 までに) - **15 件**
- V18/V19 race-level normalize 本番統合 (`predict_core.py`、30min)
- V18/V19 feature distribution shift 調査 (90min) ← winner_top1 13pt 劣化の核心
- NAR データ stale 修復 + `--date` 引数追加 (5/12 paper 必須前提)
- `tools/scrape_nar_today.py` / `scrape_nar_results.py` 手動 dry-run (本実装直後の動作検証)
- 戦略⑦ 5/2-5/3 retro 完全版 (5/9 投入直前必須)
- `cumulative_results.csv` 書き込みバグ修正 (top1_num/score 95% 欠損)
- 累計 PnL 自動計算スクリプト + 撤退ライン Discord アラート
- TYB 観測完了判定 (5/11)
- SED260503 取得 + KKA/KAB 連結再実行 (5/9 朝の前走成績結合率に直結)
- 平日 race_auto_notify watchdog fatal alert 誤報 抑止
- DailyPredict 5/4 LastResult=1 原因調査
- `Keiba-WeeklyScrapeResume` LastResult=3221225786 原因調査
- 静音化 4 task 残存 (`JrdbHealthCheck_Sat/Sun`, `DailyResults_Sun`, `Keiba-ScrapeProgress`, `Keiba-WeeklyScrapeResume`)
- 5/9 結果集計のため `jra_payouts.csv` 自動取得復旧
- 5/15 夜 v18/v19 GO/no-go 判定 (条件 5 件のうち #1/#5 達成度で決定)

### 🟡中 (5/24 Phase 3 移行までに) - **30+ 件**
- V15.1 4-model ensemble 学習 + 全年 WF 検証 + leak audit
- V15.1 軸 top3 率 retro WF 検証 (AUC ≠ TOP1精度問題)
- V15.1 production pipeline (`predict_core.py`) 統合
- V20 統合モデル設計実装 (chihou data 必須)
- `data/v18/index.md` / `data/results/index.md` 新設 (5/9 朝の操作迷子防止)
- v15 + 戦略⑦ Monte Carlo 再シミュレーション
- TARGET Frontier JV 再インストール + feature_lookups 再生成
- validation_1〜13 を v15 対応 (リーク監査特に重要)
- `calc_actual_roi.py` を v15 対応 + 戦略⑦ filter 統合
- drift detection の year_week バグ修正 + baseline_auc 0.8939 更新
- 古い doc 一括 archive (~55 ファイル / ~3MB)

### 🟢低 - **15+ 件** (バックアップ削除 / dead code 整理 / archive 整備)

---

## 1. 領域A: モデル関連

### 1.1 V15 (本番、AUC 0.8939、150 features)
- **5/9 朝 動作確認のみ**: 工数 5min、Cookie + V15 load + 12R race_name
- 軸 top3 率 BT 57.0% → 本番 40.3% gap (-16pt) は **継続観察**、構造的解決は V15.1 SKB が候補
- `keiba_model_v15_central_live.pkl.gz.bak_v16_20260427` (4.2MB) 削除候補
- 5/9 20:30 `post_5_9_improvement_template.md` 埋め必須

### 1.2 V15.1 (試作、5/24 Phase 3 候補)
- **commit 7c5ba9f8** で SKB +0.0694 大発見 (LGB single retro)
- KKA 16f は寄与 0.0000 → race_id 変換失敗の疑い、要原因究明
- WF 検証は **未完了** (LGB single time-based split のみ)
- 5/24 Phase 3 移行 6 条件達成判定の core item:
  1. 4-model ensemble (LGB+XGB+FT+IR) で SKB 寄与 再検証 (4-6h)
  2. 軸 top3 率 retro WF (2h、AUC ≠ TOP1精度の検証)
  3. `tools/predict_core.py` に SKB merge 統合 (1h)
  4. DailyJrdbKyi で SKB daily 取得確実化 (1h)
  5. 5/16-5/24 paper trading (実投資なし、Discord 通知のみ)
- Pattern A (リークフリー) + Pattern B 両方 学習 (2h)

### 1.3 V17 (morning ULTRA-CLEAN)
- CRLF 復旧済 (commit 777cc08e)
- TYB 観測継続中、**5/11 月 結果判定**
- TYB 結果次第で復活 or 廃止 (廃止なら 192MB 解放)

### 1.4 V18/V19 (BT 295%/149%)
- 5/2-5/3 retro: 全 filter で **bet=0** (probability 過小)
- race-level normalize で bet>0 化したが winner_top1 rate 34.5% は monotonic 変換で **不変**
- distribution shift factor BT/retro 27.69x → **RANK_SHIFT** = 1着馬選定自体が劣化
- 5/16 試行は前提条件 5 件のうち **#1 (predict_core 統合) ⏳ / #4 (winner_top1 ≥40%) ⚠️ / #5 (feature shift 調査) ❌** で**実質 no-go 寄り**
- 5/15 夜 GO/no-go 判定が累計死守の生命線

### 1.5 NAR v4 (AUC 0.8145/0.8519)
- データ 1年 stale (2024-02〜12 + 2025-06〜現在 月次穴)
- 5/12 paper 必須前提:
  - データ backfill (60min/月 × 約23ヶ月)
  - `tools/scrape_nar_all.py` に `--date YYYYMMDD` 引数追加 (30min)
- `chihou_races_2020_2025.csv` 不在で strict OOS 評価不能 (60min)
- NAR 配当データ調達 (条件別 ROI 計算用、60min)
- NAR v5 再学習 (2025-2026年 + JOCKEY_OVERRIDE 構造化、6月)

### 1.6 古いモデル
- archive 化済 (commit e23b5a88、`archive/old_models_20260505/` 86MB)
- `predict_core.py` L479-487 fallback chain は古いまま (V13.5b chain) → V15 only にリファクタ推奨 (30min)
- 本番影響なし (dead code)

---

## 2. 領域B: データ関連

### 2.1 JRA-VAN DataLab (TARGET frontier JV)
- **`C:\TFJV` 完全不在 (再インストール必要)** - 領域B 報告
- `jra_races_full.csv` 5/3 末尾まで復旧済 (commit b4c4894c、532,005 行 / 178MB)
- `feature_lookups.pkl` 3/27 fixed - 4ヶ月 stale
- **JRA-VAN (一度だけ契約、2025データ取得済、現在退会)**、5/24 まで不要、6 月学習タイミングで再契約候補 (代替不在)
- 必要時期: **5/16 v17 再学習** / **5/24 v16/v17 学習データ完備**

### 2.2 JRDB
- 23 種別が 5/2-5/3 まで raw 取得済
- **SED260503.zip 欠落** → 5/9 朝の前走成績結合率に直撃 (5min で再取得)
- `jrdb_paci.csv` 4/4 stale 問題は **解消済** (5/3 09:42 更新確認)
- KKA/KAB raw は 5/5 19:04 取得完了だが連結 CSV は 5/3 09:45 で停止 → 連結再実行 (10min)
- DailyJrdbKyi の 5/4-5/5 不発火疑い (10min logs 確認)

### 2.3 netkeiba premium
- Cookie size 1836 bytes (.env mtime 5/3 10:51) - 直近更新済
- `speed_index.csv` 4/29 09:52 で **2026年 16行のみ** → 4-5月 backfill 30min (5/9 朝、条件C/D 予測精度直撃)
- `race_review.csv` 3/29 stale → 5/16 V15.1 SKB 再投入のため 1h backfill 必要
- `shinba_eval.csv` 3/29 stale (UI バッジ用、低優先度)

### 2.4 コアデータ
- `jra_races_full.csv` 5/3 まで復旧済
- `odds_history.csv` / `training_times.csv` / `blood_full.csv` 全て 3/11 stale (TARGET 由来)
- 6月以降 取得計画 → 5/12 以降 weekly 月曜朝 半自動化

### 2.5 配当データ
- `jra_payouts.csv` 5/4 07:59 更新 / 5/3 末尾 → CLAUDE.md 「4/6 stale」記述は古い、訂正必要 (5min)
- 5/4-5/5 分の取得確認 + cron 自動化 (毎日 21:00 想定)
- 5/9 結果集計に必要

### 2.6 NAR データ - **🔴緊急**
- `chihou_races_2020_2025.csv` **不在** = 5/12 paper 唯一最大 blocker
- `chihou_races_full.csv` 末尾 2020-03-19 (6 年遅れ)
- スクレイプ工数 5-10h、**5/9-5/11 週末で完遂必要**

### 2.7 特徴量 cache
- `feature_lookups.pkl` 40MB / 3/27 fixed (TARGET 同根)
- `_v15_optuna_df_cache.pkl.gz` 104MB / 4/8 (v15 運用継続中は OK)
- v16 切替狙うなら 5/19 までに lookups 再生成

---

## 3. 領域C: インフラ・運用

### 3.1 タスクスケジューラ (32 task)
- 28 task が wscript + silent_runner.vbs 静音化済
- **`ProcessWatchdog` Disabled** (v1 → v2 移行中で停止) → **🔴緊急: v2 schtasks 登録**
- `Keiba-WeeklyScrapeResume` LastResult=3221225786 (Ctrl+C 強制終了)
- `DailyPredict` LastResult=1 (5/4 月曜エラー継続)
- `WeeklyReport` LastResult=1 (月曜 08:00 連続失敗)
- NAR placeholder 5 task は LastResult=267011 (未発火、正常)

### 3.2 Discord 通知
- 3 channel 振り分け済 (BETS / UPDATES / URL fallback)
- `notify.py` の Discord 障害時 silent fail 問題 → log 化 30min
- NAR 用 channel 分離未設計 (5/12 paper 開始時 JRA と混線リスク)

### 3.3 Watchdog 体制
- v1 (PID json) は監視対象 0、Disabled
- v2 (ログ mtime ベース) 実装済だが **未スケジュール** → 🔴緊急
- 火-金 fatal alert 誤報 (race_auto_notify 平日無効化必要、HANDOFF L389)
- daily_predict_watchdog (S4) は 5/5 朝動作確認済

### 3.4 静音化
- 4 task 未適用: `JrdbHealthCheck_Sat/Sun`, `DailyResults_Sun`, `Keiba-WeeklyScrapeResume`, `Keiba-ScrapeProgress` → 15min で完了

### 3.5 SCRAPER-GUARD
- pytest 16/16 PASS、4/19 wait ループ事故修正済 (mode="exit")
- **fire_check 系 (PreFire/AM3/AM6/AM8) の caller 引数渡し未確認** → 4/19 事故と同型リスク (30min 監査)
- `weekly_scrape_resume` も caller 未登録疑い

### 3.6 cron / schedule (Claude Code 側)
- `.claude/scheduled_tasks.lock` 取得済 (5/5 19:04)
- schedule skill 未活用、設計討議のみ

### 3.7 git 戦略
- `.gitignore` 112行 + `.gitattributes` (CRLF 抑止) 整備済 (commit e20bbc0c)
- `archive/` 4 サブディレクトリ整理済 (Session #18)
- `.bak_20260427` 残置確認のみ

---

## 4. 領域D: 検証・テスト

### 4.1 WF バックテスト
- `backtest_central_leakfree.py` 3/11 stale (v15 未対応)
- V15.1 retro は LGB quick mode (200K subsample, lr=0.1) のみ
- **5/24 Phase 3 必須**: V15.1 4-model 全年 WF (6h)、KKA 16f coverage 0% 原因究明 (1h)

### 4.2 実配当 ROI
- `calc_actual_roi_v135b.py` 4/2 / v15 未対応
- `actual_roi_results.json` は 4/19 v13.5b の数字 (324R, 120.25%)
- v15 + 戦略⑦ filter 統合した計算スクリプト (3h、5/9 戦略⑦実証で必要)

### 4.3 retro 検証 (v18/v19)
- normalize 後 ROI 1450-2708% は **sample 9-25 bets で CI 広い**
- winner_top1 rate -13.3pt 劣化は normalize で **解消されない別要因**
- feature shift 調査 8h (5/16 v18/v19 投入なら必須)

### 4.4 drift detection
- W18 (4/29-5/4) ROI **31.7%** = 3週連続 ROI<100%
- 5/9 結果で trend 確認必須
- `weekly_auc` 197001/202611 のみ報告 = year_week 計算バグ
- baseline_auc=0.8856 (v13.5b) → 0.8939 (v15) 更新

### 4.5 monte carlo
- 最終実行 2026-03-09 (2ヶ月停止)
- v15 + 戦略⑦ で再実行必要 (2h、Phase 3 必須)
- 撤退ライン -50,000円 整合性確認

### 4.6 cumulative_results.csv
- top1_name 充填 **20/494 (4.0%)** = CLAUDE.md 既知バグ実測確認
- 累計 +14,140円 は `phase_2_5_session10_final.md` 値、CSV から手計算困難
- 修復 4h + 累計 PnL 自動計算 2h + 過去 backfill 3h

### 4.7 戦略⑦ 検証
- 4/27 から filter active (06_特別/京都/条件E/条件B 除外)
- 実運用 N=298R での検証未完了
- healthy 4日分析 ROI 161.0% [CI 135.9-222.4%], **n=10** = thin
- 5/9 投入直前に retro 完全版 2h 必須

### 4.8 テストスイート
- `tests/test_features.py` / `debug_all.py` 3/11 stale
- `validation_1〜13_*.py` 全て 3/11 stale (v12 ベース)
- V15.1 leak audit 2h (5/24 採用なら必須)
- `regression_test_v15_final.py` 4/11 / `fullclass_test_v15.py` 4/11 → 4/27 全20R PASS 19/FAIL 0/SKIP 1

### 4.9 healthy 4日分析
- n=10 二項検定 CI 広い → 4/4, 4/11, 4/12 等 healthy 候補日追加で n 拡張 (3h)
- bootstrap 4日 cluster 依存性 → 実質自由度 4 で CI 解釈注意

---

## 5. 領域E: ドキュメント

### 5.1 CLAUDE.md - **最大の負債**
- 1325 行 / 「現行モデル v13.5b」のまま (V14/V15/V15.1/V16/V17/V18/V19/NAR 一切なし)
- 「v16 Development Status」セクション **二重重複** (1177-1240 + 1241-1325 行、emoji 化けあり/なし両方残置)
- header "Last updated: 2026-04-19" は誤
- 5/24 Phase 3 移行までに header + 現行モデル + V17/V18/V19/NAR セクション追記必須 (合計 ~2h)
- 朝起きて運用するだけなら CLAUDE.md は読まなくて良い (README + HANDOFF で完結)

### 5.2 HANDOFF 系 (35 ファイル)
- 最新 v2 系 5 件 (5/5、合計 ~55KB) + 古い 30 件
- GW 系 8 + V162 系 4 + 4/19-4/25 旧 handoff 2 = **14 ファイル archive 候補**

### 5.3 report/ (40 ファイル、4/19-4/26)
- すべて Phase 2.5+ 移行で意味失効 → **`report/archive_pre_5_5/` 一括移動**
- 価値の高い 5 件のみ root 残置
- V15 / V15.1 関連 audit が **不在**

### 5.4 data/v18/ (38 ファイル)
- **`data/v18/index.md` 新設**で 5/9 朝の操作迷子防止 (30min)
- nar_* 5 ファイルを `data/nar_docs/` へ移動
- `phase_2_5_session10_final.md` を Session #11-#18 反映で更新

### 5.5 data/results/
- 5/9 系 8 ファイルのうち `final_plan.md` (v1) は廃止 → archive (5/9 朝混乱防止)
- `data/results/index.md` 新設 (5/9 朝 順番に開く 5 ファイル明示)

### 5.6 README.md - **最新で完璧**
- 218 行 / 5/5 18:35 更新 / V15/V17/V18/V19/NAR/Phase 2.5+/戦略⑦/cumulative +14,140 全反映
- バックアップ `README.md.bak_v161_20260428` を archive へ (1min)

### 5.7 Phase 3 / V20 構想
- `data/v18/jra_nar_integration_plan.md` 唯一の Phase 3 設計書 / **完成度 70%**
- 抜けている: chihou data 生成手順 / V20 train skeleton / A/B 評価計画 / NAR 実 ROI backtest

### 5.8 MEMORY system
- `memory/` ディレクトリ存在するが **空**
- `MEMORY.md` 不在 → 新規作成 (V15 ベースライン / 戦略⑦ / 撤退ライン / リークフリー features の 1 page、30min)

### 5.9 docs/security.md
- 4/4 更新、概ね有効
- WARNING `tools/scrape_jrdb.py:884` の JRDB_ID print 伏字化 (5min)

---

## 6. 領域F: 戦略・マイルストーン

### 6.1 5/9 (土) 本番 - **4 日後**
- V15 案B改 (12R 1勝クラスのみ、最大 2,100円) 確定
- schtasks 全 Ready、dry-run 完了
- 5/8 21:00 race_name 確認 + 5/9 朝 checklist 通り + 5/9 20:30 振り返り埋めだけ

### 6.2 5/12 (火) NAR paper - **7 日後**
- pipeline 5 task admin 登録済、`scrape_nar_today.py` / `scrape_nar_results.py` 本実装済 (commit c106f66b)
- **🔴緊急 blocker**: `chihou_races_2020_2025.csv` 不在
- 5/12 火 17:00 発火前に手動 dry-run 推奨 (本実装直後で本番未実行)
- 5/15 夜 go/no-go 判定基準 doc 化必要

### 6.3 5/16 (土) 試行候補 - **11 日後**
- (a) v18/v19 1,000 円/日 = **実質 no-go 寄り** (条件 5 件未達)
- (b) NAR 500 円/日 = paper 結果次第
- (c) V15.1 paper trading = Phase 3 (5/24+) defer
- 5/15 まで critical path: race-level normalize 本番統合 + feature shift 調査

### 6.4 5/24 Phase 3 移行 - **19 日後**
- 採用基準 6 条件 (post_5_9_improvement_template.md §5):
  1. JRA 案B改 ROI ≥ 100% (4/12-5/24 累計)
  2. race-level normalize 本番統合済
  3. NAR paper 12-14 race 蓄積
  4. v18/v19 試行 sample 30+ bets
  5. 累計 +10,000 円維持
  6. 撤退ライン余裕 30,000+ 円
- 全達成 → V15.1 SKB 4-model + V20 統合構想着手
- 未達 → Phase 2.5 延長

### 6.5 撤退ライン -50,000円 監視
- 累計 +14,140 円、余裕 +64,140 円
- 多段階基準明文化 (5/9 単日 ROI<50% → 5/10 停止 / 累計 -10k → 翌週停止 / -50k → 完全撤退)
- **アラート自動化未実装** → 撤退判定は手動チェック前提
- 累計閾値自動 Discord アラート 30min で実装可

### 6.6 既知の問題 (HANDOFF §9)
- TYB 公開時刻不明 (5/11 判定)
- jra_payouts 4/26 stale → 実は 5/3 まで更新済 (CLAUDE.md 訂正必要)
- chihou_races 2020-2025 不在
- daily_predict_watchdog 平日 fatal alert 誤報
- jrdb_paci.csv 4/4 停止 → 解消済 (CLAUDE.md 訂正)
- ot/ov/ow/oz 4 種 33日 stale (v15 未使用)
- JRDB 2026年データ未取得 (CLAUDE.md 既知、要確認)

### 6.7 Phase 2.5 → 3 移行基準
- M3 (race-level normalize 本番統合) が 5/24 採用基準のクリティカルパス
- 撤退/採用 1 枚 doc 化推奨

### 6.8 長期戦略
- JRA-VAN: **一度だけ契約 → 2025 年データ抜き取り済 → 退会済** (Session #24 ユーザー報告)。 5/24 まで再契約不要、6 月の v16/v17/v20 学習タイミングで再契約候補 (¥2,090/月、必要月数のみ)
- クラウド化不要 (現状 schtasks 安定)
- 累計目標: Phase 2.5 完了で +20-25k / Phase 3 完了で +30-50k / 年末 +100k

---

## 7. 統合 優先順位 (時系列)

### 今夜 (5/5 深夜) - 任意
- このレポートを朝読むだけ
- 何もしない選択も可

### 5/6 (火) 〜 5/8 (金) 平日 - 推奨実施
1. **🔴 ProcessWatchdog v2 schtasks 登録** (30min)
2. **🔴 fire_check 系 caller 引数監査** (30min)
3. **🟠 SED260503 取得 + KKA/KAB 連結再実行** (15min)
4. **🟠 speed_index 4-5月 backfill** (30min)
5. **🟠 5/4-5/5 jra_payouts 取得** (5min) + cron 自動化 (30min)
6. **🟠 戦略⑦ 5/2-5/3 retro 完全版** (2h) ← 5/9 投入直前必須
7. **🟠 cumulative_results.csv 書き込みバグ修正** (4h)
8. **🟠 累計 PnL 自動計算 + 撤退ライン Discord アラート** (1h)
9. **🟡 CLAUDE.md header + 既知問題 (paci/payouts) 訂正** (10min)

### 5/8 (金) 21:00 後 - **必須 (1度)**
- 5/9 12R race_name 確認 (5min)
- Cookie 有効性 check (1min)
- FridayWeekendScrape 補完手動 (10min)

### 5/9 (土) - **本番**
- 06:30 / 08:00 自動発火後 `data/results/20260509_pat_checklist.md` 通り
- 14:00-15:30 PAT 投票 (700円 × 採用R数、最大 2,100円)
- 18:00 自動レポート確認
- 20:30 `post_5_9_improvement_template.md` 埋め

### 5/10 (日) 〜 5/11 (月)
- **🔴 chihou_races_2020_2025.csv 取得 (5-10h)** ← 5/12 paper 必須
- TYB 観測完了判定 (5/11、5min)
- v17 ULTRA-CLEAN 復活 or 廃止判断
- 5/9 結果反映で drift detection 再評価

### 5/12 (火) NAR paper 開始
- scrape_nar_today/results 手動 dry-run (30min)
- NAR 5/15 夜 go/no-go 判定基準 doc 化 (30min)
- 5 task 自動発火を 4 日間観察

### 5/13 (水) 〜 5/15 (金)
- **🟠 race-level normalize 本番統合 (predict_core.py)** (30min)
- **🟠 feature distribution shift 調査** (90min)
- 複勝 odds 実値で fukusho retro 再評価 (30min)
- 5/15 夜 v18/v19 GO/no-go 判定

### 5/16 (土) 試行候補
- v18/v19 1,000円/日 (条件達成時のみ、現状 no-go 寄り)
- NAR 500円/日 (paper 結果良好なら)
- V15.1 paper trading 開始

### 5/17 (日) 〜 5/24 (金) Phase 3 移行週
- V15.1 4-model ensemble 学習 (5/17-5/19、6h)
- V15.1 全年 WF (5/19-5/21、6h)
- V15.1 軸 top3 率 retro WF (2h)
- V15.1 leak audit (validation_1 v15 対応含む、4h)
- V15.1 + 戦略⑦ Monte Carlo 再実行 (2h)
- KKA 16f coverage 0% 原因究明 (1h)
- `predict_core.py` SKB merge 統合 (1h)

### 5/24 (金) 夜 Phase 3 移行判定
- 6 条件全達成チェック → GO なら V15.1 + V20 構想着手 / 未達なら Phase 2.5 延長

### 6 月以降 (Phase 3 / 4)
- V20 統合モデル設計実装 (28h、chihou data 必須)
- TARGET Frontier JV 再インストール → feature_lookups 再生成
- NAR v5 再学習 (2025-2026年データ + JOCKEY_OVERRIDE 構造化)
- Phase 4+ ロードマップ (ELO/レーティング、リアルタイム EV、馬連/三連単拡張)

---

## 8. 古い doc / バックアップ 一括 archive (任意、~3MB cleanup)

### 5/9 後 推奨 archive 移動
- `docs/CHECKLIST_4_30_*.md`, `CHECKLIST_5_2_*.md` (3件)
- `docs/GW_PLAN_2026*.md` (2件)
- `docs/SESSION_20260428*.md` (2件)
- `docs/MAY_PLAN_V162.md`, `V162_*.md` (4件)
- `docs/V17_DEPLOY_STRATEGY.md`, `20260427_v16_prep_report.md`, `20260428_v161_session.md`
- `docs/daily_handoff_20260420.md`, `weekly_handoff_20260425.md`, `TOMORROW_MORNING_TODO.md`
- `report/*` 35 ファイル → `report/archive_pre_5_5/`
- `keiba_model_v15_central{,_live}.pkl.gz.bak_v16_20260427`
- `README.md.bak_v161_20260428`, `CLAUDE.md.bak_20260427`
- `data/_model_bak_20260503/` (135MB)
- `archive/old_models_20260505/` (86MB) - 将来 Phase 3 確定後

---

## 9. 結論 (一句)

**5/9 (土) は現状維持で完遂、心配無用**。
本格的な整備は 5/10-5/24 の 2 週間で計画通りに進めれば良い。
最大のリスクは (1) `ProcessWatchdog v2` 未スケジュール、(2) `chihou_races_2020_2025.csv` 不在、(3) feature distribution shift で V18/V19 5/16 試行不能の 3 点。
全項目を朝1分で見渡せるよう **§0 エグゼクティブサマリー** に集約済。
累計 +14,140 円死守、撤退ライン -50,000 円まで余裕 +64,140 円、慌てる必要は一切なし。

---

**生成元**: 6 並列 research agent (claude opus 4.7、合計実行時間 ~12分、tokens ~520k)
**根拠ファイル**: HANDOFF_5_5_TO_5_9.md / data/v18/*.md / report/*.md / CLAUDE.md / 実コード grep 200+ 箇所
