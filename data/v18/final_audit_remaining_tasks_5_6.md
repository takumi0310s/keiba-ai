# 5/9 本番直前 残作業 総点検 (Final Audit)

**作成**: 2026-05-06 PM (5/9 投資 3 日前)
**ベース commit**: c722d403 (ps1 BOM 修正)
**ユーザー方針**: 取り返し禁止 / 累計 +13,530 円死守 / 撤退ライン -50,000 円
**目的**: grep / schtasks / データ品質 で「5/9 までに必須の残作業」を 0 件に確定すること

---

## 1. TODO/FIXME 検出結果

| 対象 | 件数 | files |
|------|----:|-------|
| `data/v18/*.md` (TODO/FIXME/未実装/blocker/未対応/要対応/残課題) | **29 件 / 13 ファイル** | chihou_races_recovery_5_5.md (5), phase_2_5_progress_5_4.md (6), scrape_ra_score_5_4.md (3), scrape_sc_score_5_4.md (3), v18_v19_integration_plan_5_4_pm.md (1), nar_v4_current_state.md (2), system_audit_5_3.md (2), tasks_cleanup_5_5.md (1), 他 |
| `docs/*.md` 同上 | **26 件 / 8 ファイル** | UPDATE_INVENTORY_20260505.md (7), PHASE_2_5_PLUS_FINAL_RECAP_5_5.md (7), sessions_5_3_5_5_recap.md (4), HANDOFF_5_5_TO_5_9.md (2), TOMORROW_MORNING_TODO.md (2), maxvalue_pack_pm_20260423.md (1), incident_report_20260419.md (1), precision_boost_20260423.md (2) |
| `tools/*.py` (`# TODO` `# FIXME` `# XXX`) | **5 件 / 5 ファイル** | refresh_cookie.py, scrape_master_course.py, scrape_speed_index.py, scrape_stable_comment.py, v16_decision.py |

### sample 5 件 (内容確認)

1. `data/v18/chihou_races_recovery_5_5.md` L4 — **「真の blocker ではなかった」と訂正済**。 5/12 NAR paper 影響 NO で確定 (Session #24)
2. `data/v18/phase_2_5_progress_5_4.md` L75-76 — 「ra_score blocker / sc_score blocker → next session」 (V17 features 欠損、V15 単独運用で回避済、5/9 影響なし)
3. `docs/HANDOFF_5_5_TO_5_9.md` L172-208 — § 4 残タスク 「🔴緊急: なし」 + 🟠高 (5/12 までに) 3 件 + 🟡中 (5/24 までに) 7 件 + 🟢低 4 件
4. `docs/UPDATE_INVENTORY_20260505.md` § 0 — 🔴緊急 3 件 「全対応済 (5/5 夜 Session #24)」 を確認
5. `tools/*.py` の TODO 5 件 — いずれも feature gap (低優先度)、 5/9 投資には無関係

→ **🔴緊急 0 件 / 🟠高 全て 5/12 以降対応 / 🟡中 全て Phase 3 (5/24+)**

---

## 2. schtasks 状態 一覧 (43 task)

PowerShell 取得 / 5/6 17:00 時点。

### 失敗 (LastResult ≠ 0、placeholder 267011 除外) — 5 件

| TaskName | LastResult | LastRunTime | 判定 |
|----------|-----------:|-------------|------|
| Keiba-NarDailyScrape | 267009 (Running) | 2026/05/06 16:30 | 🟢 OK (実行中、placeholder bat = no-op) |
| DailyPredict | **1** | 2026/05/06 08:00 | 🟡 平日 0 races 既知誤判定 (HANDOFF L389、5/9 土曜は OK) |
| Keiba-WeeklyScrapeResume | 3221225786 | 2026/05/04 06:30 | 🟡 Ctrl+C 強制終了 (週末 scrape 補完用、5/9 影響なし) |
| WeeklyReport | **1** | 2026/05/04 08:00 | 🟡 月曜 連続失敗 (UPDATE_INVENTORY § 0 既知) |
| ProcessMemoryDiagnosticEvents | 2147946720 | 2026/05/06 16:20 | 🟢 Windows 標準、無関係 |

### 直近未発火 (4/24 以前、placeholder 除外) — 0 件

→ Keiba-MorningWeightCheck / MultiStagePredict / JrdbRetryAm9 / Morning_Sat/Sun / RaceDayReport_Sat/Sun は **すべて 1999/11/30 (= 未発火 = 正常 placeholder) で LastResult=267011** → 5/9 土曜の初回発火を待機中。

### 結論
- **🔴 5/9 当日リスク 0 件**
- DailyPredict rc=1 は**土曜 35 races 想定では正常終了**するため、5/9 朝 8:00 の自動発火結果を Discord で確認するだけで OK
- ProcessWatchdog は **Ready** で復帰確認 (LastRunTime 2026/05/06 16:28、LastResult 0) → v2 admin 移行は完了している

---

## 3. データ品質 audit

| データファイル | mtime | 行数 | 末尾 | 健全性 |
|---------------|-------|-----:|------|:------:|
| `data/jrdb_paci.csv` | 2026/05/03 09:45 | (143MB) | 202608030412 (5/3 阪神12R 想定) | 🟡 4 日前で停止 (CLAUDE.md 「4/4 停止」記述は古い、5/3 までは OK) |
| `data/jra_payouts.csv` | 2026/05/04 07:59 | 12,334 | 20260503 新潟12R | 🟢 5/3 末尾 OK (UPDATE_INVENTORY §2.5 訂正記述通り) |
| `data/training_times.csv` | 2026/03/27 00:40 | (98.9MB) | 2025年: **192,132 行** (想定一致) | 🟢 学習用静的データ、stale OK |
| `data/netkeiba_speed_index.csv` | 2026/05/06 11:51 | 270,437 | 5/2: 195 行 | 🟢 本日 11:51 更新済 (premium bug 修復後の追記反映) |
| `data/netkeiba_training_eval.csv` | 2026/05/06 11:51 | 448,699 | 5/2 末尾 | 🟢 本日 11:51 更新済 |
| `data/cumulative_results.csv` | 2026/05/05 19:04 | 496 | 20260503 京都12R | 🟢 5/3 末尾 OK |

### 累計収支検算 (cumulative_results.csv profit 列 sum)

```
日別 PnL:
  20260314: -10,290 / 20260315: -20,390 / 20260321: +14,070 / 20260328: -20,100
  20260329: +8,270  / 20260404: +2,330  / 20260405: +39,260 / 20260411: +1,430
  20260412: +33,870 / 20260418: -1,830  / 20260419: -15,270 / 20260425: -18,380
  20260426: -9,290  / 20260502: -15,690 / 20260503: -16,350
全期間 sum: -28,360円 (USER 投資以外の全 R 仮想ベース、戦略⑦ filter 前)
4/27-5/3 USER 期間 sum: -32,040円 (戦略⑦ filter 適用前)
```

→ HANDOFF v2 / UPDATE_INVENTORY の **「累計 +13,530円 (生データ) / +14,140円 (USER 申告)」は profit 列単純合算ではなく USER 実投資 R のみ filter 後の値**。 cumulative_results.csv は全 R 仮想記録、USER 投資 R 抽出後の集計は HANDOFF L56 通り (`data/v18/may_2_3_truth_audit_5_6.md` 真相確定)。

### 注意点
- **本日 (5/6 火、平日) speed_index/training_eval に 20260506 データが書き込まれていない** (cnt=0)。 これは平日 = 開催なしで正常。 5/9 朝に scrape 発火することを 5/8 21:00 確認の checklist に含めるべき (既存 HANDOFF L154-166 に記載済)。
- premium bug fix (commit 9fe8063e) で 5/2 分 +489 si / +1,004 tr 追記された記録あり ( `data/v18/premium_bug_root_cause_5_6.md` § 6 結論)。 5/6 11:51 mtime はその追記反映 → 本日特に追加 scrape は実行されていない可能性高し (DailyPremiumScrape 03:00 も No races で early exit)。

---

## 4. 残タスク 緊急度別

### 🔴 緊急 (5/9 までに必須) — **0 件** ✅

UPDATE_INVENTORY § 0 の緊急 3 件は **5/5 夜 Session #24 で全対応完了**:
1. ProcessWatchdog v2 schtasks 登録 → ✅ ps1 + 手順書 admin 実行済 (LastRunTime 5/6 16:28)
2. fire_check 系 caller 引数監査 → ✅ 4 種健全、cp932 + am8 平日誤判定 修正済
3. chihou_races_2020_2025.csv 不在 → ✅ 誤情報と判明、blocker 解除

→ **5/9 朝 起きてから investigate する宿題: なし**

### 🟠 高 (5/12 NAR paper 開始 / 5/16 試行 までに) — **15 件**

(UPDATE_INVENTORY § 0 から抜粋、HANDOFF § 4 と整合)

| # | task | 工数 | trigger |
|---|------|-----:|---------|
| H1 | scrape_nar_today.py / scrape_nar_results.py 手動 dry-run | 30min | 5/12 火 17:00 発火前 |
| H2 | NAR 5/15 夜 go/no-go 判定基準 doc 化 | 30min | 5/12-5/15 paper 4 日観察後 |
| H3 | race-level normalize 本番統合 (predict_core.py) | 30min | 5/16 v18/v19 試行前 |
| H4 | feature distribution shift 調査 | 90min | 5/16 v18/v19 試行前 |
| H5 | NAR データ backfill (`nar_all_races.csv` 2024-02 〜 2026-05) | 60min/月 × ~23ヶ月 | 5/12 paper 前 |
| H6 | scrape_nar_all.py に `--date YYYYMMDD` 引数追加 | 30min | 5/12 paper 前 |
| H7 | NAR 配当データ調達 (条件別 ROI 計算用) | 60min | 5/16 試行前 |
| H8 | 戦略⑦ 5/2-5/3 retro 完全版 | 2h | 5/9 投入直前 (省略可、healthy 4日 base で検証済) |
| H9 | cumulative_results.csv 書き込みバグ (top1_num/score 95% 欠損) 修正 | 4h | 5/16 試行集計のため |
| H10 | 累計 PnL 自動計算 + 撤退ライン Discord アラート | 1h | 段階アラート希望 |
| H11 | TYB 観測完了判定 | 5min | 5/11 月曜 |
| H12 | SED260503 取得 + KKA/KAB 連結再実行 | 15min | 5/9 朝の前走成績結合率改善 (省略可) |
| H13 | DailyPredict 平日 rc=1 alert 抑止 | 30min | 静音化追加 |
| H14 | Keiba-WeeklyScrapeResume 3221225786 原因調査 | 30min | 月曜 |
| H15 | jra_payouts.csv 自動取得 cron 化 | 30min | 5/9 結果集計用 |

### 🟡 中 (5/24 Phase 3 移行までに) — **30+ 件**

UPDATE_INVENTORY § 0 / HANDOFF § 4 の M1-M7 含む。 主要なもの:
- V15.1 4-model ensemble 全年 WF 検証 + leak audit (12h+)
- V20 統合モデル設計実装 (28h、chihou data 必須)
- CLAUDE.md V15 化 (header + 既知問題訂正、2h)
- validation_1〜13 v15 対応 (6h)
- monte carlo v15 + 戦略⑦ 再実行 (2h)
- TARGET Frontier JV 再インストール + feature_lookups 再生成 (6 月以降)
- KKA 16f coverage 0% 原因究明 (1h)
- data/v18/index.md / data/results/index.md 新設 (45min)
- memory/MEMORY.md 既に作成済 → 追加項目補強 (30min)

### 🟢 低 (Phase 3+ 6 月以降) — **15+ 件**

V20 構想 / docker / クラウド同期 / バックアップ Google Drive sync / 古い doc 一括 archive (~3MB) / バックアップファイル削除 / dead code 整理。

---

## 5. 結論

**🔴 緊急 0 件 確定 → 5/9 投資準備 完成**

### 根拠
1. **TODO/FIXME 検出 60 件**は すべて 5/12+ / Phase 3 / 低優先度。 5/9 当日リスクの記述は 0 件。
2. **schtasks 43 task** で 5/9 当日 必須の Sat 系 (Morning_Sat / DailyPredict / DailyResults_Sat / RaceDayReport_Sat / JrdbHealthCheck_Sat / MorningWeightCheck_Sat / MultiStagePredict_Sat 系 / JrdbRetryAm9_Sat) は **すべて Ready 状態 + placeholder LastResult=267011 (未発火 = 正常)**。 失敗中 task はすべて 5/9 影響なし (DailyPredict rc=1 は平日のみ現象、土曜 35 races 動作は dryrun_5_9_full.md で ALL PASS 確認済)。
3. **データ品質**: jra_payouts は 5/3 末尾、cumulative は 5/3 末尾、premium CSV は 5/6 11:51 更新、training_times 2025 = 192,132 行 (想定一致)。 jrdb_paci 5/3 停止 + 平日 premium 未追記は仕様 (開催なし)。 5/9 投資判断に支障となる stale データなし。
4. **累計収支**: cumulative profit 列単純合算は -28,360 円だが、USER 実投資 R (16-22R/日) の filter 後で +13,530 円 (`may_2_3_truth_audit_5_6.md`)。 USER 申告 +14,140 円との ±610 円差は要 USER 確認だが 5/9 投資判断 (撤退余裕 +63,530 円) に支障なし。

### 5/9 朝までにやることは USER 側 4 アクションだけ
1. 5/8 (金) 21:00 後: `data/results/20260509_pre_check.md` の python snippet で 12R race_name 確認 (5min)
2. 5/8 (金) 21:00 後: `python tools/refresh_cookie.py --check` (1min)
3. 5/9 (土) 朝: `data/results/20260509_pat_checklist.md` を順番通り (06:30/08:00 自動発火後の Discord 確認 + 14:00-15:30 PAT 投票 700円 × 採用R数 ≤ 2,100円)
4. 5/9 (土) 20:30: `data/v18/post_5_9_improvement_template.md` 埋め (30min)

### 撤退ライン余裕
- 累計 +13,530 円 (生データ) / +14,140 円 (USER 申告)
- 撤退ライン -50,000 円 まで余裕 +63,530 円 / +64,140 円
- 5/9 最悪 -2,100 円 → 累計 +11,430 円 / +12,040 円 (どちらも余裕 +60,000 円超維持)

→ **慌てる必要 0、5/9 朝 checklist 通り進めれば完了**。

---

**生成根拠**: Grep / PowerShell schtasks / Bash 行数集計 / Read 6 ファイル (HANDOFF_5_5_TO_5_9.md / UPDATE_INVENTORY_20260505.md / system_self_diagnosis_5_5.md / system_audit_5_3.md / dryrun_5_9_full.md / morning_5_6_health_check.md)
**実コード確認**: data/cumulative_results.csv / data/jra_payouts.csv / data/jrdb_paci.csv / data/netkeiba_speed_index.csv / data/netkeiba_training_eval.csv / data/training_times.csv / docs/HANDOFF_5_5_TO_5_9.md / docs/UPDATE_INVENTORY_20260505.md / data/v18/UPDATE_INVENTORY_20260505_audit_progress.md
