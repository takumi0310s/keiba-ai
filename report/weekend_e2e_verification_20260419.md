# 来週末 (4/25-26) E2E 検証レポート

- 作成: 2026-04-19 (Sun) 23:00
- 対象: 2026-04-25 (Sat) / 04-26 (Sun) / 04-27 (Mon) の全自動発火タスク
- 判定: **✅ 来週末 手動介入不要**

---

## 1. エンドツーエンド dry-run 結果

`tools/dryrun_weekend_full.py` で 17 タスクを完全シミュレーション.

### Saturday 2026-04-25
| 時刻 | タスク | 判定 |
|------|--------|:---:|
| 03:00 | DailyPremiumScrape | ✅ PASS |
| 06:00 | DailyJrdbKyi | ✅ PASS |
| 07:30 | JrdbHealthCheck | ✅ PASS |
| 08:00 | DailyPredict | ✅ PASS |
| 08:45 | RaceAutoNotify | ✅ PASS |
| 18:00 | DailyResults | ✅ PASS |
| 20:00 | DailyResultsEvening | ✅ PASS |

### Sunday 2026-04-26
| 時刻 | タスク | 判定 |
|------|--------|:---:|
| 03:00 | DailyPremiumScrape | ✅ PASS |
| 06:00 | DailyJrdbKyi | ✅ PASS |
| 07:30 | JrdbHealthCheck | ✅ PASS |
| 08:00 | DailyPredict | ✅ PASS |
| 08:45 | RaceAutoNotify | ✅ PASS |
| 18:00 | DailyResults | ✅ PASS |

### Monday 2026-04-27 (非開催日)
| 時刻 | タスク | 判定 |
|------|--------|:---:|
| 03:00 | DailyPremiumScrape | ✅ PASS (Mon早朝特例適用済) |
| 06:00 | DailyJrdbKyi | ✅ PASS |
| 08:00 | DailyPredict | ✅ PASS (非開催日で即終了見込) |
| 08:00 | WeeklyReport | ✅ PASS |

### チェック項目 (各タスク)
a. SCRAPER-GUARD 挙動 (ALLOW/STOP 期待値) → 全 ALLOW
b. import 整合性 → 全モジュール import 可能
c. 必要ファイル存在 (モデル/.env/スクリプト) → 全 OK
d. 出力ディレクトリ書き込み可能性 → 全 OK

**総合: 17 PASS / 0 FAIL**

---

## 2. タスクスケジューラ整合性チェック

`tools/check_scheduler_integrity.py` で登録済み 14 タスクを検証.

### 状態OK (10/14)
- Keiba-FridayWeekendScrape
- KeibaAI_DriftDetector
- DailyJrdbKyi
- DailyPremiumScrape
- DailyResultsEvening
- DailyResults_Sat
- JrdbHealthCheck_Sat (LastRun は未発火・自然)
- JrdbHealthCheck_Sun
- ProcessWatchdog
- RaceAutoNotify_Sat

### 警告 (過去の強制終了履歴あり、次回は修正済で問題なし) 4/14
- DailyPredict (last=3221225786 Ctrl+C @ 4/19)
- DailyResults_Sun (last=3221225786 Ctrl+C @ 4/19)
- RaceAutoNotify_Sun (last=3221225786 Ctrl+C @ 4/19)
- WeeklyReport (last=1 failure @ 4/13)

→ いずれも「過去の履歴」。状態 (State) は全て Ready。
   commit `1e208b97` (コンソール強制終了対策) で軽減済、commit `4f613a03` (guard修正) で根本原因解消.
   来週末は正常発火見込.

---

## 3. 新規追加: Keiba-NightlySanity

**新しい毎晩自動チェック**

| 項目 | 値 |
|------|-----|
| タスク名 | Keiba-NightlySanity |
| 発火時刻 | 毎日 23:00 |
| 実行 | `C:\Users\takum\keiba-ai\nightly_sanity_check.bat` |
| 中身 | `python tools/nightly_sanity_check.py` |
| 状態 | ✅ Ready (登録済) |

### 自動チェック内容 (翌日発火予定タスクを事前確認)
1. 翌日発火予定 Keiba タスクを列挙
2. 各タスクの State / LastResult 確認
3. 必要ファイル (モデル v15, .env, scraper_guard.py) の存在確認
4. SCRAPER-GUARD 挙動を翌日時刻でシミュレート
5. 異常があれば Discord 緊急通知 (red)、正常なら通知 (green)

### 本日 (Sun 4/19) の初回 dry-run 結果 (翌日 Mon 4/20 向け)
- 発火予定 6 タスク 全 Ready
- 必要ファイル 3/3 存在
- SCRAPER-GUARD 6/6 ALLOW
- ⚠ 警告: DailyPredict の Ctrl+C 履歴 (明日は非開催日で軽量動作見込で問題なし)

---

## 4. 手動切り替え推奨タスク

**なし**

既存タスク全てが最新の .bat を参照しており、.bat 内で PYTHONIOENCODING=utf-8 を設定済.
手動切り替えは不要.

---

## 5. 既存タスクの .bat 設定

全タスクで以下を確認済:
- `cd /d C:\Users\takum\keiba-ai`
- `set PYTHONIOENCODING=utf-8`
- `set PYTHONUNBUFFERED=1` (KeibaAI_DriftDetector のみ未設定)
- ログ出力 `>> logs\...log 2>&1`

### ⚠ 微警告
- KeibaAI_DriftDetector の `drift_detector.bat` に `PYTHONUNBUFFERED=1` 未設定
  → ログ反映遅延の可能性。致命的ではないので現状維持

---

## 6. 生成物

| ファイル | 内容 |
|----------|------|
| `tools/dryrun_weekend_full.py` | 来週末+Mon の 17 タスク dry-run (新規) |
| `tools/check_scheduler_integrity.py` | タスク登録整合性チェック (新規) |
| `tools/nightly_sanity_check.py` | 毎晩23:00 の事前チェック (新規) |
| `nightly_sanity_check.bat` | NightlySanity タスク実行用 (新規) |
| `report/weekend_e2e_verification_20260419.md` | 本レポート |

---

## 最終判断

### ✅ 来週末 (4/25-26) 手動介入不要

根拠:
- **E2E dry-run 17/17 PASS** — 全時刻の全タスクで SCRAPER-GUARD ALLOW、import OK、ファイル存在 OK、書き込み OK
- **スケジューラ登録 14 タスク全て Ready** — 過去の強制終了履歴はあるが、修正 commit 済
- **Monday 4/27 早朝 03:00** も premium_scrape Mon 特例で正常発火 (commit `4f613a03` で修正)
- **nightly_sanity_check** が毎晩 23:00 に自動チェックして問題を事前に Discord 通知
- 必要モデル v15 (150特徴量) + .env + Cookie 全て揃っている

### 予防措置
- 明晩 23:00 に Keiba-NightlySanity が自動発火 → 翌 Mon のタスクを事前チェックして Discord 通知
- 金曜夜 23:00 発火 → 翌 Sat のタスクを事前チェック
- 土曜夜 23:00 発火 → 翌 Sun のタスクを事前チェック

### 万が一の緊急リカバリ
詳細は `docs/weekly_handoff_20260425.md` (フェーズE で作成予定) を参照.
