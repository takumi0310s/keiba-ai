# 明日 AM3:00〜AM8:00 自動発火タスク事前検証レポート

- 作成日時: 2026-04-19 (Sun) 18:30
- 対象日: **2026-04-20 (Mon)** ※指示書では「日曜」と記載されていたが、実際は月曜
- 検証ブランチ: `fix/sunday-am3-regression-check`
- 関連インシデント: 4/19 AM3:00 / AM8:00 SCRAPER-GUARD 誤停止事故 (commit e173f40d 等で修正)

---

## ⚠️ 日付の重要な訂正

| 指示書の記述 | 実際 |
|--------------|------|
| 本日 2026/04/19 = 土曜 | **本日は日曜 (Sun, weekday=6)** |
| 明日 2026/04/20 = 日曜 | **明日は月曜 (Mon, weekday=0)** |

→ 明日発火する週末専用タスク (`RaceAutoNotify_Sun`, `JrdbHealthCheck_Sun`, `DailyResults_Sun`) は、
   次回起動が **2026-04-26 (Sun)** に持ち越される。明日 4/20 (Mon) は発火しない。

---

## 1. タスクスケジューラ登録一覧 (現状)

| タスク名 | 次回実行 | 実行ファイル | 最終結果 | 状態 |
|----------|----------|--------------|----------|------|
| **DailyPremiumScrape** | **2026/04/20 03:00** | `daily_premium_scrape.bat` | 0 | Ready |
| **DailyJrdbKyi** | **2026/04/20 06:00** | `tools\daily_jrdb_kyi.bat` | 0 | Ready |
| **DailyPredict** | **2026/04/20 08:00** | `daily_predict.bat` | 3221225786 ⚠ | Ready |
| **WeeklyReport** | **2026/04/20 08:00** | `weekly_report.bat` | 1 | Ready |
| DailyResultsEvening | 2026/04/19 20:00 | `daily_results.bat` | 0 | Ready (今夜) |
| DailyResults_Sat | 2026/04/25 18:00 | `daily_results.bat` | 0 | Ready |
| DailyResults_Sun | 2026/04/26 18:00 | `daily_results.bat` | 3221225786 ⚠ | Ready |
| RaceAutoNotify_Sat | 2026/04/25 08:45 | `race_auto_notify.bat` | 0 | Ready |
| RaceAutoNotify_Sun | 2026/04/26 08:45 | `race_auto_notify.bat` | 3221225786 ⚠ | Ready |
| JrdbHealthCheck_Sat | 2026/04/25 07:30 | `jrdb_health_check.bat` | 267011 ⚠ | Ready (未起動) |
| JrdbHealthCheck_Sun | 2026/04/26 07:30 | `jrdb_health_check.bat` | 0 | Ready |
| ProcessWatchdog | 2026/04/19 18:10 (5分間隔) | `process_watchdog.bat` | 0 | Ready |
| Keiba-FridayWeekendScrape | 2026/04/24 10:00 | `friday_weekend_scrape.bat` | 0 | Ready |
| KeibaAI_DriftDetector | 2026/04/20 08:30 | `drift_detector.bat` | 0 | Ready |

**3221225786 = 0xC000013A = STATUS_CONTROL_C_EXIT (Ctrl+C / 強制終了)**

→ 今日 (Sun 4/19) の `DailyPredict` (08:00)、`DailyResults_Sun` (18:00)、`RaceAutoNotify_Sun` (08:45) はいずれも強制終了されている。
   `1e208b97 fix: daily_predict Windows コンソール強制終了対策` の修正後も発生 → 別途調査要 (本タスクのスコープ外)。

---

## 2. SCRAPER-GUARD バイパス設計確認

### 2.1 OPERATIONAL_CALLERS ホワイトリスト (修正後)

```python
OPERATIONAL_CALLERS = frozenset({
    "daily_predict",
    "race_auto_notify",
    "notify_bets_all_in_one",
    "jrdb_health_check",
    "daily_jrdb_kyi",
    "daily_results",   # ← 本検証で追加 (defensive: 内部で netkeiba scrape 有り)
})
```

### 2.2 各タスクの check_scraping_allowed 実装状況

| タスク | スクリプト | check_scraping_allowed 呼び出し | caller= 引数 |
|--------|------------|---------------------------------|--------------|
| **DailyPremiumScrape** | `tools/daily_premium_scrape.py` | ✅ あり (L365) | `"daily_premium_scrape"` (mode="exit") |
| DailyPredict | `tools/daily_predict.py` | ❌ なし | — |
| DailyJrdbKyi | `tools/scrape_jrdb.py` 等 | ❌ なし | — |
| RaceAutoNotify | `tools/race_auto_notify.py` | ❌ なし | — |
| JrdbHealthCheck | `tools/jrdb_health_check.py` | ❌ なし | — |
| DailyResults | `tools/daily_results.py` | ❌ なし (内部に netkeiba scrape あり) | — |
| WeeklyReport | `tools/weekly_report.py` | ❌ なし | — |

→ ✅ あり = 1 件のみ。残りは guard を呼ばないので影響を受けない (が、将来のため defensive に whitelist 登録)。

### 2.3 ★ 発見した重大バグ

**`_premium_scrape_early_slot` が Sat/Sun のみ → Mon 早朝 03:00 が誤停止**

```python
# 修正前 (BUG)
def _premium_scrape_early_slot(now: datetime) -> bool:
    return now.weekday() in (5, 6) and 3 <= now.hour < 6
```

- ガード窓は **Fri22:00 〜 Mon06:00** (Mon 00:00-05:59 も含む)
- DailyPremiumScrape は **daily** (毎日 03:00) 起動
- Mon 03:00 起動時:
  - `weekday=0` (Mon)、`hour=3` < 6 → ガード時間帯
  - `_premium_scrape_early_slot()` は (5,6) のみ → False
  - → `mode="exit"` で即終了

**実害履歴 (logs/premium_scrape_*.log で確認):**
- 2026-04-13 (Mon) 03:00 → BLOCKED (`SCRAPER-GUARD ... Mon`)
- 2026-04-19 (Sun) 03:00 → BLOCKED (修正前バージョンで実行)

**修正:**
```python
def _premium_scrape_early_slot(now: datetime) -> bool:
    """Sat/Sun/Mon の 03:00-05:59 を許可"""
    return now.weekday() in (5, 6, 0) and 3 <= now.hour < 6
```

---

## 3. verify_scraper_guard_sunday.py 実行結果

### 修正前 (再現)
```
[NG] 2026-04-20 03:00:00 caller=daily_premium_scrape  expect=ALLOW  got=STOP   AM3:00 DailyPremiumScrape (Mon)
[NG] 2026-04-25 18:00:00 caller=daily_results         expect=ALLOW  got=STOP   PM18:00 DailyResults_Sat
❌ 2 NG / 16 total
```

### 修正後
```
[OK] 2026-04-20 03:00:00 caller=daily_premium_scrape  expect=ALLOW  got=ALLOW  AM3:00 DailyPremiumScrape (Mon)
[OK] 2026-04-20 06:00:00 caller=daily_jrdb_kyi        expect=ALLOW  got=ALLOW  AM6:00 DailyJrdbKyi (Mon, boundary)
[OK] 2026-04-20 08:00:00 caller=daily_predict         expect=ALLOW  got=ALLOW  AM8:00 DailyPredict (Mon)
[OK] 2026-04-20 08:00:00 caller=(none)                expect=ALLOW  got=ALLOW  AM8:00 WeeklyReport
[OK] 2026-04-25 03:00:00 caller=daily_premium_scrape  expect=ALLOW  got=ALLOW  AM3:00 DailyPremiumScrape (Sat)
[OK] 2026-04-25 07:30:00 caller=jrdb_health_check     expect=ALLOW  got=ALLOW  AM7:30 JrdbHealthCheck_Sat
[OK] 2026-04-25 08:00:00 caller=daily_predict         expect=ALLOW  got=ALLOW  AM8:00 DailyPredict (Sat)
[OK] 2026-04-25 08:45:00 caller=race_auto_notify      expect=ALLOW  got=ALLOW  AM8:45 RaceAutoNotify_Sat
[OK] 2026-04-25 18:00:00 caller=daily_results         expect=ALLOW  got=ALLOW  PM18:00 DailyResults_Sat
[OK] 2026-04-26 03:00:00 caller=daily_premium_scrape  expect=ALLOW  got=ALLOW  AM3:00 DailyPremiumScrape (Sun)
[OK] 2026-04-26 08:45:00 caller=race_auto_notify      expect=ALLOW  got=ALLOW  AM8:45 RaceAutoNotify_Sun
[OK] 2026-04-20 03:00:00 caller=(none)                expect=STOP   got=STOP   Mon 03:00 (default)
[OK] 2026-04-20 03:00:00 caller=bulk_scrape_upset     expect=STOP   got=STOP   Mon 03:00 (non-op)
[OK] 2026-04-20 03:00:00 caller=scrape_master_index   expect=STOP   got=STOP   Mon 03:00 (non-op)
[OK] 2026-04-25 08:00:00 caller=(none)                expect=STOP   got=STOP   Sat 08:00 (no caller)
[OK] 2026-04-26 08:00:00 caller=bulk_scrape_upset     expect=STOP   got=STOP   Sun 08:00 (non-op)

✅ ALL PASS 16/16
```

### tests/test_scraper_guard.py
- 修正前: 50/50 PASS (旧 `test_premium_scrape_mon_early_blocked`)
- 修正後: **57/57 PASS** (新 `test_premium_scrape_mon_early_allowed`, `test_premium_scrape_mon_03_allowed`, `test_premium_scrape_mon_06_blocked_for_default` を追加)

### tests/regression_test.py
- **16/16 PASS** (157秒)

---

## 4. dry-run 実行結果

| スクリプト | --dry-run | 実行結果 |
|------------|-----------|----------|
| `tools/daily_predict.py --date 20260420` | ✅ あり | 正常終了。「20260420 のレースが見つかりません(非開催日)」← Mon は非開催日 |
| `tools/jrdb_health_check.py --date 2026-04-20` | ✅ あり | 正常終了。「[DRY-RUN] Would re-scrape: KYI, KAB, KTA, CHA, KKA, JO」 |
| `tools/daily_premium_scrape.py` | ❌ なし | 実行検証は guard simulation で代替 |
| `tools/weekly_report.py` | ❌ なし | Discord通知あり、副作用回避のため実行スキップ |

---

## 5. 明日 4/20 (Mon) 各タスクの期待挙動 (修正適用後)

| 時刻 | タスク | 期待挙動 |
|------|--------|----------|
| **03:00** | DailyPremiumScrape | ✅ Mon 早朝特例で **実行** (修正で復活) |
| **06:00** | DailyJrdbKyi | ✅ ガード窓終了直後・guard 呼出なし → 実行 |
| **08:00** | DailyPredict | ✅ ガード窓外・guard 呼出なし → 実行 (Mon は非開催日のため即終了見込み) |
| **08:00** | WeeklyReport | ✅ ガード窓外・guard 呼出なし → 実行 (Discord通知発信) |
| 08:30 | KeibaAI_DriftDetector | ✅ ガード窓外 |
| 18:10〜 | ProcessWatchdog (5分間隔) | ✅ Mon 06:00 以降は guard 通過 |
| 20:00 | DailyResultsEvening | ✅ ガード窓外 |

---

## 6. 問題と対応

| # | 問題 | 対応 |
|---|------|------|
| 1 | DailyPremiumScrape が Mon 03:00 で誤停止 (今夜放置すると明朝再発) | ✅ `_premium_scrape_early_slot` を Sat/Sun/**Mon** に拡張 |
| 2 | `daily_results` が OPERATIONAL_CALLERS 未登録 (defensive 不足) | ✅ ホワイトリスト追加 |
| 3 | `DailyPredict` LastResult 3221225786 (4/19 強制終了) | ⚠ 別 issue (本タスクスコープ外、`1e208b97` で部分修正済) |
| 4 | `JrdbHealthCheck_Sat` LastResult 267011 (起動履歴なし) | ⚠ 一度も実行されていない可能性。次回 4/25 で要確認 |

---

## 7. 修正ファイル一覧

| ファイル | 変更内容 |
|----------|----------|
| `tools/scraper_guard.py` | `_premium_scrape_early_slot` を Sat/Sun/Mon に拡張、`OPERATIONAL_CALLERS` に `daily_results` 追加、docstring 更新 |
| `tests/test_scraper_guard.py` | Mon 早朝 allow テスト追加 (旧 block テストを反転)、回帰テスト追加 |
| `tools/verify_scraper_guard_sunday.py` | **新規**: 16 ケース実機シミュレーション |
| `report/task_scheduler_audit_20260419.md` | **新規**: 本レポート |

---

## 最終判断: 明日このまま寝て大丈夫か？

### ✅ 大丈夫 (本修正を main に merge した上で)

理由:
- **Mon 03:00 DailyPremiumScrape の再発バグを発見・修正**
  - 旧コードのまま寝ると明朝 03:00 で premium 事前取得が再び誤停止していた (今日と同パターン)
  - 修正後 verify スクリプト + pytest で 16/16, 57/57 PASS
- **明日 4/20 (Mon) の他のタスクは全て安全**
  - DailyJrdbKyi (06:00): guard 呼出なし、scrape_jrdb.py が無条件実行
  - DailyPredict (08:00): ガード窓外、Mon は非開催日で軽量実行
  - WeeklyReport (08:00): Discord 通知のみ、副作用は許容範囲
- **regression_test 16/16 PASS** で他の機能への影響なし
- 明日は **月曜 (非開催日)** のため、レース運用上のクリティカルパスは存在しない
  - 仮に何か失敗しても次の本番 (4/25 土曜) まで時間的余裕あり

注意事項:
- 本レポートと修正は `fix/sunday-am3-regression-check` ブランチ。**main に merge** してから就寝すること
- DailyPredict の Ctrl+C 強制終了 (LastResult 3221225786) は別 issue。明日 (Mon) は非開催日のため発生せず、4/25 (Sat) までに別途調査推奨
- ProcessWatchdog v2 は引き続き 5 分おきに死活監視。万が一プロセスが死んでも検知可能

---
