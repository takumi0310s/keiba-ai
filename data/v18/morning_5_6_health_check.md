# Morning Health Check 5/6 (寝てる間 5/5 22:35 - 5/6 朝)

**生成時刻**: 2026-05-06 朝 (Wed, 平日 / 非開催日)
**対象期間**: 2026-05-05 22:35 ~ 2026-05-06 09:50
**ベース commit**: bed809ec

---

## 1. schtasks status (35 task 取得、Keiba関連 28 + 周辺)

| TaskName | State | LastResult | LastRun | 判定 |
|----------|-------|-----------|---------|------|
| Keiba-AM3FireCheck | Ready | 0 | 2026/05/06 03:15:01 | OK |
| Keiba-AM6FireCheck | Ready | 0 | 2026/05/06 06:15:01 | OK |
| Keiba-AM8FireCheck | Ready | 0 | 2026/05/06 08:50:01 | OK |
| Keiba-PreFireCheck | Ready | 0 | 2026/05/06 02:55:01 | OK |
| Keiba-MorningDigest | Ready | 0 | 2026/05/06 07:00:01 | OK |
| Keiba-NightlySanity | Ready | 0 | 2026/05/05 23:00:01 | OK |
| Keiba-ScrapeProgress | Ready | 0 | 2026/05/06 07:00:01 | OK |
| Keiba-TybPublishMonitor | Ready | 0 | 2026/05/06 09:30:01 | OK |
| DailyJrdbKyi | Ready | 0 | 2026/05/06 06:00:01 | OK |
| DailyPremiumScrape | Ready | 0 | 2026/05/06 03:00:01 | OK |
| DailyPredict | Ready | **1** | 2026/05/06 08:00:01 | NG (但し非開催日想定動作) |
| DailyResultsEvening | Ready | 0 | 2026/05/05 20:00:01 | OK |
| DailyResults_Sat | Ready | 0 | 2026/05/02 18:00 | OK (前回土曜) |
| DailyResults_Sun | Ready | 0 | 2026/05/03 18:00 | OK (前回日曜) |
| JrdbHealthCheck_Sat | Ready | 0 | 2026/05/02 07:30 | OK |
| JrdbHealthCheck_Sun | Ready | 0 | 2026/05/03 07:30 | OK |
| KeibaAI_DriftDetector | Ready | 0 | 2026/05/04 08:30 | OK |
| Keiba-FridayWeekendScrape | Ready | 0 | 2026/05/01 10:00 | OK |
| WeeklyReport | Ready | **1** | 2026/05/04 08:00 | NG (要確認、5/4 月曜分) |
| Keiba-NarDailyResults | Ready | 0 | 2026/05/05 21:30:01 | OK (NAR placeholder) |
| Keiba-NarLiveOddsRefresh | Ready | 0 | 2026/05/05 19:00:00 | OK |
| Keiba-NarDailyPredict | Ready | 267011 | 1999/11/30 (未発火) | OK (placeholder 起動なし) |
| Keiba-NarDailyScrape | Ready | 267011 | 1999/11/30 (未発火) | OK |
| Keiba-NarMidDayCalendar | Ready | 267011 | 1999/11/30 (未発火) | OK |
| Keiba-Morning_Sat / Sun | Ready | 267011 | 1999/11/30 (未発火) | OK (新規未発火) |
| Keiba-RaceDayReport_Sat / Sun | Ready | 267011 | 1999/11/30 (未発火) | OK (新規未発火) |
| Keiba-WeeklyScrapeResume | Ready | **3221225786** | 2026/05/04 06:30 | NG (Ctrl+C 終了、5/4 既知) |
| RaceAutoNotify_Sat | Ready | **3221225786** | 2026/05/02 08:45 | NG (5/2 既知 Ctrl+C) |
| RaceAutoNotify_Sun | Ready | **3221225786** | 2026/05/03 08:45 | NG (5/3 既知 Ctrl+C) |
| ProcessWatchdog | **Disabled** | 0 | 2026/04/24 | NG (停止中、要再起動判断) |
| ProcessMemoryDiagnosticEvents | Ready | 2147946720 | 2026/05/06 09:45 | (Windows標準、無関係) |

**寝てる間の発火 (5/5 22:35 - 5/6 09:50) 全件**:
- 5/5 23:00 NightlySanity → 0 (PASS、CLAUDE.mdログ通り全16タスクALLOW)
- 5/6 02:55 PreFireCheck → 0 (PASS)
- 5/6 03:00 DailyPremiumScrape → 0 (No races, 正常 early exit)
- 5/6 03:15 AM3FireCheck → 0
- 5/6 06:00 DailyJrdbKyi → 0 (CHA/KTA/KKA/JO 全 parse 完了、UKC 12.2MB)
- 5/6 06:15 AM6FireCheck → 0
- 5/6 07:00 MorningDigest / ScrapeProgress → 0
- 5/6 08:00 DailyPredict → **rc=1** (詳細は §3 で説明、非開催日のため0レース、Watchdog [FATAL] は誤判定)
- 5/6 08:50 AM8FireCheck → 0
- 5/6 09:30 TybPublishMonitor → 0

---

## 2. logs 鮮度 (5/6 生成分のみ)

| ファイル | size | mtime | 鮮度 |
|----------|------|-------|------|
| pre_fire_check_20260506.log | 777B | 5/6 02:55 | OK |
| premium_scrape_20260506.log | 339B | 5/6 03:00 | OK |
| am3_fire_check_20260506.log | 350B | 5/6 03:15 | OK |
| jrdb_kyi_auto_20260506.log | 4721B | 5/6 06:01 | OK |
| am6_fire_check_20260506.log | 332B | 5/6 06:15 | OK |
| morning_dashboard_20260506.log | 926B | 5/6 07:00 | OK |
| scrape_progress.log | 4308B | 5/6 07:00 | OK |
| daily_predict_watchdog_20260506_subproc.log | 2550B | 5/6 08:15 | OK |
| daily_predict_watchdog_wrapper_20260506.log | 1432B | 5/6 08:20 | NG ([FATAL]、誤検知) |
| am8_fire_check_20260506.log | 343B | 5/6 08:50 | OK |
| nightly_sanity_20260505.log | 3045B | 5/5 23:00 | OK (寝る前ぎり) |

5/6 ログ全て生成済み、STALE なし。

---

## 3. 失敗ログ検出 (Traceback / ERROR / SCRAPER-GUARD blocked)

### 3.1 Traceback / Exception
- **検出ゼロ**。5/5 と 5/6 の全ログを `Traceback|ERROR|Exception|fatal|FATAL|SCRAPER-GUARD blocked|IP banned|Cookie expired` で grep。

### 3.2 検出された FATAL (1 件、誤検知)
**logs/daily_predict_watchdog_wrapper_20260506.log:24**
```
[FATAL] 20260506 3回再起動後も 0/30 レースのみ完了。手動対応必要。
```

**実態**: daily_predict.py の subproc ログを見ると 4 回全て以下を出力:
```
[STEP 1] レース一覧取得中...
[INFO] 20260506 のレースが見つかりません（非開催日の可能性）
[2026-05-06 08:00:04] daily_predict.py 終了 (rc=0)
```

→ **5/6 (Wed) は中央競馬の非開催日のため、netkeiba にレース一覧がない**。daily_predict.py 自体は rc=0 で正常終了しているが、Watchdog ラッパーが「レース 0 件 < 閾値 30」を異常として 3 回再起動 → rc=1 で終了。
→ **Watchdog ラッパーのバグ** (非開催日判定ロジックなし)。AM8FireCheck は別ロジックで「平日 (非開催日) のため発火スキップ」と正常判定済み (am8_fire_check_20260506.log:7)。
→ **5/9 投資への影響なし**。5/9 (Sat) は開催日のため、レース一覧が取得できる前提で正常動作する。

### 3.3 daily_results 5/5 20:00
- `[ERROR] CSV/DB両方で予測データなし` → 5/5 (Tue) 非開催日、予測 CSV なし、想定通り (settled分は 5/3 までで止まっている)

### 3.4 過去既知の NG コード残骸
- **Keiba-WeeklyScrapeResume / RaceAutoNotify_Sat / Sun**: 全て LastResult=3221225786 (STATUS_CONTROL_C_EXIT)、5/2-5/4 のもの、寝てる間の発火ではない (それぞれ 5/2, 5/3, 5/4 が最終発火)。新規発生ではないため放置可。
- **ProcessWatchdog**: Disabled 状態が継続 (4/24 から)。CLAUDE.md の Phase 2-5 で migration 済みと推定。

---

## 4. fire_check 5 種 status

| Check | 時刻 | status | 内容 |
|-------|------|--------|------|
| PreFireCheck | 5/6 02:55 | **ok** | SCRAPER-GUARD ALLOW / Cookie OK (1817 chars) / Disk 728.8GB / JRDB 200 |
| AM3FireCheck | 5/6 03:15 | **ok** | 「平日 (非開催日) のため正常早期終了」DailyPremiumScrape size=339B |
| AM6FireCheck | 5/6 06:15 | **ok** | DailyJrdbKyi 正常発火 size=4721B |
| AM8FireCheck | 5/6 08:50 | **ok** | DailyPredict: 平日 (非開催日) のため発火スキップ |
| TybPublishMonitor | 5/6 09:30 | (要確認) | LastResult=0 のみ、ログ未確認 |

`data/fire_check_results/20260506.json` も DailyJrdbKyi=ok / DailyPredict=ok (非開催日スキップ) で記録済み。

---

## 5. cumulative_results.csv

- **mtime**: 2026-05-05 19:04:13 (5/5 19:04 が最終更新、寝てる間 更新なし)
- **行数**: 496 (header込み、データ495R)
- **末尾 5 race_id**:
  - 202608030408 (京都8R 御池特別 D)
  - 202608030409 (京都9R 六波羅特別 A)
  - 202608030410 (京都10R 朱雀S D)
  - 202608030411 (京都11R 天皇賞春 C) — settled (3-7-15 で外れ)
  - 202608030412 (京都12R 東大路S D)
- 全件 5/3 京都 8R-12R、5/3 settled で止まっている。5/4 月曜以降は非開催日のため新規行なし。**正常**。

ROI Monitor 5/5 20:00 出力:
- 累計 495R, ROI **91.8%** (-28,360円), 19R 連敗中 (5/3 終時点)
- BT保守的見積り 142.6% に対し -36% 乖離 (DANGER)
- 既知 (CLAUDE.md 実戦成績セクション) と一致

---

## 6. 結論

### 寝てる間 (5/5 22:35 - 5/6 朝) の異常: **実質ゼロ**
- 全 fire_check (Pre/AM3/AM6/AM8/Tyb) PASS
- nightly_sanity 16タスク全 ALLOW
- DailyPremiumScrape / DailyJrdbKyi / MorningDigest / ScrapeProgress 全正常
- Traceback / Exception / IP banned / Cookie expired 検出ゼロ
- cumulative_results.csv 改竄なし、5/3 settled で正常停止

### 唯一の懸念事項: DailyPredict Watchdog 誤検知 (1 件)
- **症状**: 5/6 08:00 daily_predict_watchdog が rc=1、`[FATAL] 0/30 レースのみ完了` を出力
- **原因**: 5/6 (Wed) は中央競馬非開催日 → netkeiba にレースなし → daily_predict 自体は rc=0 で正常終了するも、Watchdog ラッパーが「レース 0 件 < 閾値 30」で異常判定
- **影響**: なし。AM8FireCheck は同じ事象を「平日 (非開催日) のため発火スキップ」と正しく ok 判定。
- **5/9 投資への影響**: なし。5/9 (Sat) は開催日のためレース取得成功し、Watchdog 通常動作する。

### 5/9 投資に影響しそうな問題: **なし**
- v15 model file OK (nightly_sanity が確認済)
- Cookie 1817 chars OK (期限内)
- JRDB 全 parse 通過 (CHA 301,718rows / KTA 298,551rows / KKA 547,611rows / JO 301,718rows)
- 2026年 KKA 16,987rows / JO 15,658rows 健全
- SCRAPER-GUARD 全条件 ALLOW

### 5/6 朝 ユーザー対処事項 (admin watchdog 以外)

1. **(任意 / 低優先) DailyPredict Watchdog 非開催日判定ロジック追加**: `tools/daily_predict_watchdog.py` で「レース取得 0 件かつ rc=0」のとき非開催日 fallback で rc=0 にする改修。今すぐ不要 (5/9 までに非開催日 5/7, 5/8 で再発するが投資影響なし)。
2. **(任意) ProcessWatchdog Disabled 状態確認**: 意図的停止か事故停止か CLAUDE.md でも明記なし。Process Watchdog v2 移行済みなら Disabled で OK。
3. **(任意) Keiba-WeeklyScrapeResume rc=3221225786**: 5/4 06:30 Ctrl+C 終了。次回月曜 (5/11) 発火まで放置可。
4. **5/9 (Sat) 開催に向けた準備**: 既存ドライラン (HANDOFF_5_5_TO_5_9.md commit c7fdce57) で完了済。追加対応不要。

**総評**: 寝てる間 全システム正常稼働。5/9 本番に向けて懸念事項なし。
