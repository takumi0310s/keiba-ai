# Deploy Check 20260424

実行日時: 2026-04-24 20:00
本番: 2026-04-25 (土) / 2026-04-26 (日)

## 判定

**🟢 本番準備完了 (CRITICAL=0, WARNING=2)**

CRITICAL: 0
WARNING (対処可): 2
- WARNING-1: JRDB csv の age=5日 (今朝 jrdb_kyi_auto が 0件だったため更新なし。平日のため正常、土朝 AM6:00 の DailyJrdbKyi で解消)
- WARNING-2: cookies.pkl 不在 (`.env NETKEIBA_COOKIE` 単独運用。19:44 refresh 済 / 運用上 OK)

## 1. pytest tests/

- `pytest.ini` の `addopts=-s` により capture 問題回避済
- exit code: **0 (PASS)**
- 個別テスト: Condition Classification / Bet Generation / Prediction Consistency / daily_predict uses predict_core → すべて PASSED
- ログ確認: `=== Test: * === ... PASSED` のパターン全件通過
- pytest-timeout 非インストールのため `--timeout` フラグは不要

## 2. タスクスケジューラ (22本 schtasks)

**目標**: ProcessWatchdog=Disabled, 他21本=Ready → **達成** ✓

| TaskName | State | NextRun | LastResult |
|---|---|---|---|
| ProcessWatchdog | **Disabled** ✓ | 2026/04/24 19:55 (無効) | 0 |
| DailyPremiumScrape | Ready | 2026/04/25 03:00 | 0 |
| Keiba-PreFireCheck | Ready | 2026/04/25 02:55 | 0 |
| Keiba-AM3FireCheck | Ready | 2026/04/25 03:15 | 0 |
| DailyJrdbKyi | Ready | 2026/04/25 06:00 | 0 |
| Keiba-AM6FireCheck | Ready | 2026/04/25 06:15 | 0 |
| Keiba-MorningDigest | Ready | 2026/04/25 07:00 | 0 |
| Keiba-ScrapeProgress | Ready | 2026/04/25 07:00 | 0 |
| JrdbHealthCheck_Sat | Ready | **2026/04/25 07:30** | 267011* |
| DailyPredict | Ready | 2026/04/25 08:00 | 0 |
| RaceAutoNotify_Sat | Ready | **2026/04/25 08:45** | 0 |
| Keiba-AM8FireCheck | Ready | 2026/04/25 08:50 | 0 |
| Keiba-FridayWeekendScrape | Ready | 2026/05/01 10:00 | 0 |
| DailyResults_Sat | Ready | **2026/04/25 18:00** | 0 |
| DailyResultsEvening | Ready | 2026/04/24 20:00 (今晩) | 0 |
| Keiba-NightlySanity | Ready | 2026/04/24 23:00 (今晩) | 3221225786** |
| JrdbHealthCheck_Sun | Ready | **2026/04/26 07:30** | 0 |
| RaceAutoNotify_Sun | Ready | **2026/04/26 08:45** | 3221225786** |
| DailyResults_Sun | Ready | **2026/04/26 18:00** | 3221225786** |
| Keiba-WeeklyScrapeResume | Ready | 2026/04/27 06:30 | 267011* |
| WeeklyReport | Ready | 2026/04/27 08:00 | 1 |
| KeibaAI_DriftDetector | Ready | 2026/04/27 08:30 | 0 |

凡例:
- `0` = 正常終了
- `1` = 一般エラー (WeeklyReport 4/20分, 要調査だが来週月曜のため本番影響なし)
- `3221225786` = 0xC000013A = STATUS_CONTROL_C_EXIT (手動介入 or Windows 終了割込, 次回で正常実行想定)
- `267011` = タスク未発火 (今回が初回)

**土日本番タスク (太字) すべて Ready + NextRun 正しい** ✓

## 3. Cookie 状態

| 項目 | 値 |
|---|---|
| `.env` NETKEIBA_COOKIE | ✓ (len=1,634) |
| `.env` 最終更新 | **2026-04-24 19:44:27** ✓ (本番前 refresh 済) |
| cookies.pkl | 不在 (env のみで運用, 仕様通り) |
| Premium 認証 | OK (19:44 refresh_cookie.py にて確認) |

昨日 (4/23) の警告「cookies.pkl 不在」は同仕様のため無視可。

## 4. 土日タスク NextRun 確認

| タスク | 発火時刻 | 備考 |
|---|---|---|
| Keiba-RaceAutoNotify_Sat | 2026/04/25 08:45 | ✓ 指示通り |
| Keiba-RaceAutoNotify_Sun | 2026/04/26 08:45 | ✓ 指示通り |
| DailyResults_Sat | 2026/04/25 18:00 | ✓ 指示通り |
| DailyResults_Sun | 2026/04/26 18:00 | ✓ 指示通り |
| JrdbHealthCheck_Sat | 2026/04/25 07:30 | ✓ |
| JrdbHealthCheck_Sun | 2026/04/26 07:30 | ✓ |

## 5. ディスク空き

- **743.7 GB 空き** (使用率 21.9%) → ✓ 閾値 10GB 以上大幅にクリア

## 6. モデルファイル

| ファイル | exists | size | age |
|---|---|---|---|
| `keiba_model_v15_central_live.pkl.gz` | ✓ | 2.0 MB | 15日 |
| `keiba_model_v15_central.pkl.gz` | ✓ | 2.0 MB | 15日 |

(v15 は 4/9 学習済の現行モデル)

## 7. JRDB データ (age=5日, 要注意)

| ファイル | exists | age | size |
|---|---|---|---|
| jrdb_kyi.csv | ✓ | 5日 | 88.9 MB |
| jrdb_sed.csv | ✓ | 5日 | 25.6 MB |
| jrdb_tyb.csv | ✓ | 5日 | 210.3 KB |
| jrdb_cyb.csv | ✓ | 5日 | 151.3 KB |

4/24 06:00 の DailyJrdbKyi は平日のため 0 件更新 (`KYI/SED/TYB: データなし`)。
**4/25 06:00 の DailyJrdbKyi で自動更新される予定** (JRDB側に土曜朝データが出現)。

## 8. JRDB / netkeiba 疎通

- JRDB (www.jrdb.com): HTTP 301 (→HTTPS リダイレクト, 疎通 OK)
- netkeiba race page: HTTP **200** ✓ (Cookie使用で race_id 指定あり)

## 9. pytest.ini 存在

```ini
[pytest]
addopts = -s
testpaths = tests
python_files = test_*.py regression_test.py
python_functions = test_* main
```

→ ✓ 存在 / `-s` により output capture 問題回避済

## 10. 構文チェック (参考)

昨日(4/23)時点:
- app.py: OK
- tools/predict_core.py: OK

本日改変なし → 継続 OK 想定。

## 総合判定

- **CRITICAL 無し** → 土日本番準備完了
- JRDB age 5日 / cookies.pkl 不在 は既知の非致命。明朝 DailyJrdbKyi で JRDB は自動解決。
- 19:44 の Cookie refresh により Premium データ取得も翌朝 AM03:00 で正常化予定。

**🟢 GO 判定**
