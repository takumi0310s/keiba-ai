# Session #65 A: R 別 1h 前 predict schedule (5/9)

## 1. 全 R 発走時刻 推定 (静的、 11R anchor base)

netkeiba scrape は server block (Session #62/#63) リスクのため、 既知 11R anchor から JRA 標準 ~25-30 分 interval で逆算/順算。 当日朝 race_auto_notify.py / multi_stage の発走時刻 log と照合可能だが本 doc は静的 list で進める。

### anchor (実) — Session #61 / #63 で確定済
- 京都 11R 京都新聞杯 G2: **15:30**
- 東京 11R エプソムカップ G3: **15:45**
- 新潟 11R 駿風 S OP: **15:20**

### 全 36 R 推定 schedule (1h 前 = predict timing)

| 場 | R | 発走 (推定) | 1h 前 predict timing | 既過 / 残 |
|----|---|------|------|------|
| 京都 | 1 | 09:55 | 08:55 | 既過 |
| 京都 | 2 | 10:25 | 09:25 | 既過 |
| 京都 | 3 | 10:55 | 09:55 | 既過 |
| 京都 | 4 | 11:30 | 10:30 | 既過 |
| 京都 | 5 | 12:00 | 11:00 | 既過 |
| 京都 | 6 | 12:35 | 11:35 | 既過 |
| 京都 | 7 | 13:30 | 12:30 | 既過 |
| 京都 | 8 | 14:00 | 13:00 | **残** |
| 京都 | 9 | 14:25 | 13:25 | **残** |
| 京都 | 10 | 14:55 | 13:55 | **残** |
| 京都 | 11 | **15:30** ★ | 14:30 | **残** |
| 京都 | 12 | 16:00 | 15:00 | **残** |
| 東京 | 1 | 10:10 | 09:10 | 既過 |
| 東京 | 2 | 10:40 | 09:40 | 既過 |
| 東京 | 3 | 11:15 | 10:15 | 既過 |
| 東京 | 4 | 11:50 | 10:50 | 既過 |
| 東京 | 5 | 12:25 | 11:25 | 既過 |
| 東京 | 6 | 13:00 | 12:00 | 既過 |
| 東京 | 7 | 13:45 | 12:45 | 既過 |
| 東京 | 8 | 14:15 | 13:15 | **残** |
| 東京 | 9 | 14:45 | 13:45 | **残** |
| 東京 | 10 | 15:15 | 14:15 | **残** |
| 東京 | 11 | **15:45** ★ | 14:45 | **残** |
| 東京 | 12 | 16:25 | 15:25 | **残** |
| 新潟 | 2 | 10:30 | 09:30 | 既過 |
| 新潟 | 3 | 11:00 | 10:00 | 既過 |
| 新潟 | 5 | 12:00 | 11:00 | 既過 |
| 新潟 | 6 | 12:30 | 11:30 | 既過 |
| 新潟 | 7 | 13:00 | 12:00 | 既過 |
| 新潟 | 8 | 13:30 | 12:30 | 既過 |
| 新潟 | 9 | 14:00 | 13:00 | **残** |
| 新潟 | 10 | 14:30 | 13:30 | **残** |
| 新潟 | 11 | **15:20** ★ | 14:20 | **残** |
| 新潟 | 12 | **16:10** ★★ V15 投資 | 15:10 | **残** |

注: 新潟 1R / 4R は daily_predictions に存在せず (障害 or 競走除外推定)、 schedule 対象外。

## 2. 残 R = 13 件 (5/9 12:55 時点)

13:00 以降 fire できる predict timing が **13 件**。 順次 30 分毎 watchdog で個別 R 予測 + Discord 通知。

## 3. 推奨 schedule 戦略: 30 分毎 watchdog (1 件)

### なぜ ONCE × 13 件 ではなく watchdog × 1 件か

| 比較 | ONCE × 13 件 | 30分毎 watchdog × 1 件 |
|------|-------------|----------------------|
| schtasks 数 | 13 件 (既存 49 → 62) | 1 件 (既存 49 → 50) |
| 失敗時影響 | 該当 R のみ | 全部停止リスクだが kill-switch で即停止可 |
| dedup | timing 揃えるだけ | cache JSON で重複防止必須 |
| 発走時刻ずれ吸収 | 不可 (固定 timing) | 可 (1h ± window で fire) |
| 実装複雑性 | schtasks 13 件登録のみ | watchdog logic + dedup |

→ **watchdog × 1 件 を採用** (schtasks 汚染を最小化、 発走時刻ずれにも頑健)。

### watchdog 仕様

- 名: `Keiba-PreRacePredict_Watchdog_5_9`
- Schedule: 13:00 から 30 分毎 (`/SC MINUTE /MO 30 /DU 0700:00`)、 16:30 で自然終了
- Action: `wscript.exe silent_runner.vbs pre_race_predict_runner.bat --check-next-1h`
- logic: 現在時刻 +60min window 内の R を抽出、 cache 未記録なら予測実行 + Discord 通知 + cache 記録
- kill-switch: `data/v18/pre_race_predict.kill` を touch で即 no-op (Session #64 patten 踏襲)

### dedup 設計

- `data/v18/pre_race_predict_cache_5_9.json`: `{race_id: timestamp}` 記録
- 既記録 race_id は skip。 watchdog が複数回 fire しても 1 R 1 通

### 万が一 watchdog NG → fallback ONCE 戦略

`/SC MINUTE` が Win11 で拒否された場合、 残 13 件の predict timing で ONCE schtasks を 13 件登録。 doc 内 fallback section に schtasks /Create 一括 PowerShell snippet を予め用意。

## 4. 干渉禁止確認

- 既存 schtasks 49 件 (Session #61 含む 9 件) 触らない
- daily_predict.py / race_auto_notify.py を 絶対 trigger しない (Session #64 spam 再発防止)
- ProcessWatchdog kill-switch (`process_watchdog_v2.kill`) 削除しない
- V15 投資方針 不変 (新潟 12R ¥700)
- Stage 2 予測は学習用、 投票推奨ではない
