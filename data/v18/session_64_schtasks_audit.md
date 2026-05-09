# Session #64 A: schtasks audit (5/9 12:35)

## 1. 全 keiba schtasks (49 件)

下表は 5/9 12:35 時点で `\Keiba-*` / `\keiba-ai\*` / `\ProcessWatchdog` / `\KeibaAI_DriftDetector` を全列挙した結果。

| Task | Next Run | Status | 重複? |
|------|----------|--------|------|
| \ProcessWatchdog | 2026/05/09 12:43:00 | Ready | ★ ROOT CAUSE ★ (5 分間隔 fire) |
| \keiba-ai\DailyPredict | 2026/05/10 8:00 | Ready | 単独 |
| \keiba-ai\RaceAutoNotify_Sat | 2026/05/16 8:45 | Ready | 単独 |
| \keiba-ai\RaceAutoNotify_Sun | 2026/05/10 8:45 | Ready | 単独 |
| \Keiba-MultiStagePredict_Race11_1450_Sat | 2026/05/09 14:50 | Ready | 単独 |
| \Keiba-MultiStagePredict_Race12_1545_Sat | 2026/05/09 15:45 | Ready | 単独 |
| Session #61 で追加した 9 件 (VoteCandidates / Verdict×6 / Cumulative / Summary) | 2026/05/09 各時刻 | Ready | 単独、 ONCE |
| (省略 — 全 49 件は schtasks /Query /FO LIST 参照) | | | |

→ schtasks レベルでの重複は **なし**。 daily_predict / race_auto_notify は schtasks 上は 1 件ずつ。

## 2. 過去 24h の log fire 履歴

`logs/daily_predict_watchdog_restart_20260509.log` (104 KB):
- 11:48, 11:53, 11:58, 12:03, 12:08, 12:13, 12:18, 12:23, 12:28, 12:33 — **5 分間隔** で daily_predict.py 再起動
- 各 restart で「整形済み買い目通知送信: 8 messages」を記録 (Discord push 8 通)

`logs/race_auto_notify_watchdog_restart_20260509.log` (43 KB):
- 同 5 分間隔で race_auto_notify.py 再起動 (こちらは「Found: 0 races」で no-op、 Discord push なし)

`logs/watchdog_v2_20260509.log`:
- 元締め。 各 5 分の `[watchdog_v2] target: MISSING (alive=False, stale=False, mtime=...)` ログ
- → ProcessWatchdog schtask (5 分 interval) → process_watchdog_v2.py が daily_predict / race_auto_notify を MISSING 判定 → 不要再起動 → spam

## 3. 累積影響

- 11:48 〜 12:33 (45 分) で daily_predict 10 回再起動 → 80 messages 送信
- 12 件の lingering race_auto_notify プロセス が累積 (5/9 12:35 時点で kill 済)

## 4. 結論

schtasks レベルは健全。 真の原因は `tools/process_watchdog_v2.py` の logic bug (Session #64 B 参照)。

main 不変 (dev/training-poc 専用 commit のみ)。 V15 投資保護 維持。
