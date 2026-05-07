# Session #40 B: 運用安定化 (alert + runbook + dashboard + logs)

**作成**: 2026-05-07 (Session #40 B)
**目的**: 5/9 投資 + 長期運用の堅牢性 UP

---

## 1. B1 — Discord 3 channel routing wrapper

### 1.1 ファイル

`tools/discord_routing.py` (新規、 約 90 行)

### 1.2 channel 設計

| channel | webhook env (priority) | 用途 |
|---------|----------------------|------|
| `investments` | `DISCORD_WEBHOOK_INVESTMENTS` → BETS → URL | 5/9 当日 投資進捗 / 結果 (新設) |
| `alerts` | `DISCORD_WEBHOOK_ALERTS` → UPDATES → URL | 失敗・障害 (新設) |
| `bets` | `DISCORD_WEBHOOK_BETS` → URL | 平常 race_auto_notify 買い目 (既存) |
| `updates` | `DISCORD_WEBHOOK_UPDATES` → URL | 通常進捗 (既存) |

### 1.3 既存 notify.py との互換性

- `notify.py` (bets / updates 2 channel) は完全に不変
- `discord_routing.py` は既存を内部で呼び出し、 拡張 (investments / alerts) だけ追加
- → V15 production の通知経路 (notify_done.py / race_auto_notify.py) は完全不変

### 1.4 .env 追加 (任意)

```env
# 既存 (必須)
DISCORD_WEBHOOK_BETS=https://discord.com/api/webhooks/.../...
DISCORD_WEBHOOK_UPDATES=https://discord.com/api/webhooks/.../...
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/.../...

# 新規 (任意、 設定無しなら fallback)
DISCORD_WEBHOOK_INVESTMENTS=https://discord.com/api/webhooks/.../...
DISCORD_WEBHOOK_ALERTS=https://discord.com/api/webhooks/.../...
```

### 1.5 利用例

```bash
# 5/9 投資完了通知
python tools/discord_routing.py --title "5/9 投資完了" \
  --body "12R 1勝 3R 投票, 投資 2,100円, 期待 +400-1300円" \
  --channel investments --color green

# Cookie 失効アラート
python tools/discord_routing.py --title "Cookie 失効" \
  --body "refresh_cookie 失敗、 手動更新必要" \
  --channel alerts --color red
```

---

## 2. B2 — emergency runbook 詳細版

詳細: [`docs/EMERGENCY_RUNBOOK_5_9_DETAILED.md`](../docs/EMERGENCY_RUNBOOK_5_9_DETAILED.md) (15 シナリオ)

### 2.1 シナリオ一覧

| # | シナリオ | 復旧時間 | 5/9 case 推奨 |
|---|----------|---------|--------------|
| S01 | Cookie 切れ + refresh fail | 5-10 分 | 手動 refresh |
| S02 | JRDB 全 retry 失敗 | 15-30 分 | V15 は 0 fallback で動作可 |
| S03 | 馬体重取得失敗 (一部開催) | 10-15 分 | 該当 race 投資除外 |
| S04 | Discord webhook 死亡 | 5 分 | 通知無くても投票可 |
| S05 | PAT サーバー障害 | JRA 次第 | 投票見送り (損失回避) |
| S06 | ProcessWatchdog 誤発火 | 10-15 分 | resume 再開 |
| S07 | 落雷・停電 | 数時間 | 復旧不能なら見送り |
| S08 | NW障害 | 数分〜数時間 | スマホ A-PAT 切替 |
| S09 | PC ハング | 5-10 分 | 強制再起動 |
| S10 | PowerShell 起動不能 | 10-30 分 | CMD 代替 |
| S11 | python crash | 10-30 分 | 該当 race 個別 retry |
| S12 | git 衝突 | 5-15 分 | 投資直前は不要 |
| S13 | 学習 data 破損 | 30 分〜 | V15 production には不要 |
| S14 | predict_core 異常終了 | 10 分/race | 該当 R 見送り |
| S15 | 全 system fallback | 30-60 分 | manual override or 見送り |

### 2.2 5/9 当日 朝 全体フロー (時系列、 詳細 runbook 参照)

```
05:00 PC ON
06:00 final_health_check (自動)
06:30 health 確認
07:00 morning_digest
08:00 daily_predict (V15 全レース)
08:45 race_auto_notify (戦略⑦ + 案B改 → bets/investments)
09:00 候補確定
09:30 PAT login
10:00- 投票
18:00 結果照合
20:30 振り返り
```

### 2.3 撤退判定 (5/9 単日)

| 累計 (1 日終了) | 翌日 |
|--------------|------|
| ≥ -2,100 円 | 通常運用 |
| -2,100 〜 -10,000 円 | 翌週投資停止 |
| -10,000 〜 -50,000 円 | 全停止 |
| < -50,000 円 | 撤退 |

5/9 max loss = -2,100円 想定 → **5/9 中の撤退発生せず**

---

## 3. B3 — リアルタイム監視 CLI

### 3.1 ファイル

`tools/realtime_monitor.py` (新規、 約 130 行)

### 3.2 監視項目 (5 秒 polling)

```
======================================================================
realtime_monitor  (2026-05-07 23:36:06)
======================================================================
  cumulative: rows=496, profit=-28,360 JPY, last_date=20260503
  retire margin: +21,640 JPY  (line=-50,000)
  daily_predict: daily_predict_watchdog_restart_20260507.log (mtime: 338 min ago) > ...
  JRDB: latest: 20260503
  Cookie: no cookies.json
  schtasks (Keiba-*): ...@2026/05/08 3:15:00, ...@2026/05/08 6:15:00
======================================================================
```

### 3.3 利用法 (5/9 当日)

```bash
# 朝 06:00 〜 終日 PC で開きっぱなし
python tools/realtime_monitor.py --interval 5
```

→ 5/9 朝 投資判断時に最新状況一覧、 既存 dashboard.py (Streamlit) と独立。

### 3.4 V15 production 不変保証

- read-only monitor (data/cumulative_results.csv / logs/ / data/jrdb/ を読むのみ)
- production の predict_core / daily_predict / V15 model 完全不変

---

## 4. B4 — logs cleanup (30 日 archive)

### 4.1 ファイル

`tools/logs_cleanup.py` (新規、 約 100 行)

### 4.2 機能

- `logs/` 配下を walk
- mtime > 30 日前 のファイルを `logs/archive/{YYYYMM}/` に移動 + gzip 圧縮
- 移動 log を `data/logs_cleanup_history.json` に記録
- `--dry-run` で試行、 `--days 60` で日数調整

### 4.3 動作確認

```
$ python tools/logs_cleanup.py --dry-run
  [DRY] logs\premium_scrape_20260405.log -> logs\archive\202604\premium_scrape_20260405.log.gz
  [DRY] logs\premium_scrape_20260406.log -> ...
logs_cleanup: archived=4, skipped=0, saved=0.0 MB (dry run)
```

### 4.4 schtasks 推奨追加

```cmd
schtasks /Create /TN "Keiba-LogsCleanup" ^
    /TR "powershell -ExecutionPolicy Bypass -Command \"cd C:\Users\takum\keiba-ai; python tools\logs_cleanup.py\"" ^
    /SC WEEKLY /D MON /ST 04:00 /F
```

→ 月曜 04:00 自動実行、 logs 容量を一定保つ。

---

## 5. 5/9 V15 投資保護 final 確認 (B 領域)

✅ predict_core.py 完全不変
✅ daily_predict.py 完全不変
✅ V15 model file 完全不変
✅ 既存 notify.py / discord_notifier.py 完全不変
✅ 既存 dashboard.py 完全不変
✅ schtasks 既存 task 完全不変 (新規追加 推奨のみ)
✅ 新規 tool: discord_routing / realtime_monitor / logs_cleanup (read-only or logs のみ)

→ **5/9 朝 V15 daily_predict 完全同一動作 保証**

---

## 6. 結論

✅ B1: 3 channel routing (investments / alerts / updates / bets)
✅ B2: 15 シナリオ runbook (`EMERGENCY_RUNBOOK_5_9_DETAILED.md`)
✅ B3: リアルタイム監視 CLI (5 秒 polling)
✅ B4: logs cleanup (30 日 archive + gzip)
✅ B5: 統合 doc (本ファイル)

→ **5/9 投資 + 長期運用 堅牢性 UP**

---

**Session #40 B 完了**
