# fire_check caller 監査結果

**作成**: 2026-05-05 夜 / 緊急 3 件 #2
**結論**: **大半は健全**、ただし 2 件のバグを発見・修正済

## 1. 監査対象

| TaskName | schtasks State | LastResult | 実 script |
|---|---|---|---|
| Keiba-PreFireCheck | Ready | 0 | `tools/pre_fire_check.py` (267 行、6 項目) |
| Keiba-AM3FireCheck | Ready | 0 | `tools/am3_fire_check.py` (185 行) |
| Keiba-AM6FireCheck | Ready | 0 | `tools/am6_fire_check.py` (49 行、common 経由) |
| Keiba-AM8FireCheck | Ready | 0 | `tools/am8_fire_check.py` (62 行→修正後 78 行) |

共通モジュール: `tools/fire_check_common.py` (244 行)

## 2. SCRAPER-GUARD caller 引数渡し監査

「fire_check 系の caller 引数渡し未確認 → 4/19 事故と同型リスク」 (UPDATE_INVENTORY §1) に対する監査結果。

### pre_fire_check.py L47 確認

```python
allowed = is_scraping_allowed(now=target, caller="daily_premium_scrape")
```

→ **`caller="daily_premium_scrape"` を明示的に渡している**。 4/19 事故 (caller 不在で SCRAPER-GUARD が誤停止) と同型ではない。

### am3/am6/am8 fire_check は SCRAPER-GUARD 不要

これら 3 種は **発火確認のみ** (ログ mtime + size + エラーキーワード判定)、URL fetch しない。
SCRAPER-GUARD は scrape 系にのみ作用するため、caller 引数自体が不要。

### 結論

**caller 引数渡しは健全**。 4/19 事故と同型のリスクは存在しない。

## 3. 動作確認 dry-run

### pre_fire_check (修正後)

```
PRE-FIRE-CHECK @ 2026-05-05T22:08:26
 [OK] SCRAPER-GUARD: ALLOW @ 2026-05-06 03:00 Wed (daily_premium_scrape 特例)
 [OK] Cookie: Cookie OK (1817 文字)
 [OK] Directories: 書き込み権限 OK (4 dirs)
 [OK] JRDB reachable: JRDB 疎通 OK (HTTP 200)
 [OK] Disk space: 空き 729.0 GB
 [OK] Task Scheduler: Ready, next=2026-05-06T03:00:00
OVERALL: OK
```

→ 6/6 全 OK。 5/6 水曜の AM3:00 DailyPremiumScrape は SCRAPER-GUARD `daily_premium_scrape` 特例で ALLOW、発火見込み。

### am3_fire_check

```json
{"status": "ok", "message": "平日 (非開催日) のため正常早期終了", "size": 339, "mtime": "2026-05-05T03:00:06"}
```

→ 5/5 火曜 AM3:00 発火、size 339B (No races found) で正常早期終了判定。

### am6_fire_check

```json
{"status": "ok", "message": "DailyJrdbKyi 正常発火", "size": 4721, "mtime": "2026-05-05T06:01:41"}
```

→ 5/5 火曜 AM6:00 発火、size 4721B、JRDB ダウンロード正常。

### am8_fire_check (修正後)

```json
{"status": "ok", "message": "DailyPredict: 平日 (非開催日) のため発火スキップ", "weekday": "Tuesday"}
```

→ 5/5 火曜は予測スキップ正常 (修正前は critical 誤判定だった)。

## 4. 発見・修正したバグ 2 件

### バグ 1: pre_fire_check.py UnicodeEncodeError (cp932)

**症状**: bash で実行すると `print(f" [{icon}] {name}: ...")` で `✓` (✓) を cp932 で出力できず Traceback。

**原因**: Windows の Python デフォルト stdout encoding が cp932 で、`✓ ⚠ ✗` 文字を含めない。

**修正**: `sys.stdout.reconfigure(encoding="utf-8", errors="replace")` を main() 冒頭に追加。さらに icon を `OK / WARN / NG` の ASCII に変更 (Discord 通知側 fire_check_common.py は既に ASCII 化済)。

```python
# Windows cp932 で ✓/⚠/✗ が UnicodeEncodeError になるため stdout を utf-8 に切替
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# icon を ASCII 化
if r["ok"]:
    icon = "OK"
elif r["severity"] == "warning":
    icon = "WARN"
else:
    icon = "NG"
```

### バグ 2: am8_fire_check.py 平日 critical 誤判定

**症状**: 平日 (Mon-Fri) で前日 logs/daily_predict.log が残ってると、`mtime < cfg.expected_time` で **critical 誤発報** (UPDATE_INVENTORY §3 の DailyPredict LastResult=1 5/4 月曜エラーの真因の一つ)。

**実例**: 5/5 火曜実行時、5/4 月曜の log mtime のため `mtime 2026-05-04T08:00 < expected 2026-05-05T08:00:00` で critical → Discord 誤通知。

**修正**: am8_fire_check.py に「平日 + 当日 CSV 未生成 → 早期 OK 返す」ロジック追加:

```python
if not is_weekend:
    csv_today = BASE / f"data/daily_predictions/{ymd}.csv"
    if not csv_today.exists():
        r = {
            "status": "ok",
            "message": "DailyPredict: 平日 (非開催日) のため発火スキップ",
            "csv": str(csv_today),
            "weekday": today.strftime("%A"),
        }
        # 通知 + return
```

これで火曜・水曜・木曜・金曜の false positive 抑止。

## 5. 影響評価

- 5/9 (土) は週末扱いで `min_rows=30` の通常判定が走る → 修正の影響なし、本番影響ゼロ
- 5/6 (水) - 5/8 (金) の平日 false positive が消える → Discord ノイズ減
- pre_fire_check は 5/6 水曜 02:55 発火、修正後の icon=OK で正常出力

## 6. 5/6 以降の追加 TODO

- [ ] 4/19 incident report (`report/incident_impact_analysis_20260419.md`) との相互リンク追加
- [ ] fire_check_common.py の `_check_log` で「mtime が前日のままで CSV 未生成 → 平日扱い OK」を共通化 (am3 にも適用余地)
- [ ] `--silent` 強制で平日通知抑止 (現状は OK 通知も Discord に流れる)

## 7. 結論

**fire_check 4 種は監査の結果、設計健全**。 4/19 事故と同型のリスクは存在しない。 dry-run で発見した cp932 バグと am8 平日誤判定は本日修正完了。
