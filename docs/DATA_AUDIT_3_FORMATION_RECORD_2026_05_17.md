# data-audit-3: 投票実 formation record 完全性 audit (2026-05-17)

## 0. 結論

- **真の formation record 完全性**: ★ **記録なし** ★ (race-time 実通知 formation は 永久喪失)
- **保存されているのは**: AM 8:00 morning prediction (`daily_predict.py`) で生成された formation のみ
- **記録 source**: `daily_predictions/YYYYMMDD.csv` (10 日分) + Streamlit local DB `keiba_predictions.db / race_results` (一部) → 統合先 `data/cumulative_results.csv` の `trio_bets_str` カラム
- **race_notify_log**: `20260517.json` 1 日 のみ存在、 内容 = race_id / channel / skip_reason のみ (formation 情報 ゼロ)
- **race_auto_notify.py** (race-15min 自動通知): 独立予測 → Discord 送信 → ★ formation 永久 不揮発化 されず ★
- **実投票 hit rate 計算 可否**: ★ 不可 ★ (AM 8:00 morning prediction の hit rate のみ計算可能、 race-time 再予測 (live odds / JRDB live / TYB / 直前天候 込み) と一致する保証なし)

---

## 1. record source 全探索

| source | 存在 | 期間 | formation 記録? | 完全性 |
|--------|------|------|----------------|--------|
| (a) `data/race_notify_log/YYYYMMDD.json` | YES (1 日のみ) | 5/17 のみ | ★ NO ★ (race_id/channel/skip のみ、 formation 情報 なし) | 0% |
| (b) Discord 通知 履歴 (local 保存) | NO | — | — | — |
| (c) `logs/race_auto_notify_*.log` (15 日分) | YES | 4/5-5/17 | ★ NO ★ (`Notified: race_name [cond] trio 7点` のみ、 馬番 formation 不明) | 0% |
| (d) `data/cumulative_results.csv` の `trio_bets_str` カラム | YES | 4/1-5/17 全 456 R | ★ AM 8:00 prediction のみ ★ (race-time formation ではない) | 100% (但し morning) |
| (e) `data/daily_predictions/YYYYMMDD.csv` (`trio_bets` カラム) | YES (10 日分) | 4/11, 4/12, 4/18, 4/19, 4/25, 4/26, 5/9, 5/10, 5/16, 5/17 | YES (morning prediction) | 部分 (10/19 日) |
| (f) Streamlit local DB `keiba_predictions.db / race_results / trio_bets` | YES | 4/4, 4/5, 4/12 (partial) | YES (Streamlit 経由 morning prediction) | 部分 |

---

## 2. 期間別 record audit

| 期間 | N races (cumulative) | trio_bets_str fill | top1_num fill | source 推定 |
|------|---------------------|-------------------|--------------|------------|
| 4/1-4/26 | 255 | 100% (255/255) | **0%** (0/255) | daily_predictions CSV (10 日分) + Streamlit DB (4/4, 4/5) + 5/6 backfill commit `66c78e9e` |
| 4/27-5/9 | 100 | 100% (100/100) | **0%** (0/100) | daily_predictions CSV (5/9 のみ) + DB/backfill |
| 5/10-5/15 | 34 | 100% (34/34) | **0%** (0/34) | daily_predictions/20260510.csv (5/10 のみ、 5/11-5/15 平日 開催なし) |
| 5/16 | 34 | 100% (34/34) | **100%** (34/34) | daily_predictions/20260516.csv (新 schema 適用日) |
| 5/17 | 33 | 100% (33/33) | **100%** (33/33) | daily_predictions/20260517.csv |

注: top1_num NaN = 旧 schema 期 (race_id だけで結合した記録)。 trio_bets_str は形式上 入っているが 結合元に top1 が無い (4/27-5/15 共通 schema 不一致)。

---

## 3. ★ verdict (read-only audit) ★

### 3-1. 実投票 (Discord 通知) formation = ★ 永久喪失 ★

`tools/race_auto_notify.py` の動作:

```python
# tools/race_auto_notify.py 300, 352, 354
bets = generate_trio_bets(df)
...
title, msg, color = build_rich_bet_message(df, ..., bets, ...)
send_discord(title, msg, color=color, channel="bets")   # ← formation 送信
_p0_5_notify_log(race_id, race_name, ..., channel='bets', ...)  # ← formation 含まない
```

→ Discord 送信時 の formation は どこにも persist されていない (`_p0_5_notify_log` は race_id/channel/skip_reason のみ)。

### 3-2. cumulative_results.csv の `trio_bets_str` = AM 8:00 morning prediction

`tools/daily_results.py` の動作:

```python
# tools/daily_results.py 354
'trio_bets': row.get('trio_bets', ''),   # ← daily_predictions CSV から読込
...
# tools/daily_results.py 597
'trio_bets_str': bets_str,   # ← cumulative に書込
```

5/16 実測検証: `daily_predictions/20260516.csv['trio_bets']` と `cumulative['trio_bets_str']` は **34/34 一致**。 つまり cumulative は AM 8:00 morning prediction を そのまま継承。

### 3-3. AM 8:00 vs race-time の差

race_auto_notify.py は race -15min に 独立予測 を実行 (live odds + JRDB live (KYI/TYB) + 直前天候/track condition + 馬体重)。 morning predict は AM 8:00 時点 (基本データのみ)。 → ★ TOP3 が変わる可能性 高 ★。 但し定量比較は 本 audit 範囲外 (もう一度 race-time 予測を再現する必要がある)。

---

## 4. cumulative_results.csv の trio_bets カラム

- 旧 `trio_bets` カラム: 4/1-5/17 期間 ★ 全行 NaN ★ (0%)
- 新 `trio_bets_str` カラム: 4/1-5/17 期間 ★ 100% fill ★ (456/456)
- 内容: AM 8:00 morning prediction の 7 点フォーメーション文字列 (例: `4-5-7; 4-5-8; 4-7-8; 4-7-11; 4-7-14; 4-8-11; 4-8-14`)

期間別 fill rate:

| 期間 | trio_bets_str fill | 内容質 |
|------|---------------------|--------|
| 4/1-4/26 | 100% | morning prediction (一部 Streamlit DB 経由) |
| 4/27-5/9 | 100% | morning prediction |
| 5/10-5/15 | 100% | morning prediction (5/10 のみ、 平日 開催なし) |
| 5/16 | 100% | morning prediction (新 schema、 top1-3 列 完備) |
| 5/17 | 100% | morning prediction (新 schema) |

---

## 5. 真の hit rate 計算 可否

| 観点 | 計算可? | 詳細 |
|------|---------|------|
| 仮想 morning formation の hit rate | ★ YES ★ | cumulative_results.csv の `trio_hit` / `trio_payout` で算出済 |
| ★ 実 Discord 通知 formation の hit rate ★ | ★ NO (永久不可) ★ | race-time formation が 永久喪失 |
| race-time vs morning formation 差分 audit | NO | race-time formation 自体が 不在 |
| 真の ROI (実投票額 ベース) | NO | 投票履歴 そのもの不在 (注: 本 audit は paper trade 範囲、 真の現金投票なし前提) |

★ cumulative_results.csv で計算されている `actual_payout` / `trio_hit` / `profit` は 「AM 8:00 morning formation を そのまま投票したと仮定した場合の hit/payout」 ★。 実通知時 (race -15min) に formation が違う場合、 実 hit rate は cumulative の値と乖離する可能性。

---

## 6. 重要ファイル (read-only)

| file | 用途 |
|------|------|
| `tools/race_auto_notify.py` L300, L352-354, L531-572 | Discord 送信 + log 書込 (formation persist なし) |
| `tools/daily_predict.py` L390-420 | AM 8:00 morning prediction (`trio_bets` 列 を daily_predictions CSV へ書込) |
| `tools/daily_results.py` L331-369, L530-605 | 結果照合 (`trio_bets_str` 列を cumulative へ書込) |
| `data/cumulative_results.csv` | 集約結果 (456 行 / 4/1-5/17、 trio_bets_str 100%、 morning 由来) |
| `data/race_notify_log/20260517.json` | P0-5 log (race_id/channel/skip のみ、 formation 不在) |
| `data/daily_predictions/YYYYMMDD.csv` | morning prediction 永続化 (10 日分のみ 物理 file 残存) |
| `keiba_predictions.db / race_results` | Streamlit local DB (4/4, 4/5 等 一部 backfill 源) |

---

## 7. V15 production 不変保証

★ 本 audit は read-only ★
- ファイル変更 0
- v15.2 training (PID 23528) 不変
- commit/push 0
- 親集中 ✅

---

## 8. 来週対応 推奨 (★ 本 audit の範囲外、 参考メモ ★)

実投票 formation を 永続化するには `tools/race_auto_notify.py` L353 直後で `bets` (生 list) を JSON で書き出す改修が必要。 ただし 本 audit は read-only 厳守 のため 提案のみ。

```python
# tools/race_auto_notify.py L353 直後 (★ 未実装 ★、 提案メモ)
# _p0_5_notify_log の signature 拡張、 もしくは
# data/race_notify_log/{date_str}/{race_id}.formation.json に bets を 追記
```
