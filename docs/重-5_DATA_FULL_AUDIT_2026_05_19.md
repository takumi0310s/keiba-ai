# 過去 1.5 ヶ月 data 完全 audit (重-5、2026-05-19)

> read-only audit。既存データ上書き一切なし。5/24+ paper eval の base data 確定が目的。

---

## 1. cumulative_results.csv 概要

| 項目 | 値 |
|------|-----|
| total rows | 663 |
| settled (raw) | 629 ← 5/17 二重登録 33件含む |
| settled (dedup) | **596** ← V15-audit-4 と一致 |
| pending | 33 (全件 5/9) |
| anomaly (column-shift bug) | 1 (row 0、NAR race 5/5) |
| 日付範囲 | 2026-03-14 〜 2026-05-17 |
| columns | 34 列 |

### 5/17 二重登録バグ (新発見)

5/17 の 33 レースが全件 2 行ずつ登録されている (= 66 行)。
内容は完全一致 (profit / top1_num / trio_bets_str まで同一)。
cumulative_truth.json が n=629 と報告しているのはこれが原因。

- **真の settled n = 596** (drop_duplicates subset=['race_id','status'] 後)
- **dedup 後 P&L = -19,080 + 12,160 = -6,920** ← V15-audit-4 真値と一致 ✅
- **dedup 後 ROI = 98.34%** ← V15-audit-4 真値と一致 ✅

```python
# 確認コマンド
df_dedup = df.drop_duplicates(subset=['race_id', 'status'])
settled_dedup = df_dedup[df_dedup['status'] == 'settled']  # 596 rows
```

### column-shift bug (row 0)

| field | 格納値 | 本来の値 |
|-------|--------|---------|
| status | '20260505' | 'settled' |
| profit | 'settled' | profit 数値 |
| date | 0.0 | 20260505 |

NAR レース (course=43, condition=NAR_X) の 1 件のみ。
影響: V15-audit-4 では dedup + bet_type フィルタで除外済み → 指標への影響なし。

---

## 2. 週末別 race 数 + 予測・配当カバレッジ

| 日付 | 曜日 | cumul 件数 | pred_csv | payout | 状態 |
|------|------|------------|----------|--------|------|
| 20260404 | 土 | 22 | NO | YES | pred_csv 欠落 (AM8:00 CSV 未保存、formation あり) |
| 20260405 | 日 | 24 | NO | YES | 同上 |
| 20260411 | 土 | 34 | YES | YES | OK |
| 20260412 | 日 | 35 | YES | YES | OK |
| 20260418 | 土 | 35 | YES | YES | OK |
| 20260419 | 日 | 35 | YES | YES | OK |
| 20260425 | 土 | 35 | YES | YES | OK |
| 20260426 | 日 | 35 | YES | YES | OK (50 rows = prerace+afternoon 合算) |
| 20260502 | 土 | 33 | NO | YES | pred_csv 欠落 (formation あり) |
| 20260503 | 日 | 34 | NO | YES | 同上 |
| 20260509 | 土 | 33 | YES | NO | **全件 pending** (results 未取得) |
| 20260510 | 日 | 34 | YES | NO | settled OK / payout 欠落 |
| 20260516 | 土 | 34 | YES | NO | settled OK / payout 欠落 |
| 20260517 | 日 | 33 (dedup) | YES | NO | settled OK / payout 欠落 / 二重登録バグあり |

---

## 3. AM 8:00 daily_predictions/*.csv 完全性

### 欠落日 (4 日)

| 日付 | daily_predictions CSV | 実態 |
|------|----------------------|------|
| 4/4 (土) | **欠落** | cumulative に 22 件あり。formation (trio_bets_str) も 100% 入力済み。v17_morning 等 別経路で予測実行 |
| 4/5 (日) | **欠落** | 同上 (race_auto_notify 08:45 のみ起動、AM8:00 CSV 未保存) |
| 5/2 (土) | **欠落** | v17_morning が 15:14 に実行 (AM8:00 は未実行)、formation 100% あり |
| 5/3 (日) | **欠落** | 同上 |

### 補填判定

- AM 8:00 CSV はシステム運用ログ目的のみ
- **formation (trio_bets_str)** は cumulative_results.csv に 100% 記録済み → 実質問題なし
- **top1_num / top1_score** は上記 4 日で 0/N (欠落) → スコア分析には利用不可

---

## 4. settled result 完全性

| カテゴリ | 件数 | 備考 |
|---------|------|------|
| settled (dedup) | **596** | V15-audit-4 と一致 |
| pending (5/9) | **33** | results 未取得 |
| 4/1〜5/17 期間 外 (3/14〜3/29) | 173 | settled、期間外 |
| 4/1〜5/17 within-scope settled | **423** | dedup 後、4/1 以降のみ |

### 5/9 pending (33 件) 補填可能性

- daily_results/20260509.csv に 33 件あり、全件 status='pending'、trio_payout=0
- 5/9 の JVLink HR データなし (data/jvlink/2026/ に 5/9 分未取得)
- **補填方法**: `python tools/daily_results.py --date 20260509` 再実行
  - netkeiba から results 取得 → cumulative_results.csv を pending → settled に更新
  - 注: Cookie が有効な場合のみ (期限切れなら refresh_cookie.py 先に実行)
- **推奨**: 5/24 paper eval 開始前に実行

---

## 5. jra_payouts.csv 完全性

| 項目 | 値 |
|------|-----|
| 最終取得日 | **20260503** |
| 4/1〜5/3 カバレッジ | 10 週末全日 (336 records) |
| **5/9-5/17 欠落** | 4 週末 (20260509, 20260510, 20260516, 20260517) |
| 欠落の影響 settled 件数 | 101 件 (5/10-5/17 settled) |

5/9-5/17 の actual_payout:
- 5/10: 34 件のうち 28 件 = 0 (payout 未入力 = 負け)、6 件 = 実配当あり
- 5/16, 5/17: 同様に零多数

**補填方法**: `python scrape_jra_payouts.py` 実行 (JRA公式DB から 5/9-5/17 配当取得)
実行後に `python tools/daily_results.py --date YYYYMMDD` で actual_payout を更新。

### payout 付き settled 件数

| 期間 | settled (dedup) | payout あり | カバレッジ |
|------|----------------|------------|---------|
| 3/14〜5/3 | 495 | 495 | 100% |
| 5/9〜5/17 | 101 | 28 | 27.7% |
| **合計** | **596** | **523** | **87.8%** |

---

## 6. column-shift bug 状態

race_id 12桁フォーマット異常: **0 件** (全 663 行 = 12桁)
column-shift bug: **1 件** (row 0、NAR race)
→ V15-audit-4 では除外済み。5/24+ paper eval でも除外推奨。

---

## 7. formation data 状態

| ソース | 状態 |
|--------|------|
| cumulative_results.trio_bets_str | **88.7%** (558/629 raw、3/14-3/15 の 71 件のみ欠落) |
| data/race_notify_log/ | 5/17, 5/18 のみ (race_id=null → 実データなし) |
| data/race_notify_log_v2/ | 5/17, 5/18 の phase1/2/3 あり (race_id='None' → 実データなし) |

**判定**: CLAUDE.md 記載の「5/18+ race_notify_log v2 で record 開始」は未完全。
race_id が None で実質記録なし。5/24 運用開始時に再確認要。

過去分 (〜5/17) の formation:
- 4/1 以降 = trio_bets_str 100% あり ✅
- 3/14-3/15 = trio_bets_str なし (71 件、audit 対象外期間)

---

## 8. 補填可能 file list

| data | 状態 | 補填方法 | 実行者 | 優先度 |
|------|------|---------|--------|--------|
| 4/4, 4/5 AM8 pred_csv | 永久欠落 | 不可 (formation は cumulative に保存済) | — | 低 |
| 5/2, 5/3 AM8 pred_csv | 永久欠落 | 不可 (同上) | — | 低 |
| 5/9 pending (33件) | **補填可能** | `python tools/daily_results.py --date 20260509` | user | **高** |
| jra_payouts 5/9-5/17 | **補填可能** | `python scrape_jra_payouts.py` | user | **高** |
| 5/9 JVLink HR | 補填可能 | JV-Link fetcher で 5/9 HR 取得 | user | 中 |
| race_notify_log_v2 formation | 永久欠落 (〜5/17) | 不可 (5/18+ から正式運用開始予定) | — | — |
| 5/17 二重登録 | 修正可能 | cumulative_results.csv から dedup | user | 中 |

---

## 9. 5/24+ paper eval base data 確定

### 有効 base N

| シナリオ | n | ROI | 備考 |
|---------|---|-----|------|
| V15-audit-4 準拠 (dedup + bet_type filter) | **596** | **98.34%** | 5/17 まで settled |
| + 5/9 pending 解決後 | 629 | 未計算 | daily_results.py 再実行 |
| payout 付き subset のみ | 495 | 計算可能 | 3/14-5/3 |

### 推奨 paper eval base

**n=596 (dedup settled、4/1〜5/17)** で開始可能。
以下を先に実施するとより正確:
1. `python tools/daily_results.py --date 20260509` → 33 pending 解決
2. `python scrape_jra_payouts.py` → 5/9-5/17 payout 補填
3. cumulative_results.csv の 5/17 二重登録 dedup (スクリプト実行)

---

## ★ data 完全性 verdict ★

| 指標 | 値 |
|------|-----|
| 完全性 (settled/total expected) | **596/629 = 94.7%** (5/9 pending 33 件除く) |
| payout カバレッジ (settled中) | **523/596 = 87.8%** |
| formation カバレッジ (4/1+) | **423/423 = 100%** |
| 二重登録バグ | 5/17 のみ 33 件 (補正済み真値: n=596) |
| paper eval 開始 OK | **YES (条件付き)** |
| 条件 | 5/9 pending 解決 + 5/9-5/17 payout 補填 を先に推奨 |
| 5/22 admin action への影響 | なし |

### 重要訂正

cumulative_truth.json (5/18 計算) の n_settled=629 / pnl=-19,080 / roi=95.67% は
**5/17 二重登録を含む誤値**。

真値 (V15-audit-4 = dedup済み):
- n_settled = **596**
- pnl = **-6,920**
- roi = **98.34%**

cumulative_truth.json は 5/9 pending 解決 + dedup 後に再計算推奨。
