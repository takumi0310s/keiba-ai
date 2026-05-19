# A-1 data 補填 結果 (2026-05-19)

## 実施内容

### Step 1: 5/9 pending results 補填
- `tools/daily_results.py --date 20260509` 実行
- `data/daily_results/20260509.csv` に 33 races settled 確認
- cumulative_results.csv の 5/9 pending 33 件を settled に更新
- 更新後 settled: 629 → 662

### Step 2: jra_payouts 5/9-5/17 補填
- `scrape_jra_payouts.py --year 2026` 実行
- 旧最新: 20260503 → 新最新: **20260517**
- 追加日付: 20260509, 20260510, 20260516, 20260517
- 行数: 12333 → 14949 (+2616)

### Step 3: top4_num 列追加
- trio_bets / trio_bets_str から TOP4 馬番を逆算して追加
- non-null: 611 / null: 52 (trio_bets なし = 古い行)
- 列数: 34 → 35

## 補填後の真値

| 指標 | 値 |
|------|-----|
| settled n | **662** |
| investment | 463,400 円 |
| payout | 431,560 円 |
| PnL | **-31,840 円** |
| ROI | **93.13%** |
| jra_payouts 最新 | 20260517 |
| jra_payouts 行数 | 14,949 |

注: ROI 93.13% / PnL -31,840 は 5/9 results 反映後の真値。
CLAUDE.md 記載の 98.34% / -6,920 (n=596、≤5/17 settled) とは集計対象が異なる可能性あり。
(本集計は cumulative_results.csv 全 settled rows、 CLAUDE.md は V15-audit-4 別集計)

## 変更ファイル
- `data/cumulative_results.csv` — pending 33 → settled + top4_num 列追加
- `data/jra_payouts.csv` — 5/9-5/17 補填
