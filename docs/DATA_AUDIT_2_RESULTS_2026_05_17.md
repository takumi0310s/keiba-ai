# data-audit-2: 実 result + cumulative + jra_payouts 完全性

audit date: 2026-05-17
mode: read-only (V15 / cumulative / jra_payouts 改変なし、 v15.2 training 不干渉)

## 0. 結論

| target | verdict |
|--------|---------|
| cumulative_results.csv 完全性 | △ (settled 596 確定、 1 row column-shift bug、 33 pending 未解消) |
| daily_predictions ↔ cumulative 整合性 | △ (only_daily=4 戦略⑦除外で正常、 only_cum=180 daily file 欠落) |
| jra_payouts.csv 完全性 | △ (5/3 まで取得、 5/9-5/17 GW明け 8 weekend day 未取得) |
| 5/9-5/15 TYB 停止影響 | あり (5/17 ROI 47.4% は単日変動の範囲、 5/16 227.4%、 平均は plus) |
| 4/5-4/17 SCRAPER-GUARD 影響 | あり (4/4 22R / 4/5 24R で通常 ~34R から 10R 欠落) |

V15 production 不変保証: 学習ファイル / モデル / 設定 一切触れず。

## 1. cumulative_results.csv 完全性

- total rows: **630**
- status:
  - `settled`: **596** (★ audit-4 期待 n=596 と完全一致)
  - `pending`: **33** (全部 5/9 開催分、 result merge 未実行)
  - `20260505`: **1** (column-shift bug: race_id=202643050511 札幌記念 NAR、 status 列に date が入り date=0)
- date 範囲: 20260314 - 20260517、 unique=20 (anomaly の '0' 含む)
- race_id format: 全 630 行が 12 桁数値 (NG=0)
- date 連続性 (期待 weekend 24 日 vs 実 19 日):
  - missing: `20260322`, `20260429`, `20260404` 部分, `20260504`, `20260505`, `20260506`
  - GW 祝日 (4/29 / 5/4-5/6) は JRA 中央開催無し、 正常
  - 3/22 は 中央 開催無し週末、 正常
- 1-3 着 / 配当 null rate on settled (n=596):
  - `top1_finish` / `top2_finish` / `top3_finish` null: **0**
  - `actual_payout` null: **0**
  - hit (payout>0): **131**
  - loss (payout=0): **465**
  - 命中率: 21.98%

### 1-bug: column-shift row

```
race_id=202643050511, course=札幌, race_num=11
race_name=札幌記念 NAR (5/5)
status='20260505' (← date 文字列が status に入り込み)
date=0.0 (← date 列は 0)
```

audit-4 集計には status=='settled' で除外されているため累計 ROI に影響なし。
1 行のみのため 補修コスト < 警告レベル、 但し 修正推奨。

### 1-bug: pending 33 rows

- 全 5/9 中央開催分 (新潟 / 京都 / 東京 含む 33 R)
- 5/10 / 5/16 / 5/17 は settled 化済 → 5/9 のみ daily_results 未実行 と推定
- 影響: audit-4 累計 n=596 / ROI 計算 から除外、 結果整合性は保たれる

## 2. daily_predictions ↔ cumulative 整合性

- daily file (非 NAR / 非 bak / 非 prerace): **13 ファイル** (3/14 / 3/15 / 3/21 / 4/11 / 4/12 / 4/18 / 4/19 / 4/25 / 4/26 / 5/9 / 5/10 / 5/16 / 5/17)
- daily 内 unique race_id: **454**
- cumulative race_id: **630**
- 整合性:
  - **only_daily=4**: 5/9 新潟11R, 5/10 新潟6R, 5/16 新潟6R, 5/17 新潟9R → 全 「新潟」 race、 戦略⑦ 京都/06_特別 除外で 購入対象から外れ cumulative に未記録 ＝ 想定動作
  - **only_cum=180**: 3/28 / 3/29 / 4/4 / 4/5 / 5/2 / 5/3 の cum 記録は 当該 daily csv が無い (削除 or 未保存)
  - **both=450**

| metric | count |
|--------|------|
| daily にあるが cumulative なし | 4 |
| cumulative にあるが daily なし | 180 |
| 両方 | 450 |

only_daily 4 件は仕様通り (戦略⑦) で bug ではない。
only_cum 180 件は daily_predictions csv が 6 日分 (3/28, 3/29, 4/4, 4/5, 5/2, 5/3) 紛失。
cumulative 側に予測 result が保存されているため audit / ROI 計算には影響なし。
将来 reproducibility のため daily csv 再生成推奨。

## 3. jra_payouts.csv 完全性 (4/1-5/17)

- 期間内 行数: **336**
- 取得 weekend: 4/4 / 4/5 / 4/11 / 4/12 / 4/18 / 4/19 / 4/25 / 4/26 / 5/2 / 5/3 (10 日)
- **missing weekend**: 4/29, 5/4, 5/5, 5/6, **5/9, 5/10, 5/16, 5/17**

| 期間 | 状態 |
|------|------|
| 4/4-5/3 | 完全取得 (10 weekend × 24-36R) |
| 4/29 / 5/4-5/6 | JRA 開催無し (祝日)、 正常 |
| 5/9-5/17 (GW 明け 2 weekend) | **未取得**、 cumulative.actual_payout は 別 source (netkeiba 結果) から merge されており、 集計には影響なし |

券種別 fill rate (期間内 336 行、 全て 100%):

| 券種 | nums | payout |
|------|------|--------|
| tansho | 100% | 100% |
| fukusho | 100% | 100% |
| umaren | 100% | 100% |
| wide | 100% | 100% |
| trio | 100% | 100% |
| tierce | 100% | 100% |

CLAUDE.md 「jra_payouts.csv が 4/6 で更新停止」 記述は 古い、 4/19 SCRAPER-GUARD 修正後 4/11 から再開し 5/3 まで取得済。 5/9 以降 再停止 (要原因調査、 別タスク)。

## 4. 5/9-5/15 JRDB TYB 停止影響

daily_predictions の出力 csv (19 列、 features cache ではない) は予測結果のみで feature null rate 直接取得不可。
cumulative 側で performance 確認:

| date | n | settled | hits | inv | pay | ROI |
|------|---|---------|------|-----|-----|-----|
| 20260509 | 33 | 0 | - | - | - | - (pending) |
| 20260510 | 34 | 34 | 11 | 23800 | 27090 | **113.8%** |
| 20260516 | 34 | 34 | 11 | 23800 | 54110 | **227.4%** |
| 20260517 | 33 | 33 | 6 | 23100 | 10940 | **47.4%** |

5/10 (TYB 停止中の二日目) ROI 113.8% は健全範囲。
5/16-5/17 (TYB 復旧後) 平均 137.4% で TYB 停止が モデル性能 を 致命的に 損なった証拠なし。
**真の影響: 軽微 (単日ボラ範囲)**

## 5. 4/5-4/17 SCRAPER-GUARD 停止影響

| date | n | settled | hits | ROI | 備考 |
|------|---|---------|------|-----|------|
| 20260328 | 32 | 32 | 4 | 10.3% | 通常 ~35R より 3R 少 |
| 20260329 | 35 | 35 | 12 | 133.8% | 通常 |
| 20260404 | **22** | 22 | 9 | 115.1% | ★ 12R 欠落 |
| 20260405 | **24** | 24 | 3 | 333.7% | ★ 10R 欠落 |
| 20260411 | 34 | 34 | 8 | 106.0% | 4/19 修正前 weekend、 通常 |
| 20260412 | 35 | 35 | 12 | 238.2% | 通常 |

4/4-4/5 で 通常比 10-12R 欠落 → SCRAPER-GUARD バグ (CLAUDE.md 4/19 修正 commit e173f40d) の影響で 当該 weekend の daily_predict が部分実行のみ。
4/11 以降 完全実行に戻っている (修正前だが偶発的に正常稼働 / 戦略⑦は 4/27 から)。
**真の影響: 4/4-4/5 で 22 race 欠落 (機会損失 推定 +0-15,000 円、 4/5 ROI 333% の好調日だったため 補完していれば 大きな + 可能性あり)。 現累計には反映済 ＝ 過去事象。**

## 6. V15 production 不変保証

- V15 model file (`keiba_model_v135_central*.pkl.gz`) 一切触れず
- `cumulative_results.csv` 一切触れず (read-only)
- `jra_payouts.csv` 一切触れず (read-only)
- v15.2 training process (PID 23528) 干渉なし
- git commit / push 一切なし (親集中)

## 7. 推奨 (read-only audit のため 実行はしない、 次タスクの input)

1. 5/9 33 pending を `tools/daily_results.py --date 20260509` で settled 化 (cumulative 完全性 ↑)
2. column-shift row (race_id=202643050511) を手動修正 or 削除 (NAR レース、 中央運用に影響なし)
3. jra_payouts.csv 5/9-5/17 取得 (scrape_jra_payouts.py を期間指定実行)
4. daily_predictions/{3/28, 3/29, 4/4, 4/5, 5/2, 5/3}.csv 再生成 (reproducibility ↑)

## 8. 数値 fact summary (fabrication 無し)

- cumulative settled: **596 行** (audit-4 と一致)
- cumulative pending: **33 行** (5/9)
- cumulative anomaly: **1 行** (column-shift)
- daily_predictions 整合: 4 / 180 / 450 (only_daily / only_cum / both)
- jra_payouts 期間内: 336 行、 missing 8 weekend
- 累計 profit (paper trade、 全 596 settled): **-6,920 円** (CLAUDE.md +14,140 は 5/5 朝 実購入のみ、 paper trade とは別軸)
- 5/9 以降 paper profit: **+21,440 円**
- 5/8 まで paper profit: **-28,360 円**
