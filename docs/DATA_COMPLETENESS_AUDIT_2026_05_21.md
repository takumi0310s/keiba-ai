# DATA_COMPLETENESS_AUDIT_2026_05_21.md

> 作成: 2026-05-21 (Agent [a])
> 対象: `data/cumulative_results.csv`, `data/jra_payouts.csv`, `data/daily_predictions/`
> 方針: read-only audit のみ。V15 production 系ファイル改変なし。

---

## 1. cumulative_results.csv 全体 audit

| 項目 | 値 |
|------|----|
| 総行数 (header 除く) | **663** |
| うち NAR 行 (status=20260505、horse racing Jpn1) | **1** |
| うち JRA central 有効行 (date > 20000000) | **662** |
| status=settled | **662** |
| status=pending | **0** |
| status その他 | **1** (NAR 行、status="20260505"、実質無効) |
| race_date (date 列) 最小 | **20260314** |
| race_date (date 列) 最大 | **20260517** |

### 重要: 5/9 行の重複発見

cumulative_results.csv に **race_id が 2 回記録された 33 レース** (合計 66 行) が存在する。
全て `date=20260509` の行。

| race_id prefix | 重複 race 数 | 意味 |
|----------------|-------------|------|
| 20260502 | 12 | 5/2 round races |
| 20260803 | 12 | Chukyo round races (venue=08) |
| 20260401 | 9 | 4/1 round races |
| **合計** | **33** | **33 unique race_id × 2 = 66 行** |

原因: 5/9 の `daily_results.py` または `check_results.py` 実行時に同じ 33 レースが 2 回 append された (morning prediction + race-time notify 双方からの書き込みと推定)。

### 列の存在確認

| 列 | 状況 |
|----|------|
| top4_num | **存在**。5/9 補填で追加確認済み |
| top4_num null 数 (JRA 662 行中) | **52** (7.8%) — 主に 3/14-3/15 の初期記録分 |
| trio_bets_str null 数 (JRA 662 行中) | **72** (10.9%) — 3/14 (36件) + 3/15 (35件) + NAR (1件) |
| trio_bets_str null 日付 | 3/14 (100%)、3/15 (100%) — 初期 2 日間のみ |

**解釈**: trio_bets_str は 3/21 以降は 100% 記録済み (朝 AM 8:00 morning prediction から)。3/14-3/15 は朝予測システム稼働前のため欠損。

---

## 2. jra_payouts.csv 補填状況

| 項目 | 値 |
|------|----|
| 総行数 | **15,057** |
| 最小 race_date | **20180303** |
| 最大 race_date | **20260517** |
| 2026 年分 unique 日付数 | **42 日** |

### 5/9-5/17 補填確認

| 日付 | jra_payouts 行数 | 状況 |
|------|-----------------|------|
| 20260509 | 36 | **補填済み** |
| 20260510 | 36 | **補填済み** |
| 20260516 | 36 | **補填済み** |
| 20260517 | 36 | **補填済み** |

### 未補填 race_date

**なし** — 全 19 開催 weekend の payout データが jra_payouts.csv に存在する。

---

## 3. 5/9 補填 pending R 確認

5/9 の cumulative_results 全 66 行の状況:

| 項目 | 値 |
|------|----|
| status=settled | **66** (全件) |
| status=pending | **0** |
| actual_payout > 0 (的中) | **18** |
| total investment | 46,200 円 (重複計上 → 実質 23,100 円) |
| total payout | 21,280 円 (重複計上 → 実質 10,640 円) |

**5/9 pending 残件数: 0** (全件 settled 済み)

ただし **33 race_id が 2 重登録**されているため、実際の ROI 計算からは除外が必要 (後述の n=596 clean base 参照)。

---

## 4. daily_predictions/ 欠落確認

### 存在するファイル (JRA central)

| ファイル | 行数 |
|---------|------|
| 20260314.csv | 36 race |
| 20260315.csv | 35 race |
| 20260321.csv | 35 race |
| 20260411.csv | 34 race |
| 20260412.csv | 35 race |
| 20260418.csv | 35 race |
| 20260419.csv | 35 race |
| 20260425.csv | 35 race |
| 20260426.csv | 35 race |
| 20260509.csv | 33 race |
| 20260510.csv | 34 race |
| 20260516.csv | 34 race |
| 20260517.csv | 33 race |

※ NAR ファイル (nar_20260510.csv 〜 nar_20260520.csv) は別管理、本 audit の対象外。

### 永久欠落 日付 (JRA central)

cumulative_results.csv に settled 行が存在するが daily_predictions ファイルが **ない** 日付:

| 日付 | cumulative 行数 | 欠落理由 | 判定 |
|------|----------------|---------|------|
| 20260328 | 32 | 3/28 は daily_predict ファイル未保存 | **永久欠落** |
| 20260329 | 35 | 3/29 は daily_predict ファイル未保存 | **永久欠落** |
| 20260404 | 22 | 4/4 は daily_predict ファイル未保存 | **永久欠落** |
| 20260405 | 24 | 4/5 は daily_predict ファイル未保存 | **永久欠落** |
| 20260502 | 33 | 5/2 は daily_predict ファイル未保存 | **永久欠落** |
| 20260503 | 34 | 5/3 は daily_predict ファイル未保存 | **永久欠落** |

**計 6 日 / 180 race の pred_csv が永久欠落**。ただし cumulative_results.csv に結果 (payout/profit) は記録済みのため ROI 計算には影響なし。formation (trio_bets_str) は 3/21+ は全て記録済みのため pred_csv 欠落でも formation は確認可能。

また daily_predictions に存在するが cumulative_results がない (予測したが買い目なし) 日付: **なし** (全 pred_csv 日付が cumulative に対応)。

---

## 5. data 完全性スコア

### 定義

```
completeness = settled_N_with_payout / total_unique_race_N × 100
```

### スコア計算

| 項目 | 値 |
|------|----|
| 総 unique race_id (JRA) | **629** |
| うち clean (単一登録) | **596** |
| うち 重複登録 5/9 races | **33** |
| 全 unique race_id のうち actual_payout 確認済み | **629** (100%) |
| clean race_id のうち actual_payout 確認済み | **596** (100%) |

**completeness score: 100.0%** (target 95%+ を達成)

補足: 5/9 の 33 重複レースは payout データ自体は有効 (jra_payouts.csv に収録済み)。但し cumulative への 2 重 append のため ROI 計算上は除外する。

---

## 6. 5/24+ paper eval base 確定

### True base (V15-audit-4 と一致)

| 項目 | 値 |
|------|----|
| **n** | **596** |
| **date range** | **2026-03-14 〜 2026-05-17** |
| **投資総額** | **¥417,200** |
| **払戻総額** | **¥410,280** |
| **累計 PnL** | **¥-6,920** |
| **ROI** | **98.34%** |
| **的中数** | 149 / 596 (25.0%) |
| **payout confirmed** | 596 / 596 (100%) |

### 除外された 33 races (5/9 重複)

5/9 当日 33 race は全て race_id 2 重登録のため clean base から除外。これらのレースは **20260401 / 20260502 / 20260803 の 3 round に属す** 実際のレース。結果は valid だが cumulative への append が 2 回行われた。

除外後の clean n=596 が V15-audit-4 の真値 (ROI 98.34% / PnL ¥-6,920) に完全一致することを確認。

### paper eval base 設定

5/24+ の paper eval は以下を base とする:

- **Base**: n=596, date 2026-03-14 〜 2026-05-17, ROI 98.34%, PnL ¥-6,920
- **paper eval 追加分**: 2026-05-18+ の settled races (5/24 時点で 5/18-5/23 分が加算予定)
- **除外**: 5/9 の 33 重複 race_id (double-append された行の 2 つ目を除去)
- **trio_bets_str**: 3/14-3/15 のみ欠損 (72 行)。3/21 以降は 100% 記録済み → paper eval の 5/24+ 分は trio_bets_str 完全取得可能

---

## 7. 次アクション

| 優先度 | 内容 | 理由 |
|--------|------|------|
| P1 | 5/9 の重複 33 行を cumulative_results.csv から除去 | ROI 誤計算防止。現在は `drop_duplicates(subset='race_id', keep='first')` で実質除外しているが、ファイル自体に 2 重行が残る |
| P2 | daily_predictions/ の欠落 6 日分は永久欠落として確定 | pred_csv は復元不可、trio_bets_str は cumulative に記録済みのため実害なし |
| P3 | 5/24+ races の daily_results.py 実行後に append が単一であることを確認 | 5/9 重複再発防止 |
| P3 | 5/24+ paper eval 用に `date >= 20260524` の settled 行を追跡 | 5/24-5/25 weekend 後に集計 |

---

## 付録: 日別 ROI サマリー (clean n=596)

| 日付 | n | 的中 | ROI |
|------|---|------|-----|
| 20260314 | 36 | 6 | 59.2% |
| 20260315 | 35 | 3 | 16.8% |
| 20260321 | 35 | 8 | 157.4% |
| 20260328 | 32 | 4 | 10.3% |
| 20260329 | 35 | 12 | 133.8% |
| 20260404 | 22 | 9 | 115.1% |
| 20260405 | 24 | 3 | 333.7% |
| 20260411 | 34 | 8 | 106.0% |
| 20260412 | 35 | 12 | 238.2% |
| 20260418 | 35 | 11 | 92.5% |
| 20260419 | 35 | 5 | 37.7% |
| 20260425 | 35 | 2 | 25.0% |
| 20260426 | 35 | 9 | 62.1% |
| 20260502 | 33 | 7 | 32.1% |
| 20260503 | 34 | 4 | 31.3% |
| 20260509 | 0 | — | — (33 race 全件重複 → 除外) |
| 20260510 | 34 | 11 | 113.8% |
| 20260516 | 34 | 11 | 227.4% |
| 20260517 | 33 | 6 | 47.4% |
| **累計** | **596** | **149** | **98.34%** |
