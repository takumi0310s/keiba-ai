# 5月 11R/12R (重賞除外) 全頭 V15 production saved score 一覧

**期間**: 2026-05-01 〜 2026-05-09
**対象 R**: 12 件
**重賞除外**: G1 (天皇賞春) / G2 (京王杯SC, 京都新聞杯) / G3 (ユニコーンS, エプソムC) — 5 件
**source**: production_saved_score (data/cumulative_results.csv 5/2-5/3, data/daily_predictions/20260509.csv 5/9)
**🚨 LEAK 防止**: V15 model.predict() 不使用、 predict_core / daily_predict 一切実行せず、 read-only access のみ

---

## ⚠ source 制約

- **5/2, 5/3** の `data/cumulative_results.csv` は CLAUDE.md 既知問題により `top1_num/score` 列が NaN (95%欠損)。 `top1_finish` (V15 top1 馬の実際の着順) と `trio_bets / trio_payout / profit` は populated。 **score 値 そのものは production save 失敗** → 着順 + 投票結果 のみ記載可能。
- **5/9** は `data/daily_predictions/20260509.csv` に top1/top2/top3 の score + 馬番 + trio_bets が production saved。 4 着以下の馬の score は production csv に未保存 (full v15_scores JSON は別 branch dev/training-poc にあるが本 session では干渉防止のため参照しない)。
- → 「全頭スコア」 は **production saved の範囲では top1/top2/top3 が最大**。 4 着以下は本 audit では空欄。

---

## 5/2 (土) の 11R/12R (重賞除外)

### 京都 12R 4歳以上1勝クラス (1勝)
- 芝 1600m / 馬場 A

#### V15 production saved score (★ no leak、 production csv 由来 ★)

> ⚠ cumulative_results.csv の top1_num/score 欠損 (CLAUDE.md 既知)。 production saved score 値は不明。
> trio_bets / trio_result / payout / profit は populated。

#### 実結果 / V15 投票結果 (production)

- V15 top1 → 3 着
- V15 top2 → 5 着
- V15 top3 → 2 着
- 1-2-3 着 (trio_result): `1-4-5`
- trio 判定: ✅ HIT
- trio_payout: ¥2,390
- 投資 ¥700 → 払戻 ¥2,390
- 損益: +¥1,690

### 新潟 11R 三条S (OP/特別)
- ダ 1800m / 馬場 X

#### V15 production saved score (★ no leak、 production csv 由来 ★)

> ⚠ cumulative_results.csv の top1_num/score 欠損 (CLAUDE.md 既知)。 production saved score 値は不明。
> trio_bets / trio_result / payout / profit は populated。

#### 実結果 / V15 投票結果 (production)

- V15 top1 → 6 着
- V15 top2 → 2 着
- V15 top3 → 14 着
- 1-2-3 着 (trio_result): `5-12-13`
- trio 判定: ❌ miss
- trio_payout: ¥0
- 投資 ¥700 → 払戻 ¥0
- 損益: ¥-700

### 東京 12R 4歳以上2勝クラス (2勝)
- ダ 1600m / 馬場 B

#### V15 production saved score (★ no leak、 production csv 由来 ★)

> ⚠ cumulative_results.csv の top1_num/score 欠損 (CLAUDE.md 既知)。 production saved score 値は不明。
> trio_bets / trio_result / payout / profit は populated。

#### 実結果 / V15 投票結果 (production)

- V15 top1 → 1 着
- V15 top2 → 2 着
- V15 top3 → 9 着
- 1-2-3 着 (trio_result): `2-5-7`
- trio 判定: ✅ HIT
- trio_payout: ¥1,810
- 投資 ¥700 → 払戻 ¥1,810
- 損益: +¥1,110

---

## 5/3 (日) の 11R/12R (重賞除外)

### 京都 12R 東大路S (OP/特別)
- ダ 1400m / 馬場 D

#### V15 production saved score (★ no leak、 production csv 由来 ★)

> ⚠ cumulative_results.csv の top1_num/score 欠損 (CLAUDE.md 既知)。 production saved score 値は不明。
> trio_bets / trio_result / payout / profit は populated。

#### 実結果 / V15 投票結果 (production)

- V15 top1 → 15 着
- V15 top2 → 16 着
- V15 top3 → 6 着
- 1-2-3 着 (trio_result): `10-13-16`
- trio 判定: ❌ miss
- trio_payout: ¥0
- 投資 ¥700 → 払戻 ¥0
- 損益: ¥-700

### 新潟 11R 越後S (OP/特別)
- ダ 1200m / 馬場 D

#### V15 production saved score (★ no leak、 production csv 由来 ★)

> ⚠ cumulative_results.csv の top1_num/score 欠損 (CLAUDE.md 既知)。 production saved score 値は不明。
> trio_bets / trio_result / payout / profit は populated。

#### 実結果 / V15 投票結果 (production)

- V15 top1 → 2 着
- V15 top2 → 8 着
- V15 top3 → 10 着
- 1-2-3 着 (trio_result): `3-4-10`
- trio 判定: ❌ miss
- trio_payout: ¥0
- 投資 ¥700 → 払戻 ¥0
- 損益: ¥-700

### 新潟 12R 4歳以上1勝クラス (1勝)
- ダ 1200m / 馬場 D

#### V15 production saved score (★ no leak、 production csv 由来 ★)

> ⚠ cumulative_results.csv の top1_num/score 欠損 (CLAUDE.md 既知)。 production saved score 値は不明。
> trio_bets / trio_result / payout / profit は populated。

#### 実結果 / V15 投票結果 (production)

- V15 top1 → 1 着
- V15 top2 → 6 着
- V15 top3 → 2 着
- 1-2-3 着 (trio_result): `3-4-10`
- trio 判定: ✅ HIT
- trio_payout: ¥3,680
- 投資 ¥700 → 払戻 ¥3,680
- 損益: +¥2,980

### 東京 11R プリンシパルS (OP/特別)
- 芝 2000m / 馬場 C

#### V15 production saved score (★ no leak、 production csv 由来 ★)

> ⚠ cumulative_results.csv の top1_num/score 欠損 (CLAUDE.md 既知)。 production saved score 値は不明。
> trio_bets / trio_result / payout / profit は populated。

#### 実結果 / V15 投票結果 (production)

- V15 top1 → 13 着
- V15 top2 → 11 着
- V15 top3 → 9 着
- 1-2-3 着 (trio_result): `10-12-13`
- trio 判定: ❌ miss
- trio_payout: ¥0
- 投資 ¥700 → 払戻 ¥0
- 損益: ¥-700

### 東京 12R 4歳以上2勝クラス (2勝)
- ダ 1300m / 馬場 D

#### V15 production saved score (★ no leak、 production csv 由来 ★)

> ⚠ cumulative_results.csv の top1_num/score 欠損 (CLAUDE.md 既知)。 production saved score 値は不明。
> trio_bets / trio_result / payout / profit は populated。

#### 実結果 / V15 投票結果 (production)

- V15 top1 → 1 着
- V15 top2 → 13 着
- V15 top3 → 4 着
- 1-2-3 着 (trio_result): `5-10-13`
- trio 判定: ❌ miss
- trio_payout: ¥0
- 投資 ¥700 → 払戻 ¥0
- 損益: ¥-700

---

## 5/9 (土) の 11R/12R (重賞除外)

### 京都 12R 4歳以上2勝クラス (2勝)
- 出走 13 頭 / ダ 1800m / 馬場 A

#### V15 production saved score (★ no leak、 production csv 由来 ★)

| 順位 | 馬番 | 馬名 | V15 score |
|---|---|---|---|
| 1 | 8 | ロードヴォイジャー | 0.6614 |
| 2 | 13 | サイモンシュバリエ | - |
| 3 | 3 | メイショウヤシマ | - |

> 注: top2/top3 の score は daily_predictions csv に列がないため `-`。 馬番 + 馬名は populated。

#### 実結果 / V15 投票結果 (production)

- V15 trio_bets: `2-3-8; 2-8-13; 3-7-8; 3-8-11; 3-8-13; 7-8-13; 8-11-13`
- 案B改 strict 除外 (12R 1勝以外、 投票なし)

### 新潟 11R 駿風S (OP/特別)
- 出走 16 頭 / 芝 1000m / 馬場 D

#### V15 production saved score (★ no leak、 production csv 由来 ★)

| 順位 | 馬番 | 馬名 | V15 score |
|---|---|---|---|
| 1 | 1 | パラサイコロジー | 0.5166 |
| 2 | 13 | エコロジーク | - |
| 3 | 5 | カウンターセブン | - |

> 注: top2/top3 の score は daily_predictions csv に列がないため `-`。 馬番 + 馬名は populated。

#### 実結果 / V15 投票結果 (production)

- V15 trio_bets: `1-5-8; 1-5-9; 1-5-13; 1-5-14; 1-8-13; 1-9-13; 1-13-14`
- 案B改 strict 除外 (12R 1勝以外、 投票なし)

### 新潟 12R 4歳以上1勝クラス (1勝)
- 出走 12 頭 / ダ 1800m / 馬場 A

#### V15 production saved score (★ no leak、 production csv 由来 ★)

| 順位 | 馬番 | 馬名 | V15 score |
|---|---|---|---|
| 1 | 11 | ハイクオリティ | 0.6483 |
| 2 | 12 | マテンロウミラクル | - |
| 3 | 8 | カレンラップスター | - |

> 注: top2/top3 の score は daily_predictions csv に列がないため `-`。 馬番 + 馬名は populated。

#### 実結果 / V15 投票結果 (production)

- V15 trio_bets: `6-8-11; 6-11-12; 8-9-11; 8-10-11; 8-11-12; 9-11-12; 10-11-12`
- ★ 5/9 V15 案B改 投票対象 ★
  - 軸: 11 ハイクオリティ
  - trio_bets (saved): `6-8-11; 6-11-12; 8-9-11; 8-10-11; 8-11-12; 9-11-12; 10-11-12`
  - 結果: 11 ハイクオリティ → 3着、 三連複 7点 全 miss、 損益 -¥700 (Session #67 確定)

### 東京 12R 4歳以上2勝クラス (2勝)
- 出走 16 頭 / ダ 1400m / 馬場 D

#### V15 production saved score (★ no leak、 production csv 由来 ★)

| 順位 | 馬番 | 馬名 | V15 score |
|---|---|---|---|
| 1 | 11 | フィドルファドル | 0.5624 |
| 2 | 3 | モーニングマジック | - |
| 3 | 7 | ヴィンブルレー | - |

> 注: top2/top3 の score は daily_predictions csv に列がないため `-`。 馬番 + 馬名は populated。

#### 実結果 / V15 投票結果 (production)

- V15 trio_bets: `1-3-11; 1-7-11; 3-7-11; 3-8-11; 3-11-15; 7-8-11; 7-11-15`
- 案B改 strict 除外 (12R 1勝以外、 投票なし)

---
