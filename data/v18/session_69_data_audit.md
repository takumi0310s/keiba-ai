# Session #69 A: data audit (production score 由来 + リーク risk audit)

**作成**: 2026-05-09 (Session #69 A)
**目的**: 7 点 vs 11 点 三連複 backtest の data source 確定 + リーク risk 排除
**ユーザー指摘**: 「今スコアを改めて出すとリークが入りそう」 → 完全反映

---

## 1. 利用可能な data source

### 1.1 data/daily_predictions/{date}.csv (production、 leak-free)

| 列 | 内容 |
|----|------|
| race_id | レース ID |
| top1_num/name/score | V15 top1 (production score) |
| top2_num/name | V15 top2 |
| top3_num/name | V15 top3 |
| trio_bets | 7 点 三連複 (top1-{top2,3}-{top4-6} 構造) |
| condition | A-X 条件 |
| num_horses, distance, surface | レース基本 |

available dates (production 予測 保存済):

| date | predictions | results | usable |
|------|-------------|---------|--------|
| 20260314 | OK | OK | OK |
| 20260315 | OK | OK | OK |
| 20260321 | OK | OK | OK |
| 20260411 | OK | OK | OK |
| 20260412 | OK | OK | OK |
| 20260418 | OK (+_prerace) | OK | OK |
| 20260419 | OK | OK | OK |
| 20260425 | OK | OK | OK |
| 20260426 | OK | NG | NG |
| 20260509 | OK | NG (5/9 21:00 未照合) | NG |

**実効 backtest 期間**: 3/14 - 4/25 (8 開催日)

### 1.2 data/daily_results/{date}.csv (照合済 結果)

| 列 | 内容 |
|----|------|
| trio_result | 実 1-2-3着 (sorted) |
| trio_payout | 実 三連複 配当 (100円換算) |
| trio_hit | 実 hit 判定 (0/1) |
| status | settled / etc |

→ trio_payout で任意 formation の 7 点 vs 11 点 比較可能

---

## 2. リーク risk audit (★ ユーザー指摘に基づく ★)

### 2.1 NG: V15 model 再 inference (data/v18/session69_horse_scores.csv)

source: `tools/session69_v15_inference.py` で生成 (141K rows、 17:08 作成)

**重大 LEAK**:
- V15 model は 2010-2025 全年で学習済 (year≤25)
- year>=23 の cache rows に対し再 inference → train/test 完全 overlap
- score は in-sample fit、 真の予測能力を表さない
- → **本 backtest で使用 NG** ★

→ 本 file は read 不要、 完全無視。

### 2.2 OK: production saved score 限定使用

- daily_predictions/{date}.csv は予測時点で保存
- top1-3 + 7 trio_bets は production score 由来 = 完全 out-of-sample
- → **本 backtest で使用 OK** ★

---

## 3. Top4-6 復元 (leak-free reconstruct logic)

production data は top1-3 のみ name/num 保存。 top4-6 は trio_bets パターンから逆算可能:

### 3.1 7 点 trio_bets の構造

```
top1-top2-top3, top1-top2-top4, top1-top2-top5, top1-top2-top6,
top1-top3-top4, top1-top3-top5, top1-top3-top6
```

### 3.2 horse 出現回数

| horse | 出現回数 |
|-------|---------|
| top1 | 7 (全 bet に登場) |
| top2 | 4 (top1-top2-* 4 件) |
| top3 | 4 (top1-top3-* 3 件 + top1-top2-top3 1 件) |
| top4 | 2 |
| top5 | 2 |
| top6 | 2 |

→ top4-6 は出現回数 2 で同率、 順位は trio_bets list 順から確定可能。
`tools/formation_retro_5_2_5_3.py:46-54` の `reconstruct_top16` で実装済 (Session #42)。

### 3.3 ★ Top7-10 復元 不可能 ★

production data は top1-6 までしか格納されていない:
- 11 点 variant で top7-10 を 3列目に使うと → V15 model 再 inference 必須
- → **LEAK risk 発生** ★

→ 本 Session の 11 点 formation は **top1-6 のみ使用** で leak-free 設計。

---

## 4. 11 点 formation 設計 (leak-free)

top1-6 のみ使用、 11 通り組合せ:

```
top1-{top2,3,4,5,6} の任意 2 頭ペア = C(5,2) = 10 通り
+ top2-top3-top4 = 1 通り
= 11 通り
```

これは "top1 軸 box5" + "top2-top3-top4" 1 点 の hybrid。

### 4.1 7 点 vs 11 点 比較条件

| 項目 | 7 点 (現状 V15) | 11 点 (本 Session) |
|------|----------------|-------------------|
| 軸 | top1 | top1 (10 点) + 軸なし (1 点) |
| 2列目 | top2-3 | top2-6 (拡張) |
| 3列目 | top3-6 | top2-6 のもう 1 頭 |
| total | 7 bets | 11 bets |
| invest/R | 700 円 | 1,100 円 (100 円/点) |
| LEAK risk | なし | なし (top1-6 のみ使用) |

---

## 5. user prompt 指定 spec との差異 (★ 重要 ★)

| 項目 | user prompt | 本実装 | 理由 |
|------|------------|--------|------|
| 7 点 3列目 | top4-8 (5頭) | **top3-6 (現状 V15 baseline)** | production saved bets 構造 |
| 11 点 3列目 | top4-10 (7頭) | **top4-6 + 軸非依存** | top7-10 は LEAK risk のため不採用 |

→ user の "top1-2-7" 構想 (3列目 7頭) は production 保存 score の制約上 実現不可。
→ 代わりに **top1-6 のみで 11 点に拡張** する代替設計を採用。
→ user 意図 (大幅拡張で hit 率向上 vs 配当低下の trade-off 計測) は維持。

---

## 6. 結論

- OK: data/daily_predictions/ + data/daily_results/ で 8 開催日 (3/14-4/25) backtest 可能
- OK: top4-6 は production data から leak-free 復元可能
- OK: 11 点 formation は top1-6 のみ使用設計で leak-free
- NG: 既存 session69_horse_scores.csv (V15 再 inference) は使用禁止
- NG: user prompt の "top4-10" 11 点 spec は LEAK risk のため変更 → top1-6 11 点 で代替

→ 次 step: B 領域で 7 点 / 11 点 formation logic + backtest 実装
