# paci_ninki_idx の正体 + V15 人気依存 真値

**作成**: 2026-05-23  
**目的**: paci_ninki_idx の正体確認 + 前回「1.2%」報告の矛盾を解明

---

## 1. paci_ninki_idx の正体

### ソース

JRDB **KYI ファイル** (`jrdb_paci.csv`) の `ninki_idx` 列。  
フォーマット位置: offset 140, width 5 (5桁整数、スケール不明)。  
配信タイミング: **前日夜〜当日朝 06:00** (pre-race)。

### 正式名称と内部文書の記述

| ドキュメント | 記述 |
|------------|------|
| `parse_jrdb.py:372` | `('ninki_idx', 140, 5), # 人気指数` |
| `train_v141_paci_tierA.py:11` | `paci_ninki_idx: 人気指数 (100%)` |
| `docs/T4_LEAK_AUDIT_AUTOMATION_2026_05_17.md` | `人気指数 (odds-derived だが pre-race snapshot)` |
| `docs/SUBTASK_18_V152_FE_COMPLETE_DESIGN_2026_05_16.md` | `人気指数 (= odds-derived、 LEAK 隣接)` |
| `docs/B4_V15_FULL_VERDICT_HONEST_RECHECK_2026_05_18.md` | `corr_target = +0.4477 / safe / morning_06 pre-race` |

### 結論: paci_ninki_idx の正体

**JRDB が前日に発行する「人気指数」で、早朝オッズを主要入力とした JRDB 独自の複合スコア。**

| 属性 | 内容 |
|------|------|
| データ種別 | JRDB KYI (人気指数) |
| 入力ソース | **odds-derived** — JRDB の基準オッズ (`base_odds`) 等の早朝市場オッズを主入力 |
| 配信時刻 | 前日〜当日朝 06:00 (pre-race、 リーク ❌) |
| 値の性質 | 連続スコア (この R での実測: 80〜200+ 程度) |
| 確定オッズと同一か | **NO** — 確定オッズは投票締切後。ninki_idx は前日 JRDB 推計値 |
| 生市場オッズと同一か | **NO** — JRDB の独自計算。 `base_odds` (基準オッズ) とも別フィールド |
| 市場人気と相関するか | **YES** — odds-derived のため市場人気と強く相関 |
| leak 判定 | **SAFE** (pre-race snapshot) — Pattern A でも使用可 |

### ペッパーミルの paci_ninki_idx=180 の意味

> JRDB が「このメンバー内でこの馬は相対的に人気が高いと予測する」スコアが 180 点。  
> 実際の投票日オッズは 7.1 倍 (4 番人気) で、前日 JRDB 評価とほぼ一致している。  
> → **「JRDB の事前人気予測が高い」** が正確な解釈。「市場オッズが低い」とは同義ではないが高相関。

---

## 2. V15 の「人気/オッズ系」feature 完全リスト

### 前回の「直接オッズ系 1.21%」(ODDS_DEPENDENCY_ANALYSIS.md)

| feature | LGB gain% | 種別 |
|---------|----------|------|
| `pop_rank_change` | 0.83% | 基準人気 vs 当日人気の変動 |
| `oz_base_pop_rank` | 0.14% | 基準人気順位 (OZ CSV) |
| `odds_change_rate` | 0.12% | 基準オッズ vs 当日オッズの変化率 |
| `oz_tansho_base_log` | 0.12% | 基準単勝オッズ log |
| `prev_odds_log` | 0.00% | 前走オッズ (不活性) |
| `odds_sharp_drop` | 0.00% | 急落フラグ (不活性) |
| **直接市場オッズ 合計** | **1.21%** | — |

### ★ 見落としていた特徴量 ★

| feature | LGB gain% | XGB gain | 種別 |
|---------|----------|---------|------|
| `paci_ninki_idx` | **16.93%** | 630.17 (rank 3) | **odds-derived JRDB 人気指数** |

### V15 人気/オッズ系 完全リスト (合計)

| カテゴリ | 合計 gain% |
|---------|-----------|
| 直接市場オッズ features | **1.21%** |
| JRDB 人気指数 (`paci_ninki_idx`) | **16.93%** |
| **★ 広義の人気/オッズ系 合計 ★** | **★ ~18.1% ★** |

---

## 3. 矛盾の解明

### 「人気寄与 10%」vs「paci_ninki_idx 16.9%」の矛盾

前回 REASONING_OOTANI_PEPPER.md で書いた「人気寄与 ~10%」は**誤り**。

| 元の発言 | 実際の根拠 | 誤りの原因 |
|---------|----------|---------|
| 「人気寄与 ~10%」 | スコア差 0.0126 に占める pop_rank_change 等の寄与を推定 | `paci_ninki_idx` を「JRDB 総合評価」とカテゴライズして人気系から除外 |
| 「直接オッズ依存 1.2%」 | 直接 odds 名の features のみ集計 | `paci_ninki_idx` が "odds-derived" であることを見落とした |

### 前回 ODDS_DEPENDENCY_ANALYSIS.md の「1.2%」は何を見落としたか

`paci_ninki_idx` を「JRDB 人気指数 **(間接)**」と書き、人気系から除外した。  
しかし内部文書は明確に **"odds-derived"** と記述している。

> `docs/T4_LEAK_AUDIT`: "人気指数 (odds-derived だが pre-race snapshot)"  
> `docs/SUBTASK_18`: "人気指数 (= odds-derived、 LEAK 隣接)"

→ **1.2% の報告は「直接 raw market price features のみ」の数値であり、`paci_ninki_idx` を含めていなかった。**

---

## 4. ★ 真の人気依存度 (訂正) ★

### 定義別の人気依存度

| 定義 | 対象 features | Gain合計 |
|------|-------------|---------|
| **狭義**: 直接市場価格 (raw odds) | `pop_rank_change` + `oz_base_pop_rank` + `odds_change_rate` + `oz_tansho_base_log` | **1.21%** |
| **広義**: 人気相関 features 全体 | 上記 + `paci_ninki_idx` | **~18.1%** |

### どちらが「正しい」人気依存か

両方とも正しく、**目的によって使い分ける**:

| 問い | 答え |
|------|------|
| 「V15 は当日の確定オッズを直接使っているか?」 | → **1.21%** のみ (Pattern B feature の `oz_*` 等) |
| 「V15 は市場の人気傾向 (早朝オッズ評価) を反映しているか?」 | → **18.1%** (ninki_idx 含む広義) |
| 「確定オッズがわからなくても V15 は予測できるか?」 | → できる (98.79% はオッズ非依存) |
| 「JRDB の事前人気評価を無効にすると AUC はどれだけ下がるか?」 | → 推定 -0.015 (ninki_idx gain 17%) |

### 前回レポートの訂正

| ドキュメント | 旧記述 | 訂正 |
|------------|--------|------|
| `ODDS_DEPENDENCY_ANALYSIS.md` | 「直接オッズ系は 1.2% のみ」 | → 狭義は正しい。ただし paci_ninki_idx (16.9%) が odds-derived であることを追記すべき |
| `REASONING_OOTANI_PEPPER.md` | 「人気寄与 ~10%」 | → 誤り。paci_ninki_idx 経由の人気寄与は 16.9% で最大 feature |
| `PREDICTION_REASONING.md` | 「直接オッズ系が 1.2% のみ」 | → 同上。広義では 18.1% |

---

## 5. honest verdict

### V15 の真の人気依存構造

```
V15 LGB gain 100% の内訳 (人気/能力 分類):

能力系 (速度・調教・血統等):
  paci_jockey_exp_wr:   16.34%   ← 騎手勝率 (能力 proxy)
  paci_jockey_exp_3rd:  14.78%   ← 騎手複勝率 (能力 proxy)
  jrdb_ze_idm_avg:       9.40%   ← スピード指数 (純粋能力)
  training_time_filled:  5.36%   ← 調教タイム (純粋能力)
  ...

人気/市場評価系 (odds-derived / correlated):
  paci_ninki_idx:       16.93%   ← JRDB 人気指数 (odds-derived, 前日)
  pop_rank_change:       0.83%   ← 基準〜当日人気変動 (raw market)
  oz_base_pop_rank:      0.14%   ← 基準人気順位 (raw market)
  odds_change_rate:      0.12%   ← オッズ変化率 (raw market)
  oz_tansho_base_log:    0.12%   ← 基準単勝 log (raw market)
  [合計]                ~18.1%
```

### ペッパーミルは「人気で2位」か?

**正確には: 「JRDB の事前人気評価が高いから2位」 ≒ 半分正しい**

| 要素 | 内容 |
|------|------|
| paci_ninki_idx=180 の意味 | JRDB が前日に発行した人気スコア。このメンバーで上位評価 |
| なぜ高い? | 朝時点では 3 番人気 (5.9 倍) → JRDB の評価と一致 |
| 「市場オッズが低いから高スコア」か? | odds-derived なので相関はあるが、V15 が参照するのは JRDB の前日評価で確定オッズではない |
| 「18走1勝でも2位になれた理由」 | → JRDB がこのメンバー内でこの馬を「上位人気予測」しているため。JRDB の評価 = 公衆の早朝オッズ傾向を反映 |

**結論: ペッパーミルは「JRDB が事前人気評価を高くしたから2位」= 広義の「人気で2位」が正しい。**  
前回「JRDB 総合評価 (人気ではない)」と書いたのは不正確だった。

### V15 は「人気依存が高い」モデルか?

| 観点 | 結論 |
|------|------|
| 確定オッズを直接使うか? | **NO** — Pattern A (leak-free) では除外、Pattern B でも 1.2% のみ |
| 早朝の市場人気評価を間接使用するか? | **YES** — paci_ninki_idx (16.9%) が最重要 feature |
| 人気馬が常に高スコアか? | **半分** — ninki_idx は人気と強相関 (corr=0.44) だが、速度指数・騎手等の能力系も同等の影響 |
| 「人気馬を推す AI」か? | 過剰な表現だが、 JRDB 人気評価との一致度は高い |

---

## 6. 更新すべきドキュメント

| ドキュメント | 追記内容 |
|------------|--------|
| `ODDS_DEPENDENCY_ANALYSIS.md` | paci_ninki_idx が odds-derived である旨を追記 |
| `REASONING_OOTANI_PEPPER.md` | 「人気寄与 ~10%」→「広義の人気系 ~18%」に訂正 |
| `PREDICTION_REASONING.md` | paci_ninki_idx を「PACI人気◎」と表示するのは正確 (odds-derived 確認) |

*V15 production 完全不変 — 本ドキュメントは read-only 分析*
