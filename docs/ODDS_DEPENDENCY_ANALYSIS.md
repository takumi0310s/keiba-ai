# オッズ依存分析

**作成**: 2026-05-23  
**目的**: V15/V21 のスコアにおけるオッズ系特徴量の寄与を定量化

---

## 1. V15 オッズ依存度 (5/23 feature importance 実測)

### 直接オッズ特徴量 (Pattern B に追加される 8 特徴量のうちオッズ系)

| feature | V15 LGB gain寄与 | 説明 |
|---------|----------------|------|
| `pop_rank_change` | 0.83% | 基準オッズ比での人気順変動 |
| `oz_base_pop_rank` | 0.14% | 基準人気順位 (OZ CSV) |
| `odds_change_rate` | 0.12% | オッズ変化率 |
| `oz_tansho_base_log` | 0.12% | 基準単勝オッズ log |
| `prev_odds_log` | 0.00% | 前走オッズ (不活性) |
| `odds_sharp_drop` | 0.00% | 急落フラグ (不活性) |
| **合計** | **1.21%** | — |

### V15 Top 特徴量 (オッズ系なし)

| rank | feature | gain寄与 |
|------|---------|---------|
| 1 | `paci_ninki_idx` | 16.93% ← JRDB 人気指数 (間接) |
| 2 | `paci_jockey_exp_wr` | 16.34% ← 騎手勝率 |
| 3 | `paci_jockey_exp_3rd` | 14.78% ← 騎手複勝率 |
| 4 | `jrdb_ze_idm_avg` | 9.40% ← 速度指数 |
| 5 | `training_time_filled` | 5.36% ← 調教タイム |

**結論**: V15 の直接オッズ系は **1.2%** のみ。スコア上位の主因は騎手・速度指数・調教。

### paci_ninki_idx の位置づけ

`paci_ninki_idx` (16.9%) は JRDB PACI ファイルの人気指数。発走前日配信で、レース当日の確定オッズとは異なる。
→ 間接的に公衆評価を反映するが、「当日確定オッズが高い馬を上位に置く」構造ではない。

---

## 2. V21 オッズ依存度 (TYB 含む)

### V21 LGB gain 上位 (5/23 実測)

| rank | feature | gain寄与 | 種別 |
|------|---------|---------|------|
| 1 | `tyb_tansho_odds` | 23.01% | ★ TYB 単勝オッズ ★ |
| 2 | `tyb_fukusho_odds` | 21.54% | ★ TYB 複勝オッズ ★ |
| 3 | `jrdb_ze_idm_avg` | 9.14% | 速度指数 |
| 4 | `training_time_filled` | 5.16% | 調教タイム |
| 5 | `paci_ninki_idx` | 4.03% | JRDB 人気指数 |

**V21 TYB オッズ合計**: tyb_tansho + tyb_fukusho = **44.55%**

### TYB あり / なし の影響

| 状況 | 挙動 |
|------|------|
| TYB 取得成功 | tyb_tansho/fukusho_odds が実際の直前オッズ (発走 ~15min 前) を反映 |
| **TYB 未取得 (5/23 実績)** | **これら 2 列 = 0 → gain 44.5% の情報が欠落 → スコア ≒ V15** |

**5/23 問題**: TYB なしの V21 は V15 と実質同等。比較の意味なし。

---

## 3. market_dependency_test 既知データ (V12 Pattern A)

```
Baseline (Pattern A, 67 features): AUC 0.8019
No prev_odds_log (66 features):     AUC 0.7993
→ AUC drop: -0.26 bp (前走オッズ除去の影響は微小)
```

**注**: この test は Pattern A (odds_log 除外済み) から更に `prev_odds_log` を除いた結果。
V15/V21 Pattern B (odds_log 含む) での odds 除去影響は別途検証が必要。

---

## 4. 「戦績悪いがオッズ1位の馬」が高スコアになる問題

### V15 での構造

V15 は直接オッズ系 1.2% のみ → **「オッズ1位だから高スコア」にはなりにくい**。
`paci_ninki_idx` (16.9%) が上位に効くが、これは JRDB 総合評価で純粋な市場オッズとは異なる。

**結論**: V15 で「戦績悪いがオッズ1位の馬」が上位に来る場合、主因は `paci_ninki_idx` の高値
(= JRDB がその馬を高評価している) であり、raw オッズそのものではない。

### V21 での構造 (TYB あり時)

TYB `tyb_tansho_odds` (23%) + `tyb_fukusho_odds` (21.5%) = 44.5%。
**TYB オッズが低い (人気馬) = スコアが高くなる**構造。

→ V21 は意図的に「直前 TYB オッズ」を強い信号として採用している。
→ これは「市場の直前予測を AI が利用」する設計で、leak ではない (PRE-RACE確認済)。

---

## 5. オッズ割合変更の検討 (分析のみ、V15 変更なし)

### オッズ系を下げる影響 (V21 仮説)

| シナリオ | 期待変化 |
|---------|---------|
| tyb_tansho/fukusho_odds を除外 | AUC -0.0018 〜 -0.003 程度 (推定) |
| pop_rank_change を除外 | AUC 変化 微小 (現在 0.83% gain) |
| paci_ninki_idx を除外 | AUC -0.015 程度 (推定、gain 17%) |

**結論**: V15 の直接オッズ依存は 1.2% で問題ない。
V21 の TYB オッズ依存 (44.5%) は設計上の意図 — TYB 取得を安定させることが最優先。

### 次フェーズでの検討事項

- V21 WF ablation: tyb_tansho/fukusho_odds を除いた場合の AUC 変化を計測
- 「オッズ信号」 vs 「pure 能力信号」の分離: non-odds V21 版との比較
- 現状: TYB fix (7z path + _inject_tyb_features) を 5/24 weekend で検証してから判断

---

## 6. 推奨アクション

| priority | action |
|---------|--------|
| 🔴 高 | 5/24 weekend で TYB fetch 動作確認 (7z fix + inject fix) |
| 🟡 中 | tyb_tansho/fukusho_odds ablation → V21 の TYB 依存が本当に有益か検証 |
| 🟢 低 | V21 non-odds 版 (TYB odds 除外) のサイドバイサイド比較 |
| ⚪ 保留 | V15 オッズ割合変更 → 現在 1.2% で問題なし、変更不要 |

*V15 production 完全不変 — V21 は paper 検証中*
