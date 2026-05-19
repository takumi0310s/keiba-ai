# V15 Top-5 Rerank Audit (2026-05-19)
Sub-task 強-2: V15 inference 後 post-process rerank による top-5 hit rate 向上 path 探索

---

## 1. データ確認結果

### predict_core.py 出力形式
- スコア列名: `df['スコア']` (float, 0-1 範囲)
- 馬番列名: `df['馬番']` (int)
- 最終ソート: `df['AI順位']` 昇順 (`スコア` 降順に相当)
- top-6 馬に対して trio 7点買い目を生成 (`generate_trio_bets(df_sorted)`)

### cumulative_results.csv 列構成 (n=662 settled)

| 列 | 内容 |
|----|------|
| top1_num / top2_num / top3_num / top4_num | 予測 1-4 位の馬番 |
| top1_score | 予測 1 位のスコア (non-null: 153/662) |
| top2_score / top3_score | **常に 0** (not stored properly) |
| top1_finish / top2_finish / top3_finish | 予測 1-3 位の実際の着順 |
| trio_hit | trio 7 点が的中したか (1.0=的中) |
| trio_bets_str | top-6 馬番の bet 文字列 |

**データ制約 (HONEST)**:
- top2-6 のスコアが cumulative_results.csv に記録されていない (全て 0 or null)
- top5_num, top6_num の列が存在しない
- top4-6 の実着順が不明
- 完全スコアベクトルが取得できる行: **0 件**
- スコア + 着順 両方が有効な行: **132 件** (top1_score のみ)

---

## 2. 現行ベースライン指標

| 指標 | 値 | n |
|------|-----|---|
| trio 的中率 (7点買い) | **22.2%** | 662 |
| 予測 top3 の set hit (いずれか 1 頭が実際 top3 入り) | **86.1%** | 649 |
| top1 実 top3 入り率 | 55.2% | 649 |
| top2 実 top3 入り率 | 46.2% | 649 |
| top3 実 top3 入り率 | 34.8% | 649 |

条件別 trio 的中率:

| 条件 | n | trio hit |
|------|---|----------|
| A | 204 | 29.4% |
| C | 175 | 20.6% |
| D | 225 | 20.0% |
| B | 16 | 18.8% |
| X | 18 | 5.6% |
| E | 11 | 0.0% |

---

## 3. Rerank 5 候補の設計説明

### a. Probability Tempering (`rerank_tempering`)
```
probs = scores / scores.sum()
probs_tempered = probs^(1/T) / sum(probs^(1/T))
```
- T < 1: 分布を均等化 (低スコア馬が上昇)
- T = 1: 変化なし (恒等変換)
- T > 1: 分布を尖らせる (高スコア馬がより上位へ)

**数学的制約**: tempering は単調変換 → top-5 の **順位は変化しない**。
スコアが全て正の場合、score^k の大小関係は score の大小関係と同一。
top-5 の構成馬は baseline と **完全に同じ**。

### b. Odds Divergence Boost (`rerank_odds_divergence`)
```
rerank_score = score × (1 + alpha × divergence)
divergence   = AI複勝圏確率 / 市場複勝圏確率 - 1
```
AIが市場より高く評価している馬 (divergence > 0) を上位へ。
odds が None の場合は baseline と同一。

### c. Top-5 Spread (`rerank_spread`)
```
近接ペア (|score[i] - score[i-1]| < threshold) の後続馬に penalty を乗算
```
スコア差が小さい「実質同スコア」の馬群を整理し、より差のある馬を引き上げる。

### d. Score + Rank Average (`rerank_rank_average`)
```
final_rank = (raw_score_rank + normalized_score_rank) / 2
```
normalized_score_rank: スコアを [0, 1] 正規化したランク。
同じスコアデータから 2 種のランクを取って平均するため、**順位は変化しない** (恒等変換)。

### e. Condition-Aware (`rerank_condition_aware`)
tempering の T パラメータを条件別に切り替え:

| 条件 | T | 意図 |
|------|---|------|
| A, E | 1.5 | 少頭数/中距離: top1 重視 |
| C | 1.2 | 大頭数/長距離/良: top1 やや重視 |
| D | 0.7 | 短距離: 混戦多いので分散 |
| X | 0.7 | 大頭数/悪馬場: 荒れやすいので分散 |
| B | 1.0 | 変化なし |

tempering と同様に **単調変換** のため top-5 構成は変化しない。

---

## 4. 実際の hit rate 計算 (data + simulation)

### 4-a. 実データベースの評価
- cumulative_results.csv では top2-6 のスコアが未記録
- full score vector なしに rerank は実施不可能
- **結論: 実データによる hit rate 比較は不可能** (データ制約)

### 4-b. Monte Carlo Simulation (N=10,000 races, 12頭, softmax スコア分布)

| Method | Top-5 Coverage | Delta vs Baseline |
|--------|---------------|-------------------|
| baseline | 48.10% | - |
| tempering (T=0.8) | 48.10% | **+0.00%** |
| odds_divergence (alpha=0.5) | 48.00% | -0.10% |
| spread | 47.41% | -0.69% |
| rank_average | 48.10% | **+0.00%** |
| condition_aware | 48.10% | **+0.00%** |

### 4-c. より現実的なシミュレーション (市場ノイズ σ=0.2, AI ノイズ σ=0.3)

| Method | Top-5 Coverage | Delta vs Baseline |
|--------|---------------|-------------------|
| baseline | 46.22% | - |
| odds_div (alpha=+0.3) | 44.33% | -1.89% |
| odds_div (alpha=-0.3) | 35.77% | -10.45% |

---

## 5. ★ Honest Verdict ★

### 数学的に変化なし (恒等変換)
- **Tempering**: 単調変換 → top-5 順位は baseline と同一。delta = 0%。
- **Rank Average**: 同じデータから 2 ランクを平均 → 恒等変換。delta = 0%。
- **Condition-Aware**: tempering の条件別版 → 同様に恒等変換。delta = 0%。

### シミュレーションで negative signal
- **Odds Divergence**: AI が市場より優位なシナリオでは **悪化する** (-1.9% 〜 -7.4%)。
  原因: 市場は多数の参加者の集合知 → AI 単体より calibration が高い場合が多い。
  V15 genuine WF AUC = 0.8678 は高いが、市場の平均 AUC も 0.75+ と推定される。
  結果: divergence boost = AI が外れた時のペナルティが大きくなる。
- **Spread**: top-5 から valid 馬を除外するため **常に悪化** (-0.7% 〜)。

### 真の signal (+2% 以上) を持つ rerank: **0 件**

採用候補なし。

---

## 6. 根本原因分析

V15 の post-process rerank が効果を持つには以下の条件が必要:
1. **複数馬の full score vector** が利用可能 (現在 cumulative_results には top1 のみ記録)
2. rerank に用いる**外部信号** (odds, pace, jockey change) が AI スコアより追加情報を持つ
3. その外部信号が **実際の着順と正の相関**を持つ (odds divergence は逆相関の可能性)

現状の V15 では:
- predict_core.py が 70% AI + 6% pop + 6% dist_apt + ... の複合スコアを使用
- オッズ情報 (pop_rank) は既にスコアに組み込み済み
- post-process で再度 odds を使うと **二重カウント** になる

---

## 7. データ整備 Recommendation (6/17 採用判定前に必要)

| 作業 | 優先度 | 効果 |
|------|--------|------|
| cumulative_results.csv に top2-6 のスコアを記録 | HIGH | rerank paper eval が可能になる |
| top5_num, top6_num 列の追加 | HIGH | 上に同じ |
| レース全頭スコアの記録 (race_scores_{race_id}.pkl 等) | MEDIUM | より正確な simulation |
| 実着順全頭 (1着〜全頭) の記録 | MEDIUM | top-k coverage の正確な計算 |

---

## 8. 6/17 採用判定 Path

**現時点の判定**: 全 5 候補を **NO-GO** とする。

理由:
1. 数学的に恒等変換のものは信号なし (tempering / rank_average / condition_aware)
2. odds divergence は simulation で negative signal
3. spread は valid 候補を排除し悪化

**6/17 以降に再評価できる条件**:
- `tools/daily_predict.py` に全頭スコア保存を追加 (top2-6 スコアも記録)
- 新方式でのデータを 4 週間 (≥40 レース) 蓄積
- その後 `simulate_rerank_hit_rate` で OOS hit rate 比較
- **閾値: top-5 coverage delta ≥ +2% (absolute), p < 0.05**

---

## 9. 実装ファイル

| ファイル | 内容 |
|---------|------|
| `tools/v15_top5_rerank.py` | 5 rerank 関数 + `get_top5_reranked()` + `simulate_rerank_hit_rate()` |
| `tests/test_v15_top5_rerank.py` | 42 tests (全 PASS) |

**predict_core.py / daily_predict.py / app.py / race_auto_notify.py は一切無改変**。
