# hybrid 戦略 backtest 拡張 plan (Session #82)

> Session #69 (280 R production score) を cell 別に拡張する設計。
> 作成: 2026-05-09 (Session #82)

---

## 1. Session #69 リファレンス

| 項目 | 値 |
|------|----|
| サンプル | 280 R production score (リーク防止) |
| bootstrap CI95 | [-17.55pt, +24.24pt] |
| P(11 > 7) | 56.2% |
| 結論 | 全体は 統計的同等 (NO-GO) |

→ 全体結論は NG だが、 differential が大きく hybrid 候補。

---

## 2. 拡張 backtest 設計

### 2.1 surface 別

| cell | サンプル想定 (280 R 中) | 期待 P(11>7) |
|------|---------------------|--------------|
| 芝 | ~150 R | 80%+ ★ |
| ダート | ~130 R | < 30% |

期待結果:
- 芝のみ: bootstrap CI95 lower bound > 0pt
- ダートのみ: bootstrap CI95 upper bound < 0pt

### 2.2 頭数別

| cell | サンプル想定 | 期待 P(11>7) |
|------|-----------|--------------|
| <= 7 頭 (条件 E) | ~10 R | 不明 (sample 少) |
| 8-14 頭 (条件 A/B/D) | ~200 R | 70%+ ★ |
| 15+ 頭 (条件 C/X) | ~70 R | < 40% |

### 2.3 クラス別

| cell | サンプル想定 |
|------|-----------|
| 1 勝クラス | ~80 R |
| 2 勝 / 3 勝 | ~100 R |
| OPEN / G | ~30 R (重賞投票なし) |
| 未勝利 / 新馬 | ~70 R |

→ 1 勝以下と 2 勝以上で挙動が変わる可能性。

### 2.4 distance 別

| cell | range | サンプル想定 |
|------|-------|-----------|
| 短距離 | <= 1400m | ~80 R (条件 D 中心) |
| マイル | 1500-1700m | ~50 R |
| 中距離 | 1800-2200m | ~100 R |
| 長距離 | 2300m+ | ~50 R |

---

## 3. 必要 サンプル

### 3.1 統計的有意性閾値
- 各 cell ≥ 100 R 推奨 (bootstrap CI95 が安定)
- < 50 R は noisy で結論不可

### 3.2 5/10 以降の data 蓄積 (Session #71)
- 全馬 score 完全保存 機能 ON (5/10〜)
- 月 50 R × 12 馬 = 600 score / 月
- 5/10〜6/30 で約 100 R 増加

| 期間 | 累計 R | 備考 |
|------|--------|------|
| 〜5/9 | 280 R | Session #69 baseline |
| 5/10-5/31 | +50 R = 330 R | 5/16 V18 trial 含む |
| 6/1-6/30 | +50 R = 380 R | |
| 7/1-7/14 | +25 R = 405 R | V20 投入 |

→ 7/1 V20 投入時点で **約 400 R** で再 backtest 可能。

---

## 4. 実装 plan (将来)

### 4.1 backtest 拡張 script
```python
# tools/hybrid_backtest_session82.py (将来実装、 まだ書かない)
import pandas as pd

df = pd.read_csv("data/cumulative_results.csv")

# cell 分類
def classify_cell(row):
    if row["surface"] == "芝" and row["num_horses"] <= 14:
        return "cell1_turf_small"
    elif row["surface"] == "ダート":
        return "cell2_dirt"
    elif row["num_horses"] >= 15:
        return "cell3_large"
    else:
        return "cell4_other"

df["cell"] = df.apply(classify_cell, axis=1)

# bootstrap per cell
for cell, grp in df.groupby("cell"):
    print(cell, len(grp), bootstrap_ci(grp))
```

### 4.2 KPI per cell
- ROI (mean、 median)
- bootstrap CI95
- P(11 > 7)
- 的中率
- max DD

### 4.3 採用閾値
- P(11 > 7) >= 75%
- bootstrap CI95 lower bound >= 0pt
- N >= 100 R

---

## 5. 5/10 以降の data 蓄積 (Session #71 連動)

Session #71 の全馬 score 保存機能を活用:
- `data/cumulative_results.csv` に 全 12 馬 score 列 追加済
- 5/10 から 全 R で 12 行記録
- 6/30 で 約 600-800 R 累積見込

→ 7/1 V20 投入時 **hybrid 再 backtest 可能** (cell 1 サンプル ~300 R 達成)。

---

## 6. 関連 doc

- `docs/STRATEGY_HYBRID_DESIGN.md` — 戦略確定案
- `docs/HYBRID_DEPLOYMENT_PLAN.md` — 投入 plan
- `docs/HYBRID_RISK_ANALYSIS.md` — risk 分析

---

**結論**: 5/10 〜 6/30 で data 蓄積 → 7/1 V20 投入時に backtest 拡張 実施 → 8/1 hybrid 採用判定。 backtest script 実装は 6/末 着手 (今は plan のみ)。
