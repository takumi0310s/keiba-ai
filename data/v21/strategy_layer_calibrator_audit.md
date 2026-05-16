# Strategy Layer v2 - Calibrator Audit

**実施日**: 2026-05-16
**対象 file**: `data/calibrator_v15_pilot.pkl`

## 1. file 構造

| key | 型 | 内容 |
|-----|---|------|
| `isotonic` | `sklearn.isotonic.IsotonicRegression` | 主 calibrator |
| `platt` | `sklearn.linear_model.LogisticRegression` | 副 calibrator (Platt scaling) |
| `metrics` | `dict` | before / after_iso / after_platt の Brier + ECE |
| `trained_at` | `str` | `2026-05-11T18:36:12.276578` |
| `n_samples` | `int` | **21** |

## 2. 学習 metrics

| stage | Brier | ECE |
|-------|------:|----:|
| before | 0.4698 | 0.5146 |
| after_iso | 0.1881 | 0.0000 |
| after_platt | 0.2008 | 0.0245 |

raw 確率の Brier 0.47 / ECE 0.51 は **異常に高い** (well-calibrated なら Brier 0.05-0.15, ECE 0.01-0.05 程度)。 これは raw 確率が「複勝圏」確率なのに対し、 calibration label が異なる target (おそらく 単勝 hit?) だったか、 sample size 21 だけで mean baseline が不正確 のいずれか。

## 3. predict 動作 test

```python
raw_p  = [0.05, 0.15, 0.30, 0.50, 0.70, 0.85, 0.95]
iso(p) = [0.50, 0.58, 1.00, 1.00, 1.00, 1.00, 1.00]
platt(p) = [0.558, 0.678, 0.820, 0.927, 0.973, 0.987, 0.992]
```

## 4. 重大な懸念

### 4-1. isotonic は p>=0.30 で完全飽和 (1.0)

学習 sample 21 件は **絶対的に不足**。 isotonic regression は piecewise constant のため、 sample が少ないと plateau だらけになる。 p=0.30 以上が全て 1.0 になるのは bet 戦略に致命的:

- V15 raw top1_score 分布: mean=0.171, max=0.201 (cumulative_results.csv 20 件 sample)
- 実 race の V15 確率は ほぼ **常に 0.1-0.25 範囲**
- この範囲では isotonic は 0.50 → 1.00 へ jump 直前。 監視必須

### 4-2. platt は logistic で 0.05 -> 0.56 へ膨張

raw 0.05 が calibrated 0.558 になる = 全ての低確率馬を「半分以上の確率で当たる」 と評価。 これも 21 sample の logistic fit が信頼できない 結果。

### 4-3. 学習 data が 1 日分のみ

`trained_at: 2026-05-11` から推測すると 5/10 か 5/11 試行錯誤分の sample 21 件のみ。 walk-forward 評価せず、 train data 上の Brier 改善 (0.47→0.19) も over-fit の可能性 高い。

## 5. 戦略 layer v2 での扱い

### 採用方針: **isotonic 採用、 ただし安全 clip を適用**

```python
# 元 raw を残しつつ、 isotonic を softer に
def apply_calibration_safe(raw_probs, blend=0.3):
    iso_probs = ISO.predict(raw_probs)
    return (1 - blend) * raw_probs + blend * iso_probs
```

- 純粋 isotonic 適用は **危険** (p=0.30 即飽和)
- blend=0.3 (raw 重み 0.7、 calibrated 重み 0.3) で慎重に取り込む
- ★ 5/18+ paper shadow eval で **before / after 両方記録**、 実 outcome と照合
- データ蓄積後 (n>=200 件目安) で再 train + 本格活用

### 不採用: platt

logistic で歪みが大きすぎ、 raw 0.05 → 0.56 は 戦略破壊的。 paper でも platt は適用しない。

## 6. 改善 task (out-of-scope: 後続)

- [ ] V15 確率の calibration label 再 audit (top1_finish<=3 を target にすべき)
- [ ] cumulative_results.csv の top1_score 蓄積継続 (現状 20 件 → 200+ 目標)
- [ ] 5/18 以降 paper shadow data 蓄積後 calibrator 再 train
- [ ] 条件別 (A/C/D) calibrator の必要性 検討
