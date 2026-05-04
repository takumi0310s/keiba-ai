# D. v18/v19 Platt Scaling 試作 + Retro 再評価 結果

生成: 2026-05-04 (Opus xhigh, Session#8)

## 結論

🟡 **Platt scaling は OOS calibration improvement に effective だが、5/2-5/3 retro の bet=0 問題は未解決**

### 主要数値

| 指標 | v18 単勝 | v19 複勝 |
|------|---------|---------|
| BT 2025 OOS calibration | ✅ Brier -0.0008 / LogLoss -0.0040 | ✅ Brier -0.0003 / LogLoss -0.0010 |
| 5/2-5/3 retro raw max p | 0.154 | 0.142 |
| 5/2-5/3 retro calibrated max p | 0.213 | 0.156 |
| Required p (filter 0.5+) | × 全く到達不能 | × 全く到達不能 |
| Best calibrated bet | n=1, ROI 2,660% (sample 不足) | n=0 |

→ 5/2-5/3 では **どんなフィルタでも実用的 bet 数得られず**。distribution shift が根本原因。

## Calibration 詳細

### v18 単勝 OOS 2025 (n=47,497)

```
BEFORE (raw):
  Brier = 0.0514, LogLoss = 0.1787
  全 bin で under-confidence (gap +0.007〜+0.107)

AFTER (Platt):
  Brier = 0.0506, LogLoss = 0.1747
  Δ Brier = -0.0008, Δ LogLoss = -0.0040
  全 bin で gap |≤0.05| (大幅改善)

Platt 係数: coef=0.9453, intercept=0.3029
```

### v19 複勝 OOS 2025

```
BEFORE: Brier 0.1066, LogLoss 0.3329 (mild under-conf 一貫)
AFTER:  Brier 0.1062, LogLoss 0.3319
Platt: coef=1.0252, intercept=0.1539
```

### Reliability bins (v18 before/after)

| bin | n_before | gap_before | gap_after |
|-----|--------:|-----------:|----------:|
| [0.10, 0.20) | 3,539 | +0.070 | +0.001 |
| [0.20, 0.30) | 1,581 | +0.048 | +0.020 |
| [0.30, 0.50) | 1,405 | +0.063 | -0.029 |
| [0.50, 0.70) | 490 | +0.038 | -0.015 |
| [0.70, 0.90) | 199 | +0.107 | +0.047 |

→ **after では gap がほぼゼロ + 上下に分散** (calibration 成功)

## 5/2-5/3 retro 再評価 (calibrated 版)

### 問題: distribution shift

```
BT 2025 OOS:
  v18 max p ~0.94 (calibrated 後はさらに上)
  bet 候補 (p>=0.5, EV>=1.2): 642 件 / 47,497

5/2-5/3 retro (2026):
  v18 max p (raw)  : 0.154
  v18 max p (cal)  : 0.213
  bet 候補 (p>=0.5): **0件**
  bet 候補 (p>=0.3): **0件**
  bet 候補 (p>=0.10, EV>=1.0): 1件 (n=1 で統計無意味)
```

→ Calibration 後でも 5/2-5/3 で bet 数増えず。

### v18 calibrated 1件 bet 詳細

```
p_min=0.10, ev_min=1.0:
  n=1, win=1, ROI 2,660%
  → サンプル小すぎて意味なし
```

### v19 calibrated bet=0

複勝 calibrated max p = 0.156、p>=0.30 の filter で 0件。

## 真の原因: Distribution Shift

| 期 | model output max p | base rate |
|----|------------------|----------|
| 2015-2024 train cache | ~0.94 | ~0.07 |
| 2025 OOS test | ~0.94 (BT 通り) | ~0.07 |
| **2026 5/2-5/3 (本番)** | **~0.15** | ~0.07 (実1着率) |

→ **2026 の features 分布が 2024 までと違う**。
   model が出す確率 max が 6x 縮小 → bet 不能。

### 推定要因

1. **Feature pipeline 変化**: 5/2-5/3 では parse_shutuba live で取得、cache とは特徴量計算経路違う
2. **JRDB feature 欠損**: KTA/KAA stop (4/5以降), TYB/SKB 5/3未公開, netkeiba premium stale
3. **特徴量分布シフト**: 2026年GW 特殊レース構成 (G1/G2/G3 集中) ?

## 次セッション対策案

### 🟠 案1: Race-level probability normalization (即試作可)

```python
# 各レース内で probability を sum=N (N=出走頭数) に正規化
df['p_norm'] = df.groupby('race_id')['p_tansho'].transform(
    lambda x: x / x.sum() * len(x))  # OR / x.sum() で 1.0 範囲
```

→ 最大確率馬 ~ 1.0/N の理論値に揃う、bet 候補抽出可能。

### 🟠 案2: 特徴量分布検証 (5/2-5/3 vs 2024)

```python
# 5/2-5/3 race_df の各 feature と 2024 cache の同 feature を分布比較
# KS test, Wasserstein distance 等で shift 検出
```

→ どの feature が distribution shift 主因か特定。

### 🟡 案3: Live shutuba + feature pipeline 整合性検証

cache 経由 (training) と live shutuba 経由 (本番) で同じ race_id の feature 値が一致するか比較。

## 5/9 投資判断への影響

🟢 **影響なし** — 5/9 は V15 単独運用 (案B改)、v18/v19 は本番未投入。

Phase 2.5 残作業: **race-level normalization + 特徴量分布検証** を 5/16前までに完了し、v18/v19 部分実弾の判断材料にする。

## 出力ファイル

- `tools/calibrate_v18_v19.py` (Platt scaling 試作)
- `tools/apply_calibration_retro.py` (calibrator 適用)
- `data/v18/models/v18_tansho_calibrator.pkl` (LR object)
- `data/v18/models/v19_fukusho_calibrator.pkl`
- `data/v18/calibration_5_4_summary.json`
- `data/v18/v18_v19_retro_calibrated.csv`

## TL;DR

- ✅ Platt scaling 実装、OOS calibration improvement 確認
- ❌ 5/2-5/3 retro の bet=0 は **calibration では解決せず**
- 🔴 真の原因は **distribution shift** (model out max p 6x 縮小)
- 🔧 次対策: race-level normalization or feature distribution audit
- 🟢 5/9 投資への影響なし (V15 単独運用継続)
