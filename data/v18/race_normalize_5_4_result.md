# race-level normalization 試作結果 (Phase 2.5 / 2026-05-04)

**最新 commit**: 9c88d27c (Phase 2.5 静音化スクリプト) base + race_normalize 一連
**目的**: v18/v19 5/2-5/3 retro で全 bet=0 となる distribution shift を race-level normalization で解決

---

## 1. 問題: distribution shift の正体

### 1.1 horse-level prob distribution

| dataset | n | mean | median | p95 | p99 | max |
|---|---:|---:|---:|---:|---:|---:|
| BT v18 p_ens (tansho) | 47,497 | 0.0631 | 0.0238 | 0.2823 | 0.5897 | **0.9863** |
| Retro v18 p_tansho_raw | 932 | 0.00185 | 0.00097 | 0.00637 | 0.0226 | **0.1538** |
| Retro v18 p_tansho_cal (Platt) | 932 | 0.00266 | 0.00231 | 0.00831 | 0.0153 | **0.2128** |

→ Platt scaling は max を 0.154→0.213 にしか持ち上げられず、Phase 2 filter (p≥0.5) に届かない

### 1.2 race-level distribution (核心)

| dataset | n_race | race_max_p mean | race_max_p p95 | race_max_p max | race_sum_p mean | top1/top2 ratio mean | winner_top1 | winner_top3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BT 2025 OOS | 3,455 | **0.347** | 0.748 | 0.986 | **0.753** | 4.13 | **47.8%** | 78.8% |
| Retro 5/2-5/3 raw | 67 | 0.013 | 0.035 | 0.154 | 0.025 | 4.37 | 34.5% | 72.4% |
| Retro 5/2-5/3 cal | 67 | 0.020 | - | 0.213 | - | - | 34.5% | 72.4% |

### 1.3 shift attribution

- **race_max_p factor (BT/retro): 27.7x** ← 主要因
- **top1/top2 ratio diff: -0.24** (≈ ほぼ同等) ← 構造保たれてる
- **winner_top1 rate diff (BT-retro): +13.3pt** ← 副次的に rank 劣化

### 1.4 判定

> **GLOBAL_SCALING_SHIFT が dominant** — relative ranking 構造は維持されているため race-level normalization で **bet>0 化が可能**。winner_top1 rate 13pt 差は別要因 (feature shift 疑い、後日調査)。

`data/v18/distribution_shift_analysis.json` / `.md` 参照。

---

## 2. 3案比較 (winner_known 29 races, 387 horses)

normalize 適用前 (raw): bet=0 (max=0.016 で filter 通らず)

### 2.1 案1 softmax `e^(logit(p)/T) / Σ`

| T | race_max_p mean | sum mean | bet (p≥0.5,ev≥1.2) | win | ROI |
|---:|---:|---:|---:|---:|---:|
| 0.3 | 0.795 | 1.000 | 27 | 9 | 1786% |
| 0.5 | 0.671 | 1.000 | 22 | 9 | 2141% |
| **0.7** | 0.563 | 1.000 | 16 | 6 | **2386%** |
| 1.0 | 0.440 | 1.000 | 9 | 3 | 1450% |
| 1.5 | 0.313 | 1.000 | 1 | 1 | 2660% |

### 2.2 案2 power `p^(1/T) / Σ`

T=0.3〜2.0 で測定。softmax とほぼ同一結果 (sum=1 制約から)。

### 2.3 案3 rank-scale (各レース max を target に linear rescale)

| target_max | race_sum_p mean | bet (p≥0.5,ev≥1.2) | win | ROI |
|---:|---:|---:|---:|---:|
| 0.347 (BT mean) | 0.903 | 0 | - | - |
| 0.5 | 1.302 | 26 | - | 1210% |
| 0.7 | 1.822 | 43 | - | 1512% |
| 0.9 | 2.343 | 49 | - | 1422% |

→ sum>1 で確率制約破る、bet 過多、推奨度低

---

## 3. 推奨: softmax (default T=1.0)

### 3.1 採用理由

1. **理論的整合性**: temperature scaling は標準 ML 手法、sum=1 制約満たす
2. **rank order 保持**: monotonic transform、馬選定不変
3. **race-aware**: 各レース内で再分配、cross-race calibration の影響受けない
4. **T で調整可**: 蓄積 sample で再 tuning 可能

### 3.2 T 推奨

- **default T=1.0**: 標準温度、sample 適正、ROI 中庸 (race_max_p mean 0.440 = BT 0.347 比 1.27x やや過信)
- **T=1.5 (BT 模倣)**: race_max_p mean 0.292 で BT 0.347 に最近接、ただし bet=1 で sample 不足
- **T=0.7 (アグレッシブ)**: bet 多 (16) ROI 高 (2386%)、sample 確保したいフェーズ向け

production 初期は **T=1.0**、accumulate 後に再 tuning が穏当。

---

## 4. 5/2-5/3 retro 再実行結果 (--normalize softmax)

### 4.1 単勝 (v18) — winner_known 29 races

| T | filter (p,ev) | bet | win | hit% | inv | pay | ROI |
|---:|:---:|---:|---:|---:|---:|---:|---:|
| 0.5 | (0.5, 1.2) | 22 | 9 | 40.9% | 2,200 | 47,100 | **2141%** |
| 0.7 | (0.5, 1.2) | 16 | 6 | 37.5% | 1,600 | 38,170 | **2386%** |
| **1.0** | (0.5, 1.2) | **9** | **3** | **33.3%** | **900** | **13,050** | **1450%** |
| 1.5 | (0.5, 1.2) | 1 | 1 | 100% | 100 | 2,660 | 2660% |

### 4.2 複勝 (v19) — 全 67 races (is_top3 GT)

| T | filter (p,ev) | bet | hit | hit% | inv | pay~ | ROI~ |
|---:|:---:|---:|---:|---:|---:|---:|---:|
| 0.5 | (0.7, 1.1) | 19 | 9 | 47.4% | 1,900 | 14,190 | **747%** |
| 0.7 | (0.7, 1.1) | 12 | 5 | 41.7% | 1,200 | 7,595 | **633%** |
| **1.0** | (0.7, 1.1) | **1** | **1** | **100%** | **100** | **798** | **798%** |

複勝オッズ = 単勝×0.3 仮定。実際の複勝オッズで再計算が望ましい。

### 4.3 winner_top1 rate (重要)

normalize は monotonic なので winner_top1 rate (34.5%) は **不変**。
race-level scaling で「filter 通る bet が増える」のみ。

### 4.4 注意点

- **sample サイズ**: 9-22 bets は統計的に不十分、95%CI は広い
- **高 ROI のソース**: avg winning odds ~63x (longshot wins 寄与)
- **winner_known=43%**: 残り 57% races で winner が pred top1-3 外 → 評価対象外
- **複勝 odds 推定**: tansho×0.3 は近似、実複勝 odds で再評価が望ましい

---

## 5. 結論

✅ **race-level normalization で bet>0 化を確認** (5/2-5/3 retro)
✅ **softmax T=1.0 で sum=1 + race_max_p mean 0.440 (BT 0.347 比 1.27x)**
✅ **rank order 不変、winner_top1 rate 34.5% は normalize 前後で同じ**
🟡 **sample 不足**: 9-22 bets で ROI 1450-2386% の信頼区間は広い → 5/16-5/24 蓄積で検証
⚠️ **真の calibration 改善ではない**: prob を見せかけ上 BT に近づけているだけ。1着馬選定能力 (winner_top1 34.5%) は raw / norm 同じ
⚠️ **winner_top1 13pt 劣化 (BT 47.8%→retro 34.5%)** は normalize で解消されない、別要因 (feature shift) 調査が必要

---

## 6. 関連ファイル

| 種別 | path | 役割 |
|------|------|------|
| 分析 | `data/v18/distribution_shift_analysis.json` / `.md` | shift 定量化 |
| 比較 | `data/v18/normalization_compare_results.json` | 3案 sweep |
| ツール | `tools/race_normalize.py` | normalization API + CLI |
| ツール | `tools/v18_v19_retro_full.py` | `--normalize <method> --T <T>` 追加、`--from-csv` で再評価 |
| 試作 | `tools/analyze_v18_v19_distribution.py` | 分析スクリプト |
| 試作 | `tools/compare_normalization_methods.py` | 比較スクリプト |

## 7. 使い方

```bash
# CSV から normalize + retro 評価のみ
python tools/v18_v19_retro_full.py \
    --from-csv data/v18/v18_v19_retro_full_predictions.csv \
    --normalize softmax --T 1.0 \
    --output-md data/v18/v18_v19_retro_norm_T1.md

# 単独 normalize ツール (predictions に列追加)
python tools/race_normalize.py data/v18/v18_v19_retro_full_predictions.csv \
    --prob-col p_tansho --method softmax --T 1.0
```
