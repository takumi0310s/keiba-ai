# calibrator v1 vs v2 shadow compare (5/16)

**実施日**: 2026-05-16
**source**: `tools/strategy_layer_v2.py --shadow 20260516 --calibrator {v1,v2}`
**目的**: 21 sample orig calibrator vs 315 sample retrain v2 の shadow eval 比較
**G1 day 影響**: 0% (shadow only、 Discord 通知なし、 production 不変)

## 1. 全体 stats

| metric | v1 (orig 21 sample) | v2 (retrain 315 sample) | delta |
|--------|---:|---:|---:|
| total races | 35 | 35 | 0 |
| recommended | 30 | 30 | 0 |
| total inv | 58,100 円 | 54,600 円 | **-3,500 円 (-6.0%)** |
| p_calibrated mean | 0.583 | 0.481 | -0.102 |
| p_calibrated std | 0.252 | 0.213 | -0.039 |
| ev_top1 mean | 4.53 | 3.87 | **-0.66 (より慎重)** |
| ev_top1 std | 3.04 | 2.80 | -0.24 |

## 2. bet_size 分布 (★ v2 が より graduated ★)

| bet_size | v1 | v2 | delta | 意味 |
|---------:|---:|---:|---:|------|
| 0 (skip) | 5 | 5 | 0 | EV<1.0 race 同数 |
| 700 (1x) | 1 | 3 | +2 | v2 は確信度低時に慎重 |
| 1400 (2x) | 5 | 6 | +1 | 中庸 EV race |
| 2100 (3x) | 24 | 21 | -3 | ★ v1 saturate、 v2 抑制 |

## 3. honest 解釈

### 3-1. v1 の問題
- iso(p>=0.3) = 1.00 saturated → top1_score 0.3+ の全 race で p_calibrated boost
- 結果: 30 R 中 24 R が 3x bet (saturation)
- inv 58,100 円 / race = 平均 1,936 円 (本来 1,400 円基準なら 38% 過剰)

### 3-2. v2 の改善
- iso(p>=0.3) = 0.59 → 過信解消
- ev_top1 mean 4.53 → 3.87 で 適切に下方修正
- inv 54,600 円 / race = 平均 1,820 円 (より bet 規律)

### 3-3. 真の効果は paper shadow eval 必須
- ROI 効果は 5/17-5/24 蓄積後 計測 (★ shadow csv に 実 finish merge 後 ★)
- 単一日 (5/16 35 race) では over-fit 検出不可
- 30 race 蓄積後判定が honest

## 4. top1_score range (5/16)

- min: 0.316 / max: 0.736
- ★ 全 race が p=0.3 以上 ★ → v1 だと全部 saturation 域、 v2 は適切に discriminate

## 5. 5/16-5/17 G1 day 本番影響

- 投票実体: ★ 既存 `tools/race_auto_notify.py` の戦略⑦ ★ (v1/v2 どちらも介入なし)
- 5/16 today + 5/17 ヴィクトリアM: V15 + 戦略⑦ + 案 B 改 strict のみ
- shadow csv は read-only artifact、 Discord 通知 0、 cumulative 書き込み 0
- 累計 +13,530 円 完全維持

## 6. 5/18+ 運用 plan

```powershell
# 5/18 朝 daily_predict + save_all_horse_scores 後
python tools/strategy_layer_v2.py --shadow 20260518 --calibrator v1
python tools/strategy_layer_v2.py --shadow 20260518 --calibrator v2
```

出力:
- `data/v21/strategy_v2_shadow_20260518.csv` (v1 calibrator)
- `data/v21/strategy_v2_shadow_20260518_v2.csv` (v2 calibrator)

夜 (daily_results.py 後) で 両方 evaluate、 30 race 蓄積後 honest 判定。

## 7. 採用候補

| # | scenario | 採用条件 |
|---|----------|---------|
| A | v2 default 化 | 30 race shadow eval で v2 ROI >= v1 ROI |
| B | v1/v2 並行運用継続 | 結果僅差時 |
| C | calibration off | v1/v2 共に raw 下回る時 (極端 case) |

5/16 simulation のみでは判定不可 (★ honest ★)。
