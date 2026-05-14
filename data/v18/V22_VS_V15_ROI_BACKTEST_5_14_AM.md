# V22 top 100 vs V15 実 ROI backtest 結果 (5/14 AM、 honest)

実行: 2026-05-14 AM、 Opus 4.7、 fold 24 (2024 年) quick backtest

## ★ 結論 (honest report) ★

**V22 top 100 ROI < V15 ROI、 switch 不推奨**。 V15 production 継続。

| 指標 | V22 top 100 fold 24 | V15 baseline (2023-2025 WF) | delta |
|------|---------------------|----------------------------|-------|
| **全体 ROI** | **329.3%** | **428.4%** | **-99 pt ❌** |
| 全体 N | 2,584 | 10,314 | — |
| 全体 hit | — | — | — |
| 全体 profit | +4.86 万円 | — | — |

## 条件別 比較

| 条件 | V22 fold24 N | V22 hit% | V22 ROI | V15 ROI | delta |
|------|-------------|---------|---------|---------|-------|
| A (8-14頭/1600m+/良-稍重) | 880 | 63.1% | 345.2% | 355.4% | -10 pt |
| B (重-不良) | 0 (戦略⑦ 除外) | — | — | 346.8% | — |
| C (15頭+/1600m+/良) | 683 | 50.7% | 472.7% | 623.0% | **-150 pt ❌** |
| D (1200-1400m) | 929 | 47.9% | 231.0% | 360.8% | **-130 pt ❌** |
| E (7頭以下) | 0 (戦略⑦ 除外) | — | 195.7% | — |
| X (15頭+/重-不良) | 92 | 54.3% | 255.5% | 701.2% | **-446 pt ❌** |

★ **C / D / X で 大幅 劣勢** ★

## なぜ V22 top 100 は V15 より 悪い

AUC delta は -0.013 (V22 0.8813 vs V15 0.8939) と僅か だが、 ROI で **-99 pt = -23%** の 大幅 劣化:

1. **TOP1 軸馬 の precision 低下**:
   - V22 top 100 は features 削減で TOP1 確信度 低下
   - trio 7 点 formation は TOP1 軸馬 prediction 精度 に sensitive
   - 軸馬 失敗 → trio 全 7 点 hit rate 急落

2. **fold 24 のみ 単独 評価**:
   - V15 baseline は 2023-2025 平均
   - 直接比較 で V22 disadvantage (但し fold 24 hit rate 63% は 良好)
   - **問題 は ROI、 hit rate ではない** → 配当 低 race を 多く 取って 高 race を 取り損ねている

3. **大穴 (高配当) race 取り逃し**:
   - 条件 X (15頭+/重) は 高配当 race
   - V22 X ROI 256% vs V15 701% = **-445pt 大幅 劣勢**
   - 大穴 horse の prediction 性能 が V15 より 劣

## 戦略⑦ 除外 効果 (V22 でも 動作確認)

- 京都 + 06_特別 + 条件 E + 条件 B 除外
- 3454 race → 2584 race (-870 race 除外、 約 25%)
- 除外 race の ROI 計算なし (戦略⑦ 設計通り)

## ★ V15 production 継続 推奨 ★

判断:
- V22 top 100 ROI 329% < V15 ROI 428% (delta -23%)
- V22 hit rate は 各 条件で 維持 (47-63%) だが ROI 改善 なし
- AUC 微差 (-0.013) が ROI 大差 (-99pt) に変換
- **V22 switch は 投資 損失 大、 V15 維持 確定**

5/16 (土) 戦略 = **V15 戦略⑦ 案B改 単独継続** (絶対遵守、 変更なし)

## V15 越え path (5/24+)

1. **JV-Link RT 真値化 features** (user 32-bit Python venv 必須)
2. V20 真の構築 (V15 cache + 真値 features + 動画 Phase 4)
3. V21 投入判定 9/1+

## V22 enhanced top 100 の位置付け

- **Offline reference**: V15 越え model 構築の 比較 baseline
- **Phase 3-4 ベース**: 5/24+ JV-Link RT 統合 で 再 retrain 用 architecture
- **Production 投入は 当面 なし** (V15 維持)

## V15 投資保護 完全 (本日も遵守)

- V15 .pkl.gz / predict_core / daily_predict / app.py 完全不変
- 累計 +13,530 円 / 撤退余裕 +63,530 円
- 5/16 戦略 変更なし

## 158h+ マラソン哲学 遵守

- ✅ data 駆動 (実 jra_payouts.csv で 実 ROI 計算)
- ✅ V15 投資保護 完全
- ✅ ★ fabrication 防止 ★ (V22 期待 V15 越え → 実 -99pt 大幅劣勢、 honest report)
- ✅ user 投資安全 優先 (V22 switch 拒否)

## 残課題

### V22 vs V15 full 6-fold backtest (offline 評価)
- 現 fold 24 quick (1 fold) のみ
- 全 fold (2020-2025) backtest で 一貫性 確認 可能
- 工数: 30 min GPU
- 但し fold 24 結果が 既に decisive → full は user 判断 後 (時間あれば 着手 可能)

### V20+ 真の改善 (5/24+ JV-Link 32-bit venv 必須)
- 残 10 features 真値化 (SE pace/lap、 WE/WH 天候、 O1-O6 オッズ、 UM/SK/BR 血統)
- 動画 features (Phase 4、 7-8月)
- V22 越え + V15 越え 同時 path
