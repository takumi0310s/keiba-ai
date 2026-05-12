# V22 4-ensemble full 6-fold WF 最終結果 (5/13 深夜、 93 min GPU)

## 🎯 結論

**V22 4-ens full WF mean Grid AUC 0.8800、 V15 baseline 0.8939 → delta -0.0139**。

V15 越え 未達。 fold 22 IR collapse (0.7765) が 平均 押下げ。 fold 24,25 は V15 接近 (0.889)。

## 📊 6-fold 詳細

| fold | LGB | XGB | FT | IR | 4-ens AUC-w | 4-ens Grid |
|------|-----|-----|-----|-----|-------------|-----------|
| 20 | — | — | — | — | 0.8554 | 0.8590 |
| 21 | 0.8643 | 0.8663 | 0.8641 | 0.8739 | 0.8806 | 0.8861 |
| 22 | 0.8670 | 0.8689 | 0.8591 | **0.7765**↓ | 0.8720 | 0.8722 |
| 23 | 0.8692 | 0.8705 | 0.8689 | 0.8727 | 0.8825 | 0.8858 |
| 24 | 0.8720 | 0.8728 | 0.8711 | 0.8781 | 0.8849 | **0.8893** |
| 25 | 0.8703 | 0.8714 | 0.8693 | 0.8782 | 0.8839 | **0.8874** |
| **mean** | — | — | — | — | **0.8766** | **0.8800** |

V15 baseline: **0.8939**
V22 vs V15 delta: **-0.0139**

## 🐛 main weakness

### fold 22 IR collapse (0.7765)
- 通常 IR は 0.87+ (dominant)
- fold 22 のみ 0.7765 で 突然 低下
- 原因: 学習 instability (early stopping at epoch 12)
- 修正案: seed 変更 / d_model 拡大 / より長い patience

### fold 20 全モデル 低い (Grid 0.8590)
- train 2015-2019 のみ で 小規模 (training_n=178K vs 後年 480K)
- 解決: V20 base feature が 2020+ のみのため、 Phase 24/26 features は fold 20 で 利用不可
- → fold 20 では 実質 V15 145 features のみ で 学習

## 💡 V15 越え 残 path (5/24+)

1. **IR training stability 改善**:
   - seed 変更で fold 22 collapse 回避
   - epoch 増加 + patience 拡大
   - d_model 64 → 128 拡大 (GPU 16GB 余裕あり)

2. **Phase 24/26 features を 2015-2019 にも バックフィル**:
   - 現在 V20 base に Phase 24/26 features があるが 2022+ のみ多い
   - 全年 cumulative 計算 で fold 20-21 弱点 補完

3. **hyperparameter tuning**:
   - LGB num_leaves / learning_rate 探索
   - XGB max_depth 探索
   - FT layers/heads 調整

4. **Grid weights 学習 vs grid search**:
   - 現在 Grid search 0.05 step (粗い)
   - meta-learner で 重み 推定

期待効果: 各改善 +0.005-0.015 → V22 0.89-0.91 想定 → V15 越え

## 🛡 V15 production 完全保護 (本日も遵守)

- V15 .pkl.gz / predict_core.py / daily_predict.py 完全不変
- V22 .pkl.gz は 別 file、 production 投入は 5/24+ judgment

## 5/16 (土) - 5/17 (日) 本番運用

**V15 戦略⑦ 案B改 単独継続** (絶対遵守、 変更なし)
**Strategy 8 sidecar shadow eval** (別 channel Discord、 投資 0 円)

V22 は **5/24+ tuning 後 production 投入判定**。
