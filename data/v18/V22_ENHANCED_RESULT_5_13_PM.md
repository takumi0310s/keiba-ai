# V22 enhanced 6-fold WF 結果 (5/13 深夜、 honest report)

実行: 2026-05-13 PM-night、 Opus 4.7、 user 6h 自律実行
所要: 853 s (~14 min、 FT skip のため V22 base 27min/fold 比 大幅短縮)

## ★ 結論 (honest) ★

**mean Grid AUC: 0.8776**

| baseline | AUC | vs V22 enhanced |
|---------|-----|---------------|
| **V15 (本番)** | **0.8939** | enhanced -0.0163 ❌ |
| V22 base (前 retrain) | 0.8800 | enhanced -0.0024 ❌ |
| V22 enhanced (本) | **0.8776** | — |

★ **V15 越え 未達** ★
★ **V22 base より 僅か悪化** ★

## fold 別 詳細

| fold | LGB | XGB | FT | IR | Grid AUC | 状態 |
|------|-----|-----|-----|-----|----------|-----|
| 20 | 0.858 | 0.859 | skipped | 0.782 | 0.8610 | OK |
| 21 | 0.864 | 0.867 | skipped | 0.874 | 0.8858 | OK |
| 22 | 0.867 | 0.868 | skipped | **0.779**↓ | 0.8697 | IR collapse |
| 23 | 0.869 | 0.871 | skipped | **0.650**↓↓ | 0.8709 | IR 大 collapse |
| 24 | 0.879 | 0.881 | skipped | 0.868 | **0.8875** | 健闘 (V15 接近) |
| 25 | 0.879 | 0.881 | skipped | 0.875 | **0.8910** | 健闘 (V15 接近) |
| **mean** | 0.869 | 0.871 | — | 0.804 | **0.8776** | |

## 失敗 / 学び (honest)

### 1. CUDA OOM (282 features 重すぎ)

V22 base 177 features → V22 enhanced 282 features (+105 from features_merged_all)。
FT-Transformer (d_token=64) で validation step に 9.78 GiB 必要、 GPU 15.99 GiB のうち 11.5 GiB 既使用で OOM。

修正試行:
- batch_size 512 → 256: 失敗
- d_token 64 → 32: 失敗 (val step は batch 関係なく n_features * embedded sizeで膨らむ)
- IR d_model 128 → 64: OK (これは run できた)

最終的: **FT skip** で LGB+XGB+IR 3-model Grid ensemble に変更。

→ 282 features は GPU 16 GB で FT が動かない。 features 選別 (LGBM importance top 100) or GPU 32GB 必要。

### 2. IR collapse fold 22 (0.78) + fold 23 (0.65) ★ 大問題 ★

V22 base 時 fold 22 IR collapse (0.7765) 対策で seed/d_model 修正したが、 282 features では **fold 23 で 更に悪化 (0.65)**。

原因 仮説:
- 282 features の noise が IR attention に悪影響
- d_model 128 → 64 で 表現力低下 (V22 base は d_model 64 だった、 enhanced は OOM のため 同じく 64 だが noise 増大)
- IR は 高 features count に 弱い (attention 計算は features 線形 scale だが noise 比 increase)

### 3. 105 extra features の 効果なし

期待: +0.026-0.059 AUC
実 結果: +0 to -0.0163

仮説:
- V15 cache 既存 145 features と 重複 (例: prev3_finish vs prev4/5_finish 高相関)
- expanding window で計算した features 多いが、 V15 v13.5b 既存 feature と 似た情報
- jrdb_jo features (cid_idx 等) は LGB importance で 低かった可能性

## fold 24/25 健闘 (V15 接近)

- fold 24 Grid 0.8875 (V15 0.8939 から -0.006)
- fold 25 Grid 0.8910 (V15 0.8939 から -0.003)

これは **2024/2025 data で V15 越え 寸前**。 features 選別 + GPU 強化 で 0.89+ 可能性。

## V15 投資保護 完全 (本日も遵守)

- V15 .pkl.gz / predict_core.py / daily_predict.py / app.py 完全不変
- V22 enhanced は別 file (models/v22_enhanced/enhanced_wf_summary_*.json)
- 累計収支 +13,530 円 / 撤退余裕 +63,530 円 影響なし

## 5/24+ 計画 影響

### V20 真の構築 path (修正)

audit D で case B (本 Phase 13 + V22 enhanced 試行) 採用したが、 V22 enhanced は V15 越え 未達。

修正 path:
1. **5/24+ JV-Link RT + features 選別**:
   - LGB importance top 100 features select
   - 282 → 100 で OOM 解消、 FT 復活可能性
   - sentiment_merged (272K rows × 12 cols) も 追加
2. **GPU 増強検討**:
   - 現 RTX 4070 Ti SUPER 16GB
   - 32GB+ なら 282 features full ensemble OK
3. **IR 安定化**:
   - features 選別後 IR collapse 解消可能性
   - alternative: IR モデル complete redesign

### V20 投入 timing

- orig (audit D case C): 7/1
- case B 部分実行 (本 Phase 13 + V22 enhanced): 6/15+ (audit time line)
- 本日結果 → **V15 越え 課題、 V20 真の構築は 5/24+ JV-Link + features 選別 後 再評価**

V20 投入 7/1 は **依然 妥当**、 但し 本日の results で features 単純追加だけでは V15 越え 困難 が判明。

## 158h+ マラソン哲学 遵守

- ✅ data 駆動 (実 GPU 学習 + 6-fold WF で 検証)
- ✅ V15 投資保護 完全
- ✅ ★ fabrication 防止 ★ (期待 +0.026 だったが 実 -0.016、 honest report)
- ✅ 学び 記録 (282 features 重すぎ / IR collapse 持続 / 重複 features 多い)

## next action (user 帰宅後 推奨)

1. **本 V22 enhanced 結果確認** (honest delta -0.016)
2. **5/24+ JV-Link 加入** 計画通り
3. **features 選別 戦略** 検討 (LGB importance top 100)
4. **V15 production 完全継続** (V22 enhanced は production 投入候補 から外す)

## 出力

- `models/v22_enhanced/enhanced_wf_summary_20260513_171649.json` (6 fold 結果)
- `train/train_v22_enhanced.py` (FT skip 版、 282 features)
- `train/features_merge_all.py` (105 features 統合)
- `train/features_merge_sentiment.py` (sentiment 96%+ match)
- `data/features_merged_all.csv` (180K × 107 cols)
- `data/features_sentiment_merged.csv` (272K × 12 cols)

## まとめ

★ V22 enhanced 0.8776、 V15 0.8939 から -0.016、 V15 越え 未達 ★

105 extra features 単純追加では V15 を 越えられない。 features 選別 + GPU 増強 + IR redesign が 5/24+ 真の path。

しかし fold 24/25 で 0.887-0.891 と V15 接近、 features 選別で 突破可能性は 残る。
