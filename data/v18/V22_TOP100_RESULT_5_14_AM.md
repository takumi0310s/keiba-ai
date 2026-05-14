# V22 enhanced TOP 100 features 6-fold WF 結果 (5/14 AM、 honest)

実行: 2026-05-14 AM、 Opus 4.7、 51 min GPU

## ★ 結論 ★

**mean Grid AUC: 0.8813**

| baseline | AUC | delta |
|---------|-----|-------|
| **V15 (本番)** | **0.8939** | — |
| V22 base 4-ens (177 features) | 0.8800 | V15 -0.0139 |
| V22 enhanced 282 (FT skip) | 0.8776 | V15 -0.0163 |
| **V22 enhanced TOP 100 (FT 復活)** | **0.8813** | **V15 -0.0126** ★ 最良 ★ |

★ V15 越え 依然 未達 (-0.0126) ★ だが **V22 base + V22 enhanced 282 を 共に 上回る** (+0.0013 / +0.0037)。

## fold 別 詳細

| fold | LGB | XGB | FT | IR | Grid AUC | 状態 |
|------|-----|-----|-----|-----|----------|-----|
| 20 | 0.857 | 0.859 | 0.844 | **0.679**↓↓ | 0.8593 | IR 大 collapse |
| 21 | 0.865 | 0.866 | 0.864 | 0.872 | 0.8850 | 安定 |
| 22 | 0.868 | 0.869 | 0.866 | 0.875 | 0.8855 | 安定 (282 で 0.8697、 大 改善 +0.016) |
| 23 | 0.869 | 0.870 | 0.863 | 0.857 | 0.8750 | 安定 (282 で 0.8709、 改善 +0.004) |
| 24 | 0.879 | 0.881 | 0.876 | 0.879 | **0.8916** | V15 接近 (0.8939 -0.002) |
| 25 | 0.878 | 0.880 | 0.877 | 0.878 | **0.8914** | V15 接近 (0.8939 -0.003) |
| **mean** | 0.869 | 0.871 | 0.865 | 0.840 | **0.8813** | |

## ★ 大幅 改善 ポイント ★

### 1. fold 22, 23 IR collapse 解消
- V22 enhanced 282 (FT skip): fold 22 IR=0.78、 fold 23 IR=0.65
- V22 top100 (FT 復活): fold 22 IR=0.87、 fold 23 IR=0.86 → **IR 安定化**

理由: features 削減で attention noise 解消、 attention 表現効率向上

### 2. fold 24, 25 で V15 接近 (0.892)
- V15 0.8939 と差 -0.002 / -0.003
- 2024/2025 data では features 選別 後 V15 越え 寸前

### 3. fold 20 のみ IR 0.68 (大 collapse 持続)
- 過去 data (2015-2019) のみ で 学習、 sample size 小
- Phase 24/26 features が 2020+ のみ → fold 20 では 効かない
- これは V22 enhanced 282 fold 20 (IR 0.78) より 悪化、 features 削減 で 過去 data の variance 増大

## 失敗 / 学び (honest)

### features 削減 と IR の 相性
- 282 → 100 で IR fold 22/23 安定化 ✅
- but fold 20 IR は 削減で悪化 ❌
- → fold 20 (過去 data) は features を別途調整必要 (Phase 24/26 補完)

### V15 越え 残 課題
- top 100 でも -0.013、 features 単純 選別 では 越え 困難
- **真の 越え には**:
  1. JV-Link RT 経由 SE pace/lap、 WE/WH 天候 等 真値 features
  2. 動画 features (Phase 4)
  3. V15 cache 自体 の expanding window 再計算 (sib_*_exp 改善)

## V15 投資保護 完全 (本日も遵守)

- V15 .pkl.gz / predict_core / daily_predict / app.py 完全不変
- V22 enhanced top100 は別 file (models/v22_enhanced_top100/)
- 累計 +13,530 円 影響なし

## 5/24+ 計画 (継続 修正)

| 期間 | task |
|------|------|
| 5/14-5/16 | V15 自動運用、 5/16 (土) Strategy 8 shadow eval (要 schtask 登録) |
| 5/17-5/23 | user: JV-Link 32-bit Python venv 作成 (CRITICAL unlock) |
| 5/24-5/26 | SE/WE/WH/O1-O6 真値化 (JV-Link RT) |
| 5/27-6/8 | V20 真の構築 (top 100 + 10 真値化 features) |
| 6/15+ | V20 paper trading |
| 7/1+ | V20 production 投入判定 |

## 158h+ マラソン哲学 遵守

- ✅ data 駆動 (LGB importance で 47/282 zero gain noise 確認)
- ✅ V15 投資保護 完全
- ✅ fabrication 防止 (top 100 でも V15 越え 未達 honest report)
- ✅ 段階的 改善 (282 → 100 で +0.0037 改善 確認)

## 結論

★ V22 top100 mean Grid 0.8813、 V15 -0.0126、 V22 base + 282 を 上回る ★

V15 越え には JV-Link RT 真値 features + 動画 (Phase 4) が必要。 5/14 AI 自律 範囲 では 最良 (+0.0013 over V22 base)。

fold 24/25 で V15 接近 (0.892) → 2024/2025 data に 強い model、 5/24+ 17 features 全 真値化 後 再評価 で V15 越え 可能性 残る。
