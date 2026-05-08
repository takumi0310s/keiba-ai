# Session #54 D: V5 評価 + 5/12 paper trade 投入判断

**作成**: 2026-05-09 (Session #54 D)
**前提**: A audit (V4 spec) + B features (15 候補) + C 学習 (V5 AUC 0.8187)

---

## 1. V4 vs V5 比較サマリ

| 比較軸 | V4 | V5 | delta | 判定 |
|------|----|----|------|------|
| n_features | 22 | 37 | +15 | -- |
| LGB AUC | 0.8189 | 0.8183 | -0.0006 | × |
| XGB AUC | 0.8183 | 0.8183 | 0.0000 | -- |
| Ensemble AUC | **0.8188** | **0.8187** | **-0.0001** | × |
| 学習 rows | 49,213 | 53,407 | +4,194 | (data 増分) |
| 学習 races | 4,821 | 4,827 | +6 | -- |

**判定**: V5 は V4 と **AUC 同等** (改善なし)。

---

## 2. 重要 発見 (C より、 投入判断 の根拠)

### 2-1. LEAK 1 件 検出 + 修正 (合格)

- last3f を 当該レース値で 入れた 初版 → AUC 0.867 (+0.048!)
- 修正: prev_last3f (horse 単位 shift(1)) に 変更
- 修正後 AUC 0.8187 → **leak-free 確認**

### 2-2. V4 飽和 (不合格)

- V4 22 features で 既に 高度 飽和
- odds_log 単独 で importance 70%+
- 追加 15 features は 重要度 listed but ensemble AUC 寄与なし

---

## 3. 5/12 paper trade 投入 判断

### 3-1. 投入候補 GO/NO-GO

| 軸 | 結果 | 判定 |
|---|------|------|
| AUC > V4 | × (delta -0.0001) | **NO-GO** |
| 全条件 ROI 改善 | 未確認 (低優先) | -- |
| LEAK 検出 | ✅ (last3f を pre-race 化) | OK |
| model file 保存 | ✅ (data/nar/models/keiba_model_nar_v5.pkl) | OK |
| paper trade 安全 | ✅ (NAR は paper のみ) | OK |

### 3-2. 5/12 投入判断 = **NO-GO**

**理由**:
1. V5 AUC 0.8187 は V4 AUC 0.8188 と 統計的同等 (差 -0.0001、 ノイズ範囲)
2. 投入 メリット なし (V4 維持で 同等性能)
3. paper trade では A/B test 可能だが、 改善見込まれない 候補は 投入価値 低

### 3-3. 5/12 paper trade 推奨

→ **V4 維持** (data/nar/models/keiba_model_nar_v4.pkl)
→ V5 は **保留** (data/nar/models/keiba_model_nar_v5.pkl、 archive 扱い)

---

## 4. 次 step (V5.5 or V6 候補)

V5 失敗を踏まえ、 NAR の 真の 改善には:

| 改善 path | 期待 | 工数 |
|----------|------|------|
| 1. FT-Transformer 追加 (中央 V13.5b 同様) | +0.005-0.010 | 6h |
| 2. IntraRace Attention (中央 V13.5b 同様) | +0.005-0.010 | 8h |
| 3. NAR 独自 sib_*_exp (母系 expanding、 中央と分離) | +0.002-0.005 | 6h |
| 4. NAR 独自 速度指数 (内製、 distance × time × condition) | +0.001-0.003 | 4h |

→ V6 候補: 4-model grid ensemble + NAR sib + 速度指数 = **目標 AUC 0.825-0.835**
→ 着手日: 5/30 以降 (V20 中央 投入後)

---

## 5. 5/12 paper trade 戦略

### 5-1. 推奨 戦略

```
5/12 (火) NAR paper trade 開始:
- 使用 model: V4 (data/nar/models/keiba_model_nar_v4.pkl)
- 投入金: 0 円 (paper のみ)
- 投入 race: 大井 / 船橋 / 川崎 / 浦和 全 R
- 期間: 2 週間 (5/12-5/26) で 200+ races の paper 結果
- 評価: 条件別 ROI、 AUC live、 hit rate
```

### 5-2. V5 並行 paper trade (オプション)

V5 model も並行 で paper 走らせることは可能:
- A/B 比較 で V4 と 大きな差 ないことを 実 data で確認
- 5/26 評価で V5 を archive 確定

---

## 6. NAR 開発 中期 ロードマップ

```
2026-05-09  V5 学習 (本 Session、 NO-GO)
2026-05-12  V4 paper trade 開始 (2 週間)
2026-05-26  paper 評価、 V5 archive 確定
2026-05-30  V20 中央 投入後、 V6 候補 着手 (FT + Attention + sib + 速度指数)
2026-06-30  V6 学習完了、 paper trade 開始
2026-07-15  V6 投入判断 (AUC 0.825+ 目標)
```

---

## 7. 結論

✅ V5 学習 完了 (AUC 0.8187、 V4 並み)
✅ LEAK 1 件 (last3f) 検出 + 修正
✅ V4 飽和 確認 (odds_log 70%+ 支配)

**5/12 paper trade 判定**: ★ V4 維持 NO-GO ★
- V5 は V4 と AUC 同等、 投入 メリット なし
- V5 は archive、 V4 で paper 開始

**次 step**: V6 候補 (FT + Attention + sib + 速度指数) を 5/30 以降 着手、 7/15 投入判断
