# Session #69 D: 7 点 vs 11 点 投入判定 + 推奨 plan

**作成**: 2026-05-09 (Session #69 D)
**判定 source**: Session #69 C 結果 (280 R, 3/14-4/25, leak-free)

---

## 1. 判定 logic

| 条件 | 閾値 | 結果 | 判定 |
|------|------|------|------|
| ΔROI ≥ +5pt (有意改善) | +5.00pt | +2.28pt | NG |
| bootstrap CI95 lower bound > 0 | > 0 | -17.55pt | NG |
| P(11点 > 7点) ≥ 70% | ≥ 70% | 56.2% | NG |
| EV/R Δ ≥ 0 | ≥ 0 | **-20 円/R** | **NG** ★ |
| 投資効率 (Δ ROI / Δ 投資 比) | > 0 | -5,710 / 109,600 = -5.2% | NG |

→ 5/5 NG → **11 点 投入 NO-GO 確定**

---

## 2. 結論

### 2.1 メイン結論

**11 点 拡張は 280 R で改善なし、 投入 NO-GO**

- ROI delta +2.28pt は誤差範囲 (CI95 [-17.55, +24.24])
- 投資額 +57% (192K → 302K) で絶対損失は **+5,710 円増加**
- EV/R は -20 円/R 悪化

### 2.2 surface 別 戦術示唆 (将来 検討対象)

ただし 内訳分析で **surface 別 大差** を発見:
- **芝**: 11 点 +18.1pt 改善 (131 R)
- **ダート**: 11 点 -11.2pt 悪化 (149 R)

→ 「芝のみ 11 点、 ダートは 7 点維持」 hybrid 戦術の検討余地
→ ただし 131 R で +18pt の信頼性 低 (CI 計算未実施)
→ 5/16 V18 trial 後に サンプル積上げて 再判定

### 2.3 条件別 戦術示唆

- 条件 C (15+頭 良馬場): 11 点 **-11.4pt 悪化** → 多頭数で V15 top4-6 拡張は逆効果
- 条件 A (8-14頭 良馬場): 11 点 +6.0pt → 中頭数では 11 点 やや有利
- 条件 D (1200-1400m): 11 点 +6.5pt → 短距離は 11 点 やや有利

---

## 3. 推奨 plan

### 3.1 5/16 (V18 trial) まで

→ **7 点 V15 baseline 維持** (現状運用変更なし)
→ 5/9 投票方針: 新潟 12R 案B改 ¥700 (絶対遵守)
→ 累計 +13,530 円 維持

### 3.2 5/16 以降 (V18 投入有無 で分岐)

#### case I: 5/16 V18 GO (sib_exp 順調)
→ V18 7 点 で運用開始、 11 点 検討は **後回し** (V18 sample 蓄積優先)

#### case II: 5/16 V18 NO-GO (sib_exp 不調)
→ V15 7 点 継続
→ 5/末 - 6/上 で **+200-400R 蓄積後** 再 backtest 検討
→ 検証対象: 芝のみ 11 点 / 条件 A,D のみ 11 点 (hybrid)

### 3.3 user prompt の "top4-10 11 点" spec の取り扱い

→ **fundamental 不可**: production data に top7-10 score 未保存
→ V15 model で再 inference は LEAK risk (V15 は year≤25 全期間学習済)
→ 真の "top4-10" 11 点 検証は **WF 再 train + 再 score 必須** = 数日工数

→ 6/月以降 V20 学習時に 全頭 score を production save する 設計変更を検討
→ 実現すれば top4-10 11 点 を真に leak-free で評価可能

---

## 4. 数週間後 (5月末-6月) 再 backtest 計画

### 4.1 必要 sample 数

bootstrap CI95 width を ±10pt 以下にするには:
- 現状: 280R で width 41pt
- 推定: 500R で width 約 30pt、 800R で 約 23pt
- ★ 1,000-1,200R 蓄積で結論安定 ★

### 4.2 蓄積 timeline

- 平均 35R/開催日 × 週 2 日 = 70R/週
- 5/9 - 6/末 (約 7 週) = +490 R → 累計 約 770 R
- 6/末 で 1 回目 再 backtest
- 7/末 で 2 回目 再 backtest (累計 約 1,000R)

### 4.3 再 backtest スコープ

1. **同 logic で再実行** (top1-6 のみ 11 点)
2. **surface 別 ROI** 信頼区間 (芝 +18pt が有意か)
3. **条件 A,D のみ** 11 点 hybrid 戦術 試算
4. **production score 拡張** 検討 (top10 まで save するなら全 spec 検証可)

---

## 5. 投資保護 確認

| item | 値 |
|------|----|
| V15 model file md5 | 不変 (本 Session read-only) |
| predict_core.py | 不変 |
| daily_predict.py | 不変 |
| schtasks 41 件 | 不変 |
| 5/9 投票方針 | 新潟 12R ¥700 (不変) |
| 累計収支 | +13,530 円 (5/9 朝時点) |
| 撤退余裕 | +63,530 円 |

→ ✅ V15 production 完全保護
→ ✅ 5/9 投票 不変
→ ✅ 本 Session は read-only audit 完結 (production 経路 影響 0)
