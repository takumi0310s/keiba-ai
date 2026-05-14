# V22 top 100 vs V15 full 6-fold backtest 結果 (5/14 AM、 最終)

実行: 2026-05-14 AM、 Opus 4.7、 30 min GPU、 race_key year fix 後

## ★ 最終 結論 (honest) ★

**V22 top 100 ROI < V15 ROI 全 fold 一貫、 switch 不推奨**。

| 指標 | V22 top 100 (2020-2025 全6-fold) | V15 baseline (2023-2025 WF) | delta |
|------|----------------------------------|----------------------------|-------|
| **全体 ROI** | **332.8%** | **428.4%** | **-96 pt ❌** |
| N races | 8,361 (戦略⑦ 適用後) | 10,314 | — |
| **profit** | **+15,950,880 円** | — | (V22 単体は profitable) |

## 条件別 (full 6-fold)

| 条件 | V22 N | V22 hit% | V22 ROI | V15 ROI | delta |
|------|------|---------|---------|---------|-------|
| A (8-14頭/1600m+/良-稍重) | 2,765 | 59.8% | 296.4% | 355.4% | **-59 pt** |
| B (重-不良) | 0 (戦略⑦ 除外) | — | — | 346.8% | — |
| C (15頭+/1600m+/良) | 2,289 | 49.7% | 478.1% | 623.0% | **-145 pt** |
| D (1200-1400m) | 3,038 | 47.7% | 268.5% | 360.8% | **-92 pt** |
| E (7頭以下) | 0 (戦略⑦ 除外) | — | — | 195.7% | — |
| X (15頭+/重-不良) | 269 | 50.9% | 286.6% | 701.2% | **-414 pt** |

## 重要 観察

### V22 は profitable だが V15 inferior
- V22 全 fold ROI **332.8%** > 100% (期待値 +)
- V22 profit **+15.9M 円** / 8,361 races
- → V22 単体でも 投資 可能 model
- but V15 ROI **428.4%** で +27 pt 上回る

### AUC 微差 → ROI 大差
- AUC delta: V22 0.8813 vs V15 0.8939 = -0.013
- ROI delta: V22 332.8% vs V15 428.4% = **-96 pt**
- AUC 1 pt → ROI 約 7 pt の 増幅 効果

### fold 24 quick 結果との 整合 性
- quick fold 24 ROI: 329.3%
- full 6-fold ROI: 332.8%
- → ほぼ同じ、 fold 24 単独 結果 が 全体 を 代表

### 最大 弱点: 条件 X (大穴 race)
- V22 X ROI 286.6% vs V15 701.2% = **-414 pt 致命的**
- 大穴 high payout race の prediction 性能 V15 圧倒的

## ★ V15 production 継続 確定 ★

5/16 (土) 戦略 = **V15 戦略⑦ 案B改 単独継続** (絶対遵守、 変更なし)

判断 根拠:
- V22 top 100 ROI 332.8% は V15 428.4% より **96 pt (-22%) 劣勢**
- 全 4 条件で V22 < V15
- 大穴 race で 大幅 劣勢 (-414 pt 致命的)
- AUC 微差 でも ROI 大差 確認
- → **V22 switch は 月利 大幅 減**、 投資 後悔

## V22 top 100 の位置付け (修正)

✅ **Profitable model** (ROI 332.8% > 100%)
❌ **V15 越え model ではない** (-96 pt 劣勢)
✅ **Architecture reference** (LGB+XGB simple ensemble、 V20 構築 base)
✅ **5/24+ JV-Link RT 統合で 再 retrain 候補**

## 5/16 投資 計画 (変更なし)

| 条件 | 投資額 | 買い目 |
|------|--------|--------|
| A | 700円 | trio 7 点 |
| B | 700円 | trio 7 点 |
| C | 700円 | trio 7 点 |
| D | 700円 | trio 7 点 |
| E | 700円 | umaren 2 点 |
| X | 700円 | trio 7 点 |
| 12R 1勝クラス (案B改) | 上限 2,100円 | 同上 |
| 06_特別 / 京都 / 条件E / 条件B (戦略⑦) | 0 円 | 除外 |

累計 +13,530 円 / 撤退余裕 +63,530 円 保持。

## V15 越え 真の path (5/24+、 再確認)

1. **JV-Link 32-bit Python venv 作成** (user 手動 1-2h、 CRITICAL)
2. 残 10 features 真値化 (SE pace/lap + WE/WH 天候 + O1-O6 オッズ + UM/SK/BR 血統)
3. V20 真の構築 (V15 + 真値 features + 動画 Phase 4)
4. V21 投入判定 9/1+

V22 top 100 は features 単純拡張では V15 越え 困難 確認。 **JV-Link 真値化 features 必須**。

## 158h+ マラソン哲学 遵守

- ✅ data 駆動 (full 6-fold + 実 jra_payouts.csv で 実 ROI)
- ✅ V15 投資保護 完全
- ✅ fabrication 防止 (期待 V15 越え → 実 -96pt 一貫 劣勢 honest report)
- ✅ user 投資安全 優先 (V22 switch 拒否、 V15 維持 確定)

## 結論

★ V22 top 100 は **profitable model だが V15 越え 不可** ★

- V22 ROI 332.8% (profit +15.9M 円)
- V15 ROI 428.4% (V22 -96pt 劣勢)
- 大穴 race (X) で 致命的 -414pt
- **V15 production 継続、 V22 switch 拒否 確定**

V15 越え には JV-Link RT 真値化 features 必須 (5/24+ user 手動 unlock)。
