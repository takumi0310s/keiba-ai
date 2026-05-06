# V18/V19 winner_top1 検証 (GO 条件 #2)

**作成**: 2026-05-06 PM (Session #32 C)
**判定対象**: GO 条件 #2 「winner_top1 ≥ 45% (5pt 余裕)」
**結論**: **🔴 34.5% で 45% 大幅未達 = NO**

---

## 1. 既存数値 (Session #10 distribution_shift_analysis.md)

| dataset | winner_top1 | winner_top3 |
|---------|------------|-------------|
| BT_2025_OOS | **47.8%** | 78.8% |
| Retro_raw (5/2-5/3) | **34.5%** | 72.4% |
| Retro_calibrated | **34.5%** | 72.4% |
| 5/9 GO 基準 | **≥ 45%** (5pt 余裕) | (今回基準なし) |

---

## 2. winner_top1 とは

V18 model の予測 TOP1 馬が **実際 1 着になった率**。
V18 が単勝投票機の中核なので、**TOP1 精度 = V18 期待値の根幹**。

| 値 | 解釈 |
|----|------|
| 47.8% (BT) | 学習データで 約半分 当たる、V18 BT ROI 295% の根拠 |
| 34.5% (retro) | 本番で **TOP1 が 1着になる率が大幅低下**、ROI 期待値 大幅低下 |
| -13.3pt 劣化 | 本質的に V18 は本番で動作しない (現状) |

---

## 3. monotonic 変換 (calibration / softmax) で改善しない理由

calibration / softmax T=1.0 は **rank 不変の monotonic 変換**:
```
raw: [0.1, 0.05, 0.02, ...] → cal: [0.21, 0.13, 0.05, ...]
rank: [1, 2, 3, ...] (不変)
```

→ TOP1 馬の選定 = rank に依存 → **monotonic 変換では winner_top1 改善せず**。

検証 (distribution_shift_analysis.md):
- Retro_raw winner_top1 = 34.5%
- Retro_calibrated winner_top1 = 34.5% (**完全同値**)
- → calibration では 1 馬も rank 入替なし

---

## 4. 5/9 投入時の期待 ROI 試算

仮に V18 を 5/9 投入する場合:
- 単勝 1,000 円/日 × 1 R = 1,000 円
- 期待 ROI 計算:
  - winner_top1 = 34.5% (rank 1 → 1着 確率)
  - 単勝オッズ平均 = 6.5 (人気上位の傾向)
  - 期待 payback = 0.345 × 6.5 = **2.24** = 224%
  
→ 表面 224% は 110% 超えだが:
- **sample 9-25 bets で CI 巨大** (95% 下限 ROI < 110% の可能性)
- **34.5% は BT 47.8% から 13.3pt 劣化** (model 本来の精度の 70% しか出ていない)
- **過大評価リスク 大**

---

## 5. winner_top1 ≥ 45% 達成 path

| 対策 | 改善見込み | 工数 | 5/9 までに可能? |
|------|-----------|------|--------------|
| calibration | 0 (monotonic) | 0 | 効果なし |
| feature shift 修正 | +5-10pt 期待 | 90min 調査 + 数時間 修正 | **不可** |
| V18 再学習 (5/3 まで含めた) | +3-8pt 期待 | 数時間 | **不可** |
| feature 追加 (premium 拡充等) | +2-5pt 期待 | 数日 | 不可 |

→ **5/9 までに 45% 達成は不可能**。

---

## 6. GO 条件 #2 判定

| 観点 | 値 | 判定 |
|------|----|----|
| 現状 winner_top1 | 34.5% | 45% 基準 -10.5pt 未達 |
| calibration 後 | 34.5% (不変) | 改善不可能 |
| 5/9 までに 45% 達成可能性 | 0% | **時間的に不可** |

**判定**: 🔴 **NO** (winner_top1 < 45%、5/9 までに改善不可能)

---

## 7. 5/9 投入 status (3/6 NO 確定)

| # | 条件 | 判定 |
|---|------|------|
| 1 | ROI ≥ 110% | 🔴 NO (sample 不足、過大評価リスク) |
| 2 | **winner_top1 ≥ 45%** | 🔴 **NO (34.5%、改善不可)** |
| 3 | shift 真因 calibration | 🔴 NO (monotonic 変換で rank 不変) |
| 4 | pipeline 統合 | 🟡 準備のみ |
| 5 | fall-back 機構 | 未判定 (D で) |
| 6 | 5/8 dry-run | 未判定 (5/8 で) |

3 条件 NO 確定 (#1, #2, #3) → **5/9 投入 NO-GO 確定**。

---

## 8. 結論

GO 条件 #2: 🔴 **NO 確定** (winner_top1 34.5% < 45%、5/9 までに改善不可能)

monotonic 変換 (calibration / softmax) で改善できないのが構造的問題。
**5/9 投入 NO-GO 確定** (3/6 NO で実質確定)。
5/16+ では feature shift 修正後に再評価可能、Phase 3 (5/24+) で本格対応。
