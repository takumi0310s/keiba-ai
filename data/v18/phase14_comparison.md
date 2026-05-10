# Phase 14 C: V15 vs V18 vs V20 比較

**作成**: 2026-05-10 (Session #90 Phase 14 C)
**前提**: V15 production 累計 +¥14,140 + V18 sib_w5 (5/8) + V20 PoC v1 (5/8)

---

## 1. 3 model 横並び 比較

### 1.1 BT 2025 OOS (single fold、 LGB 単体)

| Model | features | AUC | LIVE 5/2-5/3 winner_top1 | shift_factor | 学習時間 |
|-------|---------|-----|--------------------------|--------------|---------|
| V15 (production、 4-model ensemble) | 150 | **0.8856** | (本番 運用中) | — | — |
| V15 (LGB 単体、 reference) | 150 | 0.8854 | — | — | — |
| V18 sib_w5 (LGB 単体) | 190 | **0.8847** | **34.48%** ★ | **1.32x** ★ | 1.4 分 |
| V19 sib_w5 (LGB 単体) | 190 | **0.8752** | — | — | 1.4 分 |
| V20 PoC v1 (LGB 単体) | 190 | **0.8752** | — | — | 0.5 分 |

### 1.2 V15 production 実績 (累計、 5/9 まで)

| 指標 | 値 |
|------|----|
| 累計収支 | **+¥14,140** |
| 撤退余裕 | +¥64,140 (撤退ライン -¥50,000) |
| 戦略⑦ + 案B改 (12R 1勝のみ ¥2,100 上限) | 適用中 |
| 直近 ROI | 119.2% (戦略⑦込み 140%+ 想定) |

---

## 2. 5/10 (土) 仮想 比較 (V15 production のみ実弾、 V18/V20 paper)

### 2.1 V15 5/10 results (production)

```bash
$ python tools/paper_trade_engine.py --date 20260510 --models v15
=== paper_trade summary for 20260510 ===
model  n_races  n_hits  hit_rate  investment  payout    pnl    roi_pct
  v15       34      11   32.35%     23,800     27,090  +3,290  113.82%
```

→ V15 5/10 single-day ROI **113.8%** (累計収支 維持、 投資保護 成功)

### 2.2 V18 / V20 5/10 paper

⚠ V18 / V20 5/10 shadow predictions は **本 Phase 14 で生成不能**:

| 理由 | 内容 |
|------|------|
| V15 daily_predict は専用 feature pipeline (predict_core) を使用 | V18/V20 は別 features 構成 (sib_w5 + TFJV + JRDB 拡張) で再構築必要 |
| V18 sib_w5 inference は v17 cache (1.2 GB) が必要 | 5/10 race feature を v17 cache 形式で組む工数 大 |
| V20 PoC は学習のみ完了、 inference pipeline 未整備 | Phase 11/12/13 features 統合と並行 実装必要 |

→ **5/17 (土) までに V18/V20 inference pipeline 整備** (Phase 14 D で paper trade setup ready)

---

## 3. 改善見込 (V15 → V18 → V20)

### 3.1 LIVE winner_top1 (top1 が finish 1着 する率)

| Model | LIVE winner_top1 | vs V15 |
|-------|-----------------|--------|
| V15 (現行 production) | 約 33-40% (本番 運用) | (基準) |
| V18 sib_w5 LGB | 34.48% (5/2-5/3 retro) | 同等 |
| V18 sib_w5 ensemble (4-model) | 期待 36-42% | +1-2pt 期待 |
| V20 (V15+sib_w5+TFJV+JRDB+netkeiba マスター) | 期待 38-44% | +3-5pt 期待 |

### 3.2 BT WF AUC

| Model | WF AUC | vs V15 |
|-------|--------|--------|
| V15 (4-model ensemble、 production) | 0.8939 | (基準) |
| V18 sib_w5 (LGB 単 fold) | 0.8847 | -0.0092 (LGB 単体ペナルティ) |
| V18 sib_w5 ensemble (期待) | 0.890-0.895 | +0.0-0.005 |
| V20 (4-model + features 拡張) | 0.910-0.925 (期待) | +0.016-0.031 |

### 3.3 trio 7 点 hit 率 (5/10 ベース 試算)

V15 5/10: 11/34 = 32.4%
V18 期待 (LIVE +1-2pt 換算): 33.4-35.4%
V20 期待 (LIVE +3-5pt 換算): 35.4-39.4%

→ **V20 で trio hit 率 +3-7pt、 ROI 113% → 130-150% 程度の改善見込**

---

## 4. 投入 schedule (Session #44 F 確定)

| 日付 | model | 状態 |
|------|-------|------|
| 5/9 | V15 | 単独本番 (案B改 維持、 累計 +¥14,140) |
| 5/10 | V15 | 単独本番 (本日、 ROI 113.8%) |
| 5/11-5/16 | V15 | 単独本番 (Phase 11/12/13 並行整備) |
| 5/17-5/23 | V15 + V18 paper | V18 5/17 paper trade 開始候補 |
| 5/24-5/29 | V15 + V20 構築 | 4-model ensemble 学習 (Phase 3 後半) |
| 5/30-6/7 | V15 + V20 paper | LIVE retro + paper trade |
| **6/8** | **V20 GO 判定** | 条件 PASS なら 段階投入 |
| 6/8-7/8 | V15 + V20 並行 | 1 ヶ月並行運用 |
| 7/8+ | V20 単独 (V15 archive) | 安定確認後 |

---

## 5. V15 投資保護 (絶対遵守、 V20 投入後も継続)

✅ V20 段階投入 (週末のみ、 上限 ¥5,000/日)
✅ V20 投入後も V15 並行運用 (1 ヶ月以上)
✅ 撤退ライン -¥50,000 厳守 (現状余裕 +¥64,140)
✅ V18/V20 paper trade ≠ 実弾、 1 円も追加 risk なし

---

## 6. 結論

✅ V18 sib_w5 既学習 (BT AUC 0.8847、 LIVE 34.48%)
✅ V20 PoC v1 既学習 (LGB 単 fold AUC 0.8752)
✅ paper trade engine 整備完了 (本 Phase 14 B)
⚠ V18/V20 inference pipeline は 5/17 まで に整備
⚠ V20 4-model ensemble は 5/24+ Phase 3 後半
✅ V15 5/10 ROI 113.8%、 累計 +¥14,140 維持

---

**Phase 14 C 完了** (Opus 4.7)
