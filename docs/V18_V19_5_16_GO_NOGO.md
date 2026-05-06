# V18/V19 5/16 試行 GO/no-go 計画書

**作成**: 2026-05-06 PM (Session #31)
**判定日**: 2026-05-15 (金) 22:00
**目的**: V18 単勝 / V19 複勝 試行投入の前提条件 達成度評価

---

## 1. V18/V19 status (5/6 時点)

| 項目 | 値 |
|------|-----|
| BT ROI (2025) | V18 295.1% / V19 149.3% |
| BT AUC | V18 0.8954 / V19 0.8787 |
| retro 5/2-5/3 | 全 filter で **bet=0** (probability 過小) |
| race-level normalize 後 | bet>0 化 ROI 1450-2708% (sample 9-22 bets で CI 広い) |
| winner_top1 rate | BT 47.8% → retro 34.5% (-13.3pt 不変) |
| distribution shift factor | 27.69x (RANK_SHIFT 確定) |

詳細: `data/v18/v18_v19_retro_full_result.md`, `distribution_shift_analysis.md`

---

## 2. 引き継ぎ書 v2 §2.3 5 条件 (5/15 判定)

| # | 条件 | 5/6 status | 5/15 達成 工数 |
|---|------|-----------|--------------|
| 1 | race-level normalize 本番 pipeline 統合 (predict_core.py) | ❌ 未着手 | 30min |
| 2 | 5/2-5/15 paper retro で normalize 後 ROI > 120% | ❌ 5/2-5/3 のみ (sample 9 bets) | 自動蓄積 |
| 3 | sample 30+ bets 累積 | ❌ 9 bets | 自動蓄積 (5/9-5/15) |
| 4 | winner_top1 rate ≥ 40% | ⚠️ 34.5% (改善余地不明) | feature shift 解明後 |
| 5 | feature distribution shift 調査 | ❌ 未着手 | 90min |

→ **現時点で 5 条件中 0 達成**、5/15 までに #1, #5 のみ実施可能 (= 2/5 達成見込み)。

---

## 3. 5/9-5/15 達成可能なもの (緊急度 🟠)

### 3.1 #1 race-level normalize 本番統合 (30min)

`tools/predict_core.py` に softmax T=1.0 normalize 統合:

```python
# 既実装: tools/race_normalize.py (Session #10)
# from race_normalize import softmax_normalize
# pred_normalized = softmax_normalize(pred_raw, T=1.0)
```

→ 5/9-5/13 で 1 R で動作確認、5/14-5/15 で本番統合。

### 3.2 #5 feature distribution shift 調査 (90min)

`data/v18/distribution_shift_analysis.md` の続き調査:
- BT (2025) vs production (5/2-5/3) で各 features の mean/std/median 比較
- 27.7x scaling shift の真因 features 特定
- 修正方針 (例: feature standardization、再学習)

→ 5/13-5/14 で調査、結果次第で 5/15 GO/no-go 反映。

### 3.3 #2, #3 paper 自動蓄積

5/9 / 5/10 (土日) で paper trading 自動稼働、5/15 までに 30+ bets 蓄積見込み。

### 3.4 #4 winner_top1 改善

#5 解明 + #1 統合で 34.5% → 40%+ への改善期待、ただし不透明。

---

## 4. 5/15 (金) 22:00 判定基準

| 達成数 | 判定 | 5/16 投資 |
|--------|------|----------|
| 5/5 | 🟢 GO | V18 単勝 1,000 円/日 + V19 複勝 1,000 円/日 |
| 4/5 | 🟡 部分 GO | V18 のみ 500 円/日 (paper 並行) |
| 3/5 | 🟡 paper 継続 | 投入なし、5/24 再評価 |
| 2/5 | 🔴 NO-GO | V18/V19 投入 一切なし、Phase 3 (5/24+) で再検討 |
| 0-1/5 | 🔴 NO-GO | 同上 |

**5/6 時点予測**: 5/15 までに #1 (normalize) + #5 (shift 調査) で 2/5 達成、#2/#3 paper 蓄積で 4/5 まで届く可能性、#4 (winner_top1 ≥ 40%) は不透明。

→ **暫定 NO-GO 寄り**、Phase 3 (5/24+) で V15.1 本格採用が優先。

---

## 5. 5/16 NO-GO 時の代替

V18/V19 paper 継続 + V15 案B改 単独維持。 撤退ライン余裕で 5/24 Phase 3 移行へ。

```
5/16 (土) - 5/23 (金): V15 案B改 維持 + V18/V19 paper 蓄積
5/24 (金) Phase 3 移行判定
5/25 - 6/8: V15.1 本格採用 (詳細 PHASE_3_V15_1_PLAN.md)
```

---

## 6. 5/9-5/15 必須作業 (5/15 判定 のため)

| 日 | task | 工数 |
|---|------|------|
| 5/13 (火) | feature shift 調査 (#5) | 90min |
| 5/14 (水) | race-level normalize 本番統合 (#1) | 30min |
| 5/15 (金) 22:00 | 5 条件 達成度 final チェック + GO/no-go 判定 | 30min |

---

## 7. リスク評価

- distribution shift 27.7x は normalize で **解消されない別問題** (Session #30 既知)
- winner_top1 -13.3pt 劣化は feature shift 根因、修正大変
- 5/16 GO 判定で投入しても、winner_top1 < 40% なら ROI 期待値 低下

→ **5/16 NO-GO 推奨、5/24 Phase 3 で再検討**。

---

## 8. 結論

5/15 判定で 4-5/5 達成なら GO、それ以下は NO-GO。
**5/6 時点予測は NO-GO 寄り**、V15 案B改 維持で 5/24 Phase 3 (V15.1) へ移行が王道。
