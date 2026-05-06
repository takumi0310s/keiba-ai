# V18/V19 5/16 試行 GO/no-go 計画書 (Session #33 大幅更新)

**作成**: 2026-05-06 PM (Session #31 → Session #33 真因確定で更新)
**判定日**: 2026-05-15 (金) 22:00
**目的**: V18 単勝 / V19 複勝 試行投入の前提条件 達成度評価
**Session #33 更新**: 真因 Pattern A 確定 → 5/16 GO 確率 30% → **75%** に大幅上昇

---

## ★ Session #33 (5/6 PM) 重大更新

### 真因確定: Pattern A (features pipeline 破綻、model 健全)

詳細: `data/v18/v18_v19_root_cause_resolution_5_6.md`

| 真因 | 検証 |
|------|------|
| 1. features 分布差 (12 features 破綻、gain 16.7%) | **主因** ✓ |
| 2. ラベル分布差 | 否定 (false hypothesis、1着率 0.06pt 差のみ) |
| 3. data leakage | 否定 (明確な leakage 不検出) |
| 4. sample 構成シフト (Niigata 0%→28% 等) | 副因 |
| 5. PACI 取得停止 (gain ~30%) | **主因** ✓ |

→ **monotonic 変換で改善しない理由**: 12+ features が default 同値で **rank 自体が崩壊**、calibration / softmax では不変。 解決には **predict 側 pipeline 修正のみ** (model 触らない、再学習不要)。

### 解決策: 4 group patch (5/13-15、11-18h)

| group | 内容 | 工数 | 期待 winner_top1 改善 |
|-------|------|------|------|
| 1 | PACI 復旧 (jrdb_paci.csv 取得経路) | 3-5h | +5-8pt |
| 2 | sib_*/sr_* 生成 (predict_core 拡張) | 4-6h | +3-5pt |
| 3 | sire/bms lookup table fallback | 2-3h | +1-3pt |
| 4 | premium fallback (training_time_filled 等) | 2-4h | +2-4pt |
| **合計** | | **11-18h** | **+11-20pt** |

→ winner_top1 **34.5% → 45-55%** (45% 基準クリア、BT 47.8% 近い)

---

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

**5/6 時点予測** (Session #33 更新): 5/13-15 で Pattern A 4 group patch 完遂 → winner_top1 45-55% 達成見込み、#1-#5 全達成可能。

→ **暫定 GO 寄り (確率 75%)**、Pattern A 修正範囲達成時。 Phase 3 (5/24+) で V15.1 と並行運用 候補。

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
