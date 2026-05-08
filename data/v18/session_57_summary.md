# Session #57: V20 interaction PoC 統合サマリー

**作成**: 2026-05-09 (Session #57 完了)
**branch**: dev/v20-interaction (main 6c0680ad から分岐)
**main 不変**: ✅
**V15 投資保護**: ✅ (model file / predict_core / daily_predict 不変)

---

## 1. 全 5 領域 結果

| 領域 | 内容 | output | 結果 |
|------|------|--------|------|
| A | interaction 候補 10 件 選定 | data/v18/session_57_interaction_candidates.md | ✅ |
| B | tools/v20_interaction_features.py 実装 | data/v20/interaction_features.csv (65 MB) | ✅ |
| C | V20 + interaction LGB 学習 | AUC 0.8683 (vs 0.8685 baseline) | ✅ Δ=-2bp |
| D | 3-way + shrinkage tuning | best +1.8bp (noise) | ✅ 飽和確証 |
| E | 5 commits + push + Discord | 本 doc + push + 通知 | ✅ |

---

## 2. 確定事項

### 2.1 V15 145 features は **真の飽和** (3 角度から確証)

| 試行 | session | 結果 |
|------|---------|------|
| 単一 feature 追加 | #51 | 飽和 |
| 2-way interaction 10 件 | #57 C | **-2 bp** |
| 3-way interaction + shrinkage | #57 D | best **+1.8 bp = noise** |

→ **LGB single fold で V15 0.8687 が天井**

### 2.2 V20 戦略 (確定)

| breakthrough 候補 | 期待 | 状態 |
|--------------------|------|------|
| ❌ 単一 feature | +0 | 飽和 (#51) |
| ❌ 2-way / 3-way interaction | +0 | 飽和 (#57) |
| 🔵 LGB+XGB+FT+IR ensemble | +170 bp (V13.5b 復活想定) | 検討中 (#56) |
| 🔵 TFJV 新 source 統合 | +50-100 bp | Session #44 PoC 済 |
| 🟢 動画 features (Phase 4) | +100-200 bp | 7-8月 PoC 予定 |

### 2.3 LGB が既に内部で interaction 捕捉

- LGB tree split が `jockey_id × course_enc → 内部 partition` を暗黙的に学習済
- 明示的 interaction feature 追加 → 冗長 (redundant)
- V15 既存の `paci_jockey_exp_wr` / `jrdb_ze_idm_avg` 等が既に強い signal を持つ

---

## 3. 5 commits (dev/v20-interaction)

```
46eca167 Session #57 D: V20 interaction 深掘り (3-way + shrinkage)
9b0ac1fc Session #57 C: V20 + interaction LGB 学習 + AUC 比較
7063d75f Session #57 B: V20 interaction features 10 件 実装
cc4b6cdb Session #57 A: V20 interaction candidates 選定 (10 件)
6c0680ad AUDIT-1: 3 source 全要素 audit  ← main HEAD (不変)
```

E commit (本 doc) を含めて 5 commits 確定。

---

## 4. 5/9 V15 投資 完全保護

- ✅ main HEAD 6c0680ad 不変
- ✅ keiba_model_v15_central_live.pkl.gz 不変 (md5 not modified)
- ✅ tools/predict_core.py 不変
- ✅ tools/daily_predict.py 不変
- ✅ app.py 不変
- ✅ schtasks 既存 41 件 不変
- ✅ 累計 +13,530円 維持

→ **5/9 朝 V15 案B改 単独継続 絶対**

---

## 5. NEXT (Session #58+)

| 検討事項 | 優先度 |
|----------|--------|
| Session #56 ensemble 復活 結果待ち | 🔴 high |
| Phase 4 動画 features PoC (7-8月) | 🔴 high |
| TFJV features 拡張 (Session #44 後続) | 🟡 mid |
| paci_*** 系 feature 深掘り | 🟢 low |

---

**Session #57 完了 (V20 interaction PoC、 V15 飽和確証)**
