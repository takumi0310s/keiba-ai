# 比較-2: FT+IR 有効化 path 設計 (★ 真の improvement direction ★)

date: 2026-05-17 (Sun)
mode: ★ design only (実装 / git commit / push / training 一切なし、 V15 production 完全不変) ★
source:
- `docs/V15_AUDIT_1_MODEL_STRUCTURE_2026_05_17.md` (V15 .pkl 内部構造 真値)
- `docs/V15_AUDIT_2_WF_AUC_2026_05_17.md` (V15 各 component WF AUC 真値)
- `docs/V15_AUDIT_5_INTEGRATED_VERDICT_2026_05_17.md` (v15.2 NO-GO 濃厚 verdict)
- `data/v15_master_report.json` (4-model Grid 5-fold WF 実測値)
- `train/train_v15_master.py` L573-616 (production .pkl save logic)
- `tools/predict_core.py` L2162-2243 (推論 path)

---

## 0. 結論 (★ 核心 ★)

| 項目 | 値 |
|---|---|
| 推奨案 | **case D (paper shadow eval、 V15 production 完全不変)** |
| 想定 WF AUC delta (genuine LGB+XGB → 4-model Grid) | **+0.018** (0.8678 → 0.8858) |
| 想定 ROI delta (paper、 assumption) | **+5〜10pt** (検証必須、 5/24+ paper eval で確定) |
| 実装工数 | **8〜10h** (うち GPU retrain 8h + paper 蓄積 0h、 paper 用 wrapper 2h) |
| V15 production 影響 | **ゼロ** (新規 file `v15_full.pkl.gz` + 別 module、 predict_core 不変) |
| T4 LEAK audit gate | **PASS 想定** (V15 オリジナル 145 features の FT/IR 復活、 新規 feature 追加なし) |
| 着手 priority | **高** (v15.2 NO-GO 推定 → 真の improvement path は本案) |
| 着手予定 | **5/18+ sub-task** (v15.2 verdict 確定 + audit-1〜5 完了 が precondition) |

---

## 1. 現状 V15 architecture 真値整理 (audit-1/2 引用)

### 1.1 .pkl 内 component (audit-1 §2-§3)

| key | 値 / 状態 |
|---|---|
| `model` (LightGBM Booster) | 500 trees / 145 features ✅ 推論で使用 |
| `xgb_model` (XGBoost Booster) | 500 rounds / 145 features ✅ 推論で使用 |
| `mlp_model` | **None** (dead key、 weight 0) |
| `ft_model_state` | ★ **存在しない** ★ |
| `ir_model` | ★ **存在しない** ★ |
| `cb_model` | ★ **存在しない** ★ |
| `ensemble_weights` | `{lgb: 0.5036, xgb: 0.4964, mlp: 0}` |

→ ★ V15 production 推論 = LGB+XGB 2-model 加重平均のみ ★ (audit-1 §0 確定)

### 1.2 WF AUC 真値 (audit-2 §0)

| metric | 値 |
|---|---|
| stored `.pkl auc` field (= LGB train self-eval) | 0.8939 (★ WF mean ではない ★) |
| **genuine WF LGB+XGB 6-fold mean (2020-2025)** | **0.8678** |
| **genuine WF LGB+XGB 5-fold mean (2021-2025)** | **0.8696** |
| **WF Grid 4-model 5-fold mean (2021-2025)** | **0.8858** (`v15_master_report.json` grid_mean) |

★ 真の delta = 0.8858 - 0.8678 = **+0.018 AUC** (genuine LGB+XGB vs WF Grid 4-model) ★

### 1.3 WF 評価時の Grid weights (audit-1 §3.5)

| year | LGB | XGB | FT | IR | grid_auc |
|---|---|---|---|---|---|
| 2021 | 0.20 | 0.30 | 0.10 | 0.40 | 0.8836 |
| 2022 | 0.20 | 0.30 | 0.10 | 0.40 | 0.8841 |
| 2023 | 0.20 | 0.25 | 0.15 | 0.40 | 0.8860 |
| 2024 | 0.20 | 0.25 | 0.15 | 0.40 | 0.8887 |
| 2025 | 0.20 | 0.30 | 0.10 | 0.40 | 0.8868 |
| 平均 | 0.20 | 0.28 | 0.12 | 0.40 | **0.8858** |

★ IR (IntraRace Attention) が単独で最高 AUC 0.8772 (5-fold mean)、 Grid weights 0.40 dominant ★

### 1.4 production 保存時の dropped component (audit-1 §3.5 + train_v15_master.py L573-616)

train_v15_master.py L573-616 ロジック:
- WF fold ごとに **4-component (LGB+XGB+FT+IR) を学習・評価** ✅
- production .pkl save 時 **LGB+XGB のみ を pickle**、 FT/IR は WF 評価専用で破棄 ❌
- ensemble_weights は `{lgb, xgb, mlp}` の 3 key dict に simplify (FT/IR 重み dropped)

→ ★ 4-model Grid AUC 0.8858 は WF 評価では実測されたが、 production 推論には 反映されていない ★

---

## 2. FT+IR 有効化 design 3 案 + 推奨案

### case A: 既存 V15 .pkl の ensemble_weights 再調整

**前提**: V15 .pkl 内に FT/IR が保存されていれば weight 再分配のみで活性化可能。

**判定**: ❌ **不可能** (audit-1 §3.5 で V15 .pkl 内に FT/IR が **存在しない** ことを確認済)

**結論**: 不採用

---

### case B: train_v15_master.py から full 再 train + 完全 save

**実装**:
- train_v15_master.py を改変し、 FT/IR weight を含む 4-component を 1 つの .pkl に save
- 既存 V15 .pkl は別名で backup、 新規 file `keiba_model_v15_full_central.pkl.gz` に save
- predict_core.py で FT/IR optional ensemble path を有効化

**工数**: 8h GPU 学習 (V15 と同等) + 2h save logic 改修

**risk**:
- 中 (新規 file、 predict_core.py 改変必須 → ★ user 絶対遵守違反 ★)
- predict_core 改変は 「V15 production / v15.2 training 完全不変」 ルール違反

**結論**: ❌ **不採用** (predict_core 改変が違反)

---

### case C: 既存 V15 .pkl + 別 file (FT/IR) lazy load

**実装**:
- 既存 V15 .pkl: LGB+XGB のまま production 維持
- 別 file `v15_ft_ir.pkl.gz`: 新規学習で FT/IR のみ保存
- predict_core で lazy load + 4-model ensemble

**判定**: ❌ **predict_core.py 変更必要 (★ user 絶対遵守違反 ★)**

**結論**: 不採用

---

### case D: ★ paper shadow only (推奨) ★

**実装** (V15 production 完全不変):

1. 新規 file `keiba_model_v15_full_central.pkl.gz` 作成 (4-component full ensemble)
   - 既存 V15 .pkl と **共存** (上書きしない)
   - predict_core / daily_predict / race_auto_notify は **既存 V15 .pkl を使い続ける** (一切改変なし)

2. 新規 module `tools/v15_full_shadow.py` 作成 (paper inference 専用)
   - V15 .pkl から LGB+XGB を load
   - `v15_full.pkl.gz` から FT/IR を load
   - 4-component Grid ensemble で predict (paper only)
   - production には流さない (Discord 通知も別 channel 想定 / 完全独立 log file 出力)

3. paper shadow eval phase (5/24+)
   - live_orchestrator または 別 cron で v15_full ranking を 並行計算
   - V15 production ranking と paper compare
   - 30R 蓄積後 ROI / AUC / 統計的有意性 判定

4. production 投入は paper PASS 後の ★ 別 phase ★ (本 design 範囲外)

**V15 production 影響**:
- ✅ V15 .pkl: 不変
- ✅ predict_core.py: 不変
- ✅ daily_predict / race_auto_notify / app.py: 不変
- ✅ v15.2 training: 不干渉
- ✅ git commit/push: design 段では不要

**工数**: 8〜10h
- GPU retrain 8h (V15 と同等、 4-component 全 train)
- save logic + paper wrapper 2h

**risk**: 低 (production 完全独立、 paper のみ)

**結論**: ★ **採用** ★

---

### 3 案 比較 table

| case | V15 production 不変 | predict_core 不変 | 工数 | 推奨? |
|---|:---:|:---:|---|:---:|
| A: 既存 .pkl の weight 再調整 | — | — | — | ❌ 不採用 (.pkl に FT/IR 不在) |
| B: 全 4-component 再 train + production save | ❌ | ❌ | 10h | ❌ 不採用 (predict_core 改変必要) |
| C: 別 file lazy load | ✅ | ❌ | 10h | ❌ 不採用 (predict_core 改変必要) |
| **D: paper shadow only** | **✅** | **✅** | **8〜10h** | ★ **推奨** ★ |

---

## 3. 想定 +AUC / +ROI delta

### 3.1 AUC delta (v15_master_report.json 実測ベース)

| metric | 値 | source |
|---|---|---|
| 現状 V15 production WF AUC (LGB+XGB only) | **0.8678** | audit-2 §3 (6-fold mean) |
| v15_full WF Grid AUC (4-model) | **0.8858** | `v15_master_report.json` 5-fold mean (2021-2025) |
| ★ 真の delta ★ | **+0.018 AUC** | 実測値の差分 (assumption ではない) |

注: master_report は 2020 fold を含まない (FT/IR は 2020 で training data 不足のため除外)。
公平比較は 5-fold (2021-2025) で実施: LGB+XGB 5-fold = 0.8696 vs Grid 4-model = 0.8858 = **+0.0162**。
保守的に +0.016〜+0.018 を想定。

### 3.2 ROI delta (★ assumption、 5/24+ paper eval で 検証必須 ★)

過去 audit からの 経験則:
- AUC +0.005 で paper ROI 想定 +2〜3pt (v14 → v15 paper trace)
- AUC +0.018 で paper ROI 想定 +5〜10pt (linear assumption、 上限あり)

★ ROI delta は ★ assumption ★ であり、 5/24+ paper eval (30R 蓄積) で 実測する。
master_report.json は AUC のみで ROI 評価は含まれていない。

---

## 4. 実装 path (★ 5/18+ 設計、 本 design 段では着手なし ★)

### step 1: V15 .pkl から FT/IR 抽出 試行

- train_v15_master.py の training cache (data/v15_*/) に FT/IR weight が残存している可能性を 探索
- もし FT/IR の learned weight (`*.pt` / `*.pkl`) が cache に残っていれば、 retrain 不要で paper shadow eval 可能 (工数 -8h)
- ★ 探索は read-only ★ (cache 削除 / 改変 一切なし)

### step 2: FT/IR retrain が必要な場合

- train_v15_master.py を ★ paper eval 専用 wrapper ★ で実行 (production .pkl は 触らない)
- wrapper output: `v15_full.pkl.gz` (LGB+XGB+FT+IR 4-component) + `v15_full_master_report.json`
- 工数: 8h GPU (V15 と同等)

### step 3: ensemble_weights 設計

選択肢 A: master_report 5-fold mean weights を そのまま使用
- LGB: 0.20, XGB: 0.28, FT: 0.12, IR: 0.40

選択肢 B: paper eval phase で Grid Search 再最適化 (各 fold 最適)
- audit-1 §3.5 の年別 weights を そのまま継承

★ 推奨 ★: 選択肢 A (シンプル、 master_report 整合)。 paper eval phase で B に切替検討。

### step 4: tools/v15_full_shadow.py 新規 (paper inference のみ)

```python
# 概念設計のみ (★ 実装は別 sub-task ★)
def predict_v15_full_shadow(race_id):
    # 1. V15 .pkl load (LGB+XGB)
    v15_data = load_v15_pkl()  # 既存 path 使用 (read-only)

    # 2. v15_full.pkl load (FT+IR additional)
    full_data = load_v15_full_pkl()

    # 3. features を 既存 V15 builder で生成 (predict_core 不変流用)
    X = build_features_v15(race_id)

    # 4. 4-component predict
    lgb_pred = v15_data['model'].predict(X)
    xgb_pred = v15_data['xgb_model'].predict(X)
    ft_pred = predict_ft(full_data['ft_state'], X)
    ir_pred = predict_ir(full_data['ir_state'], X)

    # 5. Grid weight ensemble
    pred = 0.20 * lgb_pred + 0.28 * xgb_pred + 0.12 * ft_pred + 0.40 * ir_pred

    # 6. paper log 出力 (production 流さない)
    log_paper(race_id, pred)
    return pred
```

### step 5: regression test 拡張

- 既存 regression 23+α (commit b3fcd14b 後の 35+ tests) を ベースに
- v15_full shadow inference 用 test 追加:
  - `test_v15_full_load.py`: v15_full.pkl 正常 load
  - `test_v15_full_shadow_inference.py`: 既存 races で paper predict 完走
  - `test_v15_production_unchanged.py`: V15 .pkl が paper eval phase 中 改変されていないこと verify (md5 / mtime check)

### step 6: paper shadow eval 30R (5/24+ live_orchestrator 経由)

- live_orchestrator から v15_full_shadow.predict を 並行呼出 (production 推論と独立)
- 30R 蓄積後、 v15_full ranking vs V15 ranking を 比較:
  - top1 / top3 一致率
  - winner_top1 率 (≥ V15 + 1pt なら GO 候補)
  - paper ROI (V15 ROI と比較、 統計的有意性 Welch's t-test)
- LEAK audit T4 reverify (FT/IR 復活で 145 features の中身は変わらないが、 念のため監査)

---

## 5. T4 LEAK audit gate (★ PASS 想定 ★)

### 5.1 ★ 重要 ★: v15.2 (17 features 追加) との 差異

| 案 | 新規 features | LEAK risk |
|---|---|---|
| **v15.2** | V15 145 features + 17 件 (paci.info_idx, paci_tan_*, paci_post_*, oddrate_5min_diff 等) | ⚠ 中 (paci.info_idx は post-race 発覚済) |
| **v15_full (本案)** | V15 145 features の **そのまま (新規 feature 追加なし)** + FT/IR component 復活 | **低** (T1 audit 既 PASS の 145 features を そのまま使用) |

★ v15_full は features list を 一切変更しない ★
→ T4 LEAK audit gate は ★ PASS 想定 ★ (T1 audit 結果を継承)。

### 5.2 LEAK_FEATURES_A 除外 8 件 (audit-1 §2 確認済)

- `cond_surface, condition_enc, horse_weight, odds_log, weight_cat, weight_cat_dist, weight_change, weight_change_abs`

→ V15 .pkl `leak_removed` field と完全一致、 v15_full でも同様除外を 継承。

---

## 6. 採用判定 5 項目 (5/24+ paper eval 後 確定)

| # | 判定項目 | GO 閾値 | NO-GO 閾値 |
|---|---|---|---|
| 1 | v15_full WF AUC vs V15 LGB+XGB 6-fold | **≥ 0.8728** (V15 + 0.005) | < 0.8678 (V15 同等以下) |
| 2 | paper ROI (30R) | V15 ROI + 5pt 以上 | V15 ROI - 2pt 以下 |
| 3 | LEAK audit (T4 reverify) | PASS (145 features 同一) | FAIL (新規 LEAK 発見) |
| 4 | LIVE inference 安定性 | 95% 以上 完走、 latency < 3s | 完走率 < 90% または timeout 多発 |
| 5 | 統計的有意性 (Welch's t-test、 paper 30R) | p < 0.05 | p ≥ 0.20 |

★ 5 項目 ALL PASS で paper → production 投入判断 ★ (production 投入は本 design 範囲外)

### 6.1 GPU spec verify (項目 4 関連)

- FT-Transformer / IR は GPU 推論で latency 短縮可能 (CPU でも動作するが 5-10x 遅延)
- production 環境 (Streamlit Cloud / local Windows) の GPU spec を 5/24+ 別途 verify

---

## 7. v15.2 vs v15_full 優先順位

### 7.1 v15.2 (FE 追加) 現状 (audit-5 §0)

- LGB+XGB 改善 = **+0.000005** (≒ ±0) → ★ NO-GO 濃厚 ★

### 7.2 v15_full (FT+IR 復活) 想定

- 4-model Grid 5-fold AUC 改善 = **+0.018** (LGB+XGB 0.8678 → Grid 0.8858) ★ 真の improvement ★

### 7.3 優先順位

| condition | v15.2 verdict | next action |
|---|---|---|
| v15.2 GO (改善 +0.002 以上) | adopted | v15.2 production 投入 + v15_full は parallel exploration |
| **v15.2 NO-GO (改善 < +0.002)** ← ★ 推定 ★ | rejected | **v15_full に pivot、 5/18+ 高 priority 着手** |

★ v15.2 verdict (audit-5 で NO-GO 濃厚) 確定後、 v15_full を 最優先 improvement path として 5/18+ 着手 ★

---

## 8. V15 production 不変保証 ✅

本 design 段で 実施したのは以下のみ:
- 既存 docs 読み取り (V15_AUDIT_1/2/5)
- 新規 docs 作成 (本 file のみ)

**変更なし**:
- 🟢 V15 .pkl.gz: 一切改変なし (md5 不変 想定)
- 🟢 predict_core / daily_predict / race_auto_notify / app.py: 一切改変なし
- 🟢 v15.2 training process: 一切干渉なし
- 🟢 git commit/push: 行わない (★ 親集中 ★)
- 🟢 実装着手: なし (★ 設計のみ ★)

5/18+ 実装着手 phase でも case D path に従う限り V15 production は 完全独立 / 不変保証。

---

## 9. fabrication 防止 verify

| 数値 | source | 引用箇所 |
|---|---|---|
| 0.8678 (genuine WF LGB+XGB 6-fold) | audit-2 §3 | data/v15_2/v15_baseline_lgb_xgb_20260517_1715.json |
| 0.8696 (5-fold 2021-2025) | audit-2 §3 | 同上 |
| 0.8858 (WF Grid 4-model 5-fold) | audit-1 §3.5 + audit-2 §4 | data/v15_master_report.json |
| 0.8939 (stored .pkl auc) | audit-1 §5.1 | model['auc'] field |
| LGB: 0.20, XGB: 0.28, FT: 0.12, IR: 0.40 (Grid weights 5-fold mean) | audit-1 §3.5 | v15_master_report.json yearly weights mean |
| ensemble_weights {lgb:0.5036, xgb:0.4964, mlp:0} | audit-1 §2 | model['ensemble_weights'] field |
| +0.018 AUC delta | 計算 (0.8858 - 0.8678 = 0.0180) | audit-1/2 数値 差分 |
| +5〜10pt ROI delta | ★ assumption ★ | (5/24+ paper eval で 実測必須) |
| 8〜10h 工数 | V15 と同等 GPU 学習 (8h) + wrapper (2h) | V15 学習履歴 (2026-04-08T23:32) 参照 |

★ 全数値 audit-1/2 から正確に引用、 V15 retrain なし (paper inference 設計のみ) ★

---

## 10. 結論 (★ 親への報告 ★)

1. ★ V15 真の architecture = **LGB+XGB 2-model only**、 FT/IR は WF 評価専用で .pkl 未保存 (audit-1 確定) ★
2. ★ 真の improvement direction = FT+IR を本当に有効化する (paper shadow eval、 case D) ★
3. ★ 想定 +AUC = **+0.018** (LGB+XGB 0.8678 → Grid 4-model 0.8858)、 master_report.json 実測値の 差分 ★
4. ★ 想定 +ROI = **+5〜10pt** (assumption、 5/24+ paper eval で 検証必須) ★
5. ★ 工数 = **8〜10h** (GPU retrain 8h + paper wrapper 2h) ★
6. ★ V15 production 完全不変保証、 predict_core 不変、 git commit/push 親集中、 実装絶対なし ★
7. ★ v15.2 NO-GO 推定 → v15_full に pivot、 5/18+ 別 sub-task として高 priority 着手 ★
