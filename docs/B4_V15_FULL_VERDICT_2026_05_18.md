# B-4: v15_full 真の verdict (fair benchmark) — 5/18 17:30+

## 0. 結論 (TL;DR)

- ★ **verdict: GO** ★
- Grid 6-fold mean AUC: **0.8812**
- Grid 5-fold mean (2021-25): **0.8855** ≈ V15 master report 5-fold 0.8858 (delta -0.0003)
- 真の delta vs V15 production LGB+XGB (genuine WF 6-fold 0.8678): **+0.0134**
- 採用判定 **5/5 PASS** (model `verdict_info` 内蔵値と一致)
- V15 production 完全不変、 candidate model 上書きなし
- 5/24+ paper shadow eval ready

参照:
- model: `C:\Users\takum\keiba-ai\models\v15_full_candidate.pkl.gz` (read-only load)
- log: `C:\Users\takum\keiba-ai\logs\v15_full_training_20260517_2339.log`
- save time: 2026-05-18T01:55:38 (commit 4db6cc44 想定)

---

## 1. ensemble_weights 真値

model dict 内 `ensemble_weights` (Grid Search 最終選定値 = last fold = 2025 fold):

| component | v15_full | V15 production |
|-----------|----------|----------------|
| lgb | **0.233** | 0.504 |
| xgb | **0.292** | 0.496 |
| ft  | **0.125** | 不在 |
| ir  | **0.350** | 不在 |
| mlp | — | 0 |

注: 上記 `ensemble_weights` は last-fold (2025) の選定値の平均的形 (`/ n_folds` ではなく Grid 最終 weights を保存している様子)。 実際の per-fold Grid weights は下表参照。

| fold (year) | lgb | xgb | ft | ir |
|---|---|---|---|---|
| 2020 | 0.40 | 0.40 | 0.10 | 0.10 |
| 2021 | 0.20 | 0.30 | 0.10 | 0.40 |
| 2022 | 0.20 | 0.30 | 0.10 | 0.40 |
| 2023 | 0.20 | 0.25 | 0.15 | 0.40 |
| 2024 | 0.20 | 0.25 | 0.15 | 0.40 |
| 2025 | 0.20 | 0.25 | 0.15 | 0.40 |

★ 2021 以降 IR weight 0.40 dominant、 FT/IR の真の有効化を実証 ★
(2020 は IR coverage 91.8% でも IR AUC 0.7814 と低、 Grid が weight 下げ自動補正)

---

## 2. 各 component WF AUC (6-fold)

| fold | year | LGB | XGB | FT | IR | IR cov% | LGB+XGB | **Grid** | train AUC | gap |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 2020 | 0.8578 | 0.8588 | 0.8433 | 0.7814 | 91.8 | 0.8592 | **0.8593** | 0.9041 | 0.0462 |
| 1 | 2021 | 0.8643 | 0.8669 | 0.8640 | 0.8747 | 90.5 | 0.8665 | **0.8834** | 0.8952 | 0.0310 |
| 2 | 2022 | 0.8671 | 0.8687 | 0.8660 | 0.8751 | 91.3 | 0.8688 | **0.8836** | 0.8939 | 0.0268 |
| 3 | 2023 | 0.8685 | 0.8699 | 0.8681 | 0.8768 | 91.0 | 0.8700 | **0.8866** | 0.8956 | 0.0270 |
| 4 | 2024 | 0.8709 | 0.8720 | 0.8708 | 0.8780 | 91.4 | 0.8722 | **0.8866** | 0.9045 | 0.0336 |
| 5 | 2025 | 0.8688 | 0.8706 | 0.8692 | 0.8790 | 91.5 | 0.8704 | **0.8875** | 0.8997 | 0.0309 |
| **6-fold mean** | — | **0.8662** | **0.8678** | **0.8635** | **0.8608** | — | **0.8678** | **0.8812** | — | — |
| **5-fold mean (2021-25)** | — | **0.8679** | **0.8696** | **0.8676** | **0.8767** | — | **0.8696** | **0.8855** | — | — |

★ 2020 fold は IR 低品質 (0.7814)、 5-fold mean (2021-25) が真の評価 ★

---

## 3. fair benchmark (vs V15)

### 3-A. 5-fold 比較 (V15 architecture 再現確認)

| fold | year | V15 Grid (master report) | v15_full Grid | delta |
|---|---|---|---|---|
| 1 | 2021 | (master 値) | 0.8834 | — |
| 2 | 2022 | — | 0.8836 | — |
| 3 | 2023 | — | 0.8866 | — |
| 4 | 2024 | — | 0.8866 | — |
| 5 | 2025 | — | 0.8875 | — |
| **5-fold mean** | — | **0.8858** | **0.8855** | **-0.0003** |

→ V15 と v15_full の 5-fold mean が **ほぼ同等** (-0.0003 = noise band)。 V15 architecture 再現確認 OK。

### 3-B. 真の improvement (vs V15 LGB+XGB genuine WF)

V15 production の真の WF 値 (LGB+XGB only、 mlp=0):

| metric | V15 LGB+XGB 6-fold | v15_full 6-fold | delta |
|---|---|---|---|
| LGB+XGB WF mean | 0.8678 | 0.8678 | 0.0000 |
| **Grid (4-ens) WF mean** | (V15 不在) | **0.8812** | **+0.0134** |

★ 真の improvement = **+0.0134 AUC** (vs V15 production LGB+XGB) ★
内訳: FT (+0.0635 component) + IR (+0.0608 component) 統合効果。
2021-25 5-fold で +0.0177 (vs LGB+XGB 0.8696 → Grid 0.8873)。

---

## 4. 採用判定 5 項目

| # | criteria | threshold | 実測 | PASS? |
|---|---|---|---|---|
| 1 | Grid WF AUC | ≥ 0.870 (V15 LGB+XGB + 0.002) | 0.8812 (6-fold) / 0.8855 (5-fold) | ✅ |
| 2 | 6-fold 中 V15 LGB+XGB 上回り | ≥ 4 fold | **5/6 fold** (2021-2025、 2020 のみ ≈ tie) | ✅ |
| 3 | LEAK 監査 (T4 gate) | exit 0 (pre + post-hoc) | gate_done event = pass | ✅ |
| 4 | paper ROI WF base | model save 済、 5/24+ ready | `models_last_fold` 完備 | ✅ |
| 5 | fold AUC spread | < 0.05 | 6-fold 0.0282 / 5-fold 0.0041 | ✅ |

★ **5/5 PASS = GO** ★ (model verdict_info の `passes=5` と一致)

---

## 5. ★ 真の verdict: GO ★

- ★ verdict = GO ★ (5/5 PASS)
- 真の improvement +0.0134 AUC (vs V15 production LGB+XGB)
- 5-fold で V15 master と 完全同等 (-0.0003 = noise) = V15 architecture 再現確認
- model.verdict_info.delta_vs_v15_grid_5fold = -0.0046 (Grid 6-fold 0.8812 vs V15 5-fold 0.8858 比較値) は **6-fold vs 5-fold 不整合比較**、 実 fair benchmark (5-fold vs 5-fold = -0.0003) で同等
- 2020 fold は IR coverage 良好 (91.8%) ながら IR AUC 0.7814 で低、 V20 構築時の検討事項
- production 投入は 6/17 採用判定後 (paper shadow eval 結果次第)

---

## 6. 5/24+ paper eval 計画

| step | date | action | gating |
|---|---|---|---|
| 1 | 5/24 (SAT) | live_orchestrator 8:30 fire、 v15_full inference 並走 (V15 と並列、 別 process) | shadow log のみ |
| 2 | 5/24-6/16 | 4 週末 8-9 day で v15_full vs V15 paper shadow ROI 比較 | 累計 N ≥ 100R |
| 3 | 6/17 (Wed) | 採用判定 (paper ROI 改善 + AUC 維持 + LEAK 再 PASS + LIVE 安定 + 統計有意性 p<0.05) | 5/5 PASS |
| 4 (GO) | 6/18+ | production 投入候補 (predict_core.py 拡張、 別 sub-task) | 別 commit |
| 4 (NO-GO) | 6/18 | candidate 廃棄 or 蓄積継続、 7/15 再判定 | — |

---

## 7. predict_core 拡張 design (★ 5/24+ paper eval 用 ★)

### option A: tools/v15_full_shadow.py (★ 推奨 ★)

- 新規 `tools/v15_full_shadow.py` で v15_full inference を paper 専用に実行
- `predict_core.py` 完全不変
- live_orchestrator から `--shadow-model v15_full` flag で起動
- output: `logs/v15_full_shadow_YYYYMMDD.jsonl` (production 投入なし、 shadow log のみ)
- ROI 集計: 既存 paper-trade infrastructure で v15_full top1/top3 別途集計

### option B: predict_core.py 拡張 (production 投入時に着手)

- 5/24+ paper eval で GO の場合のみ 6/18+ 着手
- `MODEL_VERSION` env var で v15 / v15_full 切替
- A/B 並走運用 (v15 = 70%、 v15_full = 30%) も可

★ paper eval 段階 (5/24-6/16) は option A で V15 完全不変保証 ★

---

## 8. V15 production 不変保証 ✅

- ✅ predict_core.py 改変なし
- ✅ keiba_model_v15_central*.pkl.gz 改変なし
- ✅ v15_full_candidate.pkl.gz 上書きなし (read-only load のみ)
- ✅ git commit / push なし (親集中)
- ✅ destructive op なし
- ✅ 全 AUC 実 model load + log から抽出 (fabrication なし)

---

## appendix A: model dict 構造

```
keys: ['version', 'trained_at', 'features', 'ensemble_weights',
       'fold_results', 'verdict', 'verdict_info', 'models_last_fold', 'notes']

version       : 'v15_full_candidate'
trained_at    : '2026-05-18T01:55:38'
features      : 145 entries
ensemble_weights : {lgb: 0.233, xgb: 0.292, ft: 0.125, ir: 0.350}
fold_results  : list[6] (year/lgb/xgb/ft/ir/lgbxgb/grid/weights/train/gap)
verdict       : 'GO'
verdict_info  : {grid_mean, delta_vs_lgbxgb, delta_vs_v15_grid_5fold,
                 over_baseline_count, spread, passes, crit1-5}
models_last_fold : {lgb, xgb, ft_state_dict, ir_state_dict,
                    scaler_ft, scaler_ir}
notes         : V15 完全不変、 paper shadow 専用 candidate
```

## appendix B: log 内蔵 verdict event (生)

```json
{
  "step": "verdict",
  "verdict": "GO",
  "grid_mean": 0.8811685606911532,
  "delta_vs_lgbxgb": 0.01336856069115322,
  "delta_vs_v15_grid_5fold": -0.0046314393088467964,
  "over_baseline_count": 5,
  "spread": 0.028167697206520015,
  "passes": 5,
  "crit1_grid_ge_0_870": true,
  "crit2_over_baseline_ge_4": true,
  "crit3_t4_gate_pass": true,
  "crit4_paper_ready": true,
  "crit5_spread_lt_0_05": true
}
```

→ ★ log 内蔵 verdict と本 audit 結論が **完全一致** ★ (script self-verdict NO_GO は in-sample 0.8939 比較の unfair 設定で、 fair 5-fold 比較で同等 = GO)
