# Sprint 6 詳細 plan

**期間**: 2026-05-22 〜 2026-05-30 (9 日)
**目標**: V20 cache 構築 + KKA 統合 + 4-model ensemble 準備
**作成**: Session #79

---

## task 一覧

### 1. KKA parser 統合 (Day 1-2、 5/22-5/23)

| step | 内容 |
|------|------|
| 1-1 | Session #53 KKA parser 復元 (seiseki 0% → 90.4% 修復済) |
| 1-2 | race_id format 調整 (V20 cache と整合) |
| 1-3 | V20 features 表 に組み込み (heavy / class / pace / season / dam_rensho 系 12-15 件) |
| 1-4 | unit test (parse 成功率 90%+ 確認) |
| 1-5 | mini backtest (KKA 込み AUC 確認、 V15 baseline 比較) |

**期待 delta**: +0.001-0.003 AUC

### 2. V20 cache 構築 (Day 2-5、 5/23-5/26)

| step | 内容 |
|------|------|
| 2-1 | TFJV 90 年分 → V20 features 抽出 (RA/SE/HR/H1/UM/WF) |
| 2-2 | parquet 化 (storage 最適化、 6 GB → 1-2 GB 想定) |
| 2-3 | 6 年分 (2020-2025) PoC data + 過去拡張分 統合 |
| 2-4 | data integrity check (NaN 率 / dup / date 順序) |
| 2-5 | 5/10-5/22 蓄積分 (Session #71 全馬 score) を merge |

**ファイル出力**:
- `data/v20_cache/v20_features_2010_2025.parquet`
- `data/v20_cache/v20_metadata.json`

### 3. 4-model ensemble training script (Day 5-7、 5/26-5/28)

| step | 内容 |
|------|------|
| 3-1 | V13.5b script 復元 (Session #56 logic) |
| 3-2 | LGB / XGB / FT-Transformer / IntraRace Attention 4 model 定義 |
| 3-3 | hyperparameter 最適化 (Optuna 50 trials、 IR 重視) |
| 3-4 | walk-forward CV (6-fold、 2020-2025) |
| 3-5 | grid search 重み最適化 (典型 LGB=0.043 / XGB=0.043 / FT=0.087 / IR=0.826) |

**期待 AUC**: 0.90025 (Session #56 実証)

### 4. LEAK 除外監査 (Day 7-8、 5/28-5/29)

| step | 内容 |
|------|------|
| 4-1 | Session #51 LEAK 12 件 list を学習 data から除外 |
| 4-2 | 残存 features の AUC 確認 (LEAK 抜きで 0.90+ 維持確認) |
| 4-3 | corr_target / monotonic / shift 検査 (SKB 教訓) |
| 4-4 | 新 features に対する LEAK 監査 (KKA も含む) |

### 5. Sprint 6 成果物 (Day 9、 5/30)

| 出力 | path |
|------|------|
| V20 cache | `data/v20_cache/` |
| training script | `train/train_v20_4model_ensemble.py` |
| 学習結果 | `data/v20_4model_results.json` |
| LEAK audit report | `data/v18/v20_leak_audit_5_30.md` |
| Sprint 6 完了 report | `docs/SPRINT_6_COMPLETION_5_30.md` |

---

## 期待 AUC (Sprint 6 終了時点)

| model | AUC |
|-------|----|
| V20 LGB alone | 0.875 |
| V20 + KKA LGB | 0.876-0.878 |
| V20 4-model ensemble (KKA 込み) | **0.90025+** |

---

## 投資保護 (Sprint 6 中も遵守)

- V15 production 完全不変
- 5/9-5/30 の週末 V15 案B改 単独継続
- Sprint 6 は **学習のみ**、 production 反映は **6/8 paper trade 開始**

---

## 関連

- [V20_BUILD_DETAILED_PLAN.md](V20_BUILD_DETAILED_PLAN.md)
- [V20_VS_V15_COMPARISON.md](V20_VS_V15_COMPARISON.md)
- [V20_DEPLOYMENT_CHECKLIST.md](V20_DEPLOYMENT_CHECKLIST.md)
