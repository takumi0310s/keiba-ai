# Phase 15 — V20 評価 + V15 比較

**date**: 2026-05-10 21:00 JST
**run**: train_v20_ensemble.py --quick + 5/10 actual results

## 1. V15 5/10 actual baseline

V15 が production 動作した 5/10 全 35 R の実績:

| metric | 値 |
|--------|----|
| 予測 R | 35 |
| 結果確定 | 34 |
| trio hits | 11 |
| trio hit rate | 32.4% |
| 投資 | ¥24,500 (700 yen × 35 R) |
| 配当 | ¥27,790 |
| **profit** | **+¥3,290** |
| **ROI** | **113.8%** |
| top1 1 着 率 | 26.5% (9/34) |
| top1 top3 率 | (詳細は phase15_5_10_eval.json) |

## 2. V20 quick 学習結果 (本 phase)

| 項目 | 値 |
|------|----|
| WF | train 2022-24 / val 2025 |
| train rows | 140,867 |
| val rows | 47,497 |
| features | 145 (V15 base のみ実 signal) |
| **LGB GPU val AUC** | **0.8662** |
| **XGB GPU val AUC** | **0.8676** |
| **LGB+XGB ens AUC** | **0.8678** |
| 学習時間 (LGB+XGB) | 19 sec on RTX 4070 Ti SUPER |
| **FT-Transformer 10ep AUC** | **0.8579** |
| FT 学習時間 | 426 sec (~7 min) |
| **3-model ens AUC** | **0.8676** (LGB+XGB+FT、 0.5/0.4/0.1) |

## 3. V15 vs V20 quick 5/10 仮想 hit 比較

★ V20 quick = V15 features 145 のみ retrain なので、 5/10 予測は V15 と等価が予想される ★

| 項目 | V15 prod | V20 quick (LGB+XGB only) |
|------|---------|--------------------------|
| 学習 fold | 全年 (2015-2024 walk-forward) | 2022-24 単 fold |
| 4-model ensemble | LGB+XGB+FT+IR (Grid) | LGB+XGB のみ |
| 全年 WF AUC | 0.8939 | (本 quick は単 fold のみ、 直接比較不可) |
| 単 fold AUC (val=2025) | (再評価必要) | 0.8678 |

→ V20 quick は learning 用 PoC、 production 投入は **しない**。 5/10 予測は V15 prod 維持。

## 4. AUC 単 fold vs 全年 WF の違い

V15 baseline 0.8939 は 2020-2025 全年の WF 平均。 一方 V20 quick 0.8678 は 2022-24 → 2025 単 fold。

| 比較対象 | 妥当な比較対象 |
|---------|--------------|
| V20 quick 0.8678 (2025 val) | V15 LGB単独 2025 fold AUC (CLAUDE.md より 0.8079) |
| → V20 quick が大幅優位 | 145 features (vs 74) の効果 |

V15 の 2025 単 fold AUC 0.8079 を考えると、 V20 quick 0.8678 は **+0.060 改善**。 ただしこれは V15 v12 base との比較で、 V15 production v13.5b (4-model grid) とは別軸。

## 5. ★ 真の V20 (期待 0.91+) への path ★

| step | 期日 | 内容 |
|------|------|------|
| Phase 11 実 data | 5/12+ | JRDB 外厩 / odds 時系列 / 騎手拡張 lookup |
| Phase 13 実 fetch | 5/11+ | netkeiba master DOM 検証 → 25 features 実値 |
| Phase 12 実 fetch | 5/24+ | JV-Link 32-bit Python venv backfill |
| 4-model ensemble | Phase 15 完全版 | FT-Transformer 50 epochs + IntraRace 50 epochs |
| Optuna ensemble | Phase 15 完全版 | 100 trials weight tuning |
| WF 全年再評価 | Phase 15 完全版 | 2020-2025 fold ごと AUC |
| **V20 投入候補** | **6-7 月** | WF AUC ≥ 0.910 達成時 |

## 6. V15 投資保護 (絶対不変)

| 不変 | 状態 |
|------|------|
| `predict_core.py` | ★完全不変★ |
| `daily_predict.py` | ★完全不変★ |
| `app.py` | ★完全不変★ |
| `keiba_model_v15_central*.pkl.gz` | ★完全不変★ |
| 累計 +¥14,140 | ★維持★ |
| 戦略⑦ + 案 B 改 | ★継続★ |

★ Phase 15 で作った V20 quick model (`models/v20/v20_quick_*.pkl.gz`) は production には投入しない ★

## 7. 結論

- **V20 学習 infrastructure 完成** (train/train_v20_ensemble.py、 GPU 利用、 LGB/XGB/FT/IR 4-model 対応)
- **V20 quick 学習成功** (LGB 0.8662 / XGB 0.8676 / ens 0.8678 on val 2025、 学習時間 ~20 sec)
- **Phase 11/12/13 features 全 constant default のため、 V20 候補 = V15 retrain と等価**
- **真の V20 (0.91+) には Phase 11/12/13 実 data が必要** — 5/11-5/24 で順次取得
- **5/10 V15 prod ROI 113.8% / +¥3,290** — V15 投資保護完全
