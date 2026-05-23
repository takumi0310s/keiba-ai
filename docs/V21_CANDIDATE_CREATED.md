# V21 Candidate Model — Created 2026-05-23

## Summary

V21 candidate = V15 145 features + TYB 10 fields (155 total).
Saved to `models/v21_candidate.pkl.gz` for paper comparison only.
**V15 production completely unchanged.**

## Feature Composition

| Source | Count | Notes |
|--------|-------|-------|
| V15 features | 145 | All 145 booster features present in merged data |
| TYB fields | 10 | tyb_tansho_odds, tyb_fukusho_odds, tyb_odds_idx, tyb_jockey_idx, tyb_padock_idx, tyb_info_idx, tyb_padock_mark, tyb_ashimoto, tyb_sogo_idx, tyb_bagu_change |
| **V21 total** | **155** | |

## Walk-Forward CV AUC (5-fold, 2021-2025)

| Test Year | LGB | XGB | ENS |
|-----------|-----|-----|-----|
| 2021 | 0.8651 | 0.8668 | 0.8667 |
| 2022 | 0.8674 | 0.8682 | 0.8684 |
| 2023 | 0.8694 | 0.8702 | 0.8704 |
| 2024 | 0.8715 | 0.8718 | 0.8722 |
| 2025 | 0.8695 | 0.8701 | 0.8704 |
| **Mean** | | | **0.8696** |

## vs V15 Baseline

| Model | WF AUC | Delta |
|-------|--------|-------|
| V15 (genuine WF LGB+XGB) | 0.8678 | — |
| V21 candidate | 0.8696 | **+0.0018** |

Note: V21 ablation (v21_verdict.json) showed all individual TYB fields had negative delta (-0.0016 to -0.0025) when added one at a time. The +0.0018 here reflects the combined 10-field effect on the full 5-fold WF. This may be noise given the ablation NO-GO result; candidate is for paper comparison only.

## LEAK Gate

| TYB Field | corr(target) | Result |
|-----------|-------------|--------|
| tyb_tansho_odds | -0.2914 | OK |
| tyb_fukusho_odds | -0.2881 | OK |
| tyb_odds_idx | +0.4214 | OK |
| tyb_jockey_idx | +0.4564 | OK |
| tyb_padock_idx | +0.3539 | OK |
| tyb_info_idx | +0.4196 | OK |
| tyb_padock_mark | +0.1711 | OK |
| tyb_ashimoto | -0.0204 | OK |
| tyb_sogo_idx | +0.2573 | OK |
| tyb_bagu_change | -0.0313 | OK |
| **Max \|corr\|** | **0.4564** | **PASS (< 0.5)** |

## Model Details

- Path: `models/v21_candidate.pkl.gz` (2.0 MB)
- Architecture: LGB + XGB 2-model (same as V15 production)
- LGB rounds: 492, XGB rounds: 500
- Weights: w_lgb=0.4998, w_xgb=0.5002
- Training data: 527,280 rows (2015-2025)

## Verdict

**paper_only** — This model exists for paper shadow comparison against V15.
Per the ablation test (data/v21_verdict.json): all 10 TYB fields individually degraded AUC vs V15. Not for production deployment until further validation.

V15 production: **UNCHANGED** (145 features, auc=0.8939485520467574 verified).
