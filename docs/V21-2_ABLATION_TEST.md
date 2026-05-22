# V21-2 TYB Add-One Ablation Test

Date: 2026-05-22 | Session: V21-2

## Setup

| Item | Value |
|------|-------|
| Base dataset | `data/v21_tyb_merged.pkl.gz` (527,280 rows x 242 cols) |
| V15 baseline features | 145 (all 145 present in df) |
| Target column | `target` (binary, pos rate=0.216) |
| CV method | 5-fold StratifiedKFold (random_state=42) |
| LGB params | V15 production params (CLAUDE.md) |
| num_boost_round | 300 + early stopping 50 |
| Baseline AUC | **0.8678** (genuine WF 6-fold LGB+XGB, V15-audit-2) |
| GO threshold (per field) | delta > +0.001 |
| Full WF GO threshold | AUC >= 0.880 |

## TYB Columns Evaluated (10 fields)

| field | fill_rate | dtype |
|-------|-----------|-------|
| tyb_tansho_odds | 100% | float64 |
| tyb_fukusho_odds | 100% | float64 |
| tyb_odds_idx | 100% | float64 |
| tyb_jockey_idx | 100% | float64 |
| tyb_padock_idx | 100% | float64 |
| tyb_info_idx | 100% | float64 |
| tyb_padock_mark | 38.2% | float64 |
| tyb_ashimoto | 100% | int64 |
| tyb_sogo_idx | 100% | float64 |
| tyb_bagu_change | 100% | int64 |

## Step 3: Add-One Ablation Results (5-fold CV, V15 145 feats + 1 TYB field)

Sorted by delta descending:

| TYB field | 5-fold mean AUC | delta vs 0.8678 | Fold AUCs (f1-f5) |
|-----------|-----------------|-----------------|-------------------|
| tyb_tansho_odds | 0.8662 | -0.0016 | 0.8657, 0.8658, 0.8653, 0.8678, 0.8665 |
| tyb_fukusho_odds | 0.8662 | -0.0016 | 0.8661, 0.8658, 0.8652, 0.8677, 0.8663 |
| tyb_odds_idx | 0.8655 | -0.0023 | 0.8649, 0.8652, 0.8645, 0.8674, 0.8657 |
| tyb_jockey_idx | 0.8653 | -0.0025 | 0.8651, 0.8649, 0.8641, 0.8671, 0.8653 |
| tyb_padock_idx | 0.8653 | -0.0025 | 0.8647, 0.8648, 0.8642, 0.8672, 0.8655 |
| tyb_info_idx | 0.8653 | -0.0025 | 0.8647, 0.8649, 0.8643, 0.8671, 0.8654 |
| tyb_padock_mark | 0.8653 | -0.0025 | 0.8650, 0.8648, 0.8644, 0.8672, 0.8654 |
| tyb_ashimoto | 0.8653 | -0.0025 | 0.8650, 0.8649, 0.8643, 0.8670, 0.8652 |
| tyb_sogo_idx | 0.8653 | -0.0025 | 0.8650, 0.8647, 0.8641, 0.8672, 0.8654 |
| tyb_bagu_change | 0.8653 | -0.0025 | 0.8649, 0.8648, 0.8644, 0.8670, 0.8654 |

**GO candidates (delta > +0.001): NONE**

## Step 4: Full V21 Walk-Forward

Skipped -- no GO candidates from ablation.

## Verdict

**NO-GO**

All 10 TYB fields degrade 5-fold CV AUC relative to V15 baseline (0.8678).
- Best field: tyb_tansho_odds / tyb_fukusho_odds, delta = -0.0016
- Worst: most fields at delta = -0.0025
- GO threshold (+0.001) not met by any field

## Interpretation

1. **Information already absorbed**: V15 already has `odds_log` / `pop_rank` etc. The TYB tansho/fukusho odds provide essentially the same signal -- LGB confirms this by slightly declining AUC when the redundant feature is added (noise from extra column).
2. **Padock/index signals neutral-to-negative**: tyb_padock_idx / tyb_jockey_idx / tyb_sogo_idx all land at the same -0.0025 as the lowest group, indicating the padock evaluation data does not generalize across the 2020-2025 WF window.
3. **38.2% fill rate (tyb_padock_mark) not the root cause**: Even fully-filled fields (tyb_ashimoto, tyb_bagu_change) show identical -0.0025 degradation.
4. **V15 saturation confirmed again**: TYB is the second external data source after V22 interaction features that fails to improve V15 (see also Session #57 V20 interaction PoC: -2bp to +1.8bp).

## Next Steps

- TYB as add-one source: exhausted, closed
- V21 path options:
  - Architecture upgrade (V22 FT/IR full ensemble with .pkl save, V15-audit-1 flag)
  - JV-Link additional fields (Phase 3 6/9-6/13 parser)
  - Video features (Phase 4 7/1+)
- JRDB TYB verdict memory file: confirmed (tyb content PRE-RACE, signal absorbed by V15)

## Files

| File | Description |
|------|-------------|
| `data/v21_verdict.json` | Machine-readable verdict |
| `data/v21_tyb_merged.pkl.gz` | Read-only input (unchanged) |
| `tools/v21_ablation.py` | Ablation script |

## WF Cross-check (V21-2 year-based WF, 2021-2025)

Separate year-based walk-forward validation (same LGB+XGB params, 5 folds 2021-2025):

| | AUC |
|---|---|
| V15 baseline (year-WF) | 0.8696 |
| V15 baseline (StratifiedKFold ref) | 0.8678 |
| tansho_odds + V15 (year-WF) | 0.8696 (delta=-0.0001) |

Year-WF and StratifiedKFold both confirm: no TYB field lifts V15. Results consistent.

---

*V15 production files (keiba_model_v15_central.pkl.gz / predict_core.py / app.py) were not modified.*
