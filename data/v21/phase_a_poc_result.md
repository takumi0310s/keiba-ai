# Phase A - V21 Paddock PoC Result

Generated: 2026-05-16T12:26:47.162116
Elapsed: 5.7 min

## Data

- V15 cache rows: 527280
- Paddock entries (data/video_ai_features/): 89 rows
  - distinct races: 33
  - distinct horses: 85
- Merge matched (race_id+horse_id): 0 rows (0.0000% of cache)

## Features

- V15 baseline: 145 features
- V21 candidate: V15 + 12 paddock features

## WF results (6-fold, 2020-2025, LGB+XGB ensemble)

| Year | V15 LGB | V15 XGB | V15 ENS | V21 LGB | V21 XGB | V21 ENS |
|------|---------|---------|---------|---------|---------|---------|
| 2020 | 0.8579 | 0.8588 | 0.8592 | 0.8579 | 0.8585 | 0.8590 |
| 2021 | 0.8643 | 0.8669 | 0.8665 | 0.8643 | 0.8661 | 0.8660 |
| 2022 | 0.8672 | 0.8687 | 0.8688 | 0.8672 | 0.8683 | 0.8685 |
| 2023 | 0.8686 | 0.8699 | 0.8700 | 0.8686 | 0.8701 | 0.8702 |
| 2024 | 0.8704 | 0.8720 | 0.8720 | 0.8704 | 0.8719 | 0.8720 |
| 2025 | 0.8695 | 0.8706 | 0.8708 | 0.8695 | 0.8705 | 0.8708 |
| mean | 0.8663 | 0.8678 | 0.8679 | 0.8663 | 0.8676 | 0.8678 |

## Delta (V21 - V15, ENS)

- LGB: +0.000000
- XGB: -0.000240
- ENS: -0.000112

## Honest verdict

partial-success-coverage-insufficient: 0% overlap between paddock video (2026) and V15 cache (2015-2025). Cannot evaluate signal. 5/31 coverage target needed.

## Coverage caveats

- V15 cache contains 2015-2025 only; paddock video entries are all 2026 races. Direct (race_id, horse_id) merge therefore matches 0 rows.
- 12 paddock features are all NaN in the merged cache. LGB native NaN splits will not find usable signal; XGB likewise.
- Any V21 vs V15 delta observed should be treated as noise from training non-determinism, NOT as a video signal.

## 5/31 coverage plan

Target: 1,000-2,000 races with paddock features by 5/31.
- Current: 33 races
- Need: ~65 races/day (= ~130-180 paddock videos/day) for 15 days
- Pipeline: JRA RV / netkeiba paddock fetch -> frame extract -> yolov8/gait/body_condition -> data/video_ai_features/

Phase B (post-5/31) will re-run this PoC with merged paddock features on the 2025-end / 2026-start test fold, expecting >= 1-2% coverage of the fold for meaningful signal.
