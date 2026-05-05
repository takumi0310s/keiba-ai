# v18/v19 distribution shift analysis (Phase 2.5)

生成: 2026-05-05 00:34:12

## 1. horse-level distribution

| dataset | n | mean | median | p95 | p99 | max |
|---|---:|---:|---:|---:|---:|---:|
| BT_v18_p_ens (tansho) | 47497 | 0.0548 | 0.0102 | 0.2797 | 0.5806 | 0.9863 |
| Retro_v18_p_tansho_raw | 932 | 0.0018 | 0.0004 | 0.0071 | 0.0208 | 0.1538 |
| Retro_v18_p_tansho_cal | 932 | 0.0032 | 0.0008 | 0.0125 | 0.0342 | 0.2127 |
| Retro_v19_p_fukusho_raw | 932 | 0.0016 | 0.0006 | 0.0056 | 0.0153 | 0.1423 |
| Retro_v19_p_fukusho_cal | 932 | 0.0016 | 0.0005 | 0.0057 | 0.0161 | 0.1561 |

## 2. race-level distribution

| dataset | n_race | race_max_p mean | race_max_p p95 | race_max_p max | race_sum_p mean | top1/top2 ratio mean | winner_top1 | winner_top3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BT_2025_OOS | 3455 | 0.347 | 0.748 | 0.986 | 0.753 | 4.13 | 47.8% | 78.8% |
| Retro_raw | 67 | 0.013 | 0.035 | 0.154 | 0.025 | 4.37 | 34.5% | 72.4% |
| Retro_calibrated | 67 | 0.020 | 0.055 | 0.213 | 0.044 | 3.68 | 34.5% | 72.4% |

## 3. shift attribution

- race_max_p factor (BT/retro): **27.69x**
- top1/top2 ratio diff: -0.24
- winner_top1 rate diff (BT - retro): +13.3pt
- **判定: RANK_SHIFT — 1着馬の選定自体がBTより劣化。feature distribution shift 疑い**

## 4. retro winner rank (in pred top-N)

| rank | count |
|---:|---:|
| 1 | 10 |
| 2 | 8 |
| 3 | 3 |
| 4 | 3 |
| 5 | 1 |
| 6 | 2 |
| 8 | 1 |
| 9 | 1 |