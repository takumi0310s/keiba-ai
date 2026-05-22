# V21-1 TYB Retrospective Merge

**Date**: 2026-05-22
**Task**: Merge JRDB TYB (直前情報) historical data into V15 training dataset for V21 retrospective training.

---

## Result Summary

| Item | Value |
|------|-------|
| V15 training rows | 527,280 |
| V15 unique races | 38,002 |
| TYB source rows (jrdb_tyb.csv) | 550,115 |
| TYB year range | 2015-2026 (1,340 files) |
| Merged rows | 527,280 (left join, no drop) |
| New TYB columns added | 10 |
| Merged dataset | `data/v21_tyb_merged.pkl.gz` (108MB) |
| Stats JSON | `data/v21_tyb_stats.json` |

---

## Join Key Design

TYB files use 12-char race_id format: `YYYY + basho(2) + kai(2) + nichi(2) + race_num(2)`
V15 uses 10-char race_id: `basho(2) + yr2(2) + kai_hex1(1) + nichi_hex1(1) + race_num(2) + umaban(2)`

Conversion function: extract kai/nichi as integers, re-encode as single hex char (decimal 1-9 = '1'-'9', 10+ = 'A', 'B', ...).

Join: `df_v15.race_id == tyb.v15_race_id` (built from TYB race_id + umaban).

---

## Coverage

| TYB Column | Coverage |
|-----------|---------|
| tyb_tansho_odds | 100.0% |
| tyb_fukusho_odds | 100.0% |
| tyb_odds_idx | 100.0% |
| tyb_jockey_idx | 100.0% |
| tyb_padock_idx | 100.0% |
| tyb_info_idx | 100.0% |
| tyb_padock_mark | 38.2% (sparse - field not always filled) |
| tyb_ashimoto | 100.0% |
| tyb_sogo_idx | 100.0% |
| tyb_bagu_change | 100.0% |

Note: V15's existing `tansho_odds` was a fill-value placeholder (all 15.0). `tyb_tansho_odds` provides real -15min odds values.

---

## LEAK Gate (T4 Check)

Correlation with target (`finish <= 3`):

| Field | corr_target | Verdict |
|-------|------------|---------|
| tyb_tansho_odds | -0.2914 | OK |
| tyb_fukusho_odds | -0.2881 | OK |
| tyb_odds_idx | +0.4214 | FLAG (see note) |
| tyb_jockey_idx | +0.4564 | FLAG (see note) |
| tyb_padock_idx | +0.3539 | OK |
| tyb_info_idx | +0.4196 | FLAG (see note) |
| tyb_padock_mark | -0.2077 | OK |
| tyb_ashimoto | -0.0204 | OK |
| tyb_sogo_idx | +0.2573 | OK |
| tyb_bagu_change | -0.0313 | OK |

**Max |corr| = 0.4564 < 0.50 threshold**

### LEAK GATE VERDICT: PASS

**FLAG field analysis** (0.40-0.50 range):

- `tyb_jockey_idx`, `tyb_info_idx`, `tyb_odds_idx` show moderate positive correlation with target
- These are JRDB analyst prediction indices published **BEFORE the race (~15min prior)**
- All three correlate strongly with popularity: corr(jockey_idx, popularity) = -0.81, corr(info_idx, popularity) = -0.80
- Pattern shows **gradual degradation** by finish position (finish=1 mean=1.9, finish=10 mean=0.6) — expected for a prediction index
- This contrasts with SKB POST-RACE leak pattern: SKB had finish=1 count 15% vs 10-place 49% (direct result encoding)
- V15's existing `jrdb_info_idx` (from KYI/Paci, same pre-race source) has identical corr = +0.42 — confirming this is normal behavior
- **Conclusion**: High corr = good predictive signal, NOT post-race leak

---

## Merged Data Location

- **Merged dataset**: `data/v21_tyb_merged.pkl.gz` (covered by .gitignore)
- **Stats**: `data/v21_tyb_stats.json`
- **New columns**: 10 columns prefixed `tyb_*`
- **Schema**: original V15 232 columns + 10 TYB columns = 242 columns total

---

## Usage for V21 Training

```python
import pickle, gzip
with gzip.open('data/v21_tyb_merged.pkl.gz', 'rb') as f:
    df_v21 = pickle.load(f)

# New TYB features available
TYB_FEATURES = [
    'tyb_tansho_odds', 'tyb_fukusho_odds',
    'tyb_odds_idx', 'tyb_jockey_idx', 'tyb_padock_idx',
    'tyb_info_idx', 'tyb_sogo_idx',
    'tyb_padock_mark', 'tyb_ashimoto', 'tyb_bagu_change',
]
```

Note: `tyb_padock_mark` has only 38.2% coverage — consider NaN fill strategy before including in feature set.
