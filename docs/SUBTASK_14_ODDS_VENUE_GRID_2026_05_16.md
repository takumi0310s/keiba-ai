# Sub-task 14: 5-dimensional ROI Grid (read-only)

_Generated: 2026-05-16 23:24_  
_Source: cumulative_results.csv (settled, deduped) + daily_predictions/ + jrdb_kyi/oz/kab_  

## 0. Executive summary (3-line TL;DR)

- Sample is too small (563 races, mean cell N ~ 2.6) to populate a true 6,000+ cell 5-D grid; only marginal 1-3-D pockets reach interpretable N.
- Strategy 7 Case C exclusions (特別 / 京都 / 条件B / 条件E) are validated: excluded races ROI=**84.9%** (well below 100%).
- Only **1 statistically-leaning 5-D pocket** survives (福島 x 条件C x 15-18頭 x 良馬場: N=11, ROI=329.4%, p=0.091); single course-marginal **福島 x 条件C** (N=35, ROI=171%) and **中京 x 条件A x 12-14頭** (N=15, ROI=210%) are the most actionable signals.

## 1. Dataset baseline

- Settled JRA races: **563**
- Investment total: ¥394,100
- Payout total:     ¥399,340
- Overall ROI:      **101.3%**
- After Strategy 7 Case C exclusions: N=434, ROI=**106.2%**

Coverage of derived dimensions on master:
- 頭数: 541/563 (96.1%)
- top1 馬番: 384/563 (68.2%)
- top1 単勝オッズ: 317/563 (56.3%)
- 馬場: 451/563 (80.1%)

## 2. 2-dim heatmaps (sanity)

```

### Heatmap: ROI% by course x condition (all, N shown in parens; "-" if N<5)

course    |          A |          B |          C |          D |          E |          X 
----------------------------------------------------------------------------------------
中京        |   166%(N=19|          - |    45%(N=17|   109%(N=22|          - |          - 
中山        |    88%(N=32|          - |   101%(N=37|    82%(N=39|          - |     0%(N=10
京都        |    31%(N=27|          - |   376%(N=13|    36%(N=25|          - |          - 
新潟        |   216%(N=10|          - |   150%(N= 9|    59%(N=14|          - |     0%(N= 5
東京        |    36%(N=21|          - |    65%(N=21|    84%(N=25|          - |          - 
福島        |   175%(N= 8|          - |   171%(N=35|    93%(N=29|          - |          - 
阪神        |    97%(N=47|     0%(N= 5|    46%(N=20|   202%(N=47|     0%(N= 5|          - 


### Heatmap: ROI% by odds_band x condition (all, N shown in parens; "-" if N<5)

odds_band |          A |          B |          C |          D |          E |          X 
----------------------------------------------------------------------------------------
1-3       |    50%(N=52|          - |    36%(N=30|    62%(N=39|          - |          - 
10-20     |   241%(N= 8|          - |   408%(N=12|   188%(N= 8|          - |          - 
20+       |          - |          - |          - |          - |          - |          - 
3-5       |    48%(N=20|          - |   111%(N=37|    51%(N=38|          - |          - 
5-10      |   121%(N=17|          - |   115%(N=14|    67%(N=24|          - |          - 
NA        |   121%(N=67|    36%(N=12|    95%(N=56|   161%(N=88|    23%(N= 6|     0%(N=17


### Heatmap: ROI% by nh_band x condition (all, N shown in parens; "-" if N<5)

nh_band   |          A |          B |          C |          D |          E |          X 
----------------------------------------------------------------------------------------
12-14     |    92%(N=88|    48%(N= 9|          - |    93%(N=28|          - |          - 
15-18     |          - |          - |   127%(N=15|    66%(N=15|          - |    13%(N=13
8-11      |   104%(N=72|     0%(N= 5|          - |    82%(N=13|    12%(N=11|          - 
NA        |          - |          - |          - |   990%(N= 8|          - |     0%(N= 6


### Heatmap: ROI% by course x odds_band (all, N shown in parens; "-" if N<5)

course    |        1-3 |      10-20 |        20+ |        3-5 |       5-10 |         NA 
----------------------------------------------------------------------------------------
中京        |    75%(N=17|          - |          - |    93%(N=14|   147%(N= 5|   130%(N=23
中山        |    54%(N=32|          - |          - |   125%(N=26|    58%(N= 8|    78%(N=56
京都        |    67%(N=17|   790%(N= 6|          - |    45%(N= 6|     0%(N= 5|    18%(N=34
新潟        |    89%(N= 5|          - |          - |          - |   165%(N= 9|    33%(N=19
東京        |    45%(N= 9|   112%(N= 9|          - |    27%(N=12|    65%(N= 5|    71%(N=35
福島        |     9%(N=12|    26%(N= 6|          - |    39%(N=16|   179%(N=12|   210%(N=24
阪神        |    28%(N=35|          - |          - |    61%(N=20|    22%(N=13|   219%(N=55


### Heatmap: ROI% by tc x condition (all, N shown in parens; "-" if N<5)

tc        |          A |          B |          C |          D |          E |          X 
----------------------------------------------------------------------------------------
NA        |   146%(N=29|     0%(N= 5|    26%(N=23|   265%(N=40|    24%(N= 5|     0%(N=10
稍         |   186%(N=19|          - |   153%(N= 8|    70%(N=27|          - |          - 
良         |    68%(N=11|          - |   142%(N=12|    73%(N=12|     0%(N= 5|          - 
重         |          - |    48%(N= 9|          - |    12%(N=10|          - |    21%(N= 8
```

## 3a. Marginal pockets (low-dim, N>=10, ROI>=130%, p<=0.20)

_5-dim cells are too sparse (mean N ~2.6). Marginal grids reveal interpretable pockets._

### course x condition
| course | condition | N | ROI% | hit% | CI_lo% | CI_hi% | p_value |
|---|---|---|---|---|---|---|---|
| 福島 | C | 35 | 171.3 | 25.7 | 45.4 | 339.0 | 0.1813 |

### odds_band x condition
_No cells meet thresholds._

### nh_band x condition
_No cells meet thresholds._

### tc x condition
_No cells meet thresholds._

### course x odds_band x condition
| course | odds_band | condition | N | ROI% | hit% | CI_lo% | CI_hi% | p_value |
|---|---|---|---|---|---|---|---|---|
| 福島 | NA | C | 11 | 329.4 | 54.5 | 103.0 | 674.6 | 0.0911 |

### course x nh_band x condition
| course | nh_band | condition | N | ROI% | hit% | CI_lo% | CI_hi% | p_value |
|---|---|---|---|---|---|---|---|---|
| 中京 | 12-14 | A | 15 | 210.4 | 46.7 | 50.1 | 460.2 | 0.1784 |
| 福島 | 15-18 | C | 35 | 171.3 | 25.7 | 45.4 | 339.0 | 0.1813 |

### odds_band x nh_band x condition
_No cells meet thresholds._

## 3. High-ROI 5-D pockets (base, N>=10, ROI>=130%, p<=0.20)

| label | odds_band | course | nh_band | tc | condition | N | investment | payout | ROI% | hit% | CI_lo% | CI_hi% | p_value |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base | NA | 福島 | 15-18 | 良 | C | 11 | 7700 | 25360 | 329.4 | 54.5 | 103.0 | 674.6 | 0.0911 |

## 4. High-ROI pockets after Strategy 7 Case C

| label | odds_band | course | nh_band | tc | condition | N | investment | payout | ROI% | hit% | CI_lo% | CI_hi% | p_value |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| strat7c | NA | 福島 | 15-18 | 良 | C | 10 | 7000 | 25360 | 362.3 | 60.0 | 114.6 | 732.7 | 0.082 |

## 5. Exploratory 5-D pockets (5<=N<10, ROI>=200%; exploratory only)

_These are candidates that warrant continued tracking but are NOT yet statistically reliable._

| label | odds_band | course | nh_band | tc | condition | N | investment | payout | ROI% | hit% | CI_lo% | CI_hi% | p_value |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base | NA | 阪神 | NA | NA | D | 5 | 3500 | 54840 | 1566.9 | 20.0 | 0.0 | 4700.6 | 0.2011 |
| base | NA | 中京 | 12-14 | NA | A | 6 | 4200 | 14190 | 337.9 | 33.3 | 0.0 | 930.2 | 0.2253 |
| base | NA | 中山 | 12-14 | NA | A | 5 | 3500 | 10250 | 292.9 | 60.0 | 77.1 | 508.6 | 0.1044 |

## 5b. Strong 5-D pockets (N>=15, ROI>=150%, p<=0.10)

_No 5-D cell meets the strong threshold (expected — sample is sparse)._

## 6. "Unexpected" pockets (potential discoveries)

Cells with ROI>=150% AND N>=20 that do NOT involve the known A/C/X-condition + 良 馬場 sweet spot.
Specifically: track_condition in {稍,重,不良} OR condition in {B,D,E,X}.

_No surprises._

## 7. Strategy 7 Case C consistency check

Does Strategy 7 Case C accidentally exclude high-ROI pockets?  
Build a small grid restricted to excluded races and compute their ROI.

- Excluded N: 129
- Excluded investment: ¥90,300
- Excluded payout:     ¥76,660
- Excluded ROI:        **84.9%**
  - 特別: N=40, ROI=93.7%
  - 京都: N=69, ROI=98.0%
  - 条件B: N=16, ROI=27.0%
  - 条件E: N=11, ROI=12.3%

## 8. 5/18+ paper-eval prompt (pocket pickup strategy)

Suggested prompt template for paper-eval starting 2026-05-18:

```
When daily_predict / race_auto_notify runs, additionally label each race with its 5-tuple:
  (top1_odds_band, course, num_horses_band, track_condition, condition)
Pocket priority list (this doc, prioritized; >=100% ROI, N>=10, p<=0.20):

  P1 (strong)  : 福島 x C x 15-18頭 x 良     (N=11, ROI=329%, p=0.09)
                 also passes Strategy 7 C  (N=10, ROI=362%, p=0.08)
  P2 (medium)  : 福島 x C                   (N=35, ROI=171%, p=0.18)
  P3 (medium)  : 中京 x A x 12-14頭          (N=15, ROI=210%, p=0.18)
  P4 (explor)  : 京都 x C (deduped)          (N=13, ROI=376%, hit=3/13)  <- 97% of payout from 1 race (5/16 上賀茂S); KEEP exclusion
  P5 (explor)  : 阪神 x D (NA-band)          (N=5,  ROI=1567%)        <- single ¥54,840 outlier

Notify in #updates when next race is a POCKET HIT.
Track: pocket-hit ROI vs non-pocket ROI for 4 weeks (target N>=80 pocket-hit races).
Decision rule (2026-06-15): if pocket-hit ROI > non-pocket ROI + 20pt with p<=0.05
  -> propose pocket-aware bet sizing rule for V20 deployment (e.g. P1+P2 races: 700->1400 yen).
Critical: P4 (京都 x C) conflicts with current Strategy 7 Case C exclusion.
  Excluded 京都 overall = 98% ROI (border-zone) but 京都 x C subset = 376% on N=13,
  driven by 1 large trio hit (¥33,200 / ¥34,210 = 97%). KEEP exclusion; track in shadow.
```

## 9. Fabrication checks

- All ROI values computed from settled rows in cumulative_results.csv (investment=¥700/race fixed).
- top1 odds = jrdb_oz tansho_NN final (preferred) -> jrdb_kyi 基準オッズ (fallback).
- track_condition = daily_predictions (preferred) -> jrdb_kab turf/dirt_baba_code (fallback).
- num_horses = daily_predictions (preferred) -> jrdb_kyi count (fallback).
- p-value: Welch one-sided t-test on (payout - investment) vs 0 (H1: ROI>100%).
- CI: 2000-iteration bootstrap of mean payout / investment.
- Cells with N<5 reported as "-" in heatmaps.
- All CSVs saved to data/subtask14_{master,grid_base,grid_strat7c}.csv for audit.
- Today's (2026-05-16) cumulative_results contained **34 duplicated race_ids** (in-flight re-runs); deduped before analysis. Without dedup, overall ROI was inflated to 108.5%; clean ROI is 101.3%.

## 10. Caveats

- 5-D grid is severely under-sampled at the current betting volume (563 settled JRA races). 6,000+ cells is the theoretical product space but only ~230 are populated and 186/230 have N<5.
- top1 odds coverage is only 56.3% — many races have null top1_num in both cumulative and daily_predictions, so the odds band drops into the "NA" bucket. NA-band aggregations should be treated as a "missing-odds" proxy, not as an actual odds-tier signal.
- Single-race lottery hits (e.g. ¥54,840 trio at 阪神 D condition on 5/16) dominate small-cell ROIs. Bootstrap CIs reflect this; p-values do not. Treat any pocket with N<30 as exploratory only.
- No fabrication: every payout, investment, and odds value originates from the listed CSV. Welch t-test is one-sided (H1: ROI>100%) on per-race payout-minus-investment.
