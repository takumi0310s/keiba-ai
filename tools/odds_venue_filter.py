#!/usr/bin/env python3
"""odds_venue_filter.py - TASK 5: Odds Band x Venue Filter Refinement

Walk-forward backtest cache (wf_race_results_cache.json) を用いて、
オッズ帯 x 会場 x AI予測順位 x 条件 x 券種 の多次元交差で
収益性の高いパターンを抽出する。

分析軸:
  - Odds band: 1-3, 3-10, 10-30, 30-100, 100+ (AI top-ranked horse odds)
  - Venue: 10 JRA courses
  - AI rank position: top1 ~ top6
  - Bet type: trio (三連複), umaren (馬連)
  - Condition: A-E, X

出力:
  - data/odds_venue_profitable_patterns.json
  - コンソールにサマリーテーブル

Usage:
    python tools/odds_venue_filter.py              # フル分析
    python tools/odds_venue_filter.py --min-n 50   # 最低サンプル数変更
    python tools/odds_venue_filter.py --years 2023-2025  # 対象年を絞る
    python tools/odds_venue_filter.py --rebuild    # キャッシュ再構築
"""

import argparse
import json
import os
import sys
import time
import warnings
from collections import defaultdict
from datetime import datetime
from itertools import product

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# Force unbuffered output
import functools
print = functools.partial(print, flush=True)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, 'train'))
sys.path.insert(0, BASE_DIR)
DATA_DIR = os.path.join(BASE_DIR, 'data')

CACHE_PATH = os.path.join(DATA_DIR, 'wf_race_results_cache.json')
OUTPUT_PATH = os.path.join(DATA_DIR, 'odds_venue_profitable_patterns.json')

# ============================================================
# Constants
# ============================================================

ODDS_BANDS = [
    ('1-3',    1.0,    3.0),
    ('3-10',   3.0,   10.0),
    ('10-30', 10.0,   30.0),
    ('30-100', 30.0, 100.0),
    ('100+',  100.0, 99999.0),
]

JRA_VENUES = ['札幌', '函館', '福島', '新潟', '東京', '中山', '中京', '京都', '阪神', '小倉']

CONDITIONS = ['A', 'B', 'C', 'D', 'E', 'X']

# Investment per race: 7 tickets x 100 yen (trio), 2 tickets x 350 yen (umaren)
TRIO_INVESTMENT = 700
UMAREN_INVESTMENT = 700


# ============================================================
# Helper Functions
# ============================================================

def get_odds_band(odds):
    """Map odds value to band label."""
    if odds is None or (isinstance(odds, float) and np.isnan(odds)):
        return None
    for label, lo, hi in ODDS_BANDS:
        if lo <= odds < hi:
            return label
    return None


def classify_condition(num_horses, distance, condition_enc):
    """Condition classification matching backtest_central_leakfree.py."""
    heavy = condition_enc >= 2
    if num_horses <= 7:
        return 'E'
    if distance <= 1400:
        return 'D'
    if 8 <= num_horses <= 14 and distance >= 1600 and not heavy:
        return 'A'
    if 8 <= num_horses <= 14 and distance >= 1600 and heavy:
        return 'B'
    if num_horses >= 15 and distance >= 1600 and not heavy:
        return 'C'
    return 'X'


def calc_trio_bets(ranking):
    """三連複フォーメーション: TOP1軸-TOP2,3-TOP2~6 (最大7点)."""
    if len(ranking) < 3:
        return []
    nums = ranking[:min(6, len(ranking))]
    n1 = nums[0]
    second = nums[1:3]
    third = nums[1:min(6, len(nums))]
    return sorted(set(
        tuple(sorted({n1, s, t}))
        for s in second for t in third
        if len(set({n1, s, t})) == 3
    ))


def calc_umaren_bets(ranking):
    """馬連: TOP1-TOP2, TOP1-TOP3."""
    if len(ranking) < 3:
        return []
    return [tuple(sorted([ranking[0], ranking[1]])),
            tuple(sorted([ranking[0], ranking[2]]))]


def get_recommendation(roi, n):
    """Star rating based on ROI and sample size."""
    if roi > 200 and n >= 100:
        return 3
    if roi > 150 and n >= 50:
        return 2
    if roi > 100 and n >= 30:
        return 1
    return 0


def confidence_label(n):
    """Confidence label based on sample size."""
    if n >= 200:
        return 'HIGH'
    if n >= 100:
        return 'MEDIUM'
    if n >= 50:
        return 'LOW'
    return 'VERY_LOW'


# ============================================================
# Cache Loading / Building
# ============================================================

def load_cache(year_range=None):
    """Load wf_race_results_cache.json."""
    if not os.path.exists(CACHE_PATH):
        return None
    with open(CACHE_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    races = data.get('races', [])
    if year_range:
        lo, hi = year_range
        races = [r for r in races if lo <= r['year'] <= hi]
    print(f"Cache loaded: {len(races)} races"
          f" (generated {data.get('generated', 'unknown')})")
    return races


def rebuild_cache():
    """Run WF backtest via profitable_patterns.py to rebuild cache."""
    pp_path = os.path.join(BASE_DIR, 'tools', 'profitable_patterns.py')
    if not os.path.exists(pp_path):
        print("ERROR: profitable_patterns.py not found. Cannot rebuild cache.")
        return None
    print("Rebuilding cache via profitable_patterns.py --rebuild ...")
    import subprocess
    result = subprocess.run(
        [sys.executable, pp_path, '--rebuild', '--summary'],
        cwd=BASE_DIR, capture_output=False)
    if result.returncode != 0:
        print("ERROR: Cache rebuild failed.")
        return None
    return load_cache()


# ============================================================
# Enrich race data with per-rank odds info
# ============================================================

def enrich_races(races):
    """For each race, compute per-AI-rank odds bands and
    whether each rank's horse finished in top 3."""
    enriched = []
    for race in races:
        if not race.get('has_payout'):
            continue
        horses = race.get('horses', [])
        if len(horses) < 3:
            continue

        # Build lookup by ai_rank
        by_rank = {}
        for h in horses:
            by_rank[h['ai_rank']] = h

        # Get top-1 odds band for the race
        top1 = by_rank.get(1)
        if not top1:
            continue
        top1_odds = top1.get('tansho_odds')
        top1_odds_band = get_odds_band(top1_odds)

        # Ranking list
        ranking = [by_rank[r]['umaban']
                    for r in sorted(by_rank.keys())
                    if r <= len(horses)]

        entry = {
            'race_id': race['race_id_str'],
            'year': race['year'],
            'venue': race['course'],
            'distance': race['distance'],
            'condition_enc': race['condition_enc'],
            'num_horses': race['num_horses'],
            'cond_class': race.get('cond_class') or classify_condition(
                race['num_horses'], race['distance'], race['condition_enc']),
            'surface': race.get('surface', ''),
            'top1_odds_band': top1_odds_band,
            'trio_hit': race.get('trio_hit', False),
            'trio_return': race.get('trio_return', 0),
            'umaren_hit': race.get('umaren_hit', False),
            'umaren_return': race.get('umaren_return', 0),
            'ranking': ranking,
        }

        # Per-rank info (rank 1-6)
        for rank in range(1, 7):
            h = by_rank.get(rank)
            if h:
                odds_val = h.get('tansho_odds')
                entry[f'rank{rank}_odds'] = odds_val
                entry[f'rank{rank}_odds_band'] = get_odds_band(odds_val)
                entry[f'rank{rank}_finish'] = h.get('finish')
                entry[f'rank{rank}_top3'] = (h.get('finish', 99) <= 3)
                entry[f'rank{rank}_umaban'] = h.get('umaban')
            else:
                entry[f'rank{rank}_odds'] = None
                entry[f'rank{rank}_odds_band'] = None
                entry[f'rank{rank}_finish'] = None
                entry[f'rank{rank}_top3'] = False
                entry[f'rank{rank}_umaban'] = None

        enriched.append(entry)

    return enriched


# ============================================================
# Core Analysis
# ============================================================

def analyze_odds_venue_patterns(enriched, min_n=30):
    """Cross-analyze odds band x venue x AI rank x condition x bet type."""

    df = pd.DataFrame(enriched)
    if len(df) == 0:
        print("ERROR: No enriched data.")
        return None

    total_races = len(df)
    print(f"\nAnalyzing {total_races} races with payout data")
    print(f"Year range: {df['year'].min()} - {df['year'].max()}")
    print(f"Venues: {sorted(df['venue'].unique())}")
    print(f"Conditions: {sorted(df['cond_class'].unique())}")

    all_patterns = []
    total_tested = 0

    # ================================================================
    # Analysis 1: Trio ROI by top1_odds_band x venue x condition
    # ================================================================
    print("\n[1/6] Trio by top1_odds_band x venue x condition ...")
    odds_bands_list = [b[0] for b in ODDS_BANDS]
    venues_in_data = sorted(df['venue'].unique())
    conds_in_data = sorted(df['cond_class'].unique())

    for ob, venue, cond in product(odds_bands_list, venues_in_data, conds_in_data):
        mask = ((df['top1_odds_band'] == ob) &
                (df['venue'] == venue) &
                (df['cond_class'] == cond))
        subset = df[mask]
        total_tested += 1
        n = len(subset)
        if n < min_n:
            continue
        hits = int(subset['trio_hit'].sum())
        total_return = int(subset['trio_return'].sum())
        investment = n * TRIO_INVESTMENT
        roi = (total_return / investment * 100) if investment > 0 else 0
        hit_rate = hits / n * 100 if n > 0 else 0

        if roi > 100:
            all_patterns.append({
                'analysis': 'trio_odds_venue_cond',
                'bet_type': 'trio',
                'odds_band': ob,
                'venue': venue,
                'condition': cond,
                'ai_rank': 'top1_odds',
                'roi': round(roi, 1),
                'hit_rate': round(hit_rate, 1),
                'n': n,
                'hits': hits,
                'investment': investment,
                'total_return': total_return,
                'confidence': confidence_label(n),
                'recommendation': get_recommendation(roi, n),
            })

    # ================================================================
    # Analysis 2: Trio ROI by top1_odds_band x venue (condition=all)
    # ================================================================
    print("[2/6] Trio by top1_odds_band x venue ...")
    for ob, venue in product(odds_bands_list, venues_in_data):
        mask = ((df['top1_odds_band'] == ob) & (df['venue'] == venue))
        subset = df[mask]
        total_tested += 1
        n = len(subset)
        if n < min_n:
            continue
        hits = int(subset['trio_hit'].sum())
        total_return = int(subset['trio_return'].sum())
        investment = n * TRIO_INVESTMENT
        roi = (total_return / investment * 100) if investment > 0 else 0
        hit_rate = hits / n * 100 if n > 0 else 0

        if roi > 100:
            all_patterns.append({
                'analysis': 'trio_odds_venue',
                'bet_type': 'trio',
                'odds_band': ob,
                'venue': venue,
                'condition': 'all',
                'ai_rank': 'top1_odds',
                'roi': round(roi, 1),
                'hit_rate': round(hit_rate, 1),
                'n': n,
                'hits': hits,
                'investment': investment,
                'total_return': total_return,
                'confidence': confidence_label(n),
                'recommendation': get_recommendation(roi, n),
            })

    # ================================================================
    # Analysis 3: Umaren ROI by top1_odds_band x venue x condition
    # ================================================================
    print("[3/6] Umaren by top1_odds_band x venue x condition ...")
    for ob, venue, cond in product(odds_bands_list, venues_in_data, conds_in_data):
        mask = ((df['top1_odds_band'] == ob) &
                (df['venue'] == venue) &
                (df['cond_class'] == cond))
        subset = df[mask]
        total_tested += 1
        n = len(subset)
        if n < min_n:
            continue
        hits = int(subset['umaren_hit'].sum())
        total_return = int(subset['umaren_return'].sum())
        # umaren: 2 tickets x 350 yen, but payout is for 100-yen unit
        # so multiply return by 3.5 (350 / 100)
        total_return_adj = int(total_return * 3.5)
        investment = n * UMAREN_INVESTMENT
        roi = (total_return_adj / investment * 100) if investment > 0 else 0
        hit_rate = hits / n * 100 if n > 0 else 0

        if roi > 100:
            all_patterns.append({
                'analysis': 'umaren_odds_venue_cond',
                'bet_type': 'umaren',
                'odds_band': ob,
                'venue': venue,
                'condition': cond,
                'ai_rank': 'top1_odds',
                'roi': round(roi, 1),
                'hit_rate': round(hit_rate, 1),
                'n': n,
                'hits': hits,
                'investment': investment,
                'total_return': total_return_adj,
                'confidence': confidence_label(n),
                'recommendation': get_recommendation(roi, n),
            })

    # ================================================================
    # Analysis 4: Per-rank top3 accuracy by odds_band x venue
    # (Which AI ranks are most accurate at which venues/odds?)
    # ================================================================
    print("[4/6] AI rank accuracy by odds_band x venue ...")
    for rank in range(1, 7):
        ob_col = f'rank{rank}_odds_band'
        top3_col = f'rank{rank}_top3'
        rank_label = f'top{rank}'

        for ob, venue in product(odds_bands_list, venues_in_data):
            mask = ((df[ob_col] == ob) & (df['venue'] == venue))
            subset = df[mask]
            total_tested += 1
            n = len(subset)
            if n < min_n:
                continue
            top3_count = int(subset[top3_col].sum())
            top3_rate = top3_count / n * 100 if n > 0 else 0

            # top3 rate > 33% is better than random
            if top3_rate > 33.3:
                all_patterns.append({
                    'analysis': 'rank_accuracy_odds_venue',
                    'bet_type': 'accuracy',
                    'odds_band': ob,
                    'venue': venue,
                    'condition': 'all',
                    'ai_rank': rank_label,
                    'roi': round(top3_rate, 1),  # use top3_rate as "roi" for sorting
                    'hit_rate': round(top3_rate, 1),
                    'n': n,
                    'hits': top3_count,
                    'investment': 0,
                    'total_return': 0,
                    'confidence': confidence_label(n),
                    'recommendation': get_recommendation(top3_rate * 3, n),  # scale for stars
                })

    # ================================================================
    # Analysis 5: Trio by top1_odds_band x condition (venue=all)
    # ================================================================
    print("[5/6] Trio by top1_odds_band x condition ...")
    for ob, cond in product(odds_bands_list, conds_in_data):
        mask = ((df['top1_odds_band'] == ob) & (df['cond_class'] == cond))
        subset = df[mask]
        total_tested += 1
        n = len(subset)
        if n < min_n:
            continue
        hits = int(subset['trio_hit'].sum())
        total_return = int(subset['trio_return'].sum())
        investment = n * TRIO_INVESTMENT
        roi = (total_return / investment * 100) if investment > 0 else 0
        hit_rate = hits / n * 100 if n > 0 else 0

        if roi > 100:
            all_patterns.append({
                'analysis': 'trio_odds_cond',
                'bet_type': 'trio',
                'odds_band': ob,
                'venue': 'all',
                'condition': cond,
                'ai_rank': 'top1_odds',
                'roi': round(roi, 1),
                'hit_rate': round(hit_rate, 1),
                'n': n,
                'hits': hits,
                'investment': investment,
                'total_return': total_return,
                'confidence': confidence_label(n),
                'recommendation': get_recommendation(roi, n),
            })

    # ================================================================
    # Analysis 6: Umaren by top1_odds_band x condition (venue=all)
    # ================================================================
    print("[6/6] Umaren by top1_odds_band x condition ...")
    for ob, cond in product(odds_bands_list, conds_in_data):
        mask = ((df['top1_odds_band'] == ob) & (df['cond_class'] == cond))
        subset = df[mask]
        total_tested += 1
        n = len(subset)
        if n < min_n:
            continue
        hits = int(subset['umaren_hit'].sum())
        total_return = int(subset['umaren_return'].sum())
        total_return_adj = int(total_return * 3.5)
        investment = n * UMAREN_INVESTMENT
        roi = (total_return_adj / investment * 100) if investment > 0 else 0
        hit_rate = hits / n * 100 if n > 0 else 0

        if roi > 100:
            all_patterns.append({
                'analysis': 'umaren_odds_cond',
                'bet_type': 'umaren',
                'odds_band': ob,
                'venue': 'all',
                'condition': cond,
                'ai_rank': 'top1_odds',
                'roi': round(roi, 1),
                'hit_rate': round(hit_rate, 1),
                'n': n,
                'hits': hits,
                'investment': investment,
                'total_return': total_return_adj,
                'confidence': confidence_label(n),
                'recommendation': get_recommendation(roi, n),
            })

    # ================================================================
    # Build year stability for top patterns
    # ================================================================
    year_stability = {}
    for ob in odds_bands_list:
        for cond in conds_in_data:
            key = f"trio_{ob}_{cond}"
            mask = ((df['top1_odds_band'] == ob) & (df['cond_class'] == cond))
            sub = df[mask]
            if len(sub) < min_n:
                continue
            yearly = {}
            for yr in sorted(df['year'].unique()):
                yr_sub = sub[sub['year'] == yr]
                n_yr = len(yr_sub)
                if n_yr < 5:
                    continue
                hits_yr = int(yr_sub['trio_hit'].sum())
                ret_yr = int(yr_sub['trio_return'].sum())
                inv_yr = n_yr * TRIO_INVESTMENT
                roi_yr = (ret_yr / inv_yr * 100) if inv_yr > 0 else 0
                yearly[yr] = {'n': n_yr, 'roi': round(roi_yr, 1), 'hits': hits_yr}
            if yearly:
                year_stability[key] = {int(k): v for k, v in yearly.items()}

    # Sort all patterns by ROI descending
    bet_patterns = [p for p in all_patterns if p['bet_type'] != 'accuracy']
    acc_patterns = [p for p in all_patterns if p['bet_type'] == 'accuracy']
    bet_patterns.sort(key=lambda x: (-x['recommendation'], -x['roi']))
    acc_patterns.sort(key=lambda x: (-x['hit_rate'], -x['n']))

    # Star string conversion
    for p in all_patterns:
        stars = p['recommendation']
        p['recommendation_str'] = '\u2605' * stars if stars > 0 else '-'

    # Build result
    result = {
        'generated_at': datetime.now().isoformat(),
        'total_patterns_tested': total_tested,
        'profitable_patterns': len(bet_patterns),
        'accuracy_patterns_count': len(acc_patterns),
        'total_races_analyzed': total_races,
        'year_range': [int(df['year'].min()), int(df['year'].max())],
        'min_n': min_n,
        'patterns': bet_patterns,
        'accuracy_patterns': acc_patterns,
        'year_stability': year_stability,
    }

    return result


# ============================================================
# Display
# ============================================================

def print_summary(result):
    """Print summary table to console."""
    if result is None:
        print("No results.")
        return

    print("\n" + "=" * 80)
    print("  TASK 5: Odds Band x Venue Filter Refinement")
    print("=" * 80)
    print(f"  Generated: {result['generated_at']}")
    print(f"  Races analyzed: {result['total_races_analyzed']}")
    print(f"  Year range: {result['year_range'][0]}-{result['year_range'][1]}")
    print(f"  Min N: {result['min_n']}")
    print(f"  Patterns tested: {result['total_patterns_tested']}")
    print(f"  Profitable bet patterns: {result['profitable_patterns']}")

    patterns = result.get('patterns', [])
    star3 = [p for p in patterns if p['recommendation'] == 3]
    star2 = [p for p in patterns if p['recommendation'] == 2]
    star1 = [p for p in patterns if p['recommendation'] == 1]

    print(f"    3-star (ROI>200%, N>=100): {len(star3)}")
    print(f"    2-star (ROI>150%, N>=50):  {len(star2)}")
    print(f"    1-star (ROI>100%, N>=30):  {len(star1)}")

    # ---- Top bet patterns ----
    if patterns:
        print(f"\n{'='*80}")
        print(f"  Top 40 Profitable Bet Patterns")
        print(f"{'='*80}")
        hdr = (f"{'#':>3} {'Type':>7} {'Odds':>7} {'Venue':>6} {'Cond':>4} "
               f"{'N':>5} {'Hits':>5} {'Rate%':>6} {'ROI%':>8} {'Conf':>8} {'Stars':>5}")
        print(hdr)
        print("-" * 80)
        for i, p in enumerate(patterns[:40]):
            print(f"{i+1:>3} {p['bet_type']:>7} {p['odds_band']:>7} "
                  f"{p['venue']:>6} {p['condition']:>4} "
                  f"{p['n']:>5} {p['hits']:>5} {p['hit_rate']:>5.1f}% "
                  f"{p['roi']:>7.1f}% {p['confidence']:>8} "
                  f"{p['recommendation_str']:>5}")

    # ---- Accuracy patterns ----
    acc = result.get('accuracy_patterns', [])
    if acc:
        print(f"\n{'='*80}")
        print(f"  AI Rank Accuracy (top3 rate > 33%)")
        print(f"{'='*80}")
        hdr = (f"{'Rank':>6} {'Odds':>7} {'Venue':>6} "
               f"{'N':>5} {'Top3':>5} {'Rate%':>7} {'Conf':>8}")
        print(hdr)
        print("-" * 60)
        for p in acc[:30]:
            print(f"{p['ai_rank']:>6} {p['odds_band']:>7} {p['venue']:>6} "
                  f"{p['n']:>5} {p['hits']:>5} {p['hit_rate']:>6.1f}% "
                  f"{p['confidence']:>8}")

    # ---- Year stability for key combos ----
    ys = result.get('year_stability', {})
    if ys:
        print(f"\n{'='*80}")
        print(f"  Year Stability (Trio ROI% by odds_band x condition)")
        print(f"{'='*80}")
        # Show only combos where overall ROI > 100%
        stable_keys = []
        for key, yearly in ys.items():
            rois = [v['roi'] for v in yearly.values()]
            total_n = sum(v['n'] for v in yearly.values())
            if total_n >= 30 and np.mean(rois) > 100:
                stable_keys.append((key, yearly, np.mean(rois)))
        stable_keys.sort(key=lambda x: -x[2])

        if stable_keys:
            all_years = sorted(set(
                y for _, yearly, _ in stable_keys for y in yearly.keys()))
            hdr = f"{'Combo':>25}" + "".join(f"{y:>10}" for y in all_years) + f"{'  Avg':>8}"
            print(hdr)
            print("-" * (25 + 10 * len(all_years) + 8))
            for key, yearly, avg_roi in stable_keys[:20]:
                row = f"{key:>25}"
                for y in all_years:
                    if y in yearly:
                        r = yearly[y]['roi']
                        row += f"{r:>8.0f}% "
                    else:
                        row += f"{'N/A':>10}"
                row += f"{avg_roi:>7.0f}%"
                print(row)

    # ---- Venue ranking ----
    print(f"\n{'='*80}")
    print(f"  Venue Trio ROI Ranking (all conditions)")
    print(f"{'='*80}")
    venue_stats = defaultdict(lambda: {'n': 0, 'hits': 0, 'investment': 0, 'return': 0})
    for p in patterns:
        if p['bet_type'] == 'trio' and p['venue'] != 'all' and p['condition'] == 'all':
            v = p['venue']
            venue_stats[v]['n'] += p['n']
            venue_stats[v]['hits'] += p['hits']
            venue_stats[v]['investment'] += p['investment']
            venue_stats[v]['return'] += p['total_return']

    # If no venue-level "all condition" patterns, aggregate from trio_odds_venue
    if not venue_stats:
        # Manually aggregate from enriched data via patterns
        pass

    if venue_stats:
        print(f"{'Venue':>6} {'N':>6} {'Hits':>5} {'Rate%':>7} {'ROI%':>8}")
        print("-" * 40)
        for venue, s in sorted(venue_stats.items(),
                                key=lambda x: -x[1]['return'] / max(x[1]['investment'], 1)):
            roi = s['return'] / s['investment'] * 100 if s['investment'] > 0 else 0
            rate = s['hits'] / s['n'] * 100 if s['n'] > 0 else 0
            marker = ' <-- HIGH' if roi > 200 else ''
            print(f"{venue:>6} {s['n']:>6} {s['hits']:>5} {rate:>6.1f}% "
                  f"{roi:>7.1f}%{marker}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='TASK 5: Odds Band x Venue Filter Refinement')
    parser.add_argument('--min-n', type=int, default=30,
                        help='Minimum sample size for a pattern (default: 30)')
    parser.add_argument('--years', type=str, default=None,
                        help='Year range, e.g. "2023-2025" (default: all)')
    parser.add_argument('--rebuild', action='store_true',
                        help='Rebuild WF backtest cache')
    args = parser.parse_args()

    print("=" * 80)
    print("  TASK 5: Odds Band x Venue Filter Refinement")
    print(f"  Keiba AI - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # Parse year range
    year_range = None
    if args.years:
        parts = args.years.split('-')
        if len(parts) == 2:
            year_range = (int(parts[0]), int(parts[1]))
            print(f"  Year filter: {year_range[0]}-{year_range[1]}")
        elif len(parts) == 1:
            yr = int(parts[0])
            year_range = (yr, yr)
            print(f"  Year filter: {yr}")

    # Load or rebuild cache
    races = None
    if args.rebuild:
        races = rebuild_cache()
    if races is None:
        races = load_cache(year_range)
    if races is None:
        print("\nNo cache found. Attempting to rebuild...")
        races = rebuild_cache()
        if races is None:
            print("ERROR: Cannot load or build race results cache.")
            print("Run `python tools/profitable_patterns.py --rebuild` first.")
            sys.exit(1)

    # Apply year filter after loading (if not already done in load_cache)
    if year_range and races:
        lo, hi = year_range
        races = [r for r in races if lo <= r.get('year', 0) <= hi]
        print(f"  After year filter: {len(races)} races")

    # Enrich with per-rank odds data
    print("\nEnriching race data with per-rank odds ...")
    t0 = time.time()
    enriched = enrich_races(races)
    print(f"  Enriched: {len(enriched)} races with payout data "
          f"({time.time()-t0:.1f}s)")

    if not enriched:
        print("ERROR: No enriched data. Check payout coverage.")
        sys.exit(1)

    # Run analysis
    print("\nRunning cross-dimensional analysis ...")
    t0 = time.time()
    result = analyze_odds_venue_patterns(enriched, min_n=args.min_n)
    print(f"  Analysis complete ({time.time()-t0:.1f}s)")

    if result is None:
        print("ERROR: Analysis failed.")
        sys.exit(1)

    # Save JSON
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {OUTPUT_PATH}")

    # Display
    print_summary(result)

    # Final stats
    patterns = result.get('patterns', [])
    print(f"\n{'='*80}")
    print(f"  DONE: {result['profitable_patterns']} profitable patterns found "
          f"out of {result['total_patterns_tested']} tested")
    if patterns:
        best = patterns[0]
        print(f"  Best: {best['bet_type']} {best['odds_band']} "
              f"{best['venue']} {best['condition']} "
              f"ROI={best['roi']}% N={best['n']} "
              f"{best['recommendation_str']}")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
