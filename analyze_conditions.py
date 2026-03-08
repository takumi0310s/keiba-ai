#!/usr/bin/env python
"""V9.1 5年バックテスト結果を条件別(A-E/X)に再集計するスクリプト"""
import json
import sys

RESULTS_PATH = 'backtest_results_5year.json'


def classify_condition(race):
    """レースを条件A-E/Xに分類"""
    n = race.get('num_horses', 14)
    dist = race.get('distance', 1600)
    cond = str(race.get('condition', '良'))
    heavy = any(c in cond for c in ['重', '不'])
    if n <= 7:
        return 'E'
    if dist <= 1400:
        return 'D'
    if 8 <= n <= 14 and dist >= 1600 and not heavy:
        return 'A'
    if 8 <= n <= 14 and dist >= 1600 and heavy:
        return 'B'
    if n >= 15 and dist >= 1600 and not heavy:
        return 'C'
    return 'X'  # 15頭+/重~不良


COND_MAP = {
    'A': {'label': '8-14頭/1600m+/良~稍', 'bet': 'trio', 'n_bets': 7, 'cost': 700},
    'B': {'label': '8-14頭/1600m+/重~不良', 'bet': 'wide', 'n_bets': 2, 'cost': 700},
    'C': {'label': '15頭+/1600m+/良~稍', 'bet': 'wide', 'n_bets': 2, 'cost': 700},
    'D': {'label': '1400m以下', 'bet': 'skip', 'n_bets': 0, 'cost': 0},
    'E': {'label': '7頭以下', 'bet': 'umaren', 'n_bets': 2, 'cost': 700},
    'X': {'label': '15頭+/重~不良', 'bet': 'wide', 'n_bets': 2, 'cost': 700},
}


def calc_condition_roi(results, ver='v9'):
    """条件別にROIを計算"""
    grouped = {}
    for r in results:
        ckey = classify_condition(r)
        grouped.setdefault(ckey, []).append(r)

    print(f"\n{'='*100}")
    print(f"  V9.1 CENTRAL 5-YEAR CONDITION-BASED ANALYSIS")
    print(f"{'='*100}")

    # All bet types for each condition
    bet_types = ['trio', 'wide', 'umaren']

    # First, show full matrix of all conditions x all bet types
    print(f"\n  === FULL MATRIX: ALL CONDITIONS x ALL BET TYPES ===")
    print(f"  {'COND':<6} {'DESC':<25} {'N':>4}  | {'TRIO HIT':>10} {'TRIO ROI':>10} | {'WIDE HIT':>10} {'WIDE ROI':>10} | {'UMA HIT':>10} {'UMA ROI':>10}")
    print(f"  {'-'*120}")

    best_bets = {}
    for ckey in ['A', 'B', 'C', 'D', 'E', 'X']:
        races = grouped.get(ckey, [])
        info = COND_MAP[ckey]
        n = len(races)
        if n == 0:
            continue

        results_by_bet = {}
        for bt in bet_types:
            if bt == 'trio':
                hits = sum(1 for r in races if r.get(f'{ver}_trio_hit', False))
                payouts = sum(r.get('trio_payout', 0) for r in races if r.get(f'{ver}_trio_hit'))
                investment = n * 7 * 100  # 7 bets x 100
            elif bt == 'wide':
                hits = sum(1 for r in races if len(r.get(f'{ver}_wide_hits', [])) > 0)
                payouts = sum(r.get(f'{ver}_wide_payout', 0) for r in races)
                investment = n * 2 * 100  # 2 bets x 100
            elif bt == 'umaren':
                hits = sum(1 for r in races if len(r.get(f'{ver}_umaren_hits', [])) > 0)
                payouts = 0
                for r in races:
                    if len(r.get(f'{ver}_umaren_hits', [])) > 0:
                        race_payouts = r.get('payouts', {})
                        uma_data = race_payouts.get('umaren', [])
                        hit_pairs = r.get(f'{ver}_umaren_hits', [])
                        for hp in hit_pairs:
                            hp_set = set(hp)
                            for pair_payout in uma_data:
                                if set(pair_payout[0]) == hp_set:
                                    payouts += pair_payout[1]
                                    break
                investment = n * 2 * 100  # 2 bets x 100

            hit_rate = hits / n * 100 if n > 0 else 0
            roi = payouts / investment * 100 if investment > 0 else 0
            results_by_bet[bt] = {
                'hits': hits, 'hit_rate': hit_rate,
                'investment': investment, 'payouts': payouts, 'roi': roi
            }

        t = results_by_bet['trio']
        w = results_by_bet['wide']
        u = results_by_bet['umaren']
        print(f"  {ckey:<6} {info['label']:<25} {n:>4}  | {t['hit_rate']:>8.1f}% {t['roi']:>8.1f}%  | {w['hit_rate']:>8.1f}% {w['roi']:>8.1f}%  | {u['hit_rate']:>8.1f}% {u['roi']:>8.1f}%")

        # Find best bet type
        best_bt = max(bet_types, key=lambda bt: results_by_bet[bt]['roi'])
        best_bets[ckey] = {
            'best_bet': best_bt,
            'roi': results_by_bet[best_bt]['roi'],
            'hit_rate': results_by_bet[best_bt]['hit_rate'],
            'all': results_by_bet,
            'n': n,
        }

    # Show recommended configuration
    print(f"\n\n  === RECOMMENDED BET CONFIGURATION ===")
    print(f"  {'COND':<6} {'DESC':<25} {'N':>4} {'BEST BET':<10} {'HIT%':>8} {'ROI%':>8} {'REC':>6}")
    print(f"  {'-'*75}")
    for ckey in ['A', 'B', 'C', 'D', 'E', 'X']:
        if ckey not in best_bets:
            continue
        b = best_bets[ckey]
        info = COND_MAP[ckey]
        rec = 'YES' if b['roi'] >= 80 else 'NO'
        print(f"  {ckey:<6} {info['label']:<25} {b['n']:>4} {b['best_bet']:<10} {b['hit_rate']:>7.1f}% {b['roi']:>7.1f}%  {rec}")

    # Also show current bet assignment vs optimal
    print(f"\n\n  === CURRENT vs OPTIMAL ===")
    print(f"  {'COND':<6} {'CURRENT BET':<15} {'CURRENT ROI':>12} {'OPTIMAL BET':<15} {'OPTIMAL ROI':>12}")
    print(f"  {'-'*65}")
    for ckey in ['A', 'B', 'C', 'D', 'E', 'X']:
        if ckey not in best_bets:
            continue
        b = best_bets[ckey]
        info = COND_MAP[ckey]
        curr_bt = info['bet']
        if curr_bt == 'skip':
            curr_roi = 'N/A'
        else:
            curr_roi = f"{b['all'][curr_bt]['roi']:.1f}%"
        opt_bt = b['best_bet']
        opt_roi = f"{b['roi']:.1f}%"
        print(f"  {ckey:<6} {curr_bt:<15} {curr_roi:>12} {opt_bt:<15} {opt_roi:>12}")

    # Yearly breakdown for each condition
    print(f"\n\n  === YEARLY BREAKDOWN (V9.1 - RECOMMENDED BET) ===")
    for ckey in ['A', 'B', 'C', 'E', 'X']:
        races = grouped.get(ckey, [])
        if not races:
            continue
        info = COND_MAP[ckey]
        # Use current bet type for yearly analysis
        bt = info['bet']
        if bt == 'skip':
            continue

        yearly = {}
        for r in races:
            yr = r.get('test_year', 0)
            yearly.setdefault(yr, []).append(r)

        print(f"\n  Condition {ckey} ({info['label']}) - {bt}")
        print(f"  {'YEAR':<8} {'N':>4} {'HIT':>5} {'RATE':>8} {'INVEST':>10} {'PAYOUT':>10} {'ROI':>8}")
        print(f"  {'-'*55}")
        for yr in sorted(yearly.keys()):
            yr_races = yearly[yr]
            n = len(yr_races)
            if bt == 'trio':
                hits = sum(1 for r in yr_races if r.get(f'{ver}_trio_hit', False))
                payouts = sum(r.get('trio_payout', 0) for r in yr_races if r.get(f'{ver}_trio_hit'))
                investment = n * 7 * 100
            elif bt == 'wide':
                hits = sum(1 for r in yr_races if len(r.get(f'{ver}_wide_hits', [])) > 0)
                payouts = sum(r.get(f'{ver}_wide_payout', 0) for r in yr_races)
                investment = n * 2 * 100
            elif bt == 'umaren':
                hits = sum(1 for r in yr_races if len(r.get(f'{ver}_umaren_hits', [])) > 0)
                payouts = 0
                for r in yr_races:
                    if len(r.get(f'{ver}_umaren_hits', [])) > 0:
                        race_payouts = r.get('payouts', {})
                        uma_data = race_payouts.get('umaren', [])
                        hit_pairs = r.get(f'{ver}_umaren_hits', [])
                        for hp in hit_pairs:
                            hp_set = set(hp)
                            for pair_payout in uma_data:
                                if set(pair_payout[0]) == hp_set:
                                    payouts += pair_payout[1]
                                    break
                investment = n * 2 * 100
            else:
                continue

            hit_rate = hits / n * 100 if n > 0 else 0
            roi = payouts / investment * 100 if investment > 0 else 0
            print(f"  {yr:<8} {n:>4} {hits:>5} {hit_rate:>7.1f}% {investment:>9,} {payouts:>9,} {roi:>7.1f}%")

    # Also try D condition with different bet types
    d_races = grouped.get('D', [])
    if d_races:
        print(f"\n\n  === CONDITION D DEEP DIVE (Currently skipped) ===")
        for bt in ['trio', 'wide', 'umaren']:
            n = len(d_races)
            if bt == 'trio':
                hits = sum(1 for r in d_races if r.get(f'{ver}_trio_hit', False))
                payouts = sum(r.get('trio_payout', 0) for r in d_races if r.get(f'{ver}_trio_hit'))
                investment = n * 7 * 100
            elif bt == 'wide':
                hits = sum(1 for r in d_races if len(r.get(f'{ver}_wide_hits', [])) > 0)
                payouts = sum(r.get(f'{ver}_wide_payout', 0) for r in d_races)
                investment = n * 2 * 100
            elif bt == 'umaren':
                hits = sum(1 for r in d_races if len(r.get(f'{ver}_umaren_hits', [])) > 0)
                payouts = 0
                for r in d_races:
                    if len(r.get(f'{ver}_umaren_hits', [])) > 0:
                        race_payouts = r.get('payouts', {})
                        uma_data = race_payouts.get('umaren', [])
                        hit_pairs = r.get(f'{ver}_umaren_hits', [])
                        for hp in hit_pairs:
                            hp_set = set(hp)
                            for pair_payout in uma_data:
                                if set(pair_payout[0]) == hp_set:
                                    payouts += pair_payout[1]
                                    break
                investment = n * 2 * 100

            hit_rate = hits / n * 100 if n > 0 else 0
            roi = payouts / investment * 100 if investment > 0 else 0
            print(f"  {bt:<10} N={n}, Hit={hits} ({hit_rate:.1f}%), ROI={roi:.1f}%")

    return best_bets


def main():
    with open(RESULTS_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    results = data.get('central_results_5year', [])
    print(f"Loaded {len(results)} race results")

    best_bets = calc_condition_roi(results, ver='v9')

    # Summary
    print(f"\n\n{'='*60}")
    print(f"  FINAL RECOMMENDATION SUMMARY")
    print(f"{'='*60}")
    total_invest = 0
    total_payout = 0
    total_races = 0
    for ckey in ['A', 'B', 'C', 'E', 'X']:
        if ckey not in best_bets:
            continue
        b = best_bets[ckey]
        info = COND_MAP[ckey]
        bt = info['bet']
        if bt == 'skip':
            continue
        stats = b['all'][bt]
        total_invest += stats['investment']
        total_payout += stats['payouts']
        total_races += b['n']
        print(f"  {ckey}: {info['label']} → {bt} (ROI {stats['roi']:.1f}%, Hit {stats['hit_rate']:.1f}%)")

    if total_invest > 0:
        overall_roi = total_payout / total_invest * 100
        print(f"\n  Filtered Overall: {total_races} races, ROI {overall_roi:.1f}%")
        print(f"  Total Investment: {total_invest:,}en -> Payout: {total_payout:,}en")
        print(f"  Profit: {total_payout - total_invest:,}en")


if __name__ == '__main__':
    main()
