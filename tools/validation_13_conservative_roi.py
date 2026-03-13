#!/usr/bin/env python
"""Phase 13: 実運用保守ROI
バックテストROIから実運用想定ROIを算出。
補正: オッズ差(-5~10%), モデル劣化(-10%), 条件過学習(-10%)
保守ROI = Backtest ROI x 0.7
"""
import json
import os
import time
from datetime import datetime

BASE_DIR = os.path.join(os.path.dirname(__file__), '..')


def main():
    print("=" * 60)
    print("  Phase 13: Conservative ROI Estimate")
    print("=" * 60)

    # Load actual ROI results
    roi_path = os.path.join(BASE_DIR, 'data', 'actual_roi_results.json')
    with open(roi_path, 'r', encoding='utf-8') as f:
        roi_data = json.load(f)

    conditions = roi_data.get('conditions', {})

    # Correction factors
    corrections = {
        'odds_gap': {
            'factor': 0.925,  # -7.5% (midpoint of -5 to -10%)
            'description': 'Purchase-time odds vs confirmed odds gap',
            'range': '[-5%, -10%]',
        },
        'model_degradation': {
            'factor': 0.90,
            'description': 'Model accuracy degradation over time (data drift)',
            'range': '-10%',
        },
        'condition_overfitting': {
            'factor': 0.90,
            'description': 'Condition classification may be overfit to backtest period',
            'range': '-10%',
        },
    }

    total_correction = 1.0
    for name, corr in corrections.items():
        total_correction *= corr['factor']

    # Round to match the 0.7 target
    # 0.925 * 0.90 * 0.90 = 0.749 ≈ 0.7 (conservative)
    conservative_factor = 0.70  # Use flat 0.7 as specified

    print(f"\n  Correction factors:")
    for name, corr in corrections.items():
        print(f"    {name}: x{corr['factor']} ({corr['description']})")
    print(f"  Combined: x{total_correction:.3f} (using conservative x{conservative_factor})")

    # Calculate conservative ROI per condition
    print(f"\n  {'Cond':<6} {'Backtest ROI':>13} {'Conservative ROI':>16} {'Verdict':>10}")
    print(f"  {'-' * 50}")

    condition_results = {}
    total_bt_weighted_roi = 0
    total_n = 0

    for cond_key in ['A', 'B', 'C', 'D', 'E', 'X']:
        if cond_key not in conditions:
            continue

        cond = conditions[cond_key]
        n = cond.get('n', 0)
        bt = cond.get('best_bet', 'trio')
        bt_data = cond.get('actual_roi', {}).get(bt, {})
        backtest_roi = bt_data.get('roi', 0)
        conservative_roi = round(backtest_roi * conservative_factor, 1)
        hit_rate = bt_data.get('hit_rate', 0)

        verdict = 'PROFIT' if conservative_roi > 100 else 'BREAK-EVEN' if conservative_roi > 90 else 'LOSS'

        condition_results[cond_key] = {
            'n_races': n,
            'bet_type': bt,
            'backtest_roi': backtest_roi,
            'conservative_roi': conservative_roi,
            'hit_rate': hit_rate,
            'verdict': verdict,
        }

        total_bt_weighted_roi += backtest_roi * n
        total_n += n

        print(f"  {cond_key:<6} {backtest_roi:>12.1f}% {conservative_roi:>15.1f}% {verdict:>10}")

    # Overall weighted conservative ROI
    overall_bt_roi = total_bt_weighted_roi / total_n if total_n > 0 else 0
    overall_conservative = round(overall_bt_roi * conservative_factor, 1)

    print(f"\n  Overall (weighted): Backtest {overall_bt_roi:.1f}% -> Conservative {overall_conservative:.1f}%")

    # Recommended bet sizing
    # Kelly criterion simplified: f* = (p * b - q) / b
    # where p = hit_rate, b = avg_payout/investment - 1, q = 1 - p
    print(f"\n  Recommended Bet Sizing:")
    print(f"  {'Cond':<6} {'BetType':>8} {'Rec/Race':>10} {'Monthly Est':>12}")

    monthly_races = {
        'A': 30,  # ~360/year / 12
        'B': 5,   # ~60/year / 12
        'C': 22,  # ~264/year / 12
        'D': 40,  # ~480/year / 12
        'E': 2,   # ~28/year / 12
        'X': 4,   # ~48/year / 12
    }

    bet_sizing = {}
    total_monthly_invest = 0
    total_monthly_profit = 0

    for cond_key, data in condition_results.items():
        conservative_roi = data['conservative_roi']
        # Fixed bet: 700 yen per race (current strategy)
        rec_per_race = 700
        monthly_n = monthly_races.get(cond_key, 10)
        monthly_invest = rec_per_race * monthly_n
        monthly_profit = int(monthly_invest * (conservative_roi / 100 - 1))

        bet_sizing[cond_key] = {
            'bet_per_race': rec_per_race,
            'monthly_races': monthly_n,
            'monthly_investment': monthly_invest,
            'monthly_expected_profit': monthly_profit,
        }

        total_monthly_invest += monthly_invest
        total_monthly_profit += monthly_profit

        print(f"  {cond_key:<6} {data['bet_type']:>8} {rec_per_race:>9}Y {monthly_profit:>+11,}Y")

    print(f"\n  Monthly total: invest {total_monthly_invest:,}Y, expected profit {total_monthly_profit:>+,}Y")
    print(f"  Monthly ROI: {(total_monthly_invest + total_monthly_profit) / total_monthly_invest * 100:.1f}%")

    # Risk-adjusted recommendation
    if overall_conservative > 120:
        risk_level = 'LOW'
        recommendation = 'Strong edge. Proceed with fixed bet sizing.'
    elif overall_conservative > 100:
        risk_level = 'MEDIUM'
        recommendation = 'Positive edge but thin margin. Monitor closely.'
    else:
        risk_level = 'HIGH'
        recommendation = 'Conservative estimate below break-even. Consider reducing bet size or pausing.'

    # Save
    output = {
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'correction_factors': corrections,
        'conservative_factor': conservative_factor,
        'computed_factor': round(total_correction, 3),
        'condition_results': condition_results,
        'overall': {
            'backtest_roi': round(overall_bt_roi, 1),
            'conservative_roi': overall_conservative,
            'risk_level': risk_level,
            'recommendation': recommendation,
        },
        'bet_sizing': bet_sizing,
        'monthly_summary': {
            'total_investment': total_monthly_invest,
            'expected_profit': total_monthly_profit,
            'monthly_roi': round((total_monthly_invest + total_monthly_profit) / max(total_monthly_invest, 1) * 100, 1),
            'annual_expected_profit': total_monthly_profit * 12,
        },
    }

    out_path = os.path.join(BASE_DIR, 'data', 'conservative_roi_estimate.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n  Saved: {out_path}")

    return output


if __name__ == '__main__':
    main()
