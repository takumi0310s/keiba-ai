#!/usr/bin/env python
"""Phase 11: 購入レース数検証（統計信頼性）
6年間バックテストでの購入レース数・的中数・ROI信頼区間を算出。
"""
import numpy as np
import json
import os
import time
from datetime import datetime

BASE_DIR = os.path.join(os.path.dirname(__file__), '..')


def bootstrap_roi_ci(hits, payouts, n_races, investment_per_race=700,
                     n_bootstrap=10000, ci=0.95, seed=42):
    """Bootstrap法でROIの95%信頼区間を計算"""
    rng = np.random.default_rng(seed)

    # 各レースのリターン（的中時: payout, 不的中時: 0）
    returns = np.array(payouts)  # length = n_races
    investment = investment_per_race

    roi_samples = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        # リサンプリング
        idx = rng.choice(n_races, size=n_races, replace=True)
        sample_returns = returns[idx]
        total_return = sample_returns.sum()
        total_invest = n_races * investment
        roi_samples[i] = total_return / total_invest * 100

    alpha = (1 - ci) / 2
    ci_lower = np.percentile(roi_samples, alpha * 100)
    ci_upper = np.percentile(roi_samples, (1 - alpha) * 100)
    mean_roi = np.mean(roi_samples)

    return {
        'mean_roi': round(mean_roi, 1),
        'ci_lower': round(ci_lower, 1),
        'ci_upper': round(ci_upper, 1),
        'std': round(np.std(roi_samples), 1),
    }


def binomial_ci(hits, n, ci=0.95):
    """二項分布の信頼区間（Wilson score interval）"""
    from scipy import stats
    z = stats.norm.ppf(1 - (1 - ci) / 2)
    p_hat = hits / n
    denominator = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denominator
    spread = z * np.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n)) / n) / denominator
    return {
        'hit_rate': round(p_hat * 100, 2),
        'ci_lower': round(max(0, (center - spread)) * 100, 2),
        'ci_upper': round(min(1, (center + spread)) * 100, 2),
    }


def main():
    print("=" * 60)
    print("  Phase 11: Sample Size Validation")
    print("=" * 60)

    # Load actual ROI results
    roi_path = os.path.join(BASE_DIR, 'data', 'actual_roi_results.json')
    with open(roi_path, 'r', encoding='utf-8') as f:
        roi_data = json.load(f)

    # Load yearly performance
    yearly_path = os.path.join(BASE_DIR, 'data', 'yearly_performance.json')
    with open(yearly_path, 'r', encoding='utf-8') as f:
        yearly_data = json.load(f)

    total_races_in_data = roi_data.get('total_races', 0)
    matched_races = roi_data.get('matched_races', 0)

    conditions = roi_data.get('conditions', {})

    print(f"\n  Total test races: {total_races_in_data}")
    print(f"  Matched with payouts: {matched_races}")

    results = {
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_test_races': total_races_in_data,
        'matched_races': matched_races,
        'conditions': {},
        'overall': {},
    }

    total_purchased = 0
    total_trio_hits = 0
    total_trio_payout = 0
    total_investment = 0

    print(f"\n  {'Cond':<6} {'N':>6} {'Hits':>6} {'HitRate':>8} {'ROI':>8} {'95%CI':>20} {'Reliability':>12}")
    print(f"  {'-' * 75}")

    for cond_key in ['A', 'B', 'C', 'D', 'E', 'X']:
        if cond_key not in conditions:
            continue
        cond = conditions[cond_key]
        n = cond.get('n', 0)
        bt = cond.get('best_bet', 'trio')
        bt_data = cond.get('actual_roi', {}).get(bt, {})
        hits = bt_data.get('hits', 0)
        roi = bt_data.get('roi', 0)

        total_purchased += n
        if bt == 'trio':
            total_trio_hits += hits
            # Estimate payout from ROI and investment
            inv = n * 700
            total_trio_payout += int(roi / 100 * inv)
            total_investment += inv
        else:
            inv = n * 700
            total_investment += inv
            total_trio_payout += int(roi / 100 * inv)

        # Hit rate CI (binomial)
        hit_ci = binomial_ci(hits, n) if n > 0 else {'hit_rate': 0, 'ci_lower': 0, 'ci_upper': 0}

        # ROI CI (bootstrap approximation using normal approx)
        # For bootstrap we'd need per-race payouts; use normal approximation instead
        # ROI std ≈ ROI * CV, where CV ≈ 1/sqrt(hits) for payout variance
        roi_std = roi * (1 / np.sqrt(max(hits, 1))) if hits > 0 else 0
        roi_ci_lower = max(0, roi - 1.96 * roi_std)
        roi_ci_upper = roi + 1.96 * roi_std

        # Reliability
        if n >= 5000:
            reliability = 'HIGH'
        elif n >= 1000:
            reliability = 'MEDIUM'
        else:
            reliability = 'LOW'

        cond_result = {
            'n_races': n,
            'hits': hits,
            'hit_rate': hit_ci['hit_rate'],
            'hit_rate_ci': [hit_ci['ci_lower'], hit_ci['ci_upper']],
            'roi': roi,
            'roi_ci_95': [round(roi_ci_lower, 1), round(roi_ci_upper, 1)],
            'roi_std': round(roi_std, 1),
            'reliability': reliability,
            'bet_type': bt,
        }
        results['conditions'][cond_key] = cond_result

        ci_str = f"[{roi_ci_lower:.0f}%, {roi_ci_upper:.0f}%]"
        print(f"  {cond_key:<6} {n:>6} {hits:>6} {hit_ci['hit_rate']:>6.1f}% {roi:>7.1f}% {ci_str:>20} {reliability:>12}")

    # Overall
    purchase_rate = total_purchased / total_races_in_data * 100 if total_races_in_data > 0 else 0
    overall_roi = total_trio_payout / total_investment * 100 if total_investment > 0 else 0

    # Overall reliability
    if total_purchased >= 5000:
        overall_reliability = 'HIGH'
    elif total_purchased >= 1000:
        overall_reliability = 'MEDIUM'
    else:
        overall_reliability = 'LOW'

    results['overall'] = {
        'total_purchased': total_purchased,
        'purchase_rate': round(purchase_rate, 1),
        'total_hits': total_trio_hits,
        'overall_hit_rate': round(total_trio_hits / max(total_purchased, 1) * 100, 1),
        'overall_roi': round(overall_roi, 1),
        'total_investment': total_investment,
        'total_payout': total_trio_payout,
        'reliability': overall_reliability,
    }

    print(f"\n  Overall:")
    print(f"    Total purchased: {total_purchased} races")
    print(f"    Purchase rate: {purchase_rate:.1f}% of all test races")
    print(f"    Overall ROI: {overall_roi:.1f}%")
    print(f"    Reliability: {overall_reliability}")

    # Reliability assessment
    results['assessment'] = {
        'total_sample_verdict': f"N={total_purchased}: {overall_reliability} reliability",
        'weakest_condition': min(results['conditions'].items(),
                                key=lambda x: x[1]['n_races'])[0] if results['conditions'] else 'N/A',
        'strongest_condition': max(results['conditions'].items(),
                                  key=lambda x: x[1]['n_races'])[0] if results['conditions'] else 'N/A',
        'note': 'Conditions B and E have relatively few samples (<1100). Monitor these closely in live trading.',
    }

    out_path = os.path.join(BASE_DIR, 'data', 'sample_size_validation.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n  Saved: {out_path}")

    return results


if __name__ == '__main__':
    main()
