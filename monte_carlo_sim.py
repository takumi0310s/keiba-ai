"""
モンテカルロ破産確率シミュレーション
条件別の的中率・平均配当から資金推移をシミュレートする。

Usage:
    python monte_carlo_sim.py
    python monte_carlo_sim.py --trials 50000
"""
import numpy as np
import json
import os
from datetime import datetime

# バックテスト実績（simulation_results_jra.csv + NAR backtest結果）
# 形式: {cond: (hit_rate, avg_payout_per100, investment_per_race)}
# avg_payout_per100 = 的中時の100円あたり平均配当
JRA_CONDITIONS = {
    'A': {'hit_rate': 0.451, 'avg_return': 25429227 / 3890, 'investment': 700, 'n': 8634, 'label': '8-14頭/1600m+/良~稍'},
    'B': {'hit_rate': 0.454, 'avg_return': 3539139 / 484, 'investment': 700, 'n': 1067, 'label': '8-14頭/1600m+/重~不良'},
    'C': {'hit_rate': 0.334, 'avg_return': 22355894 / 2137, 'investment': 700, 'n': 6405, 'label': '15頭+/1600m+/良~稍'},
    'D': {'hit_rate': 0.282, 'avg_return': 16958143 / 2769, 'investment': 700, 'n': 9807, 'label': '1400m以下'},
    'E': {'hit_rate': 0.757, 'avg_return': 1273853 / 431, 'investment': 700, 'n': 569, 'label': '7頭以下'},
    'X': {'hit_rate': 0.361, 'avg_return': 4418076 / 381, 'investment': 700, 'n': 1055, 'label': '15頭+/重~不良'},
}

# 条件出現頻度（バックテストの実績から）
TOTAL_RACES = sum(c['n'] for c in JRA_CONDITIONS.values())
COND_WEIGHTS = {k: v['n'] / TOTAL_RACES for k, v in JRA_CONDITIONS.items()}


def simulate_once(initial_fund, num_races, conditions, cond_weights, rng):
    """1回のシミュレーション
    Returns: (final_fund, max_drawdown, min_fund, ruin)
    """
    fund = initial_fund
    peak = initial_fund
    max_dd = 0
    min_fund = initial_fund

    cond_keys = list(cond_weights.keys())
    cond_probs = [cond_weights[k] for k in cond_keys]

    for _ in range(num_races):
        # 条件をランダムに選択（実績頻度に基づく）
        cond = rng.choice(cond_keys, p=cond_probs)
        c = conditions[cond]
        investment = c['investment']

        if fund < investment:
            # 破産
            return fund, max_dd, 0, True

        fund -= investment

        # 的中判定
        if rng.random() < c['hit_rate']:
            # 的中: 平均配当の対数正規分布でばらつきを付与
            # 配当は対数正規分布に従うことが知られている
            avg_return = c['avg_return']
            # 標準偏差はROIの条件に応じて調整
            log_mean = np.log(avg_return) - 0.5
            log_std = 0.8  # 配当のばらつき
            payout = rng.lognormal(log_mean, log_std)
            # 最低100円配当、最大上限なし
            payout = max(100, payout)
            fund += payout

        if fund > peak:
            peak = fund
        dd = (peak - fund) / peak if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd
        if fund < min_fund:
            min_fund = fund

    return fund, max_dd, min_fund, fund <= 0


def run_monte_carlo(initial_funds=None, num_races=1000, num_trials=10000,
                    conditions=None, cond_weights=None):
    """モンテカルロシミュレーション実行"""
    if initial_funds is None:
        initial_funds = [10000, 30000, 100000]
    if conditions is None:
        conditions = JRA_CONDITIONS
    if cond_weights is None:
        cond_weights = COND_WEIGHTS

    rng = np.random.default_rng(42)
    results = {}

    # 全条件のROI確認
    print("=== 条件別バックテスト実績 ===")
    total_invest = 0
    total_return = 0
    for cond, c in conditions.items():
        roi = c['avg_return'] * c['hit_rate'] / c['investment'] * 100
        total_invest += c['n'] * c['investment']
        total_return += c['avg_return'] * c['hit_rate'] * c['n']
        print(f"  {cond}: 的中率 {c['hit_rate']*100:.1f}%, 平均配当 {c['avg_return']:.0f}円, ROI {roi:.1f}%, N={c['n']}")
    print(f"  全体ROI: {total_return/total_invest*100:.1f}%")
    print()

    for fund in initial_funds:
        print(f"\n{'='*60}")
        print(f"初期資金: {fund:,}円 × {num_races}レース × {num_trials:,}回")
        print(f"{'='*60}")

        final_funds = np.zeros(num_trials)
        max_drawdowns = np.zeros(num_trials)
        ruin_count = 0

        for trial in range(num_trials):
            final, max_dd, min_f, ruin = simulate_once(
                fund, num_races, conditions, cond_weights, rng
            )
            final_funds[trial] = final
            max_drawdowns[trial] = max_dd
            if ruin:
                ruin_count += 1

        ruin_prob = ruin_count / num_trials * 100
        avg_final = np.mean(final_funds)
        median_final = np.median(final_funds)
        p5 = np.percentile(final_funds, 5)
        p25 = np.percentile(final_funds, 25)
        p75 = np.percentile(final_funds, 75)
        p95 = np.percentile(final_funds, 95)
        avg_dd = np.mean(max_drawdowns) * 100
        max_dd_worst = np.max(max_drawdowns) * 100
        profit_prob = np.mean(final_funds > fund) * 100

        total_investment = fund  # 初期資金全体でのROI
        expected_roi = avg_final / fund * 100

        print(f"\n結果:")
        print(f"  破産確率:      {ruin_prob:.2f}%")
        print(f"  利益確率:      {profit_prob:.1f}%")
        print(f"  期待最終資金:  {avg_final:,.0f}円 (中央値: {median_final:,.0f}円)")
        print(f"  期待収益率:    {expected_roi:.1f}%")
        print(f"  95%信頼区間:   {p5:,.0f}円 〜 {p95:,.0f}円")
        print(f"  四分位範囲:    {p25:,.0f}円 〜 {p75:,.0f}円")
        print(f"  平均最大DD:    {avg_dd:.1f}%")
        print(f"  最悪最大DD:    {max_dd_worst:.1f}%")

        results[str(fund)] = {
            'initial_fund': fund,
            'num_races': num_races,
            'num_trials': num_trials,
            'ruin_probability': round(ruin_prob, 2),
            'profit_probability': round(profit_prob, 1),
            'avg_final_fund': round(avg_final),
            'median_final_fund': round(median_final),
            'expected_roi': round(expected_roi, 1),
            'ci95_lower': round(p5),
            'ci95_upper': round(p95),
            'q25': round(p25),
            'q75': round(p75),
            'avg_max_drawdown': round(avg_dd, 1),
            'worst_max_drawdown': round(max_dd_worst, 1),
        }

    # 結果保存
    output = {
        'generated_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'conditions': {k: {
            'hit_rate': v['hit_rate'],
            'avg_return': round(v['avg_return']),
            'investment': v['investment'],
            'n': v['n'],
            'label': v['label'],
            'roi': round(v['avg_return'] * v['hit_rate'] / v['investment'] * 100, 1),
        } for k, v in conditions.items()},
        'condition_weights': {k: round(v, 4) for k, v in cond_weights.items()},
        'simulations': results,
    }

    os.makedirs("data", exist_ok=True)
    with open("data/monte_carlo_results.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n結果を data/monte_carlo_results.json に保存しました。")
    return output


def run_condition_specific():
    """条件別の個別シミュレーション"""
    rng = np.random.default_rng(42)

    print("\n=== 条件別 単独運用シミュレーション (初期3万円, 500レース) ===\n")
    print(f"{'条件':>4} {'破産率':>8} {'利益率':>8} {'期待ROI':>9} {'期待最終':>12} {'最大DD':>8}")
    print("-" * 60)

    for cond, c in JRA_CONDITIONS.items():
        single_cond = {cond: c}
        single_weight = {cond: 1.0}
        fund = 30000
        num_races = 500
        num_trials = 10000

        finals = np.zeros(num_trials)
        ruins = 0
        max_dds = np.zeros(num_trials)

        for trial in range(num_trials):
            final, max_dd, _, ruin = simulate_once(
                fund, num_races, single_cond, single_weight, rng
            )
            finals[trial] = final
            max_dds[trial] = max_dd
            if ruin:
                ruins += 1

        ruin_p = ruins / num_trials * 100
        profit_p = np.mean(finals > fund) * 100
        avg_final = np.mean(finals)
        exp_roi = avg_final / fund * 100
        avg_dd = np.mean(max_dds) * 100

        print(f"{cond:>4} {ruin_p:>7.1f}% {profit_p:>7.1f}% {exp_roi:>8.1f}% {avg_final:>11,.0f}円 {avg_dd:>7.1f}%")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="モンテカルロ破産確率シミュレーション")
    parser.add_argument("--trials", type=int, default=10000, help="試行回数 (default: 10000)")
    parser.add_argument("--races", type=int, default=1000, help="レース数 (default: 1000)")
    parser.add_argument("--condition-only", action="store_true", help="条件別単独シミュレーション")
    args = parser.parse_args()

    run_monte_carlo(num_races=args.races, num_trials=args.trials)
    if args.condition_only:
        run_condition_specific()
    else:
        run_condition_specific()
