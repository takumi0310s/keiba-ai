"""N5: Monte Carlo drawdown re-evaluation (read-only).

10,000 iter, 4 weeks horizon (28 days x 6 R = 168 R), bet=¥700.
Bootstrap from R-level ROI samples per case.
"""

import json
import pandas as pd
import numpy as np
from multiprocessing import Pool


CSV = 'data/cumulative_results.csv'
N_ITER = 10000
N_RACES = 168  # 28 day x 6 R/day
BET = 700
SEED_BASE = 42


def load_cases():
    cum = pd.read_csv(CSV)
    cum = cum[cum['status'] == 'settled'].copy()
    cum['inv'] = pd.to_numeric(cum['investment'], errors='coerce').fillna(0)
    cum['pay'] = pd.to_numeric(cum['actual_payout'], errors='coerce').fillna(0)
    cum_dd = cum.drop_duplicates(subset=['race_id', 'bet_type'], keep='first').copy()
    cum_dd['pnl'] = cum_dd['pay'] - cum_dd['inv']
    cum_dd['course_code'] = cum_dd['race_id'].astype(str).str[4:6]
    cum_dd = cum_dd[cum_dd['inv'] > 0].copy()
    cum_dd['r_roi'] = cum_dd['pay'] / cum_dd['inv']

    def s7c(df):
        return df[(df['course_code'] != '08') & (df['condition'] != 'X')]

    cases = {
        'A_all':    cum_dd,
        'B_s7c':    s7c(cum_dd),
        'C_pre516': cum_dd[cum_dd['date'].astype(str).str.startswith('20260516') == False],
        'D_pre516_s7c': s7c(cum_dd[cum_dd['date'].astype(str).str.startswith('20260516') == False]),
    }
    return cases


def simulate_one(args):
    seed, r_roi_samples, n_races, bet = args
    rng = np.random.default_rng(seed)
    chosen = rng.choice(r_roi_samples, size=n_races, replace=True)
    payouts = chosen * bet
    invs = np.full(n_races, bet, dtype=float)
    pnl_per = payouts - invs
    cum_pnl = np.cumsum(pnl_per)
    run_max = np.maximum.accumulate(cum_pnl)
    drawdown = run_max - cum_pnl
    max_dd = drawdown.max()
    min_pnl = cum_pnl.min()
    final_pnl = cum_pnl[-1]
    # touch flags (lowest cum PnL <= threshold)
    touch_50k = bool(min_pnl <= -50000)
    touch_100k = bool(min_pnl <= -100000)
    touch_200k = bool(min_pnl <= -200000)
    # ROI achieved at end
    total_inv = bet * n_races
    end_roi = (final_pnl + total_inv) / total_inv * 100
    # daily ROI for Sharpe (group by 6)
    daily_pnl = pnl_per.reshape(-1, 6).sum(axis=1)
    daily_mean = daily_pnl.mean()
    daily_std = daily_pnl.std(ddof=1) if len(daily_pnl) > 1 else 0.0
    neg = daily_pnl[daily_pnl < 0]
    neg_std = neg.std(ddof=1) if len(neg) > 1 else 0.0
    return {
        'final_pnl': final_pnl,
        'max_dd': max_dd,
        'min_pnl': min_pnl,
        'touch_50k': touch_50k,
        'touch_100k': touch_100k,
        'touch_200k': touch_200k,
        'end_roi': end_roi,
        'daily_mean': daily_mean,
        'daily_std': daily_std,
        'neg_std': neg_std,
    }


def run_case(name, df, n_iter=N_ITER, n_races=N_RACES, bet=BET):
    r_roi = df['r_roi'].values.astype(float)
    args_list = [(SEED_BASE + i, r_roi, n_races, bet) for i in range(n_iter)]
    with Pool(8) as p:
        results = p.map(simulate_one, args_list)
    return results


def summarize(results):
    fp = np.array([r['final_pnl'] for r in results])
    dd = np.array([r['max_dd'] for r in results])
    mp = np.array([r['min_pnl'] for r in results])
    er = np.array([r['end_roi'] for r in results])
    dm = np.array([r['daily_mean'] for r in results])
    ds = np.array([r['daily_std'] for r in results])
    ns = np.array([r['neg_std'] for r in results])
    t50 = sum(r['touch_50k'] for r in results) / len(results)
    t100 = sum(r['touch_100k'] for r in results) / len(results)
    t200 = sum(r['touch_200k'] for r in results) / len(results)
    # Sharpe / Sortino / Calmar (per-day, annualized x sqrt(252/28)?? -> raw 4-week numbers)
    sharpe_pi = (dm / np.where(ds > 0, ds, 1e-9))
    sortino_pi = (dm / np.where(ns > 0, ns, 1e-9))
    calmar_pi = (fp / np.where(dd > 0, dd, 1e-9))
    return {
        'final_pnl_p5': float(np.percentile(fp, 5)),
        'final_pnl_p50': float(np.percentile(fp, 50)),
        'final_pnl_p95': float(np.percentile(fp, 95)),
        'final_pnl_mean': float(fp.mean()),
        'max_dd_p50': float(np.percentile(dd, 50)),
        'max_dd_p95': float(np.percentile(dd, 95)),
        'max_dd_mean': float(dd.mean()),
        'min_pnl_p5': float(np.percentile(mp, 5)),
        'min_pnl_p50': float(np.percentile(mp, 50)),
        'end_roi_p5': float(np.percentile(er, 5)),
        'end_roi_p50': float(np.percentile(er, 50)),
        'end_roi_p95': float(np.percentile(er, 95)),
        'p_touch_50k': float(t50),
        'p_touch_100k': float(t100),
        'p_touch_200k': float(t200),
        'p_final_negative': float((fp < 0).sum() / len(fp)),
        'p_final_below_neg_25k': float((fp < -25000).sum() / len(fp)),
        'p_final_below_neg_50k': float((fp < -50000).sum() / len(fp)),
        'sharpe_p50': float(np.percentile(sharpe_pi, 50)),
        'sortino_p50': float(np.percentile(sortino_pi, 50)),
        'calmar_p50': float(np.percentile(calmar_pi, 50)),
    }


def main():
    cases = load_cases()
    summary = {}
    for name, df in cases.items():
        n = len(df)
        r_roi = df['r_roi'].values
        emp_roi = df['pay'].sum() / df['inv'].sum() * 100
        emp_pnl = (df['pay'] - df['inv']).sum()
        emp_mean_rroi = r_roi.mean() * 100
        emp_std_rroi = r_roi.std(ddof=1) * 100
        emp_hit_rate = (r_roi > 0).mean()
        print(f'=== {name}: n={n}, emp_ROI={emp_roi:.2f}%, emp_PnL={emp_pnl:.0f}, R-mean={emp_mean_rroi:.2f}%, R-std={emp_std_rroi:.2f}%, hit={emp_hit_rate:.3f}')
        results = run_case(name, df)
        s = summarize(results)
        s['n_sample'] = n
        s['emp_roi'] = emp_roi
        s['emp_pnl'] = float(emp_pnl)
        s['emp_r_roi_mean'] = emp_mean_rroi
        s['emp_r_roi_std'] = emp_std_rroi
        s['emp_hit_rate'] = float(emp_hit_rate)
        summary[name] = s
        print(json.dumps(s, indent=2, ensure_ascii=True))

    # Case E: D + P0-5 (+3pt assumption)
    df = cases['D_pre516_s7c'].copy()
    df['r_roi_boost'] = df['r_roi'] * 1.03
    df_boost = df.copy()
    df_boost['r_roi'] = df_boost['r_roi_boost']
    r_roi = df_boost['r_roi'].values.astype(float)
    args_list = [(SEED_BASE + i, r_roi, N_RACES, BET) for i in range(N_ITER)]
    with Pool(8) as p:
        results = p.map(simulate_one, args_list)
    s = summarize(results)
    s['n_sample'] = len(df)
    s['emp_roi_boosted'] = float(df_boost['r_roi'].mean() * 100)
    s['note'] = 'D + P0-5 (+3pt assumption - paper eval pending)'
    summary['E_D_plus_P05'] = s
    print('=== E_D_plus_P05')
    print(json.dumps(s, indent=2, ensure_ascii=False))

    with open('data/n5_mc_drawdown_results.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print('Wrote data/n5_mc_drawdown_results.json')


if __name__ == '__main__':
    main()
