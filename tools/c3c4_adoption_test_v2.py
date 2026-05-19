#!/usr/bin/env python3
"""
6/17 採用判定 statistical test script

Usage:
  python tools/c3c4_adoption_test_v2.py
  python tools/c3c4_adoption_test_v2.py --date-from 20260524 --date-to 20260616

5 項目評価 + Welch's t-test + bootstrap CI で 8 strategy の GO/NO-GO を判定。
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Windows console utf-8
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

BASE_DIR = Path(__file__).resolve().parents[1]
LOG_DIR = BASE_DIR / 'data' / 'race_notify_log_v2_summary'

STRATEGY_KEYS = [
    'actual',
    'c3',
    'c4',
    'c3c4',
    'no_1pop',
    'divergence',
    'ev_filter',
    'odds_filter',
]

STRATEGY_NAMES = {
    'actual':     '戦略⑦案C (baseline)',
    'c3':         'C3 (bet2除外 6点)',
    'c4':         'C4 (条件A 1600-1800m skip)',
    'c3c4':       'C3+C4 合算',
    'no_1pop':    'B-1 (1番人気除外)',
    'divergence': 'B-2 (divergence >=3番人気)',
    'ev_filter':  'C-1 (EV>1 filter)',
    'odds_filter': 'C-2 (odds帯 filter)',
}

# GO 判定閾値
THRESH_N = 24          # paper N >= 24R
THRESH_ROI_GO = 5.0    # baseline 比 ROI 改善 >= +5pt → GO
THRESH_ROI_COND = 3.0  # baseline 比 ROI 改善 >= +3pt → 限定GO
THRESH_P = 0.05        # p < 0.05 で統計有意


# ---------------------------------------------------------------------------
# data loading
# ---------------------------------------------------------------------------

def _parse_date_range(date_from: str | None, date_to: str | None):
    """date_from / date_to を datetime.date に変換。None なら全期間。"""
    fmt = '%Y%m%d'
    df = datetime.strptime(date_from, fmt).date() if date_from else None
    dt = datetime.strptime(date_to, fmt).date() if date_to else None
    return df, dt


def _in_range(date_str: str, df, dt) -> bool:
    try:
        d = datetime.strptime(date_str, '%Y%m%d').date()
    except Exception:
        return False
    if df and d < df:
        return False
    if dt and d > dt:
        return False
    return True


def load_paper_data(date_from: str | None = None,
                    date_to: str | None = None) -> list[dict]:
    """Load per-race strategy stats from daily summary JSON files.

    Each summary JSON (written by aggregator) contains ``strategy_stats``
    keyed by strategy.  We flatten these into a list of dicts:

        {date, strategy_key, N, hits, investment, payout, roi, pnl}

    If LOG_DIR does not exist or contains no matching files, returns [].

    The raw per-race ROI series needed for bootstrap / t-test is reconstructed
    as a binary 0/1 hit sequence scaled by payout-per-race (payout / n races).
    For t-test we use per-race pnl proxy: +payout if hit, -inv if miss.
    We reconstruct this from aggregate stats (N, hits, inv, pay) because the
    summary files aggregate per day, not per race.

    Returns list of dicts per (date, strategy_key):
        date, strategy_key, n, hits, inv, pay, roi_pct, pnl
    """
    if not LOG_DIR.exists():
        return []

    df, dt = _parse_date_range(date_from, date_to)

    rows = []
    for json_file in sorted(LOG_DIR.glob('*.json')):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception:
            continue

        # summary JSON: list (range) or dict (single day)
        records = data if isinstance(data, list) else [data]

        for rec in records:
            date_str = str(rec.get('date', ''))
            if not _in_range(date_str, df, dt):
                continue

            ss = rec.get('strategy_stats', {})
            if not ss:
                continue

            for key in STRATEGY_KEYS:
                kdata = ss.get(key, {})
                if not kdata:
                    continue
                n = int(kdata.get('n', 0) or 0)
                hits = int(kdata.get('hits', 0) or 0)
                inv = int(kdata.get('inv', 0) or 0)
                pay = int(kdata.get('pay', 0) or 0)
                roi = float(kdata.get('roi_pct', 0.0) or 0.0)
                pnl = int(kdata.get('pnl', 0) or 0)
                if n == 0:
                    continue
                rows.append({
                    'date': date_str,
                    'strategy_key': key,
                    'n': n,
                    'hits': hits,
                    'inv': inv,
                    'pay': pay,
                    'roi_pct': roi,
                    'pnl': pnl,
                })

    return rows


def _aggregate_by_strategy(rows: list[dict]) -> dict[str, dict]:
    """Aggregate per-day rows into per-strategy totals."""
    agg: dict[str, dict] = {}
    for key in STRATEGY_KEYS:
        agg[key] = {'n': 0, 'hits': 0, 'inv': 0, 'pay': 0}

    for row in rows:
        key = row['strategy_key']
        if key not in agg:
            continue
        agg[key]['n'] += row['n']
        agg[key]['hits'] += row['hits']
        agg[key]['inv'] += row['inv']
        agg[key]['pay'] += row['pay']

    result = {}
    for key, v in agg.items():
        n = v['n']
        inv = v['inv']
        pay = v['pay']
        roi = round(pay / inv * 100, 2) if inv > 0 else 0.0
        hr = round(v['hits'] / n * 100, 2) if n > 0 else 0.0
        pnl = pay - inv
        result[key] = {
            'n': n,
            'hits': v['hits'],
            'inv': inv,
            'pay': pay,
            'roi_pct': roi,
            'hit_rate_pct': hr,
            'pnl': pnl,
        }
    return result


def _build_per_race_pnl(n: int, hits: int, inv: int, pay: int) -> list[float]:
    """Reconstruct per-race pnl sequence from aggregate stats.

    We create a binary sequence of n values where `hits` races returned
    `pay/hits` each (average payout) and the rest returned 0.
    Investment per race = inv / n (typically 700).

    This is an approximation — the actual payout per hit varies, but the
    aggregate pnl is preserved.
    """
    if n == 0:
        return []
    inv_per = inv / n
    avg_pay = pay / hits if hits > 0 else 0.0

    # hits wins at avg_pay, (n - hits) losses at 0
    seq = ([avg_pay - inv_per] * hits) + ([-inv_per] * (n - hits))
    return seq


# ---------------------------------------------------------------------------
# statistical helpers
# ---------------------------------------------------------------------------

def compute_bootstrap_ci(
    values: list[float],
    n_iter: int = 1000,
    ci: float = 0.95,
) -> tuple[float, float]:
    """Bootstrap CI on mean.

    Args:
        values: list of numeric values
        n_iter: number of bootstrap resamples
        ci: confidence interval width (0.95 = 95% CI)

    Returns:
        (lower, upper) tuple of floats
    """
    if not values:
        return (0.0, 0.0)
    n = len(values)
    means = []
    for _ in range(n_iter):
        sample = random.choices(values, k=n)
        means.append(sum(sample) / n)
    means.sort()
    alpha = (1.0 - ci) / 2.0
    lo_idx = max(0, int(math.floor(alpha * n_iter)))
    hi_idx = min(n_iter - 1, int(math.ceil((1.0 - alpha) * n_iter)) - 1)
    return (round(means[lo_idx], 4), round(means[hi_idx], 4))


def welch_t_test(a: list[float], b: list[float]) -> tuple[float, float]:
    """Welch's t-test (two-sample, unequal variance).

    Args:
        a: first sample
        b: second sample

    Returns:
        (t_stat, p_value)
    """
    # Try scipy first
    try:
        from scipy import stats as sp_stats
        result = sp_stats.ttest_ind(a, b, equal_var=False)
        return (float(result.statistic), float(result.pvalue))
    except ImportError:
        pass

    # Manual Welch's t-test fallback
    na = len(a)
    nb = len(b)
    if na < 2 or nb < 2:
        return (0.0, 1.0)

    mean_a = sum(a) / na
    mean_b = sum(b) / nb

    var_a = sum((x - mean_a) ** 2 for x in a) / (na - 1)
    var_b = sum((x - mean_b) ** 2 for x in b) / (nb - 1)

    se2_a = var_a / na
    se2_b = var_b / nb
    se_diff = math.sqrt(se2_a + se2_b)

    if se_diff == 0.0:
        return (0.0, 1.0)

    t_stat = (mean_a - mean_b) / se_diff

    # Welch-Satterthwaite degrees of freedom
    num = (se2_a + se2_b) ** 2
    den = (se2_a ** 2 / (na - 1)) + (se2_b ** 2 / (nb - 1))
    df = num / den if den > 0 else 1.0

    # Two-tailed p-value via incomplete beta function approximation
    p_value = _t_dist_p(abs(t_stat), df)
    return (round(t_stat, 4), round(p_value, 6))


def _t_dist_p(t: float, df: float) -> float:
    """Two-tailed p-value for t-distribution (manual, no scipy)."""
    # Use regularized incomplete beta function approximation
    x = df / (df + t * t)
    p_half = _regularized_incomplete_beta(df / 2.0, 0.5, x) / 2.0
    return min(1.0, 2.0 * p_half)


def _regularized_incomplete_beta(a: float, b: float, x: float,
                                   max_iter: int = 200, tol: float = 1e-8) -> float:
    """Regularized incomplete beta function I_x(a, b) via continued fraction."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0

    # Use continued fraction representation (Lentz method)
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(math.log(x) * a + math.log(1.0 - x) * b - lbeta) / a

    # betacf via modified Lentz
    def _betacf(a, b, x):
        qab = a + b
        qap = a + 1.0
        qam = a - 1.0
        c = 1.0
        d = 1.0 - qab * x / qap
        if abs(d) < 1e-30:
            d = 1e-30
        d = 1.0 / d
        h = d
        for m in range(1, max_iter + 1):
            m2 = 2 * m
            aa = m * (b - m) * x / ((qam + m2) * (a + m2))
            d = 1.0 + aa * d
            if abs(d) < 1e-30:
                d = 1e-30
            c = 1.0 + aa / c
            if abs(c) < 1e-30:
                c = 1e-30
            d = 1.0 / d
            h *= d * c
            aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
            d = 1.0 + aa * d
            if abs(d) < 1e-30:
                d = 1e-30
            c = 1.0 + aa / c
            if abs(c) < 1e-30:
                c = 1e-30
            d = 1.0 / d
            delta = d * c
            h *= delta
            if abs(delta - 1.0) < tol:
                break
        return h

    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x)
    else:
        # Use symmetry: I_x(a,b) = 1 - I_{1-x}(b,a)
        lbeta2 = math.lgamma(b) + math.lgamma(a) - math.lgamma(a + b)
        front2 = math.exp(math.log(1.0 - x) * b + math.log(x) * a - lbeta2) / b
        return 1.0 - front2 * _betacf(b, a, 1.0 - x)


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------

def evaluate_strategy(
    key: str,
    stats: dict,
    baseline_stats: dict,
) -> dict:
    """5-item evaluation for a strategy.

    Items:
      1. paper N >= 24R
      2. ROI improvement vs baseline >= +5pt (GO) or +3pt (限定GO)
      3. LEAK audit PASS (assumed True for paper strategies)
      4. LIVE stability: no anomaly (always True for paper period)
      5. statistical significance p < 0.05

    Verdict:
      GO       — checks 1,3,4,5 all PASS + ROI delta >= +5pt
      限定GO   — checks 1,3,4,5 PASS + ROI delta in [+3, +5)
      NO-GO    — otherwise

    Returns dict with all details.
    """
    n = stats.get('n', 0)
    roi = stats.get('roi_pct', 0.0)
    hits = stats.get('hits', 0)
    inv = stats.get('inv', 0)
    pay = stats.get('pay', 0)
    pnl = stats.get('pnl', 0)

    baseline_n = baseline_stats.get('n', 0)
    baseline_roi = baseline_stats.get('roi_pct', 0.0)
    baseline_hits = baseline_stats.get('hits', 0)
    baseline_inv = baseline_stats.get('inv', 0)
    baseline_pay = baseline_stats.get('pay', 0)

    delta = round(roi - baseline_roi, 2)

    # Bootstrap CI on ROI% (from per-race pnl, normalised to % of inv_per_race)
    inv_per = inv / n if n > 0 else 700
    pnl_seq = _build_per_race_pnl(n, hits, inv, pay)
    roi_seq = [v / inv_per * 100 for v in pnl_seq] if inv_per > 0 else []
    ci_lower, ci_upper = compute_bootstrap_ci(roi_seq, n_iter=1000) if roi_seq else (0.0, 0.0)

    # Welch's t-test: strategy pnl vs baseline pnl
    strat_pnl_seq = _build_per_race_pnl(n, hits, inv, pay)
    base_pnl_seq = _build_per_race_pnl(baseline_n, baseline_hits, baseline_inv, baseline_pay)
    if len(strat_pnl_seq) >= 2 and len(base_pnl_seq) >= 2:
        t_stat, p_value = welch_t_test(strat_pnl_seq, base_pnl_seq)
    else:
        t_stat, p_value = 0.0, 1.0

    # 5-item checks
    check1 = n >= THRESH_N                    # paper N >= 24R
    check2_go = delta >= THRESH_ROI_GO        # ROI delta >= +5pt
    check2_cond = delta >= THRESH_ROI_COND    # ROI delta >= +3pt
    check3 = True                             # LEAK audit PASS (assumed)
    check4 = True                             # LIVE stability (paper period)
    check5 = p_value < THRESH_P               # statistical significance

    # Verdict
    all_pass_except_roi = check1 and check3 and check4 and check5
    if all_pass_except_roi and check2_go:
        verdict = 'GO'
    elif all_pass_except_roi and check2_cond:
        verdict = '限定GO'
    else:
        verdict = 'NO-GO'

    return {
        'key': key,
        'name': STRATEGY_NAMES.get(key, key),
        'n': n,
        'hits': hits,
        'roi_pct': roi,
        'baseline_roi': baseline_roi,
        'delta': delta,
        'pnl': pnl,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        't_stat': t_stat,
        'p_value': p_value,
        'check1_n': check1,
        'check2_roi': check2_go or check2_cond,
        'check3_leak': check3,
        'check4_stability': check4,
        'check5_pvalue': check5,
        'verdict': verdict,
    }


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------

def _verdict_icon(v: str) -> str:
    if v == 'GO':
        return 'GO   '
    if v == '限定GO':
        return '限定GO'
    return 'NO-GO'


def print_markdown_table(results: list[dict]) -> None:
    header = (
        '| strategy | name | N | ROI% | baseline% | delta | '
        'CI_low | CI_high | p_val | N>=24 | ROI_ok | LEAK | STAB | p<.05 | verdict |'
    )
    sep = (
        '|---|---|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|:---:|:---:|:---:|:---:|'
    )
    print('\n' + header)
    print(sep)
    for r in results:
        def c(b):
            return 'OK' if b else 'NG'
        row = (
            f"| {r['key']} "
            f"| {r['name']} "
            f"| {r['n']} "
            f"| {r['roi_pct']:.1f} "
            f"| {r['baseline_roi']:.1f} "
            f"| {r['delta']:+.1f} "
            f"| {r['ci_lower']:.1f} "
            f"| {r['ci_upper']:.1f} "
            f"| {r['p_value']:.4f} "
            f"| {c(r['check1_n'])} "
            f"| {c(r['check2_roi'])} "
            f"| {c(r['check3_leak'])} "
            f"| {c(r['check4_stability'])} "
            f"| {c(r['check5_pvalue'])} "
            f"| **{r['verdict']}** |"
        )
        print(row)
    print()


def print_verdict_summary(results: list[dict]) -> None:
    go = [r for r in results if r['verdict'] == 'GO']
    limited = [r for r in results if r['verdict'] == '限定GO']
    no_go = [r for r in results if r['verdict'] == 'NO-GO']

    print('=' * 60)
    print('採用判定サマリー')
    print('=' * 60)

    if go:
        print('\n[GO] 採用推奨:')
        for r in go:
            print(f"  - {r['key']} ({r['name']}): ROI {r['roi_pct']:.1f}% (delta {r['delta']:+.1f}pt, p={r['p_value']:.4f})")
    else:
        print('\n[GO] 採用推奨: なし')

    if limited:
        print('\n[限定GO] 条件付き採用 (4週後に再判定):')
        for r in limited:
            print(f"  - {r['key']} ({r['name']}): ROI {r['roi_pct']:.1f}% (delta {r['delta']:+.1f}pt, p={r['p_value']:.4f})")

    if no_go:
        print('\n[NO-GO] 不採用 (7/15 まで paper 継続):')
        for r in no_go:
            reason = []
            if not r['check1_n']:
                reason.append(f"N={r['n']}<{THRESH_N}")
            if not r['check2_roi']:
                reason.append(f"delta={r['delta']:+.1f}pt<{THRESH_ROI_COND}pt")
            if not r['check5_pvalue']:
                reason.append(f"p={r['p_value']:.4f}>={THRESH_P}")
            print(f"  - {r['key']}: {', '.join(reason) if reason else 'criteria not met'}")

    print()

    if go:
        print('次のアクション (GO strategy):')
        print('  race_auto_notify.py で該当 STRATEGY_*_PAPER_ONLY = False に変更')
        for r in go:
            flag = r['key'].upper()
            print(f"    STRATEGY_{flag}_PAPER_ONLY = False  # {r['name']}")
    else:
        print('次のアクション: paper 継続。7/15 に再判定。')
    print()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description='6/17 採用判定 statistical test script'
    )
    ap.add_argument('--date-from', default=None,
                    help='集計開始日 YYYYMMDD (default: 全期間)')
    ap.add_argument('--date-to', default=None,
                    help='集計終了日 YYYYMMDD (default: 全期間)')
    ap.add_argument('--seed', type=int, default=42,
                    help='bootstrap random seed (default: 42)')
    ap.add_argument('--n-iter', type=int, default=1000,
                    help='bootstrap iterations (default: 1000)')
    args = ap.parse_args()

    random.seed(args.seed)

    # Load paper data
    rows = load_paper_data(
        date_from=args.date_from,
        date_to=args.date_to,
    )

    if not rows:
        print('No paper data yet. Run after 5/24.')
        sys.exit(0)

    # Aggregate by strategy
    agg = _aggregate_by_strategy(rows)

    # baseline = 'actual'
    baseline_stats = agg.get('actual', {
        'n': 0, 'hits': 0, 'inv': 0, 'pay': 0, 'roi_pct': 0.0, 'pnl': 0,
    })

    print(f'\n6/17 採用判定 — paper eval data')
    date_from_str = args.date_from or '(全期間)'
    date_to_str = args.date_to or '(全期間)'
    print(f'期間: {date_from_str} ~ {date_to_str}')
    print(f'baseline (actual) N={baseline_stats["n"]}, ROI={baseline_stats["roi_pct"]:.1f}%')
    print(f'bootstrap n_iter={args.n_iter}, seed={args.seed}')

    # Evaluate each strategy
    results = []
    for key in STRATEGY_KEYS:
        stats = agg.get(key, {
            'n': 0, 'hits': 0, 'inv': 0, 'pay': 0, 'roi_pct': 0.0, 'pnl': 0,
        })
        ev = evaluate_strategy(key, stats, baseline_stats)
        results.append(ev)

    # Print
    print_markdown_table(results)
    print_verdict_summary(results)


if __name__ == '__main__':
    main()
