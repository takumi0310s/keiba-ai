"""Kelly Criterion for keiba-ai (D-1 2026-05-19)
Fractional Kelly with safety caps for trio/umaren betting.
Standalone module — no dependency on predict_core.
"""


def calc_kelly_fraction(score, odds, safety_factor=0.25):
    """Simplified Kelly for a single bet.

    Args:
        score: V15 probability estimate (0-1)
        odds: gross payout odds (e.g. 50.0 means ¥5000 per ¥100 ticket)
        safety_factor: fractional Kelly multiplier (default 0.25)

    Returns:
        float: Kelly fraction f (>= 0)
    """
    if odds <= 1.0 or score <= 0.0:
        return 0.0
    b = odds - 1.0          # net odds
    p = min(float(score), 0.95)
    q = 1.0 - p
    kelly_f = (p * b - q) / b
    return max(0.0, kelly_f * safety_factor)


def calc_dynamic_bet_amount(scores, bankroll=50000, safety_factor=0.25,
                             min_bet=300, max_bet=700,
                             rolling_roi=None):
    """Calculate dynamic bet amount using Kelly criterion.

    Args:
        scores: list of V15 scores for top horses (descending order preferred)
        bankroll: current bankroll estimate (yen)
        safety_factor: fractional Kelly (0.25 default = quarter-Kelly)
        min_bet: minimum bet per race (yen, multiple of 100)
        max_bet: maximum bet per race (yen, multiple of 100)
        rolling_roi: 3-day rolling ROI float (e.g. 0.80 = 80%) or None

    Returns:
        int: bet amount per ticket in yen (multiple of 100)
    """
    if not scores:
        return min_bet

    # Rolling ROI adjustment — tighten/loosen safety factor
    effective_safety = safety_factor
    if rolling_roi is not None:
        if rolling_roi < 0.80:      # < 80%: halve the Kelly fraction
            effective_safety = safety_factor * 0.5
        elif rolling_roi > 1.20:    # > 120%: double the Kelly fraction
            effective_safety = safety_factor * 2.0

    # Use the top horse score as probability proxy
    sorted_scores = sorted(scores, reverse=True)
    top_score = float(sorted_scores[0]) if sorted_scores else 0.3

    # Conservative trio payout estimate: ¥5000 → gross odds = 50x
    est_odds = 50.0
    kelly_f = calc_kelly_fraction(top_score, est_odds, effective_safety)

    # Divide bankroll fraction across 7 trio bets
    n_bets = 7
    raw_amount = int(bankroll * kelly_f / n_bets) if kelly_f > 0 else 0

    # Cap to [min_bet, max_bet] and round to nearest ¥100
    capped = max(min_bet, min(max_bet, raw_amount))
    rounded = round(capped / 100) * 100
    return max(min_bet, rounded)


def get_rolling_roi(race_log_path=None, days=3):
    """Load recent race results and compute rolling ROI.

    Args:
        race_log_path: path to CSV file (optional; falls back to defaults)
        days: look-back window in days

    Returns:
        float: rolling ROI (e.g. 0.95 = 95%) or None if insufficient data
    """
    try:
        import pandas as pd
        import os

        df = None
        if race_log_path and os.path.exists(race_log_path):
            df = pd.read_csv(race_log_path)
        else:
            default_paths = [
                'data/cumulative_results.csv',
                'data/track_record.csv',
            ]
            for path in default_paths:
                if os.path.exists(path):
                    df = pd.read_csv(path)
                    break

        if df is None:
            return None

        # Detect columns by name
        date_col = next(
            (c for c in df.columns if 'date' in c.lower()), None)
        pay_col = next(
            (c for c in df.columns
             if c in ('trio_paid', 'payout', 'paid', 'actual_payout')),
            None)
        inv_col = next(
            (c for c in df.columns
             if c in ('investment', 'bet_amount', 'cost', 'invested')),
            None)

        if not all([date_col, pay_col, inv_col]):
            return None

        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        cutoff = df[date_col].max() - pd.Timedelta(days=days)
        recent = df[df[date_col] >= cutoff]

        if len(recent) == 0:
            return None

        total_pay = recent[pay_col].fillna(0).sum()
        total_inv = recent[inv_col].fillna(700).sum()
        if total_inv <= 0:
            return None
        return float(total_pay) / float(total_inv)

    except Exception:
        return None


if __name__ == '__main__':
    # Quick smoke-test
    test_scores = [0.35, 0.22, 0.15, 0.10, 0.08, 0.05]
    bet = calc_dynamic_bet_amount(test_scores, bankroll=50000)
    print(f'Kelly bet (normal): ¥{bet}')

    bet_bad = calc_dynamic_bet_amount(test_scores, bankroll=50000, rolling_roi=0.70)
    print(f'Kelly bet (bad ROI): ¥{bet_bad}')

    bet_good = calc_dynamic_bet_amount(test_scores, bankroll=50000, rolling_roi=1.30)
    print(f'Kelly bet (good ROI): ¥{bet_good}')

    rolling = get_rolling_roi()
    print(f'Rolling ROI (live): {rolling}')
