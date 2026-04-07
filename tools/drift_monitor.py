"""Model Drift Detection - 週次モデルドリフト検知

予測精度・的中率・ROI・特徴量分布の変動を監視し、
閾値超過時にDiscord警告を送信する。

Usage:
    python tools/drift_monitor.py              # 直近4週チェック
    python tools/drift_monitor.py --verbose    # 詳細出力
    python tools/drift_monitor.py --weeks 8    # 直近8週チェック
"""

import argparse
import gzip
import json
import os
import pickle
import sqlite3
import sys
from datetime import datetime, timedelta

import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, 'tools'))

try:
    from notify import send_discord
except ImportError:
    def send_discord(title, message, color="green", channel="updates"):
        return False

# --- Constants ---

BASELINE_AUC = 0.8856  # v14.1 WF AUC
AUC_DROP_THRESHOLD = 0.02  # Alert if AUC drops by this amount

EXPECTED_HIT_RATES = {
    'A': 0.632, 'B': 0.613, 'C': 0.516,
    'D': 0.482, 'E': 0.727, 'X': 0.540,
}

ROI_FLOOR = 80.0  # Alert if ROI below this for 4 consecutive weeks
ROI_CONSECUTIVE_WEEKS = 4

KS_THRESHOLD = 0.1  # Flag features with KS statistic above this

DB_PRIMARY = os.path.join(BASE_DIR, 'keiba_predictions.db')
DB_FALLBACK = os.path.join(BASE_DIR, 'keiba_ai.db')

MODEL_PATH = os.path.join(BASE_DIR, 'keiba_model_v141_central.pkl.gz')

REPORT_PATH = os.path.join(BASE_DIR, 'data', 'drift_report_weekly.json')


def get_db_path():
    """Return the first database path that exists and has data."""
    for db in [DB_PRIMARY, DB_FALLBACK]:
        if os.path.exists(db):
            try:
                conn = sqlite3.connect(db)
                cur = conn.cursor()
                cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = [t[0] for t in cur.fetchall()]
                conn.close()
                if 'predictions' in tables:
                    return db
            except Exception:
                continue
    return None


def load_predictions(db_path, start_date, end_date):
    """Load predictions with actual results for the given date range.

    Returns list of dicts with keys: race_id, race_date, horse_num,
    ai_score, actual_finish, is_top3_pred.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("""
        SELECT race_id, race_date, horse_num, ai_score, actual_finish, is_top3_pred
        FROM predictions
        WHERE race_date >= ? AND race_date <= ? AND actual_finish IS NOT NULL
        ORDER BY race_date, race_id, horse_num
    """, (start_date, end_date))
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def load_race_results(db_path, start_date, end_date):
    """Load race-level results for the given date range.

    Returns list of dicts from race_results table.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # Determine date column: race_results may not have race_date directly,
    # use predicted_at or join with predictions.
    cur.execute("PRAGMA table_info(race_results)")
    cols = [c[1] for c in cur.fetchall()]

    if 'race_date' in cols:
        date_col = 'race_date'
    else:
        # Use predicted_at (datetime string), extract date portion
        date_col = "substr(predicted_at, 1, 10)"

    cur.execute(f"""
        SELECT *
        FROM race_results
        WHERE {date_col} >= ? AND {date_col} <= ?
        ORDER BY race_id
    """, (start_date, end_date))
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def calc_auc(predictions):
    """Calculate AUC from predictions. Returns (auc, n_races, n_predictions)."""
    if not predictions:
        return None, 0, 0

    scores = np.array([p['ai_score'] for p in predictions])
    labels = np.array([1 if p['actual_finish'] <= 3 else 0 for p in predictions])

    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None, len(set(p['race_id'] for p in predictions)), len(predictions)

    # Mann-Whitney U AUC calculation
    # Rank scores in ascending order (rank 1 = lowest score)
    order = np.argsort(scores)
    ranked = np.empty_like(order, dtype=float)
    # Assign ranks with tie handling
    i = 0
    while i < len(order):
        j = i
        while j < len(order) and scores[order[j]] == scores[order[i]]:
            j += 1
        avg_rank = (i + j + 1) / 2.0  # 1-indexed average rank
        for k in range(i, j):
            ranked[order[k]] = avg_rank
        i = j

    pos_rank_sum = ranked[labels == 1].sum()
    # Higher score = higher rank = better; U = sum_pos_ranks - n_pos*(n_pos+1)/2
    auc = (pos_rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    n_races = len(set(p['race_id'] for p in predictions))
    return float(auc), n_races, len(predictions)


def calc_hit_rates(race_results):
    """Calculate hit rates by condition class.

    Returns dict: {condition: {count, hits, rate}}.
    """
    by_cond = {}
    for r in race_results:
        cond = r.get('bet_condition', 'Unknown')
        if cond not in by_cond:
            by_cond[cond] = {'count': 0, 'hits': 0}
        by_cond[cond]['count'] += 1

        bet_type = r.get('bet_type', 'trio')
        if bet_type == 'umaren':
            hit = r.get('hit_combo', 0) or 0
        else:
            hit = r.get('hit_trio', 0) or 0
        by_cond[cond]['hits'] += (1 if hit else 0)

    for cond in by_cond:
        c = by_cond[cond]
        c['rate'] = c['hits'] / c['count'] if c['count'] > 0 else 0.0
    return by_cond


def calc_weekly_roi(race_results):
    """Calculate overall ROI from race results.

    Returns (roi_pct, investment, payout).
    """
    total_investment = 0
    total_payout = 0
    for r in race_results:
        recommended = r.get('buy_recommended', 1)
        if not recommended:
            continue
        bet_type = r.get('bet_type', 'trio')
        investment = 700  # standard per race
        payout = r.get('payout', 0) or 0
        total_investment += investment
        total_payout += payout

    if total_investment == 0:
        return None, 0, 0
    roi = (total_payout / total_investment) * 100
    return roi, total_investment, total_payout


def calc_weekly_roi_trend(db_path, end_date, num_weeks):
    """Calculate ROI for each of the last num_weeks weeks.

    Returns list of (week_label, roi_pct) tuples, oldest first.
    """
    trend = []
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    for i in range(num_weeks - 1, -1, -1):
        week_end = end_dt - timedelta(weeks=i)
        week_start = week_end - timedelta(days=6)
        results = load_race_results(
            db_path,
            week_start.strftime('%Y-%m-%d'),
            week_end.strftime('%Y-%m-%d'),
        )
        roi, inv, pay = calc_weekly_roi(results)
        label = week_start.strftime('%m/%d')
        trend.append((label, roi if roi is not None else 0.0))
    return trend


def run_ks_tests(predictions, verbose=False):
    """Run KS tests on feature distributions between model training and recent predictions.

    Returns dict: {feature_name: {statistic, p_value, alert}}.
    """
    try:
        from scipy.stats import ks_2samp
    except ImportError:
        if verbose:
            print("  [SKIP] scipy not installed, skipping KS test")
        return {}

    # Load model to get feature names
    if not os.path.exists(MODEL_PATH):
        if verbose:
            print(f"  [SKIP] Model not found: {MODEL_PATH}")
        return {}

    try:
        with gzip.open(MODEL_PATH, 'rb') as f:
            model_data = pickle.load(f)
    except Exception as e:
        if verbose:
            print(f"  [SKIP] Model load error: {e}")
        return {}

    # Extract feature names from model dict
    feature_names = model_data.get('features')
    if not feature_names:
        # Fallback: try lgb_model.feature_name_ or model.feature_name()
        lgb_model = model_data.get('lgb_model') or model_data.get('model')
        if lgb_model is None:
            if verbose:
                print("  [SKIP] No model or features key found")
            return {}
        try:
            feature_names = lgb_model.feature_name_
        except AttributeError:
            try:
                feature_names = lgb_model.feature_name()
            except Exception:
                if verbose:
                    print("  [SKIP] Cannot extract feature names from model")
                return {}

    if verbose:
        print(f"  Model has {len(feature_names)} features")

    # Get training data distributions from model's training dataset if available
    # Fallback: use historical predictions as reference (older data vs recent data)
    if not predictions or len(predictions) < 50:
        if verbose:
            print("  [SKIP] Not enough predictions for KS test")
        return {}

    # Use ai_score distribution as a proxy for drift detection
    # Split predictions into two halves: first half (reference) vs second half (recent)
    sorted_preds = sorted(predictions, key=lambda p: p['race_date'])
    mid = len(sorted_preds) // 2
    ref_scores = np.array([p['ai_score'] for p in sorted_preds[:mid]])
    recent_scores = np.array([p['ai_score'] for p in sorted_preds[mid:]])

    results = {}

    # Test ai_score distribution shift
    stat, pval = ks_2samp(ref_scores, recent_scores)
    results['ai_score'] = {
        'statistic': round(float(stat), 4),
        'p_value': round(float(pval), 6),
        'alert': bool(stat > KS_THRESHOLD),
    }

    # Test score distribution by top3 outcome
    ref_labels = np.array([1 if p['actual_finish'] <= 3 else 0 for p in sorted_preds[:mid]])
    recent_labels = np.array([1 if p['actual_finish'] <= 3 else 0 for p in sorted_preds[mid:]])

    # Positive class scores
    ref_pos = ref_scores[ref_labels == 1]
    recent_pos = recent_scores[recent_labels == 1]
    if len(ref_pos) > 10 and len(recent_pos) > 10:
        stat, pval = ks_2samp(ref_pos, recent_pos)
        results['ai_score_positive'] = {
            'statistic': round(float(stat), 4),
            'p_value': round(float(pval), 6),
            'alert': bool(stat > KS_THRESHOLD),
        }

    # Negative class scores
    ref_neg = ref_scores[ref_labels == 0]
    recent_neg = recent_scores[recent_labels == 0]
    if len(ref_neg) > 10 and len(recent_neg) > 10:
        stat, pval = ks_2samp(ref_neg, recent_neg)
        results['ai_score_negative'] = {
            'statistic': round(float(stat), 4),
            'p_value': round(float(pval), 6),
            'alert': bool(stat > KS_THRESHOLD),
        }

    # Test finish position distribution (data drift proxy)
    ref_finish = np.array([p['actual_finish'] for p in sorted_preds[:mid]])
    recent_finish = np.array([p['actual_finish'] for p in sorted_preds[mid:]])
    stat, pval = ks_2samp(ref_finish, recent_finish)
    results['actual_finish'] = {
        'statistic': round(float(stat), 4),
        'p_value': round(float(pval), 6),
        'alert': bool(stat > KS_THRESHOLD),
    }

    return results


def build_report(auc_info, hit_rates_info, roi_info, ks_results, period_start, period_end):
    """Build the drift report dict."""
    today = datetime.now().strftime('%Y-%m-%d')

    alerts = []

    # AUC section
    auc_section = {
        'current': auc_info['auc'],
        'baseline': BASELINE_AUC,
        'delta': None,
        'alert': False,
        'n_races': auc_info['n_races'],
        'n_predictions': auc_info['n_predictions'],
    }
    if auc_info['auc'] is not None:
        delta = auc_info['auc'] - BASELINE_AUC
        auc_section['delta'] = round(delta, 4)
        if delta < -AUC_DROP_THRESHOLD:
            auc_section['alert'] = True
            alerts.append(f"AUC dropped by {abs(delta):.4f} below baseline "
                          f"({auc_info['auc']:.4f} vs {BASELINE_AUC})")
    else:
        auc_section['delta'] = None

    # Hit rates section
    hit_section = {}
    for cond in ['A', 'B', 'C', 'D', 'E', 'X']:
        expected = EXPECTED_HIT_RATES.get(cond, 0)
        actual_data = hit_rates_info.get(cond, {})
        current = actual_data.get('rate', None)
        n = actual_data.get('count', 0)
        entry = {
            'current': round(current, 4) if current is not None else None,
            'expected': expected,
            'delta': None,
            'n': n,
        }
        if current is not None and n >= 5:
            delta = current - expected
            entry['delta'] = round(delta, 4)
            # Warn if hit rate drops by 15+ percentage points with sufficient sample
            if delta < -0.15 and n >= 10:
                alerts.append(f"Condition {cond} hit rate {current:.1%} "
                              f"vs expected {expected:.1%} (N={n})")
        hit_section[cond] = entry

    # ROI section
    roi_section = {
        'current': round(roi_info['roi'], 1) if roi_info['roi'] is not None else None,
        'investment': roi_info['investment'],
        'payout': roi_info['payout'],
        '4week_trend': roi_info.get('trend', []),
        'alert': False,
    }
    trend_values = [t[1] for t in roi_info.get('trend', []) if t[1] is not None and t[1] > 0]
    if len(trend_values) >= ROI_CONSECUTIVE_WEEKS:
        last_n = trend_values[-ROI_CONSECUTIVE_WEEKS:]
        if all(r < ROI_FLOOR for r in last_n):
            roi_section['alert'] = True
            alerts.append(f"ROI below {ROI_FLOOR}% for {ROI_CONSECUTIVE_WEEKS} "
                          f"consecutive weeks: {[round(r, 1) for r in last_n]}")

    # KS test section
    ks_section = {}
    for feat, info in ks_results.items():
        ks_section[feat] = info
        if info.get('alert'):
            alerts.append(f"KS drift detected for '{feat}': "
                          f"statistic={info['statistic']:.4f}, p={info['p_value']:.6f}")

    report = {
        'date': today,
        'period': f"{period_start} to {period_end}",
        'auc': auc_section,
        'hit_rates': hit_section,
        'roi': roi_section,
        'ks_test': ks_section,
        'alerts': alerts,
    }
    return report


def format_report(report, verbose=False):
    """Format report as human-readable string."""
    lines = []
    lines.append(f"=== Drift Monitor Report ===")
    lines.append(f"Date: {report['date']}")
    lines.append(f"Period: {report['period']}")
    lines.append("")

    # AUC
    auc = report['auc']
    status = "ALERT" if auc['alert'] else "OK"
    if auc['current'] is not None:
        lines.append(f"[AUC] {status}")
        lines.append(f"  Current: {auc['current']:.4f} (baseline: {auc['baseline']})")
        lines.append(f"  Delta: {auc['delta']:+.4f}")
        lines.append(f"  Races: {auc['n_races']}, Predictions: {auc['n_predictions']}")
    else:
        lines.append("[AUC] NO DATA")
    lines.append("")

    # Hit rates
    lines.append("[Hit Rates by Condition]")
    for cond in ['A', 'B', 'C', 'D', 'E', 'X']:
        info = report['hit_rates'].get(cond, {})
        current = info.get('current')
        expected = info.get('expected', 0)
        n = info.get('n', 0)
        if current is not None and n > 0:
            delta = info.get('delta', 0) or 0
            flag = " *" if abs(delta) > 0.10 else ""
            lines.append(f"  {cond}: {current:.1%} (expected {expected:.1%}, "
                         f"delta {delta:+.1%}, N={n}){flag}")
        else:
            lines.append(f"  {cond}: no data")
    lines.append("")

    # ROI
    roi = report['roi']
    status = "ALERT" if roi['alert'] else "OK"
    if roi['current'] is not None:
        lines.append(f"[ROI] {status}")
        lines.append(f"  Current: {roi['current']:.1f}%")
        lines.append(f"  Investment: {roi['investment']:,}, Payout: {roi['payout']:,}")
        trend = roi.get('4week_trend', [])
        if trend:
            trend_str = " -> ".join(f"{t[0]}:{t[1]:.0f}%" for t in trend if t[1])
            lines.append(f"  Trend: {trend_str}")
    else:
        lines.append("[ROI] NO DATA")
    lines.append("")

    # KS tests
    ks = report.get('ks_test', {})
    if ks:
        lines.append("[KS Distribution Tests]")
        for feat, info in ks.items():
            flag = "DRIFT" if info.get('alert') else "ok"
            lines.append(f"  {feat}: KS={info['statistic']:.4f} "
                         f"p={info['p_value']:.4f} [{flag}]")
    else:
        lines.append("[KS Distribution Tests] skipped")
    lines.append("")

    # Alerts
    alerts = report.get('alerts', [])
    if alerts:
        lines.append(f"[ALERTS: {len(alerts)}]")
        for a in alerts:
            lines.append(f"  ! {a}")
    else:
        lines.append("[STATUS] All OK - no alerts")

    return "\n".join(lines)


def build_discord_message(report):
    """Build Discord notification message."""
    alerts = report.get('alerts', [])
    has_alerts = len(alerts) > 0

    title = "Drift Monitor: " + ("ALERT" if has_alerts else "OK")
    color = "red" if has_alerts else "green"

    lines = [f"Period: {report['period']}"]

    # AUC summary
    auc = report['auc']
    if auc['current'] is not None:
        emoji = "!" if auc['alert'] else "o"
        lines.append(f"[{emoji}] AUC: {auc['current']:.4f} "
                     f"(baseline {auc['baseline']}, delta {auc['delta']:+.4f})")
    else:
        lines.append("[-] AUC: no data")

    # ROI summary
    roi = report['roi']
    if roi['current'] is not None:
        emoji = "!" if roi['alert'] else "o"
        lines.append(f"[{emoji}] ROI: {roi['current']:.1f}%")
    else:
        lines.append("[-] ROI: no data")

    # Hit rate summary (only conditions with data)
    hit_parts = []
    for cond in ['A', 'B', 'C', 'D', 'E', 'X']:
        info = report['hit_rates'].get(cond, {})
        current = info.get('current')
        n = info.get('n', 0)
        if current is not None and n > 0:
            hit_parts.append(f"{cond}:{current:.0%}({n})")
    if hit_parts:
        lines.append(f"Hit: {' '.join(hit_parts)}")

    # KS alerts
    ks_alerts = [f for f, info in report.get('ks_test', {}).items() if info.get('alert')]
    if ks_alerts:
        lines.append(f"[!] KS drift: {', '.join(ks_alerts)}")

    # Alert details
    if alerts:
        lines.append("")
        lines.append(f"--- Alerts ({len(alerts)}) ---")
        for a in alerts:
            lines.append(f"- {a}")

    message = "\n".join(lines)
    return title, message, color


def main():
    parser = argparse.ArgumentParser(description='Model Drift Monitor')
    parser.add_argument('--verbose', '-v', action='store_true', help='Detailed output')
    parser.add_argument('--weeks', type=int, default=4, help='Number of weeks to analyze (default: 4)')
    parser.add_argument('--no-discord', action='store_true', help='Skip Discord notification')
    args = parser.parse_args()

    verbose = args.verbose
    num_weeks = args.weeks

    if verbose:
        print(f"Drift Monitor - analyzing last {num_weeks} weeks")
        print(f"Baseline AUC: {BASELINE_AUC}")
        print()

    # Find database
    db_path = get_db_path()
    if db_path is None:
        print("ERROR: No prediction database found")
        sys.exit(1)
    if verbose:
        print(f"Database: {db_path}")

    # Date range
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_dt = datetime.now() - timedelta(weeks=num_weeks)
    start_date = start_dt.strftime('%Y-%m-%d')

    if verbose:
        print(f"Period: {start_date} to {end_date}")
        print()

    # 1. Load predictions and calculate AUC
    if verbose:
        print("[1] AUC Monitoring...")
    predictions = load_predictions(db_path, start_date, end_date)
    auc, n_races, n_preds = calc_auc(predictions)
    auc_info = {'auc': round(auc, 4) if auc is not None else None,
                'n_races': n_races, 'n_predictions': n_preds}
    if verbose:
        if auc is not None:
            print(f"  AUC: {auc:.4f} ({n_races} races, {n_preds} predictions)")
        else:
            print(f"  No AUC available ({n_preds} predictions)")

    # 2. Load race results and calculate hit rates
    if verbose:
        print("[2] Hit Rate Monitoring...")
    race_results = load_race_results(db_path, start_date, end_date)
    hit_rates = calc_hit_rates(race_results)
    if verbose:
        for cond, info in sorted(hit_rates.items()):
            print(f"  {cond}: {info['rate']:.1%} ({info['hits']}/{info['count']})")

    # 3. ROI monitoring
    if verbose:
        print("[3] ROI Monitoring...")
    roi, investment, payout = calc_weekly_roi(race_results)
    roi_trend = calc_weekly_roi_trend(db_path, end_date, num_weeks)
    roi_info = {
        'roi': roi, 'investment': investment, 'payout': payout,
        'trend': roi_trend,
    }
    if verbose:
        if roi is not None:
            print(f"  ROI: {roi:.1f}% (invest={investment:,}, payout={payout:,})")
        trend_str = ", ".join(f"{t[0]}:{t[1]:.0f}%" for t in roi_trend if t[1])
        if trend_str:
            print(f"  Trend: {trend_str}")

    # 4. KS distribution tests
    if verbose:
        print("[4] KS Distribution Tests...")
    ks_results = run_ks_tests(predictions, verbose=verbose)

    # 5. Build and save report
    if verbose:
        print("[5] Building report...")
    report = build_report(auc_info, hit_rates, roi_info, ks_results, start_date, end_date)

    # Save JSON report
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    if verbose:
        print(f"  Saved: {REPORT_PATH}")

    # Print human-readable report
    print()
    print(format_report(report, verbose=verbose))

    # 6. Discord notification
    if not args.no_discord:
        title, message, color = build_discord_message(report)
        sent = send_discord(title, message, color=color, channel="updates")
        if verbose:
            print(f"\nDiscord notification: {'sent' if sent else 'skipped (no webhook)'}")

    # Exit code: 1 if alerts present
    if report['alerts']:
        sys.exit(1)
    sys.exit(0)


if __name__ == '__main__':
    main()
