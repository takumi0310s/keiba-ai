#!/usr/bin/env python
"""モデルドリフト検出スクリプト

週次でAUC・的中率・ROIを監視し、ベースラインからの乖離をDiscord警告。
KSテストで予測スコア分布のドリフトを検出。

チェック項目:
1. 週次AUC（予測スコア vs 実結果）がベースラインから0.02以上低下 → 警告
2. KSテスト（訓練時分布 vs 直近分布）でp < 0.01 → ドリフト検出
3. 週次的中率がBT的中率から15%以上低下 → 警告
4. 週次ROIが100%未満（N>=10） → 警告

Usage:
    python tools/drift_detector.py                  # 週次チェック
    python tools/drift_detector.py --dry-run        # Discord送信せず
    python tools/drift_detector.py --weeks 4        # 直近4週間分析
"""
import pandas as pd
import numpy as np
import os
import sys
import json
import argparse
import sqlite3
from datetime import datetime, timedelta
from scipy import stats as scipy_stats

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

DATA_DIR = os.path.join(BASE_DIR, 'data')
CUMUL_PATH = os.path.join(DATA_DIR, 'cumulative_results.csv')
DRIFT_LOG_PATH = os.path.join(DATA_DIR, 'drift_detection_log.json')

# ベースライン（v14.1 WF mean）
BASELINE_AUC = 0.8856
AUC_DROP_THRESHOLD = 0.02  # AUC低下アラート閾値

# BT的中率ベースライン（v13.5b全体）
BT_HIT_RATES = {
    'A': 0.632, 'B': 0.613, 'C': 0.516,
    'D': 0.482, 'E': 0.727, 'X': 0.540,
    'overall': 0.50,
}
HIT_RATE_DROP_THRESHOLD = 0.15  # 15%低下で警告

# KSテスト閾値
KS_PVALUE_THRESHOLD = 0.01

INVESTMENT_PER_RACE = 700


def load_predictions_db():
    """SQLite DBから予測データを読み込む。"""
    db_paths = [
        os.path.join(BASE_DIR, 'keiba_predictions.db'),
        os.path.join(BASE_DIR, 'keiba_race_results.db'),
    ]

    for db_path in db_paths:
        if not os.path.exists(db_path):
            continue
        try:
            conn = sqlite3.connect(db_path)
            # Check what tables exist
            tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", conn)
            table_names = tables['name'].tolist()

            if 'predictions' in table_names:
                df = pd.read_sql("SELECT * FROM predictions", conn)
                conn.close()
                return df
            elif 'race_predictions' in table_names:
                df = pd.read_sql("SELECT * FROM race_predictions", conn)
                conn.close()
                return df
        except Exception as e:
            print(f"  WARNING: Failed to read {db_path}: {e}")
            continue
    return None


def load_cumulative():
    """累積結果CSVを読み込む。"""
    if not os.path.exists(CUMUL_PATH):
        return None
    df = pd.read_csv(CUMUL_PATH, encoding='utf-8-sig')
    if 'status' in df.columns:
        df = df[df['status'] == 'settled']
    return df


def compute_weekly_auc(df, n_weeks=4):
    """週次AUCを計算する。

    予測スコア(score/pred_score)と実際の的中(hit/trio_hit)からAUCを計算。
    """
    # Find score column
    score_col = None
    for col in ['pred_score', 'score', 'top1_score', 'ai_score']:
        if col in df.columns:
            score_col = col
            break

    if score_col is None:
        return None, "No score column found"

    # Find hit column
    hit_col = None
    for col in ['trio_hit', 'hit', 'is_hit']:
        if col in df.columns:
            hit_col = col
            break

    if hit_col is None:
        return None, "No hit column found"

    # Find date column
    date_col = None
    for col in ['date', 'race_date', 'predict_date']:
        if col in df.columns:
            date_col = col
            break

    if date_col is None:
        return None, "No date column found"

    df = df.copy()
    df['_date'] = pd.to_datetime(df[date_col].astype(str), format='%Y%m%d', errors='coerce')
    df = df.dropna(subset=['_date', score_col, hit_col])

    df[score_col] = pd.to_numeric(df[score_col], errors='coerce')
    df[hit_col] = pd.to_numeric(df[hit_col], errors='coerce').clip(0, 1)

    if len(df) == 0:
        return None, "No valid data after cleaning"

    # Group by ISO week
    df['_week'] = df['_date'].dt.isocalendar().week.astype(int)
    df['_year'] = df['_date'].dt.isocalendar().year.astype(int)
    df['_yw'] = df['_year'] * 100 + df['_week']

    # Get last n_weeks
    weeks = sorted(df['_yw'].unique())
    if len(weeks) > n_weeks:
        weeks = weeks[-n_weeks:]

    results = []
    for yw in weeks:
        week_df = df[df['_yw'] == yw]
        if len(week_df) < 5:
            continue

        scores = week_df[score_col].values
        hits = week_df[hit_col].values

        # AUC requires both classes
        if len(np.unique(hits)) < 2:
            continue

        try:
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(hits, scores)
        except Exception:
            continue

        n = len(week_df)
        hit_rate = hits.mean()
        date_range = (week_df['_date'].min().strftime('%m/%d'),
                      week_df['_date'].max().strftime('%m/%d'))

        results.append({
            'year_week': int(yw),
            'date_range': f"{date_range[0]}-{date_range[1]}",
            'n': n,
            'auc': float(auc),
            'hit_rate': float(hit_rate),
            'scores': scores.tolist(),
        })

    return results, None


def run_ks_test(reference_scores, recent_scores):
    """KSテストで予測スコア分布のドリフトを検出。

    Returns: (statistic, p_value, is_drifted)
    """
    if len(reference_scores) < 20 or len(recent_scores) < 10:
        return None, None, False

    ks_stat, p_value = scipy_stats.ks_2samp(reference_scores, recent_scores)
    is_drifted = p_value < KS_PVALUE_THRESHOLD

    return float(ks_stat), float(p_value), is_drifted


def compute_weekly_roi(df, n_weeks=4):
    """週次ROIを計算する。"""
    # Find columns
    date_col = None
    for col in ['date', 'race_date', 'predict_date']:
        if col in df.columns:
            date_col = col
            break
    if date_col is None:
        return None

    df = df.copy()
    df['_date'] = pd.to_datetime(df[date_col].astype(str), format='%Y%m%d', errors='coerce')
    df = df.dropna(subset=['_date'])

    df['_week'] = df['_date'].dt.isocalendar().week.astype(int)
    df['_year'] = df['_date'].dt.isocalendar().year.astype(int)
    df['_yw'] = df['_year'] * 100 + df['_week']

    weeks = sorted(df['_yw'].unique())
    if len(weeks) > n_weeks:
        weeks = weeks[-n_weeks:]

    results = []
    for yw in weeks:
        week_df = df[df['_yw'] == yw]
        n = len(week_df)
        if n == 0:
            continue

        investment = n * INVESTMENT_PER_RACE
        payout = 0
        if 'trio_payout' in week_df.columns:
            payout += week_df['trio_payout'].fillna(0).sum()
        if 'umaren_payout' in week_df.columns:
            payout += week_df['umaren_payout'].fillna(0).sum()

        roi = (payout / investment * 100) if investment > 0 else 0

        results.append({
            'year_week': int(yw),
            'n': int(n),
            'investment': int(investment),
            'payout': int(payout),
            'roi': float(roi),
        })

    return results


def run_drift_detection(dry_run=False, n_weeks=4):
    """ドリフト検出メインロジック。"""
    print(f"{'=' * 60}")
    print(f"Model Drift Detection - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'=' * 60}")
    print(f"  Baseline AUC: {BASELINE_AUC}")
    print(f"  AUC drop threshold: {AUC_DROP_THRESHOLD}")
    print(f"  KS p-value threshold: {KS_PVALUE_THRESHOLD}")

    alerts = []
    info = []
    report = {
        'timestamp': datetime.now().isoformat(),
        'baseline_auc': BASELINE_AUC,
        'checks': {},
    }

    # Load data
    df = load_cumulative()
    if df is None or len(df) == 0:
        print("  [INFO] No cumulative data found. Trying predictions DB...")
        df = load_predictions_db()

    if df is None or len(df) == 0:
        print("  [SKIP] No prediction data available for drift detection.")
        return []

    print(f"  Data: {len(df)} records")

    # === Check 1: Weekly AUC ===
    print(f"\n  --- Weekly AUC Analysis (last {n_weeks} weeks) ---")
    auc_results, auc_err = compute_weekly_auc(df, n_weeks)

    if auc_results and len(auc_results) > 0:
        for r in auc_results:
            auc_delta = r['auc'] - BASELINE_AUC
            status = 'OK' if auc_delta > -AUC_DROP_THRESHOLD else 'ALERT'
            print(f"    W{r['year_week']%100} ({r['date_range']}): "
                  f"AUC={r['auc']:.4f} ({auc_delta:+.4f}) N={r['n']} [{status}]")

            if auc_delta < -AUC_DROP_THRESHOLD:
                alerts.append({
                    'level': 'WARNING',
                    'type': 'auc_drop',
                    'message': f"週次AUC {r['auc']:.4f}がベースライン{BASELINE_AUC}から"
                               f"{abs(auc_delta):.4f}低下 (W{r['year_week']%100}, N={r['n']})",
                })

        # Overall recent AUC
        recent_auc = np.mean([r['auc'] for r in auc_results])
        info.append(f"直近{len(auc_results)}週平均AUC: {recent_auc:.4f}")
        report['checks']['weekly_auc'] = auc_results
    elif auc_err:
        print(f"    AUC calculation skipped: {auc_err}")
    else:
        print(f"    No weekly AUC data available")

    # === Check 2: KS Test (score distribution drift) ===
    print(f"\n  --- KS Test (Score Distribution Drift) ---")

    if auc_results and len(auc_results) >= 2:
        # Use first half as reference, second half as recent
        all_scores = []
        for r in auc_results:
            all_scores.extend(r['scores'])

        mid = len(auc_results) // 2
        ref_scores = []
        recent_scores = []
        for r in auc_results[:mid]:
            ref_scores.extend(r['scores'])
        for r in auc_results[mid:]:
            recent_scores.extend(r['scores'])

        if ref_scores and recent_scores:
            ks_stat, p_value, is_drifted = run_ks_test(
                np.array(ref_scores), np.array(recent_scores))

            if ks_stat is not None:
                status = 'DRIFT' if is_drifted else 'OK'
                print(f"    KS statistic: {ks_stat:.4f}, p-value: {p_value:.6f} [{status}]")
                info.append(f"KS test: stat={ks_stat:.4f}, p={p_value:.4f}")

                report['checks']['ks_test'] = {
                    'statistic': ks_stat,
                    'p_value': p_value,
                    'is_drifted': is_drifted,
                    'n_reference': len(ref_scores),
                    'n_recent': len(recent_scores),
                }

                if is_drifted:
                    alerts.append({
                        'level': 'WARNING',
                        'type': 'distribution_drift',
                        'message': f"予測スコア分布ドリフト検出: KS={ks_stat:.4f}, "
                                   f"p={p_value:.6f} (ref={len(ref_scores)}, "
                                   f"recent={len(recent_scores)})",
                    })
            else:
                print(f"    KS test skipped: insufficient data")
    else:
        print(f"    KS test skipped: need >= 2 weeks of AUC data")

    # === Check 3: Weekly hit rate ===
    print(f"\n  --- Weekly Hit Rate Analysis ---")
    hit_col = None
    for col in ['trio_hit', 'hit', 'is_hit']:
        if col in df.columns:
            hit_col = col
            break

    if hit_col and 'condition' in df.columns:
        date_col = None
        for col in ['date', 'race_date']:
            if col in df.columns:
                date_col = col
                break

        if date_col:
            df_copy = df.copy()
            df_copy['_date'] = pd.to_datetime(df_copy[date_col].astype(str), format='%Y%m%d', errors='coerce')
            df_copy = df_copy.dropna(subset=['_date'])
            cutoff = df_copy['_date'].max() - timedelta(weeks=n_weeks)
            recent = df_copy[df_copy['_date'] >= cutoff]

            if len(recent) > 0:
                overall_hit = pd.to_numeric(recent[hit_col], errors='coerce').clip(0, 1).mean()
                bt_overall = BT_HIT_RATES['overall']
                drop = bt_overall - overall_hit

                print(f"    全体: {overall_hit:.3f} (BT: {bt_overall:.3f}, 差: {drop:+.3f})")

                if drop > HIT_RATE_DROP_THRESHOLD:
                    alerts.append({
                        'level': 'WARNING',
                        'type': 'hit_rate_drop',
                        'message': f"的中率 {overall_hit:.1%}がBT基準{bt_overall:.1%}から"
                                   f"{drop:.1%}低下 (直近{n_weeks}週, N={len(recent)})",
                    })

                report['checks']['hit_rate'] = {
                    'overall': float(overall_hit),
                    'baseline': bt_overall,
                    'n': len(recent),
                }

    # === Check 4: Weekly ROI ===
    print(f"\n  --- Weekly ROI Analysis ---")
    roi_results = compute_weekly_roi(df, n_weeks)

    if roi_results:
        for r in roi_results:
            status = 'OK' if r['roi'] >= 100 else 'LOW'
            print(f"    W{r['year_week']%100}: ROI={r['roi']:.1f}% N={r['n']} [{status}]")

            if r['roi'] < 100 and r['n'] >= 10:
                alerts.append({
                    'level': 'INFO',
                    'type': 'weekly_roi_low',
                    'message': f"週次ROI {r['roi']:.1f}% < 100% "
                               f"(W{r['year_week']%100}, N={r['n']})",
                })

        report['checks']['weekly_roi'] = roi_results

    # === Summary ===
    print(f"\n{'=' * 60}")

    danger_alerts = [a for a in alerts if a['level'] == 'DANGER']
    warn_alerts = [a for a in alerts if a['level'] == 'WARNING']
    info_alerts = [a for a in alerts if a['level'] == 'INFO']

    if danger_alerts:
        print(f"  DANGER アラート: {len(danger_alerts)}件")
        for a in danger_alerts:
            print(f"    !! {a['message']}")
    if warn_alerts:
        print(f"  WARNING アラート: {len(warn_alerts)}件")
        for a in warn_alerts:
            print(f"    !  {a['message']}")
    if info_alerts:
        print(f"  INFO: {len(info_alerts)}件")
        for a in info_alerts:
            print(f"       {a['message']}")

    if not alerts:
        print(f"  全指標正常 - ドリフトなし")

    # Discord通知
    if alerts and not dry_run:
        _send_drift_discord(alerts, info)

    # Save log
    report['alerts'] = alerts
    _save_drift_log(report)

    print(f"{'=' * 60}")
    return alerts


def _send_drift_discord(alerts, info):
    """Discord通知を送信する。"""
    try:
        from notify import send_discord
    except ImportError:
        return

    has_danger = any(a['level'] == 'DANGER' for a in alerts)
    has_warning = any(a['level'] == 'WARNING' for a in alerts)
    color = 'red' if has_danger else 'yellow' if has_warning else 'blue'
    title = f"Drift Detection: {len(alerts)} alerts"

    lines = []
    for a in alerts:
        prefix = '!!' if a['level'] == 'DANGER' else '!' if a['level'] == 'WARNING' else ''
        lines.append(f"{prefix} [{a['type']}] {a['message']}")

    send_discord(title, "\n".join(lines), color=color)


def _save_drift_log(report):
    """ドリフトログをJSONに追記する。"""
    log = []
    if os.path.exists(DRIFT_LOG_PATH):
        try:
            with open(DRIFT_LOG_PATH, 'r', encoding='utf-8') as f:
                log = json.load(f)
        except Exception:
            pass

    # Remove scores from report to save space
    report_copy = json.loads(json.dumps(report, default=str))
    if 'checks' in report_copy and 'weekly_auc' in report_copy['checks']:
        for r in report_copy['checks']['weekly_auc']:
            r.pop('scores', None)

    log.append(report_copy)
    log = log[-52:]  # Keep last 52 weeks (~1 year)

    with open(DRIFT_LOG_PATH, 'w', encoding='utf-8') as f:
        json.dump(log, f, ensure_ascii=False, indent=2, default=str)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='モデルドリフト検出')
    parser.add_argument('--dry-run', action='store_true', help='Discord送信せず')
    parser.add_argument('--weeks', type=int, default=4, help='分析対象週数')
    args = parser.parse_args()

    run_drift_detection(dry_run=args.dry_run, n_weeks=args.weeks)
