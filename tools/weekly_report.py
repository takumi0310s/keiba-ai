"""
週次レポート生成スクリプト
過去7日間の実運用成績を集計し、条件別ROI・的中率・モデル健全性を確認する。

Usage:
    python tools/weekly_report.py                  # 直近7日分
    python tools/weekly_report.py --date 20260309  # 指定日から過去7日分
"""
import pandas as pd
import numpy as np
import os
import sys
import json
import argparse
import pickle
import gzip
from datetime import datetime, timedelta

# === パス設定 ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

# === 定数 ===
INVESTMENT_PER_RACE = 700
AUC_THRESHOLD = 0.78  # モデルAUC最低ライン

CONDITION_PROFILES = {
    'A': {'label': '条件A', 'desc': '8-14頭/1600m+/良~稍', 'bet': 'trio 7点',
           'expected_hit': 44.5, 'expected_roi': 205.3, 'conservative_roi': 143.7},
    'B': {'label': '条件B', 'desc': '8-14頭/1600m+/重~不良', 'bet': 'trio 7点',
           'expected_hit': 45.2, 'expected_roi': 236.9, 'conservative_roi': 165.8},
    'C': {'label': '条件C', 'desc': '15頭+/1600m+/良~稍', 'bet': 'trio 7点',
           'expected_hit': 33.7, 'expected_roi': 285.6, 'conservative_roi': 199.9},
    'D': {'label': '条件D', 'desc': '1400m以下', 'bet': 'trio 7点',
           'expected_hit': 27.0, 'expected_roi': 136.0, 'conservative_roi': 95.2},
    'E': {'label': '条件E', 'desc': '7頭以下', 'bet': 'umaren 2点',
           'expected_hit': 53.4, 'expected_roi': 118.0, 'conservative_roi': 82.6},
    'X': {'label': '条件X', 'desc': '15頭+/重~不良', 'bet': 'trio 7点',
           'expected_hit': 35.5, 'expected_roi': 330.5, 'conservative_roi': 231.3},
}

# バックテスト全体保守的ROI
CONSERVATIVE_ROI_OVERALL = 142.6


def check_backtest_divergence(cumul_df):
    """累積ROIとバックテスト保守的見積りの統計的乖離を検定する。

    各レースの収益をベルヌーイ試行として、累積ROIの信頼区間を計算。
    バックテスト保守的ROIが2σ外にある場合に警告を出す。

    Returns:
        dict: {overall: {live_roi, bt_roi, z_score, significant, n},
               by_condition: {cond: {live_roi, bt_roi, z_score, significant, n}}}
    """
    result = {'overall': None, 'by_condition': {}}
    if len(cumul_df) == 0:
        return result

    def _z_test(df_sub, expected_roi):
        """ROIのz検定。H0: 実ROI = expected_roi"""
        n = len(df_sub)
        if n < 10:
            return {'n': n, 'live_roi': 0, 'bt_roi': expected_roi,
                    'z_score': 0, 'significant': False, 'msg': 'N<10'}

        # 各レースの利益率を計算
        investment = df_sub.get('investment', pd.Series([INVESTMENT_PER_RACE] * n))
        if 'trio_payout' in df_sub.columns:
            payout = df_sub['trio_payout'].fillna(0)
        elif 'payout' in df_sub.columns:
            payout = df_sub['payout'].fillna(0)
        else:
            payout = pd.Series([0] * n)

        inv_total = investment.sum()
        pay_total = payout.sum()
        if inv_total == 0:
            return {'n': n, 'live_roi': 0, 'bt_roi': expected_roi,
                    'z_score': 0, 'significant': False, 'msg': 'no investment'}

        live_roi = pay_total / inv_total * 100
        # 各レースのROI（利益率）
        race_returns = (payout / investment * 100).values
        std = np.std(race_returns, ddof=1)
        if std == 0:
            return {'n': n, 'live_roi': round(live_roi, 1), 'bt_roi': expected_roi,
                    'z_score': 0, 'significant': False, 'msg': 'zero variance'}

        se = std / np.sqrt(n)
        z = (live_roi - expected_roi) / se
        return {
            'n': n,
            'live_roi': round(live_roi, 1),
            'bt_roi': expected_roi,
            'z_score': round(z, 2),
            'significant': abs(z) > 2.0,
            'direction': 'above' if z > 0 else 'below',
            'msg': f'z={z:.2f} {"⚠ SIGNIFICANT" if abs(z) > 2.0 else "within 2σ"}',
        }

    # Overall
    result['overall'] = _z_test(cumul_df, CONSERVATIVE_ROI_OVERALL)

    # By condition
    if 'condition' in cumul_df.columns:
        for cond in sorted(cumul_df['condition'].unique()):
            sub = cumul_df[cumul_df['condition'] == cond]
            profile = CONDITION_PROFILES.get(cond, {})
            c_roi = profile.get('conservative_roi', 100)
            result['by_condition'][cond] = _z_test(sub, c_roi)

    return result


def check_model_auc():
    """モデルのAUCを確認（V12対応）"""
    result = {'pattern_a': None, 'pattern_b': None, 'version': 'unknown'}

    # V12 (pkl.gz)
    for key, fname in [('pattern_b', 'keiba_model_v12_central_live.pkl.gz'),
                       ('pattern_a', 'keiba_model_v12_central.pkl.gz')]:
        fpath = os.path.join(BASE_DIR, fname)
        if os.path.exists(fpath):
            try:
                with gzip.open(fpath, 'rb') as f:
                    data = pickle.load(f)
                if isinstance(data, dict):
                    result[key] = data.get('auc', None)
                    result['version'] = data.get('version', 'v12')
            except Exception:
                pass

    # V9 fallback (pkl)
    if result['pattern_a'] is None:
        for key, fname in [('pattern_b', 'keiba_model_v9_central_live.pkl'),
                           ('pattern_a', 'keiba_model_v9_central.pkl')]:
            fpath = os.path.join(BASE_DIR, fname)
            if os.path.exists(fpath):
                try:
                    with open(fpath, 'rb') as f:
                        data = pickle.load(f)
                    if isinstance(data, dict):
                        result[key] = data.get('auc', None)
                        result['version'] = data.get('version', 'v9')
                except Exception:
                    pass

    return result


def _collect_data(start_date, end_date):
    """期間内のデータを収集する。"""
    start_str = start_date.strftime("%Y%m%d")
    end_str = end_date.strftime("%Y%m%d")

    # 累積結果ファイル
    cumul_path = os.path.join(BASE_DIR, "data", "cumulative_results.csv")
    if os.path.exists(cumul_path):
        df = pd.read_csv(cumul_path, encoding='utf-8-sig')
        df['date'] = df['date'].astype(str)
        return df[(df['date'] >= start_str) & (df['date'] <= end_str)]

    # 日別結果から集計
    all_results = []
    for d in range(7):
        dt = start_date + timedelta(days=d)
        dt_str = dt.strftime("%Y%m%d")
        daily_path = os.path.join(BASE_DIR, "data", "daily_results", f"{dt_str}.csv")
        if os.path.exists(daily_path):
            df_day = pd.read_csv(daily_path, encoding='utf-8-sig')
            df_day['date'] = dt_str
            all_results.append(df_day)

    return pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()


def _collect_all_cumulative():
    """全期間の累積データを収集する。"""
    cumul_path = os.path.join(BASE_DIR, "data", "cumulative_results.csv")
    if os.path.exists(cumul_path):
        return pd.read_csv(cumul_path, encoding='utf-8-sig')

    # daily_results から全データ収集
    results_dir = os.path.join(BASE_DIR, "data", "daily_results")
    if not os.path.exists(results_dir):
        return pd.DataFrame()

    all_results = []
    for fname in sorted(os.listdir(results_dir)):
        if fname.endswith('.csv'):
            fpath = os.path.join(results_dir, fname)
            try:
                df_day = pd.read_csv(fpath, encoding='utf-8-sig')
                df_day['date'] = fname.replace('.csv', '')
                all_results.append(df_day)
            except Exception:
                pass
    return pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()


def _calc_stats(df):
    """DataFrameから的中率・ROI等の統計を計算する。"""
    total = len(df)
    if total == 0:
        return {'count': 0, 'hit': 0, 'hit_rate': 0, 'investment': 0,
                'payout': 0, 'profit': 0, 'roi': 0}

    hit = 0
    if 'trio_hit' in df.columns and 'umaren_hit' in df.columns:
        hit = int(((df['trio_hit'].fillna(0) > 0) | (df['umaren_hit'].fillna(0) > 0)).sum())
    elif 'trio_hit' in df.columns:
        hit = int(df['trio_hit'].fillna(0).sum())

    inv = int(df.get('investment', pd.Series([INVESTMENT_PER_RACE] * total)).sum())
    payout = 0
    if 'trio_payout' in df.columns:
        payout += int(df['trio_payout'].fillna(0).sum())
    if 'umaren_payout' in df.columns:
        payout += int(df['umaren_payout'].fillna(0).sum())

    roi = (payout / inv * 100) if inv > 0 else 0
    return {
        'count': total, 'hit': hit,
        'hit_rate': round(hit / total * 100, 1) if total > 0 else 0,
        'investment': inv, 'payout': payout,
        'profit': payout - inv, 'roi': round(roi, 1),
    }


def _check_feature_rates():
    """最近の予測における特徴量取得率を確認する。"""
    rates = {}
    predictions_dir = os.path.join(BASE_DIR, "data", "daily_predictions")
    if not os.path.exists(predictions_dir):
        return rates

    recent_files = sorted(os.listdir(predictions_dir))[-7:]
    total_races = 0
    feature_counts = {
        'premium_training': 0,    # 調教実タイム取得
        'speed_index': 0,         # タイム指数
        'stable_comment': 0,      # 厩舎コメント
        'weather': 0,             # 気象データ
        'track_condition': 0,     # 馬場情報
        'odds': 0,                # オッズ
    }

    for fname in recent_files:
        if not fname.endswith('.json'):
            continue
        fpath = os.path.join(predictions_dir, fname)
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, list):
                for pred in data:
                    total_races += 1
                    features = pred.get('features', {})
                    if features.get('has_training', 0) > 0:
                        feature_counts['premium_training'] += 1
                    if features.get('index_max_filled', 0) > 0:
                        feature_counts['speed_index'] += 1
                    if pred.get('stable_comment_score', None) is not None:
                        feature_counts['stable_comment'] += 1
                    if features.get('temperature', 0) > 0:
                        feature_counts['weather'] += 1
                    if features.get('condition_enc', -1) >= 0:
                        feature_counts['track_condition'] += 1
                    if features.get('odds_log', 0) > 0:
                        feature_counts['odds'] += 1
        except Exception:
            pass

    if total_races > 0:
        for key in feature_counts:
            rates[key] = round(feature_counts[key] / total_races * 100, 1)
    rates['total_races'] = total_races
    return rates


def run_weekly_report(end_date_str):
    """週次レポート生成"""
    end_date = datetime.strptime(end_date_str, "%Y%m%d")
    start_date = end_date - timedelta(days=6)
    start_str = start_date.strftime("%Y%m%d")

    print(f"{'=' * 60}")
    print(f"KEIBA AI 週次レポート")
    print(f"期間: {start_date.strftime('%Y/%m/%d')} - {end_date.strftime('%Y/%m/%d')}")
    print(f"{'=' * 60}")

    # データ収集
    df = _collect_data(start_date, end_date)

    if len(df) == 0:
        print(f"\n[INFO] 対象期間のデータがありません")
        print(f"  先に daily_predict.py と daily_results.py を実行してください")
        _generate_empty_report(start_str, end_date_str)
        return

    # 確定済みのみ
    if 'status' in df.columns:
        settled = df[df['status'] == 'settled']
    else:
        settled = df

    if len(settled) == 0:
        settled = df

    # ===== 全体集計 =====
    overall = _calc_stats(settled)
    profit_sign = '+' if overall['profit'] >= 0 else ''

    print(f"\n--- 全体成績 ---")
    print(f"  レース数: {overall['count']}")
    print(f"  的中: {overall['hit']}/{overall['count']} ({overall['hit_rate']}%)")
    print(f"  投資: {overall['investment']:,}円")
    print(f"  払戻: {overall['payout']:,}円")
    print(f"  収支: {profit_sign}{overall['profit']:,}円")
    print(f"  ROI: {overall['roi']:.1f}%")

    # ===== 条件別集計 =====
    cond_stats = {}
    warnings = []

    if 'condition' in settled.columns:
        print(f"\n--- 条件別成績 ---")
        print(f"  {'条件':>4} {'N':>4} {'的中率':>10} {'ROI':>8} {'BT ROI':>8} {'保守ROI':>8} {'乖離':>8} {'判定':>6}")
        print(f"  {'-' * 66}")

        for cond in sorted(settled['condition'].unique()):
            sub = settled[settled['condition'] == cond]
            s = _calc_stats(sub)
            profile = CONDITION_PROFILES.get(cond, {})
            expected_roi = profile.get('expected_roi', 0)
            conservative_roi = profile.get('conservative_roi', 0)

            # 乖離率 = (実ROI - BT ROI) / BT ROI
            divergence = ((s['roi'] - expected_roi) / expected_roi * 100) if expected_roi > 0 else 0

            # 判定
            verdict = 'OK'
            if s['count'] < 5:
                verdict = 'N/A'
            elif s['roi'] < conservative_roi and s['count'] >= 10:
                verdict = 'WARN'
                warnings.append(f"条件{cond}: ROI {s['roi']:.1f}% < 保守的ROI {conservative_roi}%")
            elif s['count'] >= 5 and s['hit_rate'] < profile.get('expected_hit', 0) * 0.5:
                verdict = 'WARN'

            cond_stats[cond] = {
                'count': s['count'], 'hit': s['hit'],
                'hit_rate': s['hit_rate'], 'investment': s['investment'],
                'payout': s['payout'], 'roi': s['roi'],
                'expected_hit': profile.get('expected_hit', 0),
                'expected_roi': expected_roi,
                'conservative_roi': conservative_roi,
                'divergence': round(divergence, 1),
            }

            print(f"  {cond:>4} {s['count']:>4} "
                  f"{s['hit']}/{s['count']:>3} ({s['hit_rate']:>5.1f}%) "
                  f"{s['roi']:>7.1f}% {expected_roi:>7.1f}% {conservative_roi:>7.1f}% "
                  f"{divergence:>+7.1f}% {verdict:>6}")

    # ===== 累積ROI（全期間） =====
    cumul_df = _collect_all_cumulative()
    cumul_stats = _calc_stats(cumul_df) if len(cumul_df) > 0 else None
    cumul_cond_stats = {}

    if cumul_stats and cumul_stats['count'] > 0:
        print(f"\n--- 累積成績（全期間） ---")
        print(f"  レース数: {cumul_stats['count']}")
        print(f"  ROI: {cumul_stats['roi']:.1f}%")
        ps = '+' if cumul_stats['profit'] >= 0 else ''
        print(f"  収支: {ps}{cumul_stats['profit']:,}円")

        if cumul_stats['roi'] < CONSERVATIVE_ROI_OVERALL and cumul_stats['count'] >= 50:
            warnings.append(f"累積ROI {cumul_stats['roi']:.1f}% < 保守的見積り {CONSERVATIVE_ROI_OVERALL}%")

        # 条件別累積
        if 'condition' in cumul_df.columns:
            print(f"\n  {'条件':>4} {'N':>5} {'累積ROI':>10} {'保守ROI':>10} {'判定':>6}")
            print(f"  {'-' * 40}")
            for cond in sorted(cumul_df['condition'].unique()):
                sub = cumul_df[cumul_df['condition'] == cond]
                cs = _calc_stats(sub)
                profile = CONDITION_PROFILES.get(cond, {})
                c_roi = profile.get('conservative_roi', 0)
                verdict = 'OK'
                if cs['count'] >= 20 and cs['roi'] < c_roi:
                    verdict = 'WARN'
                    warnings.append(f"累積 条件{cond}: ROI {cs['roi']:.1f}% < 保守ROI {c_roi}%")
                elif cs['count'] < 20:
                    verdict = 'N/A'
                cumul_cond_stats[cond] = {'count': cs['count'], 'roi': cs['roi']}
                print(f"  {cond:>4} {cs['count']:>5} {cs['roi']:>9.1f}% {c_roi:>9.1f}% {verdict:>6}")

    # ===== バックテスト乖離モニタリング =====
    divergence_result = check_backtest_divergence(cumul_df) if len(cumul_df) > 0 else {'overall': None, 'by_condition': {}}
    if divergence_result['overall'] and divergence_result['overall']['n'] >= 10:
        ov = divergence_result['overall']
        print(f"\n--- バックテスト乖離検定 (2σ) ---")
        print(f"  全体: 実ROI {ov['live_roi']:.1f}% vs 保守的BT {ov['bt_roi']:.1f}%  {ov['msg']}")
        if ov['significant'] and ov['direction'] == 'below':
            warnings.append(f"⚠ 累積ROI {ov['live_roi']:.1f}%がBT保守見積り{ov['bt_roi']:.1f}%を統計的有意に下回る (z={ov['z_score']:.2f})")

        for cond, cv in sorted(divergence_result['by_condition'].items()):
            if cv['n'] >= 10:
                sig_mark = ' ⚠' if cv['significant'] else ''
                print(f"  条件{cond}: 実ROI {cv['live_roi']:.1f}% vs BT {cv['bt_roi']:.1f}%  z={cv['z_score']:.2f}{sig_mark}")
                if cv['significant'] and cv['direction'] == 'below':
                    warnings.append(f"⚠ 条件{cond} ROI {cv['live_roi']:.1f}%がBT {cv['bt_roi']:.1f}%を有意に下回る (z={cv['z_score']:.2f})")

    # ===== 日別集計 =====
    daily_stats = {}
    if 'date' in settled.columns:
        print(f"\n--- 日別成績 ---")
        for dt in sorted(settled['date'].unique()):
            sub = settled[settled['date'] == dt]
            s = _calc_stats(sub)
            daily_stats[dt] = s
            ps = '+' if s['profit'] >= 0 else ''
            print(f"  {dt}: {s['hit']}/{s['count']}的中 投資{s['investment']:,}円 "
                  f"払戻{s['payout']:,}円 {ps}{s['profit']:,}円 (ROI {s['roi']:.1f}%)")

    # ===== 特徴量取得率 =====
    feature_rates = _check_feature_rates()
    if feature_rates.get('total_races', 0) > 0:
        print(f"\n--- 特徴量取得率 (直近7日, {feature_rates['total_races']}レース) ---")
        for key, label in [
            ('premium_training', '調教実タイム'),
            ('speed_index', 'タイム指数'),
            ('stable_comment', '厩舎コメント'),
            ('weather', '気象データ'),
            ('track_condition', '馬場情報'),
            ('odds', 'オッズ'),
        ]:
            rate = feature_rates.get(key, 0)
            status = 'OK' if rate >= 80 else ('WARN' if rate >= 50 else 'LOW')
            print(f"  {label:<12} {rate:>6.1f}% [{status}]")

    # ===== モデル健全性 =====
    model_info = check_model_auc()
    print(f"\n--- モデル健全性チェック ---")
    print(f"  バージョン: {model_info['version']}")
    auc_ok = True
    if model_info['pattern_a'] is not None:
        auc_a = model_info['pattern_a']
        status = 'OK' if auc_a >= AUC_THRESHOLD else 'WARNING'
        if auc_a < AUC_THRESHOLD:
            auc_ok = False
            warnings.append(f"Pattern A AUC {auc_a:.4f} < 閾値 {AUC_THRESHOLD}")
        print(f"  Pattern A AUC: {auc_a:.4f} [{status}] (閾値: {AUC_THRESHOLD})")
    else:
        print(f"  Pattern A AUC: 未確認")
        auc_ok = False

    if model_info['pattern_b'] is not None:
        print(f"  Pattern B AUC: {model_info['pattern_b']:.4f} (参考値)")

    # ===== 警告サマリー =====
    # ===== BT乖離20%チェック (Discord警告) =====
    bt_divergence_alerts = []
    if cumul_stats and cumul_stats['count'] >= 30:
        overall_roi = cumul_stats['roi']
        if CONSERVATIVE_ROI_OVERALL > 0:
            div_pct = (CONSERVATIVE_ROI_OVERALL - overall_roi) / CONSERVATIVE_ROI_OVERALL
            if div_pct >= 0.20:
                bt_divergence_alerts.append(
                    f"累積ROI {overall_roi:.1f}%がBT保守的見積り{CONSERVATIVE_ROI_OVERALL}%を"
                    f"{div_pct:.0%}下回る (N={cumul_stats['count']})")
        # 条件別
        if 'condition' in cumul_df.columns:
            for cond in sorted(cumul_df['condition'].unique()):
                sub = cumul_df[cumul_df['condition'] == cond]
                cs = _calc_stats(sub)
                if cs['count'] < 20:
                    continue
                profile = CONDITION_PROFILES.get(cond, {})
                c_target = profile.get('conservative_roi', 0)
                if c_target > 0:
                    c_div = (c_target - cs['roi']) / c_target
                    if c_div >= 0.20:
                        bt_divergence_alerts.append(
                            f"条件{cond}: ROI {cs['roi']:.1f}%がBT保守{c_target}%を"
                            f"{c_div:.0%}下回る (N={cs['count']})")
    if bt_divergence_alerts:
        warnings.extend(bt_divergence_alerts)

    if warnings:
        print(f"\n{'!' * 60}")
        print(f"  警告 ({len(warnings)}件)")
        print(f"{'!' * 60}")
        for w in warnings:
            print(f"  - {w}")
    else:
        print(f"\n  全項目正常")

    # BT乖離アラートがある場合はDiscord通知
    if bt_divergence_alerts:
        try:
            from notify import send_discord
            alert_lines = [f"BT乖離アラート ({len(bt_divergence_alerts)}件)", ""]
            alert_lines.extend(bt_divergence_alerts)
            send_discord("BT乖離警告", "\n".join(alert_lines), color="red")
        except Exception:
            pass

    # ===== JSON保存 =====
    report = {
        'period': {'start': start_str, 'end': end_date_str},
        'generated_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'summary': overall,
        'condition_stats': cond_stats,
        'daily_stats': daily_stats,
        'cumulative': {
            'overall': cumul_stats,
            'by_condition': cumul_cond_stats,
        },
        'divergence': divergence_result,
        'feature_rates': feature_rates,
        'model': {
            'version': model_info['version'],
            'pattern_a_auc': model_info['pattern_a'],
            'pattern_b_auc': model_info['pattern_b'],
            'auc_above_threshold': auc_ok,
            'threshold': AUC_THRESHOLD,
        },
        'warnings': warnings,
    }

    out_dir = os.path.join(BASE_DIR, "data", "weekly_reports")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{end_date_str}_report.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nレポート保存: {out_path}")

    # ===== Markdown保存 =====
    md_dir = os.path.join(BASE_DIR, "results", "weekly")
    os.makedirs(md_dir, exist_ok=True)
    period_str = f"{start_date.strftime('%Y/%m/%d')}-{end_date.strftime('%Y/%m/%d')}"
    md = f"# KEIBA AI Weekly Report: {period_str}\n\n"
    md += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"

    md += f"## Summary\n\n"
    md += f"| Metric | Value |\n|---|---|\n"
    md += f"| Races | {overall['count']} |\n"
    md += f"| Hits | {overall['hit']}/{overall['count']} ({overall['hit_rate']}%) |\n"
    md += f"| Investment | {overall['investment']:,} JPY |\n"
    md += f"| Payout | {overall['payout']:,} JPY |\n"
    md += f"| P&L | {profit_sign}{overall['profit']:,} JPY |\n"
    md += f"| ROI | {overall['roi']:.1f}% |\n\n"

    if cond_stats:
        md += f"## Condition Breakdown\n\n"
        md += f"| Cond | N | Hit% | ROI | BT ROI | Conservative | Divergence | Status |\n"
        md += f"|---|---|---|---|---|---|---|---|\n"
        for c in sorted(cond_stats.keys()):
            s = cond_stats[c]
            verdict = 'OK'
            if s['count'] < 5:
                verdict = 'N/A'
            elif s['count'] >= 10 and s['roi'] < s['conservative_roi']:
                verdict = 'WARN'
            md += (f"| {c} | {s['count']} | {s['hit_rate']}% | {s['roi']}% | "
                   f"{s['expected_roi']}% | {s['conservative_roi']}% | "
                   f"{s['divergence']:+.1f}% | {verdict} |\n")
        md += "\n"

    if cumul_stats and cumul_stats['count'] > 0:
        md += f"## Cumulative Performance\n\n"
        md += f"- Total Races: {cumul_stats['count']}\n"
        md += f"- Cumulative ROI: {cumul_stats['roi']:.1f}%\n"
        md += f"- Conservative Target: {CONSERVATIVE_ROI_OVERALL}%\n\n"

    if daily_stats:
        md += f"## Daily Results\n\n"
        md += f"| Date | N | Hits | ROI | P&L |\n|---|---|---|---|---|\n"
        for dt in sorted(daily_stats.keys()):
            s = daily_stats[dt]
            ps2 = '+' if s['profit'] >= 0 else ''
            md += f"| {dt} | {s['count']} | {s['hit']} | {s['roi']}% | {ps2}{s['profit']:,} |\n"
        md += "\n"

    if feature_rates.get('total_races', 0) > 0:
        md += f"## Feature Acquisition Rates\n\n"
        md += f"| Feature | Rate | Status |\n|---|---|---|\n"
        for key, label in [
            ('premium_training', 'Premium Training'),
            ('speed_index', 'Speed Index'),
            ('stable_comment', 'Stable Comment'),
            ('weather', 'Weather'),
            ('track_condition', 'Track Condition'),
            ('odds', 'Odds'),
        ]:
            rate = feature_rates.get(key, 0)
            status = 'OK' if rate >= 80 else ('WARN' if rate >= 50 else 'LOW')
            md += f"| {label} | {rate:.1f}% | {status} |\n"
        md += "\n"

    if divergence_result['overall'] and divergence_result['overall']['n'] >= 10:
        md += f"## Backtest Divergence Monitor\n\n"
        ov = divergence_result['overall']
        md += f"| Scope | Live ROI | BT ROI | z-score | Status |\n|---|---|---|---|---|\n"
        ov_status = 'ALERT' if ov['significant'] and ov['direction'] == 'below' else 'OK'
        md += f"| Overall | {ov['live_roi']:.1f}% | {ov['bt_roi']:.1f}% | {ov['z_score']:.2f} | {ov_status} |\n"
        for cond, cv in sorted(divergence_result['by_condition'].items()):
            if cv['n'] >= 10:
                c_status = 'ALERT' if cv['significant'] and cv['direction'] == 'below' else 'OK'
                md += f"| Cond {cond} (N={cv['n']}) | {cv['live_roi']:.1f}% | {cv['bt_roi']:.1f}% | {cv['z_score']:.2f} | {c_status} |\n"
        md += "\n"

    md += f"## Model Health\n\n"
    md += f"- Version: {model_info['version']}\n"
    if model_info['pattern_a'] is not None:
        md += f"- Pattern A AUC: {model_info['pattern_a']:.4f} {'OK' if auc_ok else 'WARNING'}\n"
    if model_info['pattern_b'] is not None:
        md += f"- Pattern B AUC: {model_info['pattern_b']:.4f}\n"

    if warnings:
        md += f"\n## Warnings\n\n"
        for w in warnings:
            md += f"- {w}\n"

    md_path = os.path.join(md_dir, f"{end_date_str}_report.md")
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md)
    print(f"Markdown保存: {md_path}")
    print(f"{'=' * 60}")


def _generate_empty_report(start_str, end_str):
    """データなし時の空レポート生成"""
    report = {
        'period': {'start': start_str, 'end': end_str},
        'generated_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'summary': {
            'count': 0, 'hit': 0, 'hit_rate': 0,
            'investment': 0, 'payout': 0, 'profit': 0, 'roi': 0,
        },
        'condition_stats': {},
        'daily_stats': {},
        'cumulative': {'overall': None, 'by_condition': {}},
        'feature_rates': {},
        'model': check_model_auc(),
        'warnings': [],
    }
    out_dir = os.path.join(BASE_DIR, "data", "weekly_reports")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{end_str}_report.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n空レポート保存: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KEIBA AI 週次レポート")
    parser.add_argument("--date", type=str, default=None,
                        help="レポート終了日 YYYYMMDD (デフォルト: 今日)")
    args = parser.parse_args()

    if args.date:
        date_str = args.date
    else:
        date_str = datetime.now().strftime("%Y%m%d")

    try:
        datetime.strptime(date_str, "%Y%m%d")
    except ValueError:
        print(f"[ERROR] 日付形式が不正です: {date_str} (YYYYMMDD)")
        sys.exit(1)

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] weekly_report.py 開始")
    run_weekly_report(date_str)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] weekly_report.py 終了")
    try:
        from notify import send_discord
        send_discord("週次レポート完了", f"{date_str} 週次レポート生成完了", color="blue")
    except Exception:
        pass
