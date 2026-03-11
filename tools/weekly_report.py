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
from datetime import datetime, timedelta

# === パス設定 ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

# === 定数 ===
INVESTMENT_PER_RACE = 700
AUC_THRESHOLD = 0.78  # モデルAUC最低ライン

CONDITION_PROFILES = {
    'A': {'label': '条件A', 'desc': '8-14頭/1600m+/良~稍', 'expected_hit': 44.5, 'expected_roi': 205.3},
    'B': {'label': '条件B', 'desc': '8-14頭/1600m+/重~不良', 'expected_hit': 45.2, 'expected_roi': 236.9},
    'C': {'label': '条件C', 'desc': '15頭+/1600m+/良~稍', 'expected_hit': 33.7, 'expected_roi': 285.6},
    'D': {'label': '条件D', 'desc': '1400m以下', 'expected_hit': 27.0, 'expected_roi': 136.0},
    'E': {'label': '条件E', 'desc': '7頭以下', 'expected_hit': 53.4, 'expected_roi': 118.0},
    'X': {'label': '条件X', 'desc': '15頭+/重~不良', 'expected_hit': 35.5, 'expected_roi': 330.5},
}


def check_model_auc():
    """モデルのAUCを確認"""
    result = {'pattern_a': None, 'pattern_b': None, 'version': 'unknown'}
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


def run_weekly_report(end_date_str):
    """週次レポート生成"""
    end_date = datetime.strptime(end_date_str, "%Y%m%d")
    start_date = end_date - timedelta(days=6)
    start_str = start_date.strftime("%Y%m%d")

    print(f"{'=' * 60}")
    print(f"KEIBA AI 週次レポート")
    print(f"期間: {start_date.strftime('%Y/%m/%d')} - {end_date.strftime('%Y/%m/%d')}")
    print(f"{'=' * 60}")

    # 累積結果ファイルの読み込み
    cumul_path = os.path.join(BASE_DIR, "data", "cumulative_results.csv")
    if not os.path.exists(cumul_path):
        # 日別結果から集計を試行
        print(f"\n[INFO] 累積結果ファイルなし。日別結果を確認します。")
        all_results = []
        for d in range(7):
            dt = start_date + timedelta(days=d)
            dt_str = dt.strftime("%Y%m%d")
            daily_path = os.path.join(BASE_DIR, "data", "daily_results", f"{dt_str}.csv")
            if os.path.exists(daily_path):
                df_day = pd.read_csv(daily_path, encoding='utf-8-sig')
                df_day['date'] = dt_str
                all_results.append(df_day)

        if not all_results:
            print(f"[INFO] 対象期間のデータがありません")
            print(f"  先に daily_predict.py と daily_results.py を実行してください")
            _generate_empty_report(start_str, end_date_str)
            return

        df = pd.concat(all_results, ignore_index=True)
    else:
        df = pd.read_csv(cumul_path, encoding='utf-8-sig')
        # 期間フィルタ
        df['date'] = df['date'].astype(str)
        df = df[(df['date'] >= start_str) & (df['date'] <= end_date_str)]

    if len(df) == 0:
        print(f"[INFO] 対象期間のデータがありません")
        _generate_empty_report(start_str, end_date_str)
        return

    # 確定済みのみ
    settled = df[df.get('status', pd.Series(['settled'] * len(df))) == 'settled']
    if len(settled) == 0:
        settled = df  # statusカラムがなければ全件

    # 全体集計
    total_races = len(settled)
    hit_count = 0
    if 'trio_hit' in settled.columns:
        hit_count += int(settled['trio_hit'].fillna(0).sum())
    if 'umaren_hit' in settled.columns:
        hit_count += int(settled['umaren_hit'].fillna(0).sum())
    # trio_hitとumaren_hitの重複を避ける（同一レースで両方1になることはないが念のため）
    if 'trio_hit' in settled.columns and 'umaren_hit' in settled.columns:
        hit_count = int(((settled['trio_hit'].fillna(0) > 0) | (settled['umaren_hit'].fillna(0) > 0)).sum())

    total_inv = int(settled.get('investment', pd.Series([INVESTMENT_PER_RACE] * len(settled))).sum())
    total_payout = 0
    if 'trio_payout' in settled.columns:
        total_payout += int(settled['trio_payout'].fillna(0).sum())
    if 'umaren_payout' in settled.columns:
        total_payout += int(settled['umaren_payout'].fillna(0).sum())

    total_profit = total_payout - total_inv
    hit_rate = (hit_count / total_races * 100) if total_races > 0 else 0
    roi = (total_payout / total_inv * 100) if total_inv > 0 else 0

    print(f"\n--- 全体成績 ---")
    print(f"  レース数: {total_races}")
    print(f"  的中: {hit_count}/{total_races} ({hit_rate:.1f}%)")
    print(f"  投資: {total_inv:,}円")
    print(f"  払戻: {total_payout:,}円")
    profit_sign = '+' if total_profit >= 0 else ''
    print(f"  収支: {profit_sign}{total_profit:,}円")
    print(f"  ROI: {roi:.1f}%")

    # 条件別集計
    cond_stats = {}
    if 'condition' in settled.columns:
        for cond in sorted(settled['condition'].unique()):
            sub = settled[settled['condition'] == cond]
            c_total = len(sub)
            c_hit = 0
            if 'trio_hit' in sub.columns:
                c_hit += int(sub['trio_hit'].fillna(0).sum())
            if 'umaren_hit' in sub.columns:
                c_hit += int(sub['umaren_hit'].fillna(0).sum())
            if 'trio_hit' in sub.columns and 'umaren_hit' in sub.columns:
                c_hit = int(((sub['trio_hit'].fillna(0) > 0) | (sub['umaren_hit'].fillna(0) > 0)).sum())

            c_inv = int(sub.get('investment', pd.Series([INVESTMENT_PER_RACE] * len(sub))).sum())
            c_payout = 0
            if 'trio_payout' in sub.columns:
                c_payout += int(sub['trio_payout'].fillna(0).sum())
            if 'umaren_payout' in sub.columns:
                c_payout += int(sub['umaren_payout'].fillna(0).sum())
            c_roi = (c_payout / c_inv * 100) if c_inv > 0 else 0
            c_hit_rate = (c_hit / c_total * 100) if c_total > 0 else 0

            expected = CONDITION_PROFILES.get(cond, {})
            cond_stats[cond] = {
                'count': c_total,
                'hit': c_hit,
                'hit_rate': round(c_hit_rate, 1),
                'investment': c_inv,
                'payout': c_payout,
                'roi': round(c_roi, 1),
                'expected_hit': expected.get('expected_hit', 0),
                'expected_roi': expected.get('expected_roi', 0),
            }

        print(f"\n--- 条件別成績 ---")
        print(f"  {'条件':>4} {'N':>4} {'的中':>8} {'ROI':>8} {'期待HIT':>8} {'期待ROI':>8} {'判定':>4}")
        print(f"  {'-' * 52}")
        for cond in sorted(cond_stats.keys()):
            s = cond_stats[cond]
            # 判定: 的中率が期待値の50%未満なら警告
            verdict = 'OK'
            if s['count'] >= 5:
                if s['expected_hit'] > 0 and s['hit_rate'] < s['expected_hit'] * 0.5:
                    verdict = 'WARN'
            else:
                verdict = 'N/A'
            print(f"  {cond:>4} {s['count']:>4} {s['hit']}/{s['count']:>3} ({s['hit_rate']:>5.1f}%) {s['roi']:>7.1f}% {s['expected_hit']:>7.1f}% {s['expected_roi']:>7.1f}% {verdict:>4}")

    # 日別集計
    daily_stats = {}
    if 'date' in settled.columns:
        for dt in sorted(settled['date'].unique()):
            sub = settled[settled['date'] == dt]
            d_total = len(sub)
            d_hit = 0
            if 'trio_hit' in sub.columns and 'umaren_hit' in sub.columns:
                d_hit = int(((sub['trio_hit'].fillna(0) > 0) | (sub['umaren_hit'].fillna(0) > 0)).sum())
            elif 'trio_hit' in sub.columns:
                d_hit = int(sub['trio_hit'].fillna(0).sum())
            d_inv = int(sub.get('investment', pd.Series([INVESTMENT_PER_RACE] * len(sub))).sum())
            d_pay = 0
            if 'trio_payout' in sub.columns:
                d_pay += int(sub['trio_payout'].fillna(0).sum())
            if 'umaren_payout' in sub.columns:
                d_pay += int(sub['umaren_payout'].fillna(0).sum())
            d_roi = (d_pay / d_inv * 100) if d_inv > 0 else 0
            daily_stats[dt] = {
                'count': d_total, 'hit': d_hit,
                'investment': d_inv, 'payout': d_pay, 'roi': round(d_roi, 1),
            }

        print(f"\n--- 日別成績 ---")
        for dt in sorted(daily_stats.keys()):
            s = daily_stats[dt]
            profit = s['payout'] - s['investment']
            ps = '+' if profit >= 0 else ''
            print(f"  {dt}: {s['hit']}/{s['count']}的中 投資{s['investment']:,}円 払戻{s['payout']:,}円 {ps}{profit:,}円 (ROI {s['roi']:.1f}%)")

    # モデルAUC確認
    model_info = check_model_auc()
    print(f"\n--- モデル健全性チェック ---")
    print(f"  バージョン: {model_info['version']}")
    auc_ok = True
    if model_info['pattern_a'] is not None:
        auc_a = model_info['pattern_a']
        status = 'OK' if auc_a >= AUC_THRESHOLD else 'WARNING'
        if auc_a < AUC_THRESHOLD:
            auc_ok = False
        print(f"  Pattern A AUC: {auc_a:.4f} [{status}] (閾値: {AUC_THRESHOLD})")
    else:
        print(f"  Pattern A AUC: 未確認")
        auc_ok = False

    if model_info['pattern_b'] is not None:
        print(f"  Pattern B AUC: {model_info['pattern_b']:.4f} (参考値)")

    if not auc_ok:
        print(f"\n  [WARNING] モデルAUCが閾値{AUC_THRESHOLD}を下回っています。再学習を検討してください。")

    # JSON保存
    report = {
        'period': {'start': start_str, 'end': end_date_str},
        'generated_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'summary': {
            'total_races': total_races,
            'hit_count': hit_count,
            'hit_rate': round(hit_rate, 1),
            'total_investment': total_inv,
            'total_payout': total_payout,
            'total_profit': total_profit,
            'roi': round(roi, 1),
        },
        'condition_stats': cond_stats,
        'daily_stats': daily_stats,
        'model': {
            'version': model_info['version'],
            'pattern_a_auc': model_info['pattern_a'],
            'pattern_b_auc': model_info['pattern_b'],
            'auc_above_threshold': auc_ok,
            'threshold': AUC_THRESHOLD,
        },
    }

    out_dir = os.path.join(BASE_DIR, "data", "weekly_reports")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{end_date_str}_report.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nレポート保存: {out_path}")
    print(f"{'=' * 60}")


def _generate_empty_report(start_str, end_str):
    """データなし時の空レポート生成"""
    report = {
        'period': {'start': start_str, 'end': end_str},
        'generated_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'summary': {
            'total_races': 0, 'hit_count': 0, 'hit_rate': 0,
            'total_investment': 0, 'total_payout': 0, 'total_profit': 0, 'roi': 0,
        },
        'condition_stats': {},
        'daily_stats': {},
        'model': check_model_auc(),
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
