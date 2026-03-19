"""
月次レポート生成スクリプト
指定月の実運用成績を集計し、条件別・競馬場別・クラス別の詳細分析を行う。

Usage:
    python tools/monthly_report.py                   # 先月分
    python tools/monthly_report.py --month 202603     # 指定月
"""
import pandas as pd
import numpy as np
import os
import sys
import json
import argparse
import pickle
from datetime import datetime, timedelta
from calendar import monthrange

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

INVESTMENT_PER_RACE = 700

CONDITION_PROFILES = {
    'A': {'label': '8-14頭/1600m+/良~稍', 'expected_roi': 205.3, 'expected_hit': 44.5},
    'B': {'label': '8-14頭/1600m+/重~不良', 'expected_roi': 236.9, 'expected_hit': 45.2},
    'C': {'label': '15頭+/1600m+/良~稍', 'expected_roi': 285.6, 'expected_hit': 33.7},
    'D': {'label': '1200-1400m', 'expected_roi': 136.0, 'expected_hit': 27.0},
    'E': {'label': '7頭以下', 'expected_roi': 118.0, 'expected_hit': 53.4},
    'X': {'label': '15頭+/重~不良', 'expected_roi': 330.5, 'expected_hit': 35.5},
}

COURSE_NAMES = ['札幌', '函館', '福島', '新潟', '東京', '中山', '中京', '京都', '阪神', '小倉']


def load_month_data(year_month):
    """指定月のデータを読み込み"""
    year = int(year_month[:4])
    month = int(year_month[4:6])
    _, last_day = monthrange(year, month)
    start_str = f"{year_month}01"
    end_str = f"{year_month}{last_day:02d}"

    # 累積結果から読み込み
    cumul_path = os.path.join(BASE_DIR, "data", "cumulative_results.csv")
    if os.path.exists(cumul_path):
        df = pd.read_csv(cumul_path, encoding='utf-8-sig')
        df['date'] = df['date'].astype(str)
        df = df[(df['date'] >= start_str) & (df['date'] <= end_str)]
        if len(df) > 0:
            return df

    # 日別結果から集計
    all_results = []
    for day in range(1, last_day + 1):
        dt_str = f"{year_month}{day:02d}"
        daily_path = os.path.join(BASE_DIR, "data", "daily_results", f"{dt_str}.csv")
        if os.path.exists(daily_path):
            df_day = pd.read_csv(daily_path, encoding='utf-8-sig')
            df_day['date'] = dt_str
            all_results.append(df_day)

    if all_results:
        return pd.concat(all_results, ignore_index=True)
    return pd.DataFrame()


def calc_stats(sub):
    """DataFrameから集計統計を計算"""
    n = len(sub)
    if n == 0:
        return {'count': 0, 'hit': 0, 'hit_rate': 0, 'investment': 0, 'payout': 0, 'roi': 0, 'profit': 0}

    hit = 0
    if 'trio_hit' in sub.columns and 'umaren_hit' in sub.columns:
        hit = int(((sub['trio_hit'].fillna(0) > 0) | (sub['umaren_hit'].fillna(0) > 0)).sum())
    elif 'trio_hit' in sub.columns:
        hit = int(sub['trio_hit'].fillna(0).sum())

    inv = int(sub.get('investment', pd.Series([INVESTMENT_PER_RACE] * n)).sum())
    payout = 0
    for col in ['trio_payout', 'umaren_payout']:
        if col in sub.columns:
            payout += int(sub[col].fillna(0).sum())

    return {
        'count': n,
        'hit': hit,
        'hit_rate': round(hit / n * 100, 1) if n > 0 else 0,
        'investment': inv,
        'payout': payout,
        'roi': round(payout / inv * 100, 1) if inv > 0 else 0,
        'profit': payout - inv,
    }


def generate_monthly_report(year_month):
    """月次レポート生成"""
    year = int(year_month[:4])
    month = int(year_month[4:6])
    month_label = f"{year}年{month}月"

    print(f"{'=' * 60}")
    print(f"KEIBA AI 月次レポート: {month_label}")
    print(f"{'=' * 60}")

    df = load_month_data(year_month)
    if len(df) == 0:
        print("[INFO] 対象期間のデータがありません")
        _save_empty_report(year_month, month_label)
        return

    # === 全体サマリー ===
    summary = calc_stats(df)
    print(f"\n--- 全体成績 ---")
    print(f"  レース数: {summary['count']}")
    print(f"  的中: {summary['hit']}/{summary['count']} ({summary['hit_rate']}%)")
    print(f"  投資: {summary['investment']:,}円")
    print(f"  払戻: {summary['payout']:,}円")
    ps = '+' if summary['profit'] >= 0 else ''
    print(f"  収支: {ps}{summary['profit']:,}円 (ROI {summary['roi']}%)")

    # === 条件別 ===
    cond_stats = {}
    if 'condition' in df.columns:
        print(f"\n--- 条件別成績 ---")
        print(f"  {'条件':>4} {'N':>4} {'的中率':>8} {'ROI':>8} {'収支':>12} {'期待ROI':>8}")
        for cond in sorted(df['condition'].unique()):
            sub = df[df['condition'] == cond]
            s = calc_stats(sub)
            cond_stats[cond] = s
            exp = CONDITION_PROFILES.get(cond, {})
            ps = '+' if s['profit'] >= 0 else ''
            print(f"  {cond:>4} {s['count']:>4} {s['hit_rate']:>7.1f}% {s['roi']:>7.1f}% {ps}{s['profit']:>10,}円 {exp.get('expected_roi', 0):>7.1f}%")

    # === 競馬場別 ===
    course_stats = {}
    if 'course' in df.columns:
        print(f"\n--- 競馬場別成績 ---")
        for course in sorted(df['course'].unique()):
            sub = df[df['course'] == course]
            s = calc_stats(sub)
            course_stats[course] = s
            ps = '+' if s['profit'] >= 0 else ''
            print(f"  {course:<6} N={s['count']:>3} 的中{s['hit_rate']:>5.1f}% ROI {s['roi']:>6.1f}% {ps}{s['profit']:>8,}円")

    # === 日別推移 ===
    daily_stats = {}
    if 'date' in df.columns:
        print(f"\n--- 日別成績 ---")
        for dt in sorted(df['date'].unique()):
            sub = df[df['date'] == dt]
            s = calc_stats(sub)
            daily_stats[dt] = s
            ps = '+' if s['profit'] >= 0 else ''
            print(f"  {dt}: {s['hit']}/{s['count']}的中 {ps}{s['profit']:>8,}円 (ROI {s['roi']:>6.1f}%)")

    # === 最高配当・連敗分析 ===
    max_payout = 0
    max_payout_race = ''
    max_consecutive_loss = 0
    current_loss = 0

    for _, row in df.iterrows():
        hit = False
        p = 0
        if 'trio_hit' in df.columns and row.get('trio_hit', 0) > 0:
            hit = True
            p = row.get('trio_payout', 0)
        if 'umaren_hit' in df.columns and row.get('umaren_hit', 0) > 0:
            hit = True
            p = max(p, row.get('umaren_payout', 0))

        if p > max_payout:
            max_payout = int(p)
            max_payout_race = f"{row.get('race_name', '')} ({row.get('date', '')})"

        if hit:
            current_loss = 0
        else:
            current_loss += 1
            max_consecutive_loss = max(max_consecutive_loss, current_loss)

    print(f"\n--- 特記事項 ---")
    print(f"  最高配当: {max_payout:,}円 {max_payout_race}")
    print(f"  最大連敗: {max_consecutive_loss}レース")

    # === グラフ生成 ===
    chart_path = None
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.font_manager as fm

        # 日本語フォント設定
        for font_name in ['MS Gothic', 'Yu Gothic', 'Meiryo', 'IPAGothic']:
            if any(font_name in f.name for f in fm.fontManager.ttflist):
                plt.rcParams['font.family'] = font_name
                break

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'KEIBA AI {month_label} Monthly Report', fontsize=14, fontweight='bold')

        # 1. 日別累計収支
        if daily_stats:
            dates = sorted(daily_stats.keys())
            cum_profit = []
            running = 0
            for d in dates:
                running += daily_stats[d]['profit']
                cum_profit.append(running)
            ax = axes[0, 0]
            ax.plot(range(len(dates)), cum_profit, 'b-o', markersize=4)
            ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
            ax.set_title('Cumulative P&L')
            ax.set_ylabel('Profit (JPY)')
            ax.set_xticks(range(len(dates)))
            ax.set_xticklabels([d[-2:] for d in dates], fontsize=7)
            ax.fill_between(range(len(dates)), cum_profit, 0,
                          where=[p >= 0 for p in cum_profit], alpha=0.2, color='green')
            ax.fill_between(range(len(dates)), cum_profit, 0,
                          where=[p < 0 for p in cum_profit], alpha=0.2, color='red')

        # 2. 条件別ROI
        if cond_stats:
            conds = sorted(cond_stats.keys())
            rois = [cond_stats[c]['roi'] for c in conds]
            expected = [CONDITION_PROFILES.get(c, {}).get('expected_roi', 0) for c in conds]
            ax = axes[0, 1]
            x = range(len(conds))
            ax.bar([i - 0.15 for i in x], rois, 0.3, label='Actual', color='steelblue')
            ax.bar([i + 0.15 for i in x], expected, 0.3, label='Expected', color='lightcoral', alpha=0.6)
            ax.axhline(y=100, color='r', linestyle='--', alpha=0.5)
            ax.set_xticks(list(x))
            ax.set_xticklabels(conds)
            ax.set_title('ROI by Condition')
            ax.set_ylabel('ROI (%)')
            ax.legend(fontsize=8)

        # 3. 競馬場別的中率
        if course_stats:
            courses = sorted(course_stats.keys())
            hit_rates = [course_stats[c]['hit_rate'] for c in courses]
            counts = [course_stats[c]['count'] for c in courses]
            ax = axes[1, 0]
            bars = ax.bar(range(len(courses)), hit_rates, color='teal')
            ax.set_xticks(range(len(courses)))
            ax.set_xticklabels(courses, fontsize=8, rotation=30)
            ax.set_title('Hit Rate by Course')
            ax.set_ylabel('Hit Rate (%)')
            for i, (bar, n) in enumerate(zip(bars, counts)):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                       f'N={n}', ha='center', fontsize=7)

        # 4. 日別ROI
        if daily_stats:
            dates = sorted(daily_stats.keys())
            daily_rois = [daily_stats[d]['roi'] for d in dates]
            ax = axes[1, 1]
            colors = ['green' if r >= 100 else 'red' for r in daily_rois]
            ax.bar(range(len(dates)), daily_rois, color=colors, alpha=0.7)
            ax.axhline(y=100, color='gray', linestyle='--', alpha=0.5)
            ax.set_xticks(range(len(dates)))
            ax.set_xticklabels([d[-2:] for d in dates], fontsize=7)
            ax.set_title('Daily ROI')
            ax.set_ylabel('ROI (%)')

        plt.tight_layout()
        chart_dir = os.path.join(BASE_DIR, 'results', 'monthly')
        os.makedirs(chart_dir, exist_ok=True)
        chart_path = os.path.join(chart_dir, f'{year_month}_chart.png')
        plt.savefig(chart_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\n  Chart saved: {chart_path}")
    except Exception as e:
        print(f"\n  [WARN] Chart generation failed: {e}")

    # === Markdown保存 ===
    md_dir = os.path.join(BASE_DIR, 'results', 'monthly')
    os.makedirs(md_dir, exist_ok=True)
    md_path = os.path.join(md_dir, f'{year_month}_report.md')

    md = f"# KEIBA AI {month_label} Monthly Report\n\n"
    md += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
    md += f"## Summary\n\n"
    md += f"| Metric | Value |\n|---|---|\n"
    md += f"| Races | {summary['count']} |\n"
    md += f"| Hits | {summary['hit']}/{summary['count']} ({summary['hit_rate']}%) |\n"
    md += f"| Investment | {summary['investment']:,} JPY |\n"
    md += f"| Payout | {summary['payout']:,} JPY |\n"
    md += f"| P&L | {ps}{summary['profit']:,} JPY |\n"
    md += f"| ROI | {summary['roi']}% |\n"
    md += f"| Max Payout | {max_payout:,} JPY |\n"
    md += f"| Max Consecutive Loss | {max_consecutive_loss} races |\n\n"

    if cond_stats:
        md += f"## Condition Breakdown\n\n"
        md += f"| Cond | N | Hit% | ROI | P&L | Expected ROI |\n|---|---|---|---|---|---|\n"
        for c in sorted(cond_stats.keys()):
            s = cond_stats[c]
            exp = CONDITION_PROFILES.get(c, {}).get('expected_roi', 0)
            ps2 = '+' if s['profit'] >= 0 else ''
            md += f"| {c} | {s['count']} | {s['hit_rate']}% | {s['roi']}% | {ps2}{s['profit']:,} | {exp}% |\n"
        md += "\n"

    if course_stats:
        md += f"## Course Breakdown\n\n"
        md += f"| Course | N | Hit% | ROI | P&L |\n|---|---|---|---|---|\n"
        for c in sorted(course_stats.keys()):
            s = course_stats[c]
            ps2 = '+' if s['profit'] >= 0 else ''
            md += f"| {c} | {s['count']} | {s['hit_rate']}% | {s['roi']}% | {ps2}{s['profit']:,} |\n"
        md += "\n"

    if chart_path and os.path.exists(chart_path):
        md += f"## Charts\n\n![Monthly Chart]({os.path.basename(chart_path)})\n\n"

    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md)
    print(f"  Markdown saved: {md_path}")

    # === JSON保存 ===
    report = {
        'period': year_month,
        'month_label': month_label,
        'generated_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'summary': summary,
        'condition_stats': cond_stats,
        'course_stats': course_stats,
        'daily_stats': daily_stats,
        'max_payout': max_payout,
        'max_payout_race': max_payout_race,
        'max_consecutive_loss': max_consecutive_loss,
        'chart_path': chart_path,
    }

    json_path = os.path.join(md_dir, f'{year_month}_report.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  JSON saved: {json_path}")
    print(f"{'=' * 60}")


def _save_empty_report(year_month, month_label):
    md_dir = os.path.join(BASE_DIR, 'results', 'monthly')
    os.makedirs(md_dir, exist_ok=True)
    report = {
        'period': year_month, 'month_label': month_label,
        'generated_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'summary': {'count': 0, 'hit': 0, 'hit_rate': 0, 'investment': 0, 'payout': 0, 'roi': 0, 'profit': 0},
    }
    json_path = os.path.join(md_dir, f'{year_month}_report.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  Empty report: {json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KEIBA AI Monthly Report")
    parser.add_argument("--month", type=str, default=None,
                        help="YYYYMM (default: last month)")
    args = parser.parse_args()

    if args.month:
        ym = args.month
    else:
        today = datetime.now()
        last_month = today.replace(day=1) - timedelta(days=1)
        ym = last_month.strftime("%Y%m")

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] monthly_report.py start")
    generate_monthly_report(ym)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] monthly_report.py done")
