"""ROI 詳細分析: 条件別/月別/競馬場別

Usage:
    python tools/roi_analysis.py
    python tools/roi_analysis.py --no-discord

出力: report/roi_analysis_20260423.md
"""
import os, sys, argparse, json
from datetime import datetime
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(BASE, 'data', 'cumulative_results.csv')
OUT = os.path.join(BASE, 'report', 'roi_analysis_20260423.md')


def fmt_pct(v):
    return f'{v*100:.1f}%' if pd.notna(v) else '—'


def summarize(df, group_cols, label):
    g = df.groupby(group_cols).agg(
        n=('investment', 'count'),
        invest=('investment', 'sum'),
        payout=('actual_payout', 'sum'),
        hits_trio=('trio_hit', 'sum'),
        hits_uma=('umaren_hit', 'sum'),
    ).reset_index()
    g['profit'] = g['payout'] - g['invest']
    g['roi'] = g['payout'] / g['invest']
    return g.sort_values('n', ascending=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--no-discord', action='store_true')
    args = ap.parse_args()

    if not os.path.exists(SRC):
        print(f'NOT FOUND: {SRC}')
        sys.exit(1)

    df = pd.read_csv(SRC, encoding='utf-8-sig')
    df = df[df['status'] == 'settled'].copy()
    print(f'settled rows: {len(df)}')

    # 日付派生
    df['date_str'] = df['date'].astype(str).str[:8]
    df['date_dt'] = pd.to_datetime(df['date_str'], format='%Y%m%d', errors='coerce')
    df['ym'] = df['date_dt'].dt.strftime('%Y-%m')

    # 全体
    total_n = len(df)
    total_inv = df['investment'].sum()
    total_pay = df['actual_payout'].sum()
    total_roi = total_pay / total_inv if total_inv > 0 else 0
    total_profit = total_pay - total_inv
    hits_trio = df['trio_hit'].sum()
    hits_uma = df['umaren_hit'].sum() if 'umaren_hit' in df.columns else 0

    # 集計
    by_cond = summarize(df, ['condition'], 'condition')
    by_month = summarize(df, ['ym'], 'month')
    by_course = summarize(df, ['course'], 'course')
    # 月×条件 (直近)
    df_recent = df.sort_values('date_dt').tail(60)
    by_recent = summarize(df_recent, ['condition'], 'recent_60')

    # 異常値
    big_wins = df[df['actual_payout'] >= 5000].sort_values('actual_payout', ascending=False).head(10)
    losing_streaks = []
    cur = 0
    streak_records = []
    for _, r in df.sort_values('date_dt').iterrows():
        if r['actual_payout'] == 0:
            cur += 1
        else:
            if cur >= 5:
                streak_records.append({'end_date': r['date_str'], 'len': cur})
            cur = 0
    if cur >= 5:
        streak_records.append({'end_date': 'ongoing', 'len': cur})

    # レポート生成
    L = []
    L.append(f'# ROI 詳細分析 (4/23時点)\n\n')
    L.append(f'生成日時: {datetime.now().strftime("%Y-%m-%d %H:%M")}\n')
    L.append(f'データソース: data/cumulative_results.csv\n\n')
    L.append(f'## 全体サマリー\n\n')
    L.append(f'- 対象レース数: **{total_n}** (settled)\n')
    L.append(f'- 投資合計: ¥{int(total_inv):,}\n')
    L.append(f'- 払戻合計: ¥{int(total_pay):,}\n')
    L.append(f'- 損益: **¥{int(total_profit):+,}**\n')
    L.append(f'- ROI: **{total_roi*100:.1f}%**\n')
    L.append(f'- 三連複的中: {int(hits_trio)} 件\n')
    L.append(f'- 馬連的中: {int(hits_uma)} 件\n\n')

    L.append(f'## 条件別\n\n')
    L.append(f'| 条件 | N | 投資 | 払戻 | 損益 | ROI |\n')
    L.append(f'|------|---|------|------|------|-----|\n')
    for _, r in by_cond.iterrows():
        L.append(f'| {r["condition"]} | {int(r["n"])} | ¥{int(r["invest"]):,} | '
                 f'¥{int(r["payout"]):,} | ¥{int(r["profit"]):+,} | '
                 f'**{r["roi"]*100:.1f}%** |\n')
    L.append(f'\n')

    L.append(f'## 月別\n\n')
    L.append(f'| 月 | N | 投資 | 払戻 | 損益 | ROI |\n')
    L.append(f'|----|---|------|------|------|-----|\n')
    for _, r in by_month.sort_values('ym').iterrows():
        L.append(f'| {r["ym"]} | {int(r["n"])} | ¥{int(r["invest"]):,} | '
                 f'¥{int(r["payout"]):,} | ¥{int(r["profit"]):+,} | '
                 f'{r["roi"]*100:.1f}% |\n')
    L.append(f'\n')

    L.append(f'## 競馬場別\n\n')
    L.append(f'| 場 | N | 投資 | 払戻 | 損益 | ROI |\n')
    L.append(f'|----|---|------|------|------|-----|\n')
    for _, r in by_course.iterrows():
        L.append(f'| {r["course"]} | {int(r["n"])} | ¥{int(r["invest"]):,} | '
                 f'¥{int(r["payout"]):,} | ¥{int(r["profit"]):+,} | '
                 f'{r["roi"]*100:.1f}% |\n')
    L.append(f'\n')

    L.append(f'## 直近60レース 条件別\n\n')
    L.append(f'| 条件 | N | 投資 | 払戻 | ROI |\n')
    L.append(f'|------|---|------|------|-----|\n')
    for _, r in by_recent.iterrows():
        L.append(f'| {r["condition"]} | {int(r["n"])} | ¥{int(r["invest"]):,} | '
                 f'¥{int(r["payout"]):,} | **{r["roi"]*100:.1f}%** |\n')
    L.append(f'\n')

    L.append(f'## 大勝レース TOP10 (払戻 5,000円以上)\n\n')
    L.append(f'| 日付 | 場 | R | 条件 | 払戻 | 名 |\n')
    L.append(f'|------|----|---|------|------|----|\n')
    for _, r in big_wins.iterrows():
        L.append(f'| {r["date_str"]} | {r["course"]} | {int(r["race_num"])} | '
                 f'{r["condition"]} | ¥{int(r["actual_payout"]):,} | '
                 f'{str(r.get("race_name", ""))[:20]} |\n')
    L.append(f'\n')

    L.append(f'## 連敗 (5R以上)\n\n')
    if streak_records:
        for s in streak_records[-5:]:
            L.append(f'- {s["end_date"]} まで {s["len"]} 連敗\n')
    else:
        L.append(f'- なし\n')
    L.append(f'\n')

    # 土曜本番への示唆
    L.append(f'## 土曜本番への示唆 (4/25)\n\n')
    best = by_cond[by_cond['n'] >= 30].sort_values('roi', ascending=False).head(2)
    worst = by_cond[by_cond['n'] >= 30].sort_values('roi').head(2)
    if len(best):
        L.append(f'**おすすめ条件** (N>=30):\n')
        for _, r in best.iterrows():
            L.append(f'- 条件{r["condition"]}: ROI {r["roi"]*100:.1f}% (N={int(r["n"])})\n')
    if len(worst):
        L.append(f'\n**警戒条件** (N>=30、ROI低):\n')
        for _, r in worst.iterrows():
            L.append(f'- 条件{r["condition"]}: ROI {r["roi"]*100:.1f}% (N={int(r["n"])})\n')

    L.append(f'\n## 注記\n')
    L.append(f'- 保守的BT見積り: 142.6% (CLAUDE.md)\n')
    L.append(f'- 現在実ROI vs BT見積り: {total_roi*100:.1f}% vs 142.6% → 差分 {(total_roi - 1.426)*100:+.1f}pt\n')
    L.append(f'- N が小さい条件 (B/E/X) は統計信頼性低、継続監視\n')

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w', encoding='utf-8') as f:
        f.write(''.join(L))
    print(f'Report: {OUT}')

    # Discord
    if not args.no_discord:
        try:
            sys.path.insert(0, BASE)
            from notify import send_discord
            best_str = ', '.join([f'{r["condition"]}({r["roi"]*100:.0f}%)' for _, r in best.iterrows()]) if len(best) else '—'
            worst_str = ', '.join([f'{r["condition"]}({r["roi"]*100:.0f}%)' for _, r in worst.iterrows()]) if len(worst) else '—'
            body = (f'対象 {total_n}R / 損益 ¥{int(total_profit):+,} / ROI {total_roi*100:.1f}%\n'
                    f'おすすめ: {best_str}\n'
                    f'警戒: {worst_str}')
            send_discord('📊 ROI分析 4/23時点', body, color='blue', channel='updates')
        except Exception as e:
            print(f'Discord失敗: {e}')


if __name__ == '__main__':
    main()
