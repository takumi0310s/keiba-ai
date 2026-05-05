"""5/9 (土) 当日 自動 レポート生成 + Discord 通知.

usage:
  python tools/race_day_report.py [--date YYYYMMDD]

生成内容:
  - data/results/<DATE>_summary.md
  - Discord #updates / #bets に通知

集計:
  - 採用 R (案B改 = 12R 1勝クラスのみ) の ROI / hit
  - 全 34 R の参考 ROI (採用しなかった分)
  - V15 軸 top3 率 (BT 57% 比較)
  - trio 7点 hit 率 (BT 22% 比較)
  - 累計収支更新

schtasks 登録案 (admin 必要):
  Keiba-RaceDayReport_Sat   土曜 18:00 daily
  Keiba-RaceDayReport_Sun   日曜 18:00 daily
"""
from __future__ import annotations

import os, sys, argparse, subprocess
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import pandas as pd
import numpy as np
from datetime import datetime

BASE = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
os.chdir(BASE)


def is_1sho_class(race_name):
    return '1勝' in str(race_name or '')


def derive_metrics(df, label='全'):
    """top3 hit 率, trio hit 率, ROI を計算"""
    n = len(df)
    if n == 0:
        return {'label': label, 'n': 0, 'top3_rate': 0, 'trio_hit_rate': 0,
                'inv': 0, 'pay': 0, 'profit': 0, 'roi': 0}
    # top3 = top1 馬が finish 1-3 着内 (top1_finish <= 3)
    top3_hits = (pd.to_numeric(df['top1_finish'], errors='coerce') <= 3).sum()
    trio_hits = pd.to_numeric(df['trio_hit'], errors='coerce').fillna(0).sum()
    inv = pd.to_numeric(df['investment'], errors='coerce').fillna(0).sum()
    pay = pd.to_numeric(df['actual_payout'], errors='coerce').fillna(0).sum()
    profit = pd.to_numeric(df['profit'], errors='coerce').fillna(0).sum()
    return {
        'label': label, 'n': int(n),
        'top3_rate': float(top3_hits / n) if n else 0,
        'trio_hit_rate': float(trio_hits / n) if n else 0,
        'inv': float(inv), 'pay': float(pay), 'profit': float(profit),
        'roi': float(pay / inv * 100) if inv > 0 else 0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', default=None, help='YYYYMMDD (default 今日)')
    parser.add_argument('--no-discord', action='store_true')
    args = parser.parse_args()

    date = args.date or datetime.now().strftime('%Y%m%d')
    print(f"=== race_day_report {date} ===")

    cum_path = 'data/cumulative_results.csv'
    if not os.path.exists(cum_path):
        print(f"[ERROR] {cum_path} not found"); sys.exit(1)
    df = pd.read_csv(cum_path, dtype={'date':str, 'race_id':str})

    df_today = df[df['date'] == date].copy()
    if len(df_today) == 0:
        print(f"[WARN] {date} 該当なし (まだ DailyResults 未実行?)")
        sys.exit(0)

    # --- 採用 R (案B改): race_num=12 かつ '1勝' in race_name かつ investment > 0 ---
    df_today['race_num_int'] = pd.to_numeric(df_today['race_num'], errors='coerce').fillna(0).astype(int)
    df_adopted = df_today[
        (df_today['race_num_int'] == 12) &
        (df_today['race_name'].apply(is_1sho_class)) &
        (pd.to_numeric(df_today['investment'], errors='coerce').fillna(0) > 0)
    ]
    df_all = df_today

    metrics_adopted = derive_metrics(df_adopted, label='採用 (案B改 12R 1勝)')
    metrics_all = derive_metrics(df_all, label='全 R 参考')
    # 11R only (重賞 G) で除外したものの参考 ROI
    df_11r = df_today[df_today['race_num_int'] == 11]
    metrics_11r = derive_metrics(df_11r, label='11R 参考 (案B改 除外)')
    df_12r_other = df_today[(df_today['race_num_int'] == 12) & ~df_today['race_name'].apply(is_1sho_class)]
    metrics_12r_other = derive_metrics(df_12r_other, label='12R 1勝以外 参考')

    # --- 累計 ---
    cum_inv = pd.to_numeric(df['investment'], errors='coerce').fillna(0).sum()
    cum_pay = pd.to_numeric(df['actual_payout'], errors='coerce').fillna(0).sum()
    cum_profit = cum_pay - cum_inv
    cum_roi = cum_pay / cum_inv * 100 if cum_inv > 0 else 0

    # --- 5/9 ROI 判定 ---
    roi = metrics_adopted['roi']
    profit_today = metrics_adopted['profit']
    if metrics_adopted['n'] == 0:
        verdict = 'NO_BETS (採用 0 R、累計 維持)'
    elif roi >= 100:
        verdict = '✅ 100%+ 翌日 同戦略 継続'
    elif roi >= 50:
        verdict = '⚠️ 50-99% 翌日 控えめ運用 (1-2R 限定)'
    elif roi > 0:
        verdict = '🔴 < 50% 翌日 投資停止'
    else:
        verdict = '🔴 0% 即停止 + 原因究明'

    # --- markdown 出力 ---
    # 既存 手書き summary を上書きしないよう、_auto suffix を付ける
    # (手書き summary がない日は通常の <date>_summary.md でも OK だが、安全側)
    out_md = f'data/results/{date}_summary_auto.md'
    if not os.path.exists(f'data/results/{date}_summary.md'):
        # 手書き不在 → 通常名で OK
        out_md = f'data/results/{date}_summary.md'
    lines = [
        f'# {date} 当日 サマリー (race_day_report 自動生成)',
        '',
        f'生成: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
        '',
        '## 1. 採用 R (案B改 12R 1勝クラスのみ)',
        '',
        '| 項目 | 値 |',
        '|------|----|',
        f'| 採用 R 数 | {metrics_adopted["n"]} R |',
        f'| 投資総額 | {int(metrics_adopted["inv"]):,} 円 |',
        f'| 払戻総額 | {int(metrics_adopted["pay"]):,} 円 |',
        f'| **収支** | **{int(profit_today):+,} 円** |',
        f'| **ROI** | **{roi:.1f}%** |',
        f'| top3 率 (top1 馬 of 採用 R) | {metrics_adopted["top3_rate"]*100:.1f}% (BT 57%) |',
        f'| trio 7点 hit 率 | {metrics_adopted["trio_hit_rate"]*100:.1f}% (BT 22%) |',
        f'| 翌日 判定 | {verdict} |',
        '',
        '## 2. 参考: 採用しなかった R',
        '',
        '| 区分 | n | top3 率 | trio hit | inv | pay | profit | ROI |',
        '|------|---:|------:|------:|----:|----:|------:|----:|',
    ]
    for m in [metrics_11r, metrics_12r_other, metrics_all]:
        lines.append(f'| {m["label"]} | {m["n"]} | {m["top3_rate"]*100:.1f}% | {m["trio_hit_rate"]*100:.1f}% | {int(m["inv"]):,} | {int(m["pay"]):,} | {int(m["profit"]):+,} | {m["roi"]:.1f}% |')
    lines += [
        '',
        '## 3. 累計収支 (4/12 〜)',
        '',
        '| 項目 | 値 |',
        '|------|----|',
        f'| 累計投資 | {int(cum_inv):,} 円 |',
        f'| 累計払戻 | {int(cum_pay):,} 円 |',
        f'| **累計収支** | **{int(cum_profit):+,} 円** |',
        f'| 累計 ROI | {cum_roi:.1f}% |',
        f'| 撤退ライン (-50,000円) まで余裕 | {50000 + int(cum_profit):,} 円 |',
        '',
        '## 4. 翌日アクション',
        '',
        f'- {verdict}',
        '- 詳細: data/v18/risk_management_5_9.md',
        '',
        '## 5. 採用 R 詳細',
        '',
    ]
    if metrics_adopted['n'] > 0:
        lines.append('| 場 | R | race_name | top1 | top1_finish | trio_result | hit | profit |')
        lines.append('|----|--:|-----------|-----:|------------:|------------:|----:|------:|')
        for _, r in df_adopted.iterrows():
            lines.append(f'| {r.get("course","")} | {r.get("race_num","")} | {str(r.get("race_name",""))[:25]} | {r.get("top1_num","")} | {r.get("top1_finish","")} | {r.get("trio_result","")} | {int(pd.to_numeric(r.get("trio_hit",0), errors="coerce") or 0)} | {int(pd.to_numeric(r.get("profit",0), errors="coerce") or 0):+,} |')
    else:
        lines.append('採用 R なし (5/9 無投資)')
    with open(out_md, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"[OK] {out_md}")

    # --- Discord 通知 ---
    if not args.no_discord:
        title = f"{date} 当日サマリー (案B改)"
        msg = f"""採用 {metrics_adopted['n']} R / 投資 {int(metrics_adopted['inv']):,} 円 / 払戻 {int(metrics_adopted['pay']):,} 円
**収支 {int(profit_today):+,} 円  ROI {roi:.1f}%**
top3 率 {metrics_adopted['top3_rate']*100:.0f}% / trio hit {metrics_adopted['trio_hit_rate']*100:.0f}%

判定: {verdict}

累計 (4/12〜): {int(cum_profit):+,} 円 (撤退ライン まで {50000 + int(cum_profit):,} 円)
詳細: {out_md}"""
        color = 'red' if (metrics_adopted['n'] > 0 and roi < 50) else None
        try:
            cmd = ['python', 'tools/notify_done.py', title, msg]
            if color: cmd += ['--color', color]
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            print(f"[Discord] rc={r.returncode}, {r.stdout[:200]}")
        except Exception as e:
            print(f"[Discord ERR] {e}")


if __name__ == '__main__':
    main()
