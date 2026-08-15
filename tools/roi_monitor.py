#!/usr/bin/env python
"""実運用ROI追跡・自動アラートシステム

累積結果を監視し、ROIが閾値を下回ったらDiscord警告を発信する。
daily_results.pyの後に自動実行する想定。

チェック項目:
1. 累積ROI（全体）が保守的見積り(142.6%)を下回る → 警告
2. 累積ROI（全体）が100%を下回る → 危険アラート
3. 条件別ROIが保守的見積りを下回る（N>=20） → 条件別警告
4. 直近30レースのROIが80%未満 → 短期ドローダウン警告
5. 連敗数が20を超える → 連敗警告
6. 累積ROIがBT保守的見積りを20%以上下回る → BT乖離アラート
7. 条件別ROIがBT保守的見積りを20%以上下回る（N>=20） → 条件別BT乖離

Usage:
    python tools/roi_monitor.py              # チェック実行
    python tools/roi_monitor.py --dry-run    # Discord送信せずチェックのみ
"""
import pandas as pd
import os
import sys
import json
import argparse
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

CUMUL_PATH = os.path.join(BASE_DIR, "data", "cumulative_results.csv")
ALERT_LOG_PATH = os.path.join(BASE_DIR, "data", "roi_alerts.json")

# 閾値
CONSERVATIVE_ROI_OVERALL = 142.6  # 旧v12基準 (現状と乖離した誤DANGER源、9月ゲートで改定予定)
ROI_DANGER_THRESHOLD = 100.0

# 2026-08-15 取捨確定: 9月ゲート開始まで Discord 通知停止 (ログ/JSON記録は継続)
DISCORD_NOTIFY_ENABLED = False
SHORT_TERM_WINDOW = 30
SHORT_TERM_ROI_THRESHOLD = 80.0
MAX_LOSING_STREAK = 20

CONDITION_CONSERVATIVE_ROI = {
    'A': 143.7, 'B': 165.8, 'C': 199.9,
    'D': 95.2, 'E': 82.6, 'X': 231.3,
}
CONDITION_MIN_N = 20  # 条件別警告の最小サンプル数


def load_cumulative():
    """累積結果を読み込む。"""
    if not os.path.exists(CUMUL_PATH):
        return None
    df = pd.read_csv(CUMUL_PATH, encoding='utf-8-sig')
    if 'status' in df.columns:
        df = df[df['status'] == 'settled']
    return df


def calc_roi(df):
    """ROIを計算する。"""
    if len(df) == 0:
        return 0, 0, 0, 0
    inv = int(df['investment'].sum())
    pay = int(df['trio_payout'].fillna(0).sum())
    if 'umaren_payout' in df.columns:
        pay += int(df['umaren_payout'].fillna(0).sum())
    roi = (pay / inv * 100) if inv > 0 else 0
    hits = 0
    if 'trio_hit' in df.columns and 'umaren_hit' in df.columns:
        hits = int(((df['trio_hit'].fillna(0) > 0) | (df['umaren_hit'].fillna(0) > 0)).sum())
    elif 'trio_hit' in df.columns:
        hits = int(df['trio_hit'].fillna(0).sum())
    return len(df), hits, inv, pay


def calc_losing_streak(df):
    """現在の連敗数を計算する。"""
    if len(df) == 0:
        return 0
    # Sort by date and race_id
    if 'date' in df.columns:
        df = df.sort_values(['date', 'race_id'])
    hit_series = ((df['trio_hit'].fillna(0) > 0) | (df.get('umaren_hit', pd.Series([0]*len(df))).fillna(0) > 0))
    streak = 0
    for hit in reversed(hit_series.values):
        if hit:
            break
        streak += 1
    return streak


def run_monitor(dry_run=False):
    """ROI監視を実行する。"""
    df = load_cumulative()
    if df is None or len(df) == 0:
        print("[INFO] 累積結果なし。監視スキップ。")
        return

    alerts = []
    info = []

    # 1. 全体ROI
    n, hits, inv, pay = calc_roi(df)
    roi = (pay / inv * 100) if inv > 0 else 0
    profit = pay - inv
    ps = '+' if profit >= 0 else ''

    info.append(f"累積: {n}R, {hits}/{n}的中, ROI {roi:.1f}%, {ps}{profit:,}円")

    if n >= 50 and roi < ROI_DANGER_THRESHOLD:
        alerts.append({
            'level': 'DANGER',
            'message': f'累積ROI {roi:.1f}% < 100% ({n}R). 運用停止を検討してください。',
        })
    elif n >= 30 and roi < CONSERVATIVE_ROI_OVERALL:
        alerts.append({
            'level': 'WARNING',
            'message': f'累積ROI {roi:.1f}% < 保守的見積り {CONSERVATIVE_ROI_OVERALL}% ({n}R)',
        })

    # 2. 条件別ROI
    if 'condition' in df.columns:
        for cond in sorted(df['condition'].unique()):
            sub = df[df['condition'] == cond]
            cn, ch, ci, cp = calc_roi(sub)
            c_roi = (cp / ci * 100) if ci > 0 else 0
            c_target = CONDITION_CONSERVATIVE_ROI.get(cond, 100)
            info.append(f"  {cond}: {cn}R ROI {c_roi:.1f}% (目標{c_target}%)")

            if cn >= CONDITION_MIN_N and c_roi < c_target:
                alerts.append({
                    'level': 'WARNING',
                    'message': f'条件{cond}: ROI {c_roi:.1f}% < 保守的ROI {c_target}% (N={cn})',
                })

    # 3. 短期ドローダウン（直近30R）
    if len(df) >= SHORT_TERM_WINDOW:
        recent = df.tail(SHORT_TERM_WINDOW)
        rn, rh, ri, rp = calc_roi(recent)
        r_roi = (rp / ri * 100) if ri > 0 else 0
        info.append(f"直近{SHORT_TERM_WINDOW}R: ROI {r_roi:.1f}%")

        if r_roi < SHORT_TERM_ROI_THRESHOLD:
            alerts.append({
                'level': 'WARNING',
                'message': f'直近{SHORT_TERM_WINDOW}R ROI {r_roi:.1f}% < {SHORT_TERM_ROI_THRESHOLD}% (短期ドローダウン)',
            })

    # 4. 連敗数
    streak = calc_losing_streak(df)
    if streak > 0:
        info.append(f"現在の連敗: {streak}R")
    if streak >= MAX_LOSING_STREAK:
        alerts.append({
            'level': 'WARNING',
            'message': f'連敗 {streak}R >= {MAX_LOSING_STREAK}R',
        })

    # 5. BT乖離チェック: 累積ROIがBT保守的見積りを20%以上下回る
    BT_DIVERGENCE_THRESHOLD = 0.20  # 20%
    if n >= 30 and roi > 0:
        divergence_pct = (CONSERVATIVE_ROI_OVERALL - roi) / CONSERVATIVE_ROI_OVERALL
        info.append(f"BT乖離: {divergence_pct:+.1%} (保守的{CONSERVATIVE_ROI_OVERALL}% vs 実績{roi:.1f}%)")
        if divergence_pct >= BT_DIVERGENCE_THRESHOLD:
            alerts.append({
                'level': 'DANGER' if divergence_pct >= 0.30 else 'WARNING',
                'message': f'累積ROI {roi:.1f}%がBT保守的見積り{CONSERVATIVE_ROI_OVERALL}%を'
                           f'{divergence_pct:.0%}下回る ({n}R)',
            })

    # 6. 条件別BT乖離チェック
    if 'condition' in df.columns:
        for cond in sorted(df['condition'].unique()):
            sub = df[df['condition'] == cond]
            cn, ch, ci, cp = calc_roi(sub)
            if cn < CONDITION_MIN_N:
                continue
            c_roi = (cp / ci * 100) if ci > 0 else 0
            c_target = CONDITION_CONSERVATIVE_ROI.get(cond, 100)
            if c_target > 0 and c_roi > 0:
                c_div = (c_target - c_roi) / c_target
                if c_div >= BT_DIVERGENCE_THRESHOLD:
                    alerts.append({
                        'level': 'WARNING',
                        'message': f'条件{cond}: ROI {c_roi:.1f}%がBT保守{c_target}%を'
                                   f'{c_div:.0%}下回る (N={cn})',
                    })

    # 出力
    print(f"{'=' * 60}")
    print(f"ROI Monitor - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'=' * 60}")
    for line in info:
        print(f"  {line}")

    if alerts:
        print(f"\n{'!' * 60}")
        print(f"  アラート ({len(alerts)}件)")
        print(f"{'!' * 60}")
        for a in alerts:
            icon = '!!' if a['level'] == 'DANGER' else '!'
            print(f"  [{a['level']}] {a['message']}")

        # Discord通知
        # 2026-08-15 取捨確定: 旧142.6%基準(v12保守的ROI)由来の恒常DANGERが誤報のため、
        # 9月の事前登録ゲート判定開始まで Discord 送信を停止 (console/roi_alerts.json は継続)。
        # 再開時は DISCORD_NOTIFY_ENABLED=True に戻す。基準改定はゲート判定と併せて行う。
        if DISCORD_NOTIFY_ENABLED and not dry_run:
            _send_alert_discord(alerts, info, n, roi)
    else:
        print(f"\n  全指標正常")

    # アラートログ保存
    log_entry = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_races': n,
        'roi': round(roi, 1),
        'losing_streak': streak,
        'alerts': alerts,
    }
    _save_alert_log(log_entry)

    print(f"{'=' * 60}")
    return alerts


def _send_alert_discord(alerts, info, total_races, roi):
    """Discord警告通知を送信する。"""
    try:
        from notify import send_discord
    except ImportError:
        return

    has_danger = any(a['level'] == 'DANGER' for a in alerts)
    color = 'red' if has_danger else 'yellow'
    title = f"ROI Alert: {'DANGER' if has_danger else 'WARNING'} ({len(alerts)}件)"

    lines = [f"累積: {total_races}R, ROI {roi:.1f}%", ""]
    for a in alerts:
        lines.append(f"[{a['level']}] {a['message']}")

    send_discord(title, "\n".join(lines), color=color)


def _save_alert_log(entry):
    """アラートログをJSONに追記する。"""
    log = []
    if os.path.exists(ALERT_LOG_PATH):
        try:
            with open(ALERT_LOG_PATH, 'r', encoding='utf-8') as f:
                log = json.load(f)
        except Exception:
            pass

    log.append(entry)
    # 直近100件のみ保持
    log = log[-100:]

    with open(ALERT_LOG_PATH, 'w', encoding='utf-8') as f:
        json.dump(log, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ROI監視・自動アラート')
    parser.add_argument('--dry-run', action='store_true', help='Discord送信せずチェックのみ')
    args = parser.parse_args()

    run_monitor(dry_run=args.dry_run)
