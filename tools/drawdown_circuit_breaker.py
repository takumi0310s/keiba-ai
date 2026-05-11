#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""E5: Drawdown circuit breaker (連敗時 / 撤退 line 接近 で 自動停止).

V15 撤退 line -50,000 円 (現 +14,140、 余裕 +64,140)。 連敗 7+ 或いは
累計 -10,000 で warning、 累計 -30,000 で halt 推奨、 -50,000 で 強制停止。

【V15 投資保護】 daily_predict / race_auto_notify から呼出可能な 確認 layer のみ。
本 script は status return、 実際 の停止は呼出側の責任。

Usage:
    # check 現状況 (cumulative_results.csv 読込)
    python tools/drawdown_circuit_breaker.py

    # JSON 出力 (schtask / hook 用)
    python tools/drawdown_circuit_breaker.py --json

    # exit code: 0=GO, 1=WARN, 2=HALT, 3=STOP
"""
import argparse
import json
import os
import sys
from datetime import datetime, timedelta

try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CUMULATIVE_PATH = os.path.join(BASE_DIR, 'data', 'cumulative_results.csv')

# Thresholds (顧客 設定 absolute 値)
THRESHOLDS = {
    'warn_cumulative': -10000,
    'halt_cumulative': -30000,
    'stop_cumulative': -50000,
    'warn_losing_streak': 7,
    'halt_losing_streak': 14,
    'stop_losing_streak': 21,
    'warn_recent_roi_window': 30,
    'warn_recent_roi_threshold': 60,   # ROI < 60% in last 30 races
    'halt_recent_roi_threshold': 40,
}


def check_status():
    import pandas as pd
    if not os.path.exists(CUMULATIVE_PATH):
        return {'status': 'UNKNOWN', 'reason': 'cumulative_results.csv not found'}

    df = pd.read_csv(CUMULATIVE_PATH, encoding='utf-8-sig')
    df = df[df['status'] == 'settled'].copy()
    if df.empty:
        return {'status': 'UNKNOWN', 'reason': 'no settled races'}

    df = df.sort_values('race_id')
    df['profit'] = pd.to_numeric(df['profit'], errors='coerce').fillna(0)
    df['investment'] = pd.to_numeric(df['investment'], errors='coerce').fillna(0)
    cumulative_pnl = float(df['profit'].sum())

    # 連敗 (profit < 0 の連続)
    losing_streak = 0
    max_streak = 0
    for p in df['profit'].iloc[::-1]:
        if p < 0:
            losing_streak += 1
            max_streak = max(max_streak, losing_streak)
        else:
            break

    # 直近 30 race ROI
    recent = df.tail(THRESHOLDS['warn_recent_roi_window'])
    if recent['investment'].sum() > 0:
        recent_roi = float(recent['profit'].sum() / recent['investment'].sum() * 100 + 100)
    else:
        recent_roi = 100.0

    # 判定
    status = 'GO'
    reasons = []

    if cumulative_pnl <= THRESHOLDS['stop_cumulative']:
        status = 'STOP'
        reasons.append(f'累計 {cumulative_pnl:.0f} <= -50,000')
    elif cumulative_pnl <= THRESHOLDS['halt_cumulative']:
        status = 'HALT'
        reasons.append(f'累計 {cumulative_pnl:.0f} <= -30,000')
    elif cumulative_pnl <= THRESHOLDS['warn_cumulative']:
        status = 'WARN'
        reasons.append(f'累計 {cumulative_pnl:.0f} <= -10,000')

    if losing_streak >= THRESHOLDS['stop_losing_streak']:
        status = max(status, 'STOP', key=lambda s: ['GO', 'WARN', 'HALT', 'STOP'].index(s))
        reasons.append(f'連敗 {losing_streak} >= 21')
    elif losing_streak >= THRESHOLDS['halt_losing_streak']:
        status = max(status, 'HALT', key=lambda s: ['GO', 'WARN', 'HALT', 'STOP'].index(s))
        reasons.append(f'連敗 {losing_streak} >= 14')
    elif losing_streak >= THRESHOLDS['warn_losing_streak']:
        status = max(status, 'WARN', key=lambda s: ['GO', 'WARN', 'HALT', 'STOP'].index(s))
        reasons.append(f'連敗 {losing_streak} >= 7')

    if recent_roi < THRESHOLDS['halt_recent_roi_threshold']:
        status = max(status, 'HALT', key=lambda s: ['GO', 'WARN', 'HALT', 'STOP'].index(s))
        reasons.append(f'直近30race ROI {recent_roi:.1f}% < 40%')
    elif recent_roi < THRESHOLDS['warn_recent_roi_threshold']:
        status = max(status, 'WARN', key=lambda s: ['GO', 'WARN', 'HALT', 'STOP'].index(s))
        reasons.append(f'直近30race ROI {recent_roi:.1f}% < 60%')

    return {
        'status': status,
        'cumulative_pnl': cumulative_pnl,
        'losing_streak_current': losing_streak,
        'losing_streak_max': max_streak,
        'recent_30_roi': recent_roi,
        'n_settled': len(df),
        'reasons': reasons,
        'thresholds': THRESHOLDS,
        'checked_at': datetime.now().isoformat(),
    }


STATUS_EXIT_CODE = {'GO': 0, 'WARN': 1, 'HALT': 2, 'STOP': 3, 'UNKNOWN': 0}


def main():
    ap = argparse.ArgumentParser(description='Drawdown circuit breaker (E5)')
    ap.add_argument('--json', action='store_true', help='JSON 出力')
    args = ap.parse_args()

    result = check_status()

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(f'[STATUS] {result["status"]}')
        for k in ['cumulative_pnl', 'losing_streak_current', 'losing_streak_max',
                  'recent_30_roi', 'n_settled']:
            if k in result:
                print(f'  {k}: {result[k]}')
        if result.get('reasons'):
            print('[Reasons]')
            for r in result['reasons']:
                print(f'  - {r}')

    return STATUS_EXIT_CODE.get(result['status'], 0)


if __name__ == '__main__':
    sys.exit(main())
