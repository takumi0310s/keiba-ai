#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""5/17 開催 朝 briefing generator: V15 + Phase 22-24 status を 1 通の Discord 通知 化.

morning_go_check (Phase 21A 既存) は 10 項目 V15 production focus。 本 script は
Phase 22-24 追加分 を含めた 統合 briefing。

【通知内容】
- 累計収支 (Drawdown Breaker)
- V15 cookie / health check status
- Phase 23 shadow log 件数
- paddock 蓄積数
- YouTube schtask 稼働確認
- 30y backtest 進捗
- 推奨 投資額 (Kelly suggest)

Usage:
    python tools/morning_briefing_5_17.py
    python tools/morning_briefing_5_17.py --json
    python tools/morning_briefing_5_17.py --discord    # Discord 送信
"""
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime

try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_breaker_status():
    try:
        r = subprocess.run([sys.executable, os.path.join(BASE_DIR, 'tools',
                                                          'drawdown_circuit_breaker.py'),
                              '--json'], capture_output=True, text=True, timeout=20,
                            encoding='utf-8', errors='replace')
        return json.loads(r.stdout)
    except Exception as e:
        return {'status': 'ERROR', 'msg': str(e)}


def get_video_health():
    try:
        r = subprocess.run([sys.executable, os.path.join(BASE_DIR, 'tools',
                                                          'check_video_sources.py'),
                              '--json'], capture_output=True, text=True, timeout=20,
                            encoding='utf-8', errors='replace')
        return json.loads(r.stdout)
    except Exception as e:
        return {'summary': {'OK': 0, 'WARN': 0, 'NG': 0, 'MISSING': 0}, 'error': str(e)}


def count_paddock_frames():
    paddock = os.path.join(BASE_DIR, 'data', 'paddock_frames')
    if not os.path.exists(paddock): return 0, 0
    dirs = [d for d in os.listdir(paddock) if os.path.isdir(os.path.join(paddock, d))]
    total_frames = 0
    for d in dirs:
        full = os.path.join(paddock, d)
        for f in os.listdir(full):
            if f.endswith('.jpg'):
                total_frames += 1
    return len(dirs), total_frames


def count_shadow_logs():
    sh = os.path.join(BASE_DIR, 'data', 'shadow_log')
    if not os.path.exists(sh): return 0
    total = 0
    for root, dirs, files in os.walk(sh):
        total += sum(1 for f in files if f.endswith('.json'))
    return total


def check_schtask(name):
    try:
        r = subprocess.run(['schtasks', '/query', '/tn', name, '/fo', 'list'],
                           capture_output=True, text=True, timeout=10,
                           encoding='cp932', errors='replace')
        return r.returncode == 0
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser(description='5/17 morning briefing')
    ap.add_argument('--json', action='store_true')
    ap.add_argument('--discord', action='store_true')
    args = ap.parse_args()

    breaker = get_breaker_status()
    health = get_video_health()
    n_paddock_dirs, n_paddock_frames = count_paddock_frames()
    n_shadow = count_shadow_logs()
    yt_sat = check_schtask('Keiba-YouTubeLiveRecord-Sat')
    yt_sun = check_schtask('Keiba-YouTubeLiveRecord-Sun')
    paddock_sun = check_schtask('Keiba-PaddockArchive-Sun')
    paddock_mon = check_schtask('Keiba-PaddockArchive-Mon')

    briefing = {
        'date': datetime.now().strftime('%Y-%m-%d'),
        'breaker_status': breaker.get('status'),
        'cumulative_pnl': breaker.get('cumulative_pnl'),
        'losing_streak': breaker.get('losing_streak_current'),
        'recent_30_roi': breaker.get('recent_30_roi'),
        'health_summary': health.get('summary'),
        'paddock_archive': {'dirs': n_paddock_dirs, 'total_frames': n_paddock_frames},
        'shadow_logs': n_shadow,
        'schtasks': {
            'YouTube-Sat': yt_sat,
            'YouTube-Sun': yt_sun,
            'Paddock-Sun': paddock_sun,
            'Paddock-Mon': paddock_mon,
        },
        'generated_at': datetime.now().isoformat(),
    }

    if args.json:
        print(json.dumps(briefing, indent=2, ensure_ascii=False))
    else:
        print('=== 5/17 開催 朝 briefing ===\n')
        print(f'[累計] 撤退保護 status = {briefing["breaker_status"]}')
        print(f'  累計 PnL: {briefing["cumulative_pnl"]} 円')
        print(f'  連敗 (現在): {briefing["losing_streak"]}')
        print(f'  直近 30 race ROI: {briefing["recent_30_roi"]}%')
        print()
        h = briefing['health_summary']
        print(f'[health] OK={h.get("OK")}, WARN={h.get("WARN")}, NG={h.get("NG")}, MISSING={h.get("MISSING")}')
        print()
        print(f'[paddock archive] {briefing["paddock_archive"]["dirs"]} dirs, '
              f'{briefing["paddock_archive"]["total_frames"]} frames')
        print(f'[shadow logs] {briefing["shadow_logs"]} 件')
        print()
        print('[schtasks]')
        for k, v in briefing['schtasks'].items():
            print(f'  {k:<20}: {"✓ registered" if v else "✗ not_registered"}')
        print()
        # 推奨 invest
        breaker_status = briefing['breaker_status']
        if breaker_status in ('STOP',):
            print('[判定] ⛔ STOP — 撤退')
        elif breaker_status in ('HALT',):
            print('[判定] 🟠 HALT — 投資 停止 推奨')
        elif breaker_status == 'WARN':
            print('[判定] ⚠️ WARN — V15 戦略⑦ 単独 継続、 投資額 抑制 推奨')
        else:
            print('[判定] ✅ GO — V15 案 B 改 + 戦略⑦ 単独 GO')

    if args.discord:
        # Discord 通知
        try:
            from tools.notify_done import send_message
        except ImportError:
            pass
        try:
            cmd = [sys.executable, os.path.join(BASE_DIR, 'tools', 'notify_done.py'),
                   f'5/17 朝 briefing',
                   f'breaker={briefing["breaker_status"]}, '
                   f'PnL={briefing["cumulative_pnl"]}, '
                   f'health=OK{h.get("OK")}/WARN{h.get("WARN")}, '
                   f'paddock={briefing["paddock_archive"]["dirs"]}race/{briefing["paddock_archive"]["total_frames"]}frame, '
                   f'shadow={briefing["shadow_logs"]}log']
            subprocess.run(cmd, timeout=15)
            print('\n[Discord] 通知送信完了')
        except Exception as e:
            print(f'\n[Discord ERROR] {e}')

    return 0


if __name__ == '__main__':
    sys.exit(main())
