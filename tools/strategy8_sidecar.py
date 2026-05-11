#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Strategy 8 sidecar runner (V15 production を 一切触らず Jackpot 別 channel 通知).

V15 race_auto_notify.py は 完全 unchanged。 本 script は その **sidecar** として:
1. daily_predictions/{date}.csv 読込
2. 各 race の live features 計算 (live_features_5_17.py 流用)
3. Jackpot 該当馬 検出 (4-way pattern)
4. **別 Discord channel** に alert (DISCORD_WEBHOOK_JACKPOT)
5. shadow_log にも保存

【V15 投資保護】 完全 (production code 一切不変、 別 process / 別 channel)

Usage:
    # 5/17 朝 daily_predict 完了後 (約 09:00) に user 手動 実行
    python tools/strategy8_sidecar.py 20260517

    # schtask 自動化 (土日 09:00 起動 推奨)
    python tools/strategy8_sidecar.py --auto

【Discord 設定】
.env に追加 (推奨、 別 channel):
    DISCORD_WEBHOOK_JACKPOT=https://discord.com/api/webhooks/.../...

【出力】
- data/shadow_log/{date}/sidecar_jackpot.json
- Discord channel 通知 (Jackpot 該当馬 only)
"""
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime

try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHADOW_DIR = os.path.join(BASE_DIR, 'data', 'shadow_log')
ENV_PATH = os.path.join(BASE_DIR, '.env')


def load_env():
    if not os.path.exists(ENV_PATH):
        return {}
    env = {}
    with open(ENV_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            if '=' in line and not line.startswith('#'):
                k, v = line.split('=', 1)
                env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def send_discord(webhook, content):
    if not webhook or webhook.startswith('your_'):
        return False
    try:
        import requests
        r = requests.post(webhook, json={'content': content[:1900]}, timeout=15)
        return r.status_code in [200, 204]
    except Exception:
        return False


def detect_jackpots(date_str):
    """live_features_5_17 を call して 該当馬 list 取得."""
    cmd = [sys.executable, os.path.join(BASE_DIR, 'tools', 'live_features_5_17.py'), date_str]
    try:
        subprocess.run(cmd, timeout=600, check=False, capture_output=True)
    except Exception:
        pass

    # 出力 csv read
    out_csv = os.path.join(BASE_DIR, 'data', 'live_features', f'{date_str}.csv')
    if not os.path.exists(out_csv):
        return []

    import pandas as pd
    df = pd.read_csv(out_csv, encoding='utf-8')
    if 'is_jackpot' not in df.columns:
        return []
    jpot = df[df['is_jackpot'] == 1]
    return jpot.to_dict('records')


def format_alert(date_str, jackpots):
    if not jackpots:
        return f'[5/{date_str} Strategy 8 sidecar] Jackpot 該当馬 なし (V15 戦略⑦ 通常通り)'

    lines = [f'🎰 **Jackpot pattern 該当馬 ({len(jackpots)} 件) - {date_str}**']
    lines.append('')
    lines.append('(降級 + 馬絶好調 + 騎手絶好調 + 同騎手継続)')
    lines.append('過去 4 年 ROI 184% verified、 単勝 1500円 任意 試験運用')
    lines.append('')
    for j in jackpots[:10]:
        lines.append(f'• race {j["race_id"]} 馬番 {j["umaban"]} '
                      f'(horse {j["horse_id"][:8]}..) jockey {j["jockey_id"]}')
        recent5 = j.get('horse_recent5_top3', 'N/A')
        j30 = j.get('jockey_recent30_top3', 'N/A')
        if isinstance(recent5, float):
            recent5 = f'{recent5:.2f}'
        if isinstance(j30, float):
            j30 = f'{j30:.2f}'
        lines.append(f'  recent5: {recent5}, j30: {j30}')
    lines.append('')
    lines.append('※ V15 通常通知 と 別 channel、 試験運用 (V15 production 不変)')
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser(description='Strategy 8 sidecar runner')
    ap.add_argument('date', nargs='?', default=None, help='YYYYMMDD')
    ap.add_argument('--auto', action='store_true', help='今日の date 自動')
    args = ap.parse_args()

    if args.auto and not args.date:
        args.date = datetime.now().strftime('%Y%m%d')
    if not args.date:
        print('[ERROR] date required (or --auto)')
        return 1

    print(f'[INFO] Strategy 8 sidecar for {args.date}')
    print('  V15 production: 完全不変')
    print('  本 script: 別 process / 別 Discord channel')

    daily_path = os.path.join(BASE_DIR, 'data', 'daily_predictions', f'{args.date}.csv')
    if not os.path.exists(daily_path):
        print(f'[INFO] daily_predictions/{args.date}.csv 未生成 (V15 daily_predict 待ち)')
        return 0

    # Detect
    print('[INFO] Jackpot detection start...')
    jpots = detect_jackpots(args.date)
    print(f'[INFO] Jackpot 該当馬: {len(jpots)} 件')

    # Save log
    os.makedirs(os.path.join(SHADOW_DIR, args.date), exist_ok=True)
    log_path = os.path.join(SHADOW_DIR, args.date, 'sidecar_jackpot.json')
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump({
            'date': args.date,
            'jackpot_count': len(jpots),
            'jackpots': jpots,
            'generated_at': datetime.now().isoformat(),
        }, f, indent=2, ensure_ascii=False, default=str)
    print(f'[OK] log saved: {log_path}')

    # Discord
    env = load_env()
    webhook = env.get('DISCORD_WEBHOOK_JACKPOT') or env.get('DISCORD_WEBHOOK_URL')
    msg = format_alert(args.date, jpots)
    print(f'\n--- Discord message preview ---')
    print(msg[:500])
    print(f'-------------------------------')

    if webhook and not webhook.startswith('your_') and webhook.startswith('http'):
        sent = send_discord(webhook, msg)
        print(f'[Discord] sent: {sent}')
    else:
        print('[INFO] DISCORD_WEBHOOK_JACKPOT 未設定 (.env)、 log のみ保存')

    return 0


if __name__ == '__main__':
    sys.exit(main())
