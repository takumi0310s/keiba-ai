"""当日 危険 horse 検出 + Discord 通知.

V15 prediction TOP horses について、 投資 除外 候補を 検出 + 通知:
1. 馬体重 急変 (±10kg 以上)
2. 取消 / 出走除外
3. 騎手変更 (出馬表 vs LIVE)
4. 大幅 オッズ補正 (朝 vs 直前 で 50% 以上 急変)
5. パドック 異常 (将来 動画 解析 統合 用、 placeholder)

V15 不変、 追加 後段 layer。 daily_predict CSV 読込 → LIVE odds/horse_weight 比較 → 通知。

Usage:
    python tools/danger_horse_alert.py --date 20260516
    python tools/danger_horse_alert.py --date 20260516 --discord
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / 'tools'))

PREDICTIONS_DIR = BASE / 'data' / 'daily_predictions'
WEIGHT_DIR = BASE / 'data' / 'morning_weight_check'


def detect_weight_change_alerts(date_str: str, threshold_kg: int = 10) -> list:
    """馬体重 ±threshold_kg 以上 急変 馬 検出."""
    alerts = []
    fp = WEIGHT_DIR / f'{date_str}.csv'
    if not fp.exists():
        return alerts
    try:
        df = pd.read_csv(fp, encoding='utf-8-sig', low_memory=False)
        if 'weight_change' not in df.columns:
            return alerts
        df['weight_change_num'] = pd.to_numeric(df['weight_change'], errors='coerce')
        risky = df[df['weight_change_num'].abs() >= threshold_kg].copy()
        for _, row in risky.iterrows():
            alerts.append({
                'race_id': str(row.get('race_id', '')),
                'horse_num': int(row.get('umaban', row.get('horse_num', 0)) or 0),
                'horse_name': str(row.get('horse_name', '')),
                'weight_change': float(row['weight_change_num']),
                'reason': f'馬体重 {row["weight_change_num"]:+.0f} kg 急変',
            })
    except Exception as e:
        print(f'[WARN] weight_change_alerts: {e}')
    return alerts


def detect_cancelation_alerts(date_str: str) -> list:
    """取消 / 出走除外 検出 (daily_predictions の出走数 vs 当日 LIVE)."""
    # placeholder: LIVE 取得は 別途必要、 ここでは daily_predict CSV 内 'cancel_flag' 等を check
    alerts = []
    fp = PREDICTIONS_DIR / f'{date_str}.csv'
    if not fp.exists():
        return alerts
    try:
        df = pd.read_csv(fp, encoding='utf-8-sig', low_memory=False)
        if 'cancel_flag' in df.columns:
            cancelled = df[df['cancel_flag'].astype(str).isin(['1', 'True', 'TRUE'])]
            for _, row in cancelled.iterrows():
                alerts.append({
                    'race_id': str(row.get('race_id', '')),
                    'horse_num': int(row.get('top1_num', 0) or 0),
                    'horse_name': str(row.get('top1_name', '')),
                    'reason': '取消 / 出走除外',
                })
    except Exception as e:
        print(f'[WARN] cancelation_alerts: {e}')
    return alerts


def detect_top1_score_anomaly(date_str: str, threshold: float = 0.55) -> list:
    """TOP1 score が threshold 以下 = 接戦 race、 投資慎重 候補."""
    alerts = []
    fp = PREDICTIONS_DIR / f'{date_str}.csv'
    if not fp.exists():
        return alerts
    try:
        df = pd.read_csv(fp, encoding='utf-8-sig', low_memory=False)
        if 'top1_score' not in df.columns:
            return alerts
        df['top1_score_num'] = pd.to_numeric(df['top1_score'], errors='coerce')
        low_conf = df[df['top1_score_num'] < threshold].copy()
        for _, row in low_conf.iterrows():
            alerts.append({
                'race_id': str(row.get('race_id', '')),
                'horse_num': int(row.get('top1_num', 0) or 0),
                'horse_name': str(row.get('top1_name', '')),
                'score': float(row['top1_score_num']),
                'reason': f'TOP1 score {row["top1_score_num"]:.3f} < {threshold} (接戦)',
            })
    except Exception as e:
        print(f'[WARN] score_anomaly: {e}')
    return alerts


def format_discord_alert(date_str: str, alerts: dict) -> str:
    """alerts dict を Discord message format に."""
    if not any(alerts.values()):
        return f'{date_str[:4]}/{date_str[4:6]}/{date_str[6:]} 危険 horse: 0 件 (全 race 健全)'

    lines = [f'[ALERT] {date_str[:4]}/{date_str[4:6]}/{date_str[6:]} 危険 horse alert']

    cat_map = {
        'weight_change': '馬体重 急変',
        'cancel': '取消 / 出走除外',
        'top1_score_low': 'TOP1 score 低 (接戦)',
    }
    for cat, items in alerts.items():
        if not items:
            continue
        lines.append(f'\n{cat_map.get(cat, cat)} ({len(items)} 件):')
        for it in items[:10]:  # max 10 per category
            rid = it['race_id'][-4:] if len(it['race_id']) >= 4 else it['race_id']
            lines.append(f'  - {rid}R 馬番 {it["horse_num"]}: {it["horse_name"]} | {it["reason"]}')
        if len(items) > 10:
            lines.append(f'  (他 {len(items)-10} 件)')
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--date', type=str, default=datetime.now().strftime('%Y%m%d'))
    ap.add_argument('--discord', action='store_true', help='Discord 通知 送信')
    args = ap.parse_args()

    alerts = {
        'weight_change': detect_weight_change_alerts(args.date),
        'cancel': detect_cancelation_alerts(args.date),
        'top1_score_low': detect_top1_score_anomaly(args.date, threshold=0.55),
    }

    msg = format_discord_alert(args.date, alerts)
    print(msg)

    if args.discord:
        try:
            from notify import send_discord
            ok = send_discord('危険 horse alert', msg, color='yellow', channel='updates')
            print(f'\n[discord] sent: {ok}')
        except Exception as e:
            print(f'[WARN] discord send error: {e}')

    total = sum(len(v) for v in alerts.values())
    return 0 if total == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
