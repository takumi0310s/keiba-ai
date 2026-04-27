#!/usr/bin/env python
"""GW期間中のシステム監視ダッシュボード

使い方:
    python tools/gw_monitor.py

確認項目:
1. タスクスケジューラー全件のステータス
2. 直近の予測ファイル
3. 直近の結果照合ファイル
4. 戦略⑦ 適用状況
"""
import os
import sys
import json
import subprocess
import pandas as pd
from datetime import datetime, timedelta

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def check_tasks():
    """タスクスケジューラー確認"""
    print("=" * 60)
    print("[1] タスクスケジューラー")
    print("=" * 60)
    result = subprocess.run(
        ['schtasks', '/query', '/v', '/fo', 'list'],
        capture_output=True, text=True, encoding='cp932', errors='ignore'
    )
    if result.returncode != 0:
        print("[ERR] schtasks 実行失敗")
        return
    
    keiba_tasks = []
    name = ''
    last_result = ''
    next_run = ''
    last_run = ''
    
    for line in result.stdout.split('\n'):
        if 'TaskName:' in line and ('Keiba' in line or 'keiba' in line):
            name = line.split(':')[1].strip() if ':' in line else ''
        elif name:
            if 'Last Run Time:' in line:
                last_run = ':'.join(line.split(':')[1:]).strip()
            elif 'Last Result:' in line:
                last_result = line.split(':')[1].strip()
            elif 'Next Run Time:' in line:
                next_run = ':'.join(line.split(':')[1:]).strip()
            elif 'Status:' in line:
                status = line.split(':')[1].strip()
                keiba_tasks.append((name, status, last_run, last_result, next_run))
                name = ''
                last_run = ''
                last_result = ''
                next_run = ''
    
    for n, s, lr, lc, nr in keiba_tasks:
        marker = '[OK]' if lc == '0' else '[!!]'
        print(f"{marker} {n[:35]:35s} {s:10s} last_result={lc:6s} next={nr[:20]}")


def check_predictions():
    """直近の予測ファイル確認"""
    print("\n" + "=" * 60)
    print("[2] 直近の予測ファイル")
    print("=" * 60)
    pred_dir = os.path.join(BASE_DIR, 'data', 'daily_predictions')
    if not os.path.exists(pred_dir):
        print("[ERR] daily_predictions ディレクトリなし")
        return
    
    files = sorted([f for f in os.listdir(pred_dir) if f.endswith('.csv')])[-5:]
    for f in files:
        path = os.path.join(pred_dir, f)
        size = os.path.getsize(path)
        df = pd.read_csv(path)
        print(f"  {f}: {len(df)}行, {size}B")


def check_results():
    """直近の結果照合確認"""
    print("\n" + "=" * 60)
    print("[3] 直近の結果照合")
    print("=" * 60)
    res_dir = os.path.join(BASE_DIR, 'data', 'daily_results')
    if not os.path.exists(res_dir):
        print("[ERR] daily_results ディレクトリなし")
        return
    
    files = sorted([f for f in os.listdir(res_dir) if f.endswith('.csv')])[-5:]
    for f in files:
        path = os.path.join(res_dir, f)
        df = pd.read_csv(path)
        if 'actual_payout' in df.columns:
            paid = (df['actual_payout'] > 0).sum()
            roi = df['actual_payout'].sum() / df['investment'].sum() * 100 if df['investment'].sum() > 0 else 0
            print(f"  {f}: {len(df)}R, 的中{paid}, ROI {roi:.1f}%")


def check_strategy7():
    """戦略⑦ 実装確認"""
    print("\n" + "=" * 60)
    print("[4] 戦略⑦ 実装状況")
    print("=" * 60)
    code_path = os.path.join(BASE_DIR, 'tools', 'race_auto_notify.py')
    with open(code_path, encoding='utf-8') as f:
        content = f.read()
    
    count = content.count('STRATEGY7')
    if count == 4:
        print(f"  [OK] STRATEGY7: 4/4 実装")
    else:
        print(f"  [!!] STRATEGY7: {count}/4 実装")


def check_jrdb():
    """JRDB データ最新性"""
    print("\n" + "=" * 60)
    print("[5] JRDB データ最新性")
    print("=" * 60)
    files = ['jrdb_paci.csv', 'jrdb_kyi.csv', 'jrdb_sed.csv', 'jrdb_tyb.csv', 'jrdb_kab.csv']
    for f in files:
        path = os.path.join(BASE_DIR, 'data', f)
        if os.path.exists(path):
            mtime = os.path.getmtime(path)
            dt = datetime.fromtimestamp(mtime)
            age_hours = (datetime.now() - dt).total_seconds() / 3600
            marker = '[OK]' if age_hours < 24*3 else '[!!]' if age_hours < 24*7 else '[WARN]'
            size_mb = os.path.getsize(path) / 1024 / 1024
            print(f"  {marker} {f}: {dt:%Y-%m-%d %H:%M}, {age_hours:.1f}時間前, {size_mb:.1f}MB")


def main():
    print("\n" + "★" * 30)
    print(f"  GW Monitor Dashboard")
    print(f"  実行時刻: {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("★" * 30)
    
    check_tasks()
    check_predictions()
    check_results()
    check_strategy7()
    check_jrdb()
    
    print("\n" + "★" * 30)
    print("  ダッシュボード表示完了")
    print("★" * 30)


if __name__ == '__main__':
    main()
