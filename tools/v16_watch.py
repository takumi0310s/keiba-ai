#!/usr/bin/env python
"""v16 学習の進捗監視ダッシュボード

使い方: python tools/v16_watch.py
"""
import os
import sys
import glob
import time
import subprocess
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def check_process():
    """v16 プロセス確認"""
    try:
        result = subprocess.run(
            ['ps', 'aux'],
            capture_output=True, text=True
        )
        v16_lines = [l for l in result.stdout.split('\n') if 'v16' in l and 'grep' not in l]
        return v16_lines
    except:
        return []


def latest_log():
    """最新のログファイル"""
    logs = sorted(glob.glob('logs/v16_wf_*.log') + glob.glob('logs/retrain_v16_*.log'))
    return logs[-1] if logs else None


def main():
    print("=" * 60)
    print(f"  v16 Learning Watch")
    print(f"  時刻: {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 60)
    
    # プロセス確認
    print("\n[1] プロセス状態")
    procs = check_process()
    if procs:
        for p in procs[:3]:
            print(f"  {p[:120]}")
    else:
        print("  プロセス未検出 (完了 or 未起動)")
    
    # 最新ログ
    log_file = latest_log()
    if log_file:
        print(f"\n[2] 最新ログ: {log_file}")
        size = os.path.getsize(log_file)
        mtime = os.path.getmtime(log_file)
        age = (datetime.now().timestamp() - mtime) / 60
        print(f"  サイズ: {size/1024:.1f}KB")
        print(f"  最終更新: {age:.1f}分前")
        
        # ログ末尾
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        lines = content.split('\n')
        print(f"\n[3] ログ末尾 (直近20行)")
        for l in lines[-20:]:
            print(f"  {l}")
    
    # 結果ファイル
    print(f"\n[4] 結果ファイル")
    results_files = [
        'data/v16_wf_results.json',
        'data/v16_wf_results_retrain.json',
    ]
    for rf in results_files:
        path = os.path.join(BASE_DIR, rf)
        if os.path.exists(path):
            mtime = os.path.getmtime(path)
            print(f"  [EXISTS] {rf} ({datetime.fromtimestamp(mtime):%Y-%m-%d %H:%M})")
            # 内容確認
            try:
                import json
                with open(path) as f:
                    data = json.load(f)
                if 'v16_auc' in data:
                    print(f"    v16 AUC: {data['v16_auc']}")
                if 'baseline_auc' in data:
                    print(f"    baseline: {data['baseline_auc']}")
                if 'adopted' in data:
                    print(f"    採用: {data['adopted']}")
            except:
                pass
        else:
            print(f"  [PENDING] {rf}")
    
    # キャッシュ状態
    print(f"\n[5] キャッシュ状態")
    caches = [
        ('data/_v15_train_df_cache.pkl', '訓練データキャッシュ'),
        ('data/_v15_optuna_df_cache.pkl.gz', 'Optuna キャッシュ'),
    ]
    for path, desc in caches:
        full = os.path.join(BASE_DIR, path)
        if os.path.exists(full):
            size = os.path.getsize(full) / 1024 / 1024
            mtime = os.path.getmtime(full)
            age = (datetime.now().timestamp() - mtime) / 60
            print(f"  [EXISTS] {desc}: {size:.1f}MB, {age:.0f}分前")
        else:
            print(f"  [MISSING] {desc} (再構築待ち)")


if __name__ == '__main__':
    main()
