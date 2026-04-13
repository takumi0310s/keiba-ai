#!/usr/bin/env python
"""v16 再評価 — スクレイピング完了後に1コマンドで再学習・WF評価

前提:
  - scrape_missing_all.py が完了し、upset/training_eval/master_index が
    data_health_check.py のトリガー閾値を満たしている
  - _v15_train_df_cache.pkl は存在または再構築可能

ワークフロー:
  1. data_health_check.py でトリガー閾値チェック
  2. _v15_train_df_cache.pkl を強制再構築 (新データ反映のため)
  3. train/features_v16_premium.py の5特徴量を再マージ
  4. 4モデル WF 実行
  5. 結果 data/v16_wf_results_retrain.json に保存
  6. adopted なら v16 本番 pkl.gz を保存 (TBD)

Usage:
    nohup python -u train/retrain_v16.py > logs/retrain_v16.log 2>&1 &

採用基準: WF AUC > 0.8858
"""
from __future__ import annotations

import os
import sys
import json
import time
import subprocess
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, 'train'))
sys.path.insert(0, os.path.join(BASE_DIR, 'tools'))

CACHE_PATH = os.path.join(DATA_DIR, '_v15_train_df_cache.pkl')


def step_health_check():
    print("\n[1] data_health_check.py")
    import importlib
    hc = importlib.import_module('data_health_check')
    report = hc.run()
    print(hc.render_report(report))
    if not report.get('v16_reeval_ready'):
        print("\n[WARN] v16 再評価トリガー閾値未達")
        triggers = report.get('v16_reeval_triggers', {})
        for fn, t in triggers.items():
            if not t['met']:
                print(f"  [NG] {fn}: {t['coverage_pct']}% < {t['threshold_pct']}%")
        ans = input("強制実行しますか? [y/N]: ").strip().lower()
        if ans != 'y':
            sys.exit(1)


def step_rebuild_cache():
    print("\n[2] cache 強制再構築")
    if os.path.exists(CACHE_PATH):
        bak = CACHE_PATH + f".bak_{datetime.now():%Y%m%d_%H%M%S}"
        os.rename(CACHE_PATH, bak)
        print(f"  既存cache退避: {bak}")


def step_run_wf():
    print("\n[3] WF 実行 (run_v16_and_am8_wf.py --skip-ablation)")
    cmd = [sys.executable, '-u', 'train/run_v16_and_am8_wf.py', '--skip-ablation']
    env = os.environ.copy()
    env.setdefault('PYTHONUNBUFFERED', '1')
    env.setdefault('PYTHONIOENCODING', 'utf-8')
    proc = subprocess.run(cmd, cwd=BASE_DIR, env=env)
    return proc.returncode == 0


def step_check_result():
    print("\n[4] 結果確認")
    p = os.path.join(DATA_DIR, 'v16_wf_results.json')
    if not os.path.exists(p):
        print("  結果JSON未生成")
        return False, None
    with open(p, 'r', encoding='utf-8') as f:
        r = json.load(f)
    all_in = r.get('v16_all_in', {})
    mean = all_in.get('mean_auc', 0)
    adopted = all_in.get('adopted_vs_user_target', False)
    print(f"  mean_auc = {mean:.4f}")
    print(f"  adopted  = {adopted}")
    return adopted, r


def step_save_production(result):
    """採用時の本番モデル保存 (TBD - 既存v15パイプライン活用)"""
    print("\n[5] 本番モデル保存")
    print("  [TODO] 既存 train_v15_master.save_production_model を呼び出す実装")
    print("  現状は scaffold のみ。採用確定後に追加実装")


def notify(adopted, mean):
    try:
        from notify import send_discord
        color = 'green' if adopted else 'yellow'
        title = f"v16 retrain {'ADOPTED' if adopted else 'REJECTED'}"
        send_discord(title, f"mean AUC = {mean:.4f}", color=color, channel='updates')
    except Exception as e:
        print(f"[WARN] notify failed: {e}")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--force', action='store_true', help='トリガー未達でも実行')
    ap.add_argument('--no-rebuild', action='store_true', help='cache再構築しない')
    args = ap.parse_args()

    t0 = time.time()
    print(f"=== retrain_v16 start {datetime.now()} ===")

    if not args.force:
        step_health_check()

    if not args.no_rebuild:
        step_rebuild_cache()

    ok = step_run_wf()
    if not ok:
        print("[ERROR] WF 実行失敗")
        sys.exit(1)

    adopted, r = step_check_result()

    if adopted and r is not None:
        step_save_production(r)
    else:
        print("\n[INFO] 不採用。v15 を維持")

    elapsed = (time.time() - t0) / 60
    mean = (r.get('v16_all_in', {}).get('mean_auc', 0)) if r else 0
    print(f"\n=== retrain_v16 done {elapsed:.1f}min ===")
    notify(adopted, mean)


if __name__ == '__main__':
    main()
