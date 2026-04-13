#!/usr/bin/env python
"""AM8:00 予測精度バックテスト

目的: AM8時点で取得可能な特徴量（カテゴリA+B）のみでWF AUCを計算し、
     全150特徴量（A+B+C+D）のAUC 0.8858との差分を定量化する。

data/feature_availability_am8.json からA+B特徴量リストを読み込み、
v15_masterのWFパイプラインをそのまま流用して4モデルアンサンブルを評価する。

Usage:
    python -u train/wf_am8_backtest.py                  # A+B のみ
    python -u train/wf_am8_backtest.py --mode baseline  # 全特徴量（ベース比較用）
    python -u train/wf_am8_backtest.py --mode diff      # 両方走らせて差分レポート

実行時間: ~4-6時間（WF 5年 × 4モデル）
"""
import os
import sys
import json
import argparse
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, 'train'))


def load_am8_features():
    """A+Bカテゴリの特徴量名リストを取得。"""
    with open(os.path.join(DATA_DIR, 'feature_availability_am8.json'), 'r', encoding='utf-8') as f:
        d = json.load(f)
    return d['A']['features'] + d['B']['features']


def run_wf(features, label):
    """train_v15_masterのWFパイプラインを流用してWF AUC計算。"""
    # v15_masterの内部関数とデータロードを流用
    from train_v15_master import walk_forward_4model, WF_YEARS
    import train_v15_master as tm

    # build_all_features に相当するロード（tm.buildなどに依存）
    # ここでは最小限: jra_races_full をロード→既存データ構築→与えられたfeaturesで評価
    print(f"\n=== WF AM8 Backtest: {label} ===")
    print(f"  features: {len(features)}")

    # データロード（v15_masterのデータ構築関数を利用）
    # 注: v15_master は build 関数が多く依存するので、ここでは実行前提として
    #     既存の学習結果キャッシュ（data/v15_feature_df_cache.pkl 等）があればそれを使う
    cache_path = os.path.join(DATA_DIR, '_v15_train_df_cache.pkl')
    if os.path.exists(cache_path):
        print(f"  load cache: {cache_path}")
        df = pd.read_pickle(cache_path)
    else:
        print(f"  cache not found → train_v15_masterのdata構築を呼ぶ必要あり")
        print(f"  注: v15_masterのbuild関数を直接呼ぶには追加実装が必要")
        return None

    # 不要な列チェック
    missing = [f for f in features if f not in df.columns]
    if missing:
        print(f"  [WARN] dfに存在しない特徴量 {len(missing)}個: {missing[:5]}...")
        features = [f for f in features if f in df.columns]

    # WF評価
    results = walk_forward_4model(df, features, years=WF_YEARS, label=label)

    # 集計
    aucs = [r['auc_grid'] for r in results if r.get('auc_grid')]
    mean_auc = float(np.mean(aucs)) if aucs else 0.0
    print(f"\n=== {label}: mean WF AUC = {mean_auc:.4f} (n={len(aucs)} years) ===")
    return {'label': label, 'mean_auc': mean_auc, 'per_year': results, 'n_features': len(features)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['am8', 'baseline', 'diff'], default='am8')
    args = ap.parse_args()

    am8_feats = load_am8_features()
    print(f"A+B特徴量 (AM8使用可能): {len(am8_feats)}個")

    out = {}
    if args.mode in ('am8', 'diff'):
        r = run_wf(am8_feats, 'AM8 (A+B only)')
        if r:
            out['am8'] = r

    if args.mode in ('baseline', 'diff'):
        # baselineは v15_masterの既存結果JSON を参照する方が速い
        base_path = os.path.join(DATA_DIR, 'v15_master_report.json')
        if os.path.exists(base_path):
            with open(base_path, 'r', encoding='utf-8') as f:
                base = json.load(f)
            out['baseline'] = {'label': 'v15 full 150', 'mean_auc': base.get('best_auc', 0.8858),
                               'source': 'v15_master_report.json'}
            print(f"  baseline: WF AUC = {out['baseline']['mean_auc']:.4f} (from {base_path})")

    if args.mode == 'diff' and 'am8' in out and 'baseline' in out:
        diff = out['am8']['mean_auc'] - out['baseline']['mean_auc']
        print(f"\n=== 差分 ===")
        print(f"  AM8:      {out['am8']['mean_auc']:.4f}")
        print(f"  baseline: {out['baseline']['mean_auc']:.4f}")
        print(f"  diff:     {diff:+.4f}")
        out['diff'] = diff

    out_path = os.path.join(DATA_DIR, 'wf_am8_backtest_results.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n結果保存: {out_path}")


if __name__ == '__main__':
    main()
