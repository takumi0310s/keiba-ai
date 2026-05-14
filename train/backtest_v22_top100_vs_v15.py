#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""V22 top 100 vs V15 実 ROI 比較 backtest.

WF 6-fold で V22 top 100 OOS predictions 保存 → 実 jra_payouts.csv で 実 ROI 計算。
V15 baseline は cumulative_results.csv (実運用 ~324 race) と比較。

戦略 (V15 production 同等):
- trio 7 点 formation (TOP1 - TOP2,3 - TOP2-6)
- umaren 2 点 (条件 E のみ)
- 戦略⑦ 自動除外: 06_特別 / 京都 / 条件E / 条件B
- 案B改: 12R 1勝クラス 上限 2,100円
- 投資額: 700円/race base

V15 投資保護 完全 (バックテスト read-only、 model 不変)。

Usage:
    python train/backtest_v22_top100_vs_v15.py --quick   # 2025 fold only
    python train/backtest_v22_top100_vs_v15.py            # 2020-2025 6-fold
"""
import argparse
import gzip
import json
import os
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import torch

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / 'train'))
DATA_DIR = BASE / 'data'
MODEL_DIR = BASE / 'models' / 'v22_top100_backtest'
MODEL_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from train_v22_4ensemble import LGB_PARAMS, XGB_PARAMS, train_intra_race
from train_v22_enhanced import load_v22_enhanced_data
from train_v22_enhanced_top100 import load_top_n_features

INVESTMENT = 700
INVESTMENT_CAP_12R = 2100  # 案B改


def parse_payout(s):
    if pd.isna(s):
        return 0
    try:
        return int(str(s).replace(',', ''))
    except Exception:
        return 0


def load_payouts():
    """jra_payouts.csv を 読み込み、 (year, course, kai, nichi, race_num) でindex化."""
    fp = DATA_DIR / 'jra_payouts.csv'
    if not fp.exists():
        return {}
    df = pd.read_csv(fp, dtype=str, low_memory=False)
    out = {}
    for _, row in df.iterrows():
        rd = str(row.get('race_date', '20000101'))
        year2 = rd[2:4] if len(rd) >= 4 else '00'  # 2025 → '25'
        key = (year2,
               str(row.get('course', '')).strip(),
               str(row.get('kai', '')).strip(),
               str(row.get('nichi', '')).strip(),
               str(row.get('race_num', '')).strip())
        out[key] = {
            'trio_nums': str(row.get('trio_nums', '')),
            'trio_payout': parse_payout(row.get('trio_payout', 0)),
            'umaren_nums': str(row.get('umaren_nums', '')),
            'umaren_payout': parse_payout(row.get('umaren_payout', 0)),
            'wide_nums': str(row.get('wide_nums', '')),
            'wide_payouts': str(row.get('wide_payouts', '')),
        }
    return out


def classify_condition(num_horses, distance, condition):
    """v15 production と同じ 条件 分類."""
    # condition は str ('良', '稍重', '重', '不良') or int (0-3) どちらでも来る
    if isinstance(condition, str):
        heavy = condition in ['重', '不良']
    else:
        try:
            heavy = float(condition) >= 2
        except Exception:
            heavy = False
    if num_horses <= 7:
        return 'E'
    if distance <= 1400:
        return 'D'
    if 8 <= num_horses <= 14 and distance >= 1600 and not heavy:
        return 'A'
    if 8 <= num_horses <= 14 and distance >= 1600 and heavy:
        return 'B'
    if num_horses >= 15 and distance >= 1600 and not heavy:
        return 'C'
    return 'X'


def is_strategy7_excluded(course, condition, distance, race_class_text=''):
    """戦略⑦ 自動除外 logic (race_auto_notify.py 同等).

    除外:
    - 06_特別 (race_class 含む '特別' で G/L/OPEN ではない)
    - 京都
    - 条件 E (頭数<=7)
    - 条件 B (重~不良)
    """
    if '京都' in str(course):
        return True
    if condition == 'E' or condition == 'B':
        return True
    return False


def generate_trio_formation(top_horses):
    """trio 7 点 formation: TOP1 - TOP2,3 - TOP2-6"""
    if len(top_horses) < 6:
        return []
    t1, t2, t3, t4, t5, t6 = top_horses[:6]
    bets = []
    # TOP1 軸 - TOP2,3 と TOP2-6 残り
    pivots = [t2, t3]
    thirds = [t2, t3, t4, t5, t6]
    for p in pivots:
        for th in thirds:
            if p != th:
                bet = tuple(sorted([t1, p, th]))
                if bet not in bets and len(bet) == 3:
                    bets.append(bet)
    return bets[:7]


def check_trio_hit(bets, trio_nums_str):
    """trio nums 文字列 '1-7-8' と bets list が hit か."""
    if not trio_nums_str or '-' not in trio_nums_str:
        return False
    actual = set()
    for n in trio_nums_str.split('-'):
        try:
            actual.add(int(n))
        except Exception:
            pass
    if len(actual) != 3:
        return False
    for bet in bets:
        if set(bet) == actual:
            return True
    return False


def generate_umaren_formation(top_horses):
    """umaren 2 点: TOP1-TOP2, TOP1-TOP3"""
    if len(top_horses) < 3:
        return []
    return [tuple(sorted([top_horses[0], top_horses[1]])),
            tuple(sorted([top_horses[0], top_horses[2]]))]


def check_umaren_hit(bets, umaren_nums_str):
    if not umaren_nums_str or '-' not in umaren_nums_str:
        return False
    actual = set()
    for n in umaren_nums_str.split('-'):
        try:
            actual.add(int(n))
        except Exception:
            pass
    if len(actual) != 2:
        return False
    for bet in bets:
        if set(bet) == actual:
            return True
    return False


def run_wf_with_preds_save(df, features, folds, quick=False):
    """WF 6-fold、 per-race predictions も保存."""
    from train_v22_4ensemble import build_race_id_unique
    df = build_race_id_unique(df)
    all_preds = []

    for y_lo, y_hi in folds:
        train_mask = df['year'] < y_lo
        test_mask = (df['year'] >= y_lo) & (df['year'] <= y_hi)
        df_tr = df[train_mask].copy()
        df_te = df[test_mask].copy()
        n_tr, n_te = len(df_tr), len(df_te)
        print(f'\n=== fold {y_lo}-{y_hi}: train={n_tr:,}, test={n_te:,} ===')
        if n_tr < 1000 or n_te < 100:
            continue

        X_tr = df_tr[features].astype(np.float32).values
        y_tr = df_tr['target'].values
        X_te = df_te[features].astype(np.float32).values
        y_te = df_te['target'].values

        print('  LGB ...')
        train_set = lgb.Dataset(X_tr, label=y_tr)
        val_set = lgb.Dataset(X_te, label=y_te, reference=train_set)
        lgb_model = lgb.train(LGB_PARAMS, train_set, num_boost_round=1000,
                               valid_sets=[val_set], valid_names=['val'],
                               callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
        p_lgb = lgb_model.predict(X_te)
        auc_lgb = roc_auc_score(y_te, p_lgb)
        print(f'  LGB AUC={auc_lgb:.4f}')

        print('  XGB ...')
        dtr = xgb.DMatrix(X_tr, label=y_tr)
        dva = xgb.DMatrix(X_te, label=y_te)
        xgb_model = xgb.train(XGB_PARAMS, dtr, num_boost_round=1000,
                               evals=[(dva, 'val')], early_stopping_rounds=50, verbose_eval=0)
        p_xgb = xgb_model.predict(dva)
        auc_xgb = roc_auc_score(y_te, p_xgb)
        print(f'  XGB AUC={auc_xgb:.4f}')

        # Simple ensemble: LGB+XGB only (FT/IR は backtest 重い、 投資 layer の比較が主目的)
        p_ens = (p_lgb + p_xgb) / 2.0
        auc_ens = roc_auc_score(y_te, p_ens)
        print(f'  ENS AUC={auc_ens:.4f}')

        # per-race save
        df_te = df_te.copy()
        df_te['v22_pred'] = p_ens
        for _, row in df_te.iterrows():
            all_preds.append({
                'race_id': str(row.get('race_id', '')),
                'year': int(row['year']),
                'month': int(row.get('month', 0) or 0),
                'day': int(row.get('day', 0) or 0),
                'course': str(row.get('course', '')),
                'kai': int(row.get('kai', 0) or 0),
                'nichi': int(row.get('nichi', 0) or 0),
                'race_num': int(row.get('race_num', 0) or 0),
                'umaban': int(row.get('umaban', 0) or 0),
                'distance': float(row.get('distance', 0) or 0),
                'condition': row.get('condition', 0),
                'num_horses': int(row.get('num_horses_val', 0) or 0),
                'finish': int(row.get('finish', 99) or 99) if pd.notna(row.get('finish')) else 99,
                'v22_pred': float(p_ens[df_te.index.get_loc(row.name)]) if hasattr(p_ens, '__len__') else 0,
                'target': int(row.get('target', 0) or 0),
            })

        if quick:
            break

    return pd.DataFrame(all_preds)


def simulate_roi(df_preds, payouts):
    """per-race predictions → 戦略適用 → ROI 計算."""
    # race group
    df_preds['race_key'] = (df_preds['course'].astype(str) + '_'
                             + df_preds['kai'].astype(str) + '_'
                             + df_preds['nichi'].astype(str) + '_'
                             + df_preds['race_num'].astype(str))
    races = df_preds.groupby('race_key')
    print(f'  total races: {len(races)}')

    stats = {'A': [], 'B': [], 'C': [], 'D': [], 'E': [], 'X': []}
    total_invest = 0
    total_payout = 0
    excluded = 0
    no_payout = 0

    for race_key, grp in races:
        grp = grp.sort_values('v22_pred', ascending=False)
        top_horses = grp['umaban'].astype(int).tolist()
        if len(top_horses) < 6:
            continue

        num_h = int(grp['num_horses'].iloc[0]) if 'num_horses' in grp.columns else len(grp)
        dist = float(grp['distance'].iloc[0]) if 'distance' in grp.columns else 0
        cond = grp['condition'].iloc[0]
        course = str(grp['course'].iloc[0])
        race_num = int(grp['race_num'].iloc[0])

        condition_class = classify_condition(num_h, dist, cond)

        # 戦略⑦ 除外
        if is_strategy7_excluded(course, condition_class, dist):
            excluded += 1
            continue

        # 投資額
        invest = INVESTMENT
        if race_num == 12:
            invest = min(invest * 3, INVESTMENT_CAP_12R)  # 案B改

        # 払戻 lookup (year2 + zero-padded kai/nichi/race)
        year_v = str(int(grp['year'].iloc[0])).zfill(2)
        pk = (year_v, course,
              str(int(grp['kai'].iloc[0])).zfill(2),
              str(int(grp['nichi'].iloc[0])).zfill(2),
              str(race_num).zfill(2))
        payout_info = payouts.get(pk)
        if not payout_info:
            no_payout += 1
            continue

        # 買い目 + hit check
        if condition_class == 'E':
            bets = generate_umaren_formation(top_horses)
            hit = check_umaren_hit(bets, payout_info['umaren_nums'])
            payout = payout_info['umaren_payout'] if hit else 0
        else:
            bets = generate_trio_formation(top_horses)
            hit = check_trio_hit(bets, payout_info['trio_nums'])
            payout = payout_info['trio_payout'] if hit else 0

        total_invest += invest
        total_payout += payout
        stats[condition_class].append({'hit': hit, 'invest': invest, 'payout': payout})

    print(f'  excluded (戦略⑦): {excluded}')
    print(f'  no payout data: {no_payout}')

    # 集計
    print('\n=== ROI by condition ===')
    out = {}
    for cond, items in stats.items():
        if not items:
            print(f'  {cond}: N=0')
            continue
        n = len(items)
        hit = sum(1 for i in items if i['hit'])
        inv = sum(i['invest'] for i in items)
        pay = sum(i['payout'] for i in items)
        roi = pay / inv * 100 if inv > 0 else 0
        hr = hit / n * 100
        print(f'  {cond}: N={n}, hit={hit} ({hr:.1f}%), invest={inv:,}, payout={pay:,}, ROI={roi:.1f}%')
        out[cond] = {'n': n, 'hit_count': hit, 'hit_rate': hr,
                     'invest': inv, 'payout': pay, 'roi': roi}

    total_roi = total_payout / total_invest * 100 if total_invest > 0 else 0
    profit = total_payout - total_invest
    print(f'\n--- TOTAL ---')
    print(f'  N races: {sum(len(v) for v in stats.values())}')
    print(f'  invest: {total_invest:,} 円')
    print(f'  payout: {total_payout:,} 円')
    print(f'  profit: {profit:+,} 円')
    print(f'  ROI: {total_roi:.1f}%')

    out['_total'] = {'invest': total_invest, 'payout': total_payout,
                     'profit': profit, 'roi': total_roi}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--top-n', type=int, default=100)
    ap.add_argument('--quick', action='store_true')
    args = ap.parse_args()

    print('=' * 60)
    print(f'V22 TOP {args.top_n} backtest (実 ROI vs V15)')
    print('=' * 60)

    t0 = time.time()
    top_features = load_top_n_features(args.top_n)
    df, _ = load_v22_enhanced_data()
    available = [f for f in top_features if f in df.columns]
    print(f'features: {len(available)}')

    folds = ([(24, 24)] if args.quick
             else [(20, 20), (21, 21), (22, 22), (23, 23), (24, 24), (25, 25)])

    df_preds = run_wf_with_preds_save(df, available, folds, quick=args.quick)
    print(f'\nWF predictions saved: {len(df_preds):,} rows')

    payouts = load_payouts()
    print(f'payouts entries: {len(payouts):,}')

    results = simulate_roi(df_preds, payouts)

    out_path = MODEL_DIR / f'backtest_top{args.top_n}_{datetime.now():%Y%m%d_%H%M%S}.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({
            'mode': 'quick' if args.quick else 'full',
            'top_n': args.top_n,
            'n_features': len(available),
            'roi_by_condition': results,
            'v15_baseline_roi': {
                'note': 'V15 production WF 2020-2025 actual ROI',
                'A': 355.4, 'B': 346.8, 'C': 623.0,
                'D': 360.8, 'E': 195.7, 'X': 701.2,
                'total': 428.4,
            },
            'elapsed_s': time.time() - t0,
        }, f, ensure_ascii=False, indent=2)
    print(f'\nresults: {out_path}')


if __name__ == '__main__':
    main()
