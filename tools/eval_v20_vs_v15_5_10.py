#!/usr/bin/env python
"""V20 vs V15 evaluator on 2026-05-10 全 35 R.

V20 quick model + V15 prod model に対し、 5/10 actual results との比較を計算。

Output:
  data/v20/phase15_5_10_eval.json
"""
from __future__ import annotations
import os
import sys
import json
import gzip
import pickle
from datetime import datetime
from typing import Dict, Any, List

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

V15_PRED = os.path.join(BASE, 'data', 'daily_predictions', '20260510.csv')
V15_RES = os.path.join(BASE, 'data', 'daily_results', '20260510.csv')
V20_DIR = os.path.join(BASE, 'models', 'v20')
OUT = os.path.join(BASE, 'data', 'v20', 'phase15_5_10_eval.json')


def latest_v20_model():
    files = sorted([f for f in os.listdir(V20_DIR) if f.startswith('v20_') and f.endswith('.pkl.gz')])
    return os.path.join(V20_DIR, files[-1]) if files else None


def main():
    if not os.path.exists(V15_PRED):
        print(f"[skip] V15 preds not found: {V15_PRED}")
        return
    if not os.path.exists(V15_RES):
        print(f"[skip] V15 results not found: {V15_RES}")
        return

    df_pred = pd.read_csv(V15_PRED, encoding='utf-8-sig')
    df_res = pd.read_csv(V15_RES, encoding='utf-8-sig')

    # V15 metrics
    n_races = len(df_pred)
    n_results = len(df_res)
    trio_hits = int(df_res['trio_hit'].sum()) if 'trio_hit' in df_res.columns else 0
    profit = float(df_res['profit'].sum()) if 'profit' in df_res.columns else 0.0
    invest = float(df_res['investment'].sum()) if 'investment' in df_res.columns else 0.0
    actual = float(df_res['actual_payout'].sum()) if 'actual_payout' in df_res.columns else 0.0
    roi = (actual / invest * 100) if invest > 0 else 0.0

    # top1 finish 分布
    top1_finishes = []
    if 'top1_finish' in df_res.columns:
        top1_finishes = df_res['top1_finish'].dropna().astype(int).tolist()
    n_top1_first = sum(1 for f in top1_finishes if f == 1)
    n_top1_top3 = sum(1 for f in top1_finishes if f <= 3)

    # condition 別
    cond_stats = {}
    if 'condition' in df_res.columns:
        for cond, grp in df_res.groupby('condition'):
            cond_stats[str(cond)] = {
                'n': len(grp),
                'trio_hits': int(grp['trio_hit'].sum()) if 'trio_hit' in grp.columns else 0,
                'profit': float(grp['profit'].sum()) if 'profit' in grp.columns else 0.0,
            }

    v20_model_path = latest_v20_model()

    out = {
        'date': '2026-05-10',
        'evaluated_at': datetime.now().isoformat(timespec='seconds'),
        'v15': {
            'n_races': n_races,
            'n_settled': n_results,
            'trio_hits': trio_hits,
            'investment': invest,
            'payout': actual,
            'profit': profit,
            'roi_pct': roi,
            'top1_first_rate': n_top1_first / max(n_results, 1),
            'top1_top3_rate': n_top1_top3 / max(n_results, 1),
            'condition_stats': cond_stats,
        },
        'v20': {
            'model_path': v20_model_path,
            'note': 'V20 quick = V15 features 145 のみ retrain。 Phase 11/12/13 57 features は constant default のため signal 寄与なし。 5/10 仮想評価は同一データの場合 V15 と等価のため未実施。',
        },
        'comparison': {
            'note': 'V20 真の改善には Phase 11/12/13 実 data 取得が必要 (5/11+ Phase 13.5、 5/12+ Phase 11 lookup、 5/24+ Phase 12 JV-Link)。',
            'v15_baseline_auc': 0.8939,
            'v20_quick_eval_auc': None,  # filled by V20 trainer separately
        }
    }

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=str)
    print(f"[save] {OUT}")
    print(f"[V15] hits={trio_hits}/{n_results}  ROI={roi:.1f}%  profit={profit:,.0f}yen  top1 first rate={n_top1_first/max(n_results,1):.1%}")


if __name__ == '__main__':
    main()
