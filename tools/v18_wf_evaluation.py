"""Phase 19: V18 vs V15 WF 評価 framework

V15 baseline (AUC 0.8939) vs V18 真値版 比較 + 5/10 全 35R 仮想評価。
"""
from __future__ import annotations
import argparse
import json
import sys
import gzip
import pickle
from pathlib import Path

BASE = Path(r"C:/Users/takum/keiba-ai")
sys.path.insert(0, str(BASE / "tools"))


def load_v15_model():
    pkl = BASE / "keiba_model_v15_central_live.pkl.gz"
    with gzip.open(pkl, 'rb') as f:
        return pickle.load(f)


def load_v18_model():
    pkl = BASE / "models" / "v18" / "v18_truevalue_model.pkl.gz"
    if not pkl.exists():
        return None
    with gzip.open(pkl, 'rb') as f:
        return pickle.load(f)


def compare_5_10_full35r():
    """5/10 全 35 R 仮想 V15 vs V18 比較"""
    import csv
    out = []

    res_path = BASE / "data" / "daily_results" / "20260510.csv"
    if not res_path.exists():
        return {'error': 'daily_results/20260510.csv 不在'}

    with open(res_path, encoding='utf-8-sig') as f:
        results = list(csv.DictReader(f))

    n_total = len(results)
    n_top1_hit = 0
    n_trio_hit = 0
    inv_total, pay_total = 0, 0
    for r in results:
        if str(r.get('trio_hit', '0')).strip() == '1':
            n_trio_hit += 1
        try:
            inv_total += int(float(r.get('investment', 0) or 0))
            pay_total += int(float(r.get('actual_payout', 0) or 0))
        except (ValueError, TypeError):
            pass

    return {
        'n_total': n_total,
        'trio_hit': n_trio_hit,
        'trio_hit_rate_pct': 100.0 * n_trio_hit / max(n_total, 1),
        'investment': inv_total, 'payout': pay_total,
        'profit': pay_total - inv_total,
        'roi_pct': 100.0 * pay_total / max(inv_total, 1),
    }


def score_band_breakdown():
    """V15 morning_top1_score 帯別 hit 率 (5/10)"""
    import csv
    pred_path = BASE / "data" / "daily_predictions" / "20260510.csv"
    res_path = BASE / "data" / "daily_results" / "20260510.csv"
    if not pred_path.exists() or not res_path.exists():
        return {'error': 'data missing'}
    with open(pred_path, encoding='utf-8-sig') as f:
        preds = {r['race_id']: r for r in csv.DictReader(f)}
    with open(res_path, encoding='utf-8-sig') as f:
        results = {r['race_id']: r for r in csv.DictReader(f)}

    bands = {'≥0.7': [], '0.6-0.7': [], '0.5-0.6': [], '<0.5': []}
    for rid, p in preds.items():
        res = results.get(rid)
        if not res:
            continue
        try:
            s = float(p.get('top1_score', 0) or 0)
            hit = str(res.get('trio_hit', '0')).strip() == '1'
            inv = int(float(res.get('investment', 0) or 0))
            pay = int(float(res.get('actual_payout', 0) or 0))
        except (ValueError, TypeError):
            continue
        if s >= 0.7: band = '≥0.7'
        elif s >= 0.6: band = '0.6-0.7'
        elif s >= 0.5: band = '0.5-0.6'
        else: band = '<0.5'
        bands[band].append({'hit': hit, 'inv': inv, 'pay': pay})

    out = {}
    for band, races in bands.items():
        n = len(races)
        h = sum(1 for r in races if r['hit'])
        inv = sum(r['inv'] for r in races)
        pay = sum(r['pay'] for r in races)
        out[band] = {
            'n': n, 'hit': h,
            'hit_rate_pct': 100.0 * h / max(n, 1),
            'investment': inv, 'payout': pay,
            'roi_pct': 100.0 * pay / max(inv, 1),
        }
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', default='all', choices=['all', '5_10', 'band'])
    args = p.parse_args()

    print("=" * 70)
    print("Phase 19: V18 真値版 WF 評価 framework")
    print("=" * 70)

    # V15 model
    try:
        v15 = load_v15_model()
        print(f"\n✓ V15 model: features={len(v15['features'])}, AUC={v15.get('auc', '?')}")
    except Exception as e:
        print(f"\n✗ V15 model load 失敗: {e}")

    # V18 model
    v18 = load_v18_model()
    if v18:
        print(f"✓ V18 model: features={len(v18.get('features', []))}, AUC={v18.get('auc', '?')}")
    else:
        print(f"✗ V18 model 不在 (5/16 user CLI 実行で生成予定)")

    if args.mode in ('all', '5_10'):
        print("\n=== 5/10 全 35 R 仮想評価 ===")
        result = compare_5_10_full35r()
        for k, v in result.items():
            print(f"  {k}: {v}")

    if args.mode in ('all', 'band'):
        print("\n=== V15 score 帯別 hit 率 (5/10) ===")
        bands = score_band_breakdown()
        if 'error' not in bands:
            print(f"  {'band':<10} {'N':>4} {'hit':>4} {'hit%':>6} {'ROI%':>7}")
            for band, m in bands.items():
                print(f"  {band:<10} {m['n']:>4} {m['hit']:>4} {m['hit_rate_pct']:>5.1f}% {m['roi_pct']:>6.1f}%")

    # save
    summary = {
        'v15_baseline': {'features': len(v15['features']), 'auc': v15.get('auc')},
        'v18_status': '訓練前 (5/16 user CLI 実行待ち)' if not v18 else '訓練済',
        'd5_10_full35r': compare_5_10_full35r() if args.mode in ('all', '5_10') else None,
        'score_bands': score_band_breakdown() if args.mode in ('all', 'band') else None,
    }
    out = BASE / "data" / "v18" / "phase19_wf_evaluation_5_10.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f"\nsaved: {out}")


if __name__ == '__main__':
    main()
