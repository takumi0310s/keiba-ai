#!/usr/bin/env python3
"""Fable監査③続き(6/11): daily_predict二重マージの影響判別テスト(読み取りのみ・本番不変)。
6/6-7のダンプ(当日実スコア入り)を (a)劣化df=素列デフォルトのまま / (b)復元df=素列←_x で再採点し、
当日実際に記録された `スコア` とどちらが一致するか → 一致した方が「実際に走った採点」。
あわせて影響量(top1/top3/買い目フォーメーションの変化R数)を計測。
"""
from __future__ import annotations
import os, sys, glob, json, warnings
warnings.filterwarnings('ignore')
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd
sys.path.insert(0, os.path.abspath('tools'))
sys.path.insert(0, os.path.abspath('.'))
from scipy.stats import spearmanr
from predict_core import load_models, predict_race


def formation(order):
    """本番形 trio: top1軸 × {top2,top3} × {top2..top6}"""
    if len(order) < 6:
        return None
    o = order
    bets = set()
    for s in o[1:3]:
        for t in o[1:6]:
            c = frozenset((o[0], s, t))
            if len(c) == 3:
                bets.add(c)
    return bets


def main():
    md = load_models()
    print(f"model loaded: features={len(md.get('features', []))}")
    n = a_match = b_match = 0
    top1_chg = top3_chg = form_chg = 0
    details = []
    for date in ['20260606', '20260607']:
        for pq in sorted(glob.glob(f'data/v15_feat_dump/{date}/*.parquet')):
            try:
                df = pd.read_parquet(pq)
            except Exception as e:
                print(f"  [skip] {os.path.basename(pq)}: {e}")
                continue
            if 'スコア' not in df.columns or len(df) < 6:
                continue
            actual = pd.to_numeric(df['スコア'], errors='coerce').values
            da = predict_race(df.copy(), md)
            sa = pd.to_numeric(da['スコア'], errors='coerce').values
            db_in = df.copy()
            for c in [c for c in db_in.columns if c.endswith('_x') and c[:-2] in db_in.columns]:
                db_in[c[:-2]] = db_in[c]
            db = predict_race(db_in, md)
            sb = pd.to_numeric(db['スコア'], errors='coerce').values
            ma = float(np.nanmax(np.abs(actual - sa)))
            mb = float(np.nanmax(np.abs(actual - sb)))
            ra = spearmanr(actual, sa)[0]
            rb = spearmanr(actual, sb)[0]
            n += 1
            if ma < mb:
                a_match += 1
            elif mb < ma:
                b_match += 1
            oa = da.sort_values('スコア', ascending=False)['馬番'].astype(int).tolist()
            ob = db.sort_values('スコア', ascending=False)['馬番'].astype(int).tolist()
            t1 = oa[0] != ob[0]
            t3 = set(oa[:3]) != set(ob[:3])
            fm = formation(oa) != formation(ob)
            top1_chg += t1; top3_chg += t3; form_chg += fm
            if t3 or fm:
                details.append({'date': date, 'race': os.path.basename(pq)[:12],
                                'deg_top3': oa[:3], 'fix_top3': ob[:3], 'top1_chg': bool(t1)})
            if n <= 3:
                print(f"  例 {date} {os.path.basename(pq)[:12]}: max|actual-(a)劣化|={ma:.2e} max|actual-(b)復元|={mb:.2e} (sp a={ra:.4f} b={rb:.4f})")
    print(f"\n判別: 全{n}R中 actualに近いのは (a)劣化df={a_match}R / (b)復元df={b_match}R")
    print(f"影響量(劣化→復元): top1変化={top1_chg}R top3変化={top3_chg}R 買い目formation変化={form_chg}R / {n}R")
    for d in details:
        print(f"   {d['date']} {d['race']}: 劣化top3={d['deg_top3']} → 復元top3={d['fix_top3']}{' ★top1変化★' if d['top1_chg'] else ''}")
    json.dump({'n': n, 'a_match': a_match, 'b_match': b_match, 'top1_chg': top1_chg,
               'top3_chg': top3_chg, 'form_chg': form_chg, 'details': details},
              open('data/fable_dpfix_discriminate.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=2, default=str)
    print("-> data/fable_dpfix_discriminate.json")


if __name__ == '__main__':
    main()
