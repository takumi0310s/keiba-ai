"""V18/V19 sib抜き LIVE retro (Session #38 B).

既存 tools/v18_v19_retro_full.py 流用、 model dir を v18v19_retraining/ に切替。
- v18 / v19 LGB sib抜き single-fold model 使用 (XGB なし)
- 5/2-5/3 全 races scrape + score
- 出力: data/v18/v18v19_retraining/no_sib_retro_5_2_5_3_predictions.csv

V15 production 完全独立、 既存 v18 model file 完全不変。
"""
import sys, os, io, contextlib, time
sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None

import pandas as pd
import numpy as np
import lightgbm as lgb
from datetime import datetime

BASE_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
os.chdir(BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, 'tools'))

from predict_core import parse_shutuba, build_features, load_models, set_horse_defaults
from predict_v17_morning_pipeline import add_v17_features_to_race_df
import importlib.util
spec = importlib.util.spec_from_file_location('pred_top', 'tools/predict_v17_top_races_5_3.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
add_v162_safe = mod.add_v162_safe

DATES = ['20260502', '20260503']
NO_SIB_DIR = 'data/v18/v18v19_retraining'
OUT_PREDICTIONS = f'{NO_SIB_DIR}/no_sib_retro_5_2_5_3_predictions.csv'


def load_no_sib_models():
    v18_lgb = lgb.Booster(model_file=f'{NO_SIB_DIR}/v18_lgb_no_sib_v1.txt')
    v19_lgb = lgb.Booster(model_file=f'{NO_SIB_DIR}/v19_lgb_no_sib_v1.txt')
    return v18_lgb, v19_lgb


def predict_one_race(race_id, model_data, v18_lgb, v19_lgb):
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            race_name, horses, _, race_info = parse_shutuba(race_id)
            for h in horses:
                set_horse_defaults(h)
            race_df = build_features(horses, race_info, model_data)
            race_df = race_df.rename(columns={'馬番': 'umaban', '馬名': 'horse_name', '血統登録番号': 'blood_num'})
            race_df['_nk_race_id'] = race_id
            race_df['race_id'] = race_id
            race_df['umaban'] = pd.to_numeric(race_df['umaban'], errors='coerce').astype('Int64')
            race_df = add_v162_safe(race_df)
            race_df = add_v17_features_to_race_df(race_df, has_nk_race_id=True)

            feats_18 = v18_lgb.feature_name()
            for f in feats_18:
                if f not in race_df.columns:
                    race_df[f] = 0
            X18df = race_df[feats_18].apply(pd.to_numeric, errors='coerce').fillna(0)
            p18 = v18_lgb.predict(X18df.values)

            feats_19 = v19_lgb.feature_name()
            for f in feats_19:
                if f not in race_df.columns:
                    race_df[f] = 0
            X19df = race_df[feats_19].apply(pd.to_numeric, errors='coerce').fillna(0)
            p19 = v19_lgb.predict(X19df.values)

        return {
            'race_id': race_id, 'race_name': race_name,
            'umaban_list': race_df['umaban'].tolist(),
            'p18': p18.tolist(), 'p19': p19.tolist(),
        }
    except Exception as e:
        return {'race_id': race_id, 'error': str(e)[:120]}


def main():
    print(f"=== V18/V19 sib抜き LIVE retro START {datetime.now()} ===")
    md = load_models()
    v18_lgb, v19_lgb = load_no_sib_models()
    print(f"v18 features: {len(v18_lgb.feature_name())}, v19: {len(v19_lgb.feature_name())}")
    # 確認: sib_ が含まれていないこと
    assert not any(f.startswith('sib_') for f in v18_lgb.feature_name()), "v18 has sib_ feature"
    assert not any(f.startswith('sib_') for f in v19_lgb.feature_name()), "v19 has sib_ feature"
    print("[OK] sib_* features 完全除外確認")

    all_horse_rows = []
    t_start = time.time()

    for d in DATES:
        pred = pd.read_csv(f'data/daily_predictions/{d}.csv', dtype={'race_id': str})
        odds = pd.read_csv(f'data/odds_base_{d}.csv', dtype={'race_id': str})
        odds['horse_num'] = pd.to_numeric(odds['horse_num'], errors='coerce').astype('Int64')
        res = pd.read_csv(f'data/daily_results/{d}.csv', encoding='utf-8-sig', dtype={'race_id': str})
        res = res.drop_duplicates(subset=['race_id'], keep='last')

        races_d = sorted(pred['race_id'].astype(str).unique())
        print(f"\n=== {d}: {len(races_d)} races ===", flush=True)

        for i, rid in enumerate(races_d):
            t0 = time.time()
            r = predict_one_race(rid, md, v18_lgb, v19_lgb)
            if 'error' in r:
                print(f"  [{i+1}/{len(races_d)}] {rid} ERROR: {r['error']}", flush=True)
                continue

            odds_r = odds[odds['race_id'] == rid][['horse_num', 'odds', 'pop_rank']]
            odds_map = {int(row['horse_num']): {'odds': float(row['odds']), 'pop': int(row['pop_rank'])}
                        for _, row in odds_r.iterrows()}
            res_r = res[res['race_id'] == rid]
            if len(res_r) == 0:
                continue
            rr = res_r.iloc[0]
            tres_str = str(rr.get('trio_result', ''))
            try:
                trio_set = set(int(x) for x in tres_str.split('-') if x.strip())
            except:
                trio_set = set()
            pred_r = pred[pred['race_id'] == rid].iloc[0] if len(pred[pred['race_id'] == rid]) else None
            winner_num = None
            if pred_r is not None:
                if int(rr.get('top1_finish', 0) or 0) == 1:
                    winner_num = int(pred_r.get('top1_num', 0) or 0)
                elif int(rr.get('top2_finish', 0) or 0) == 1:
                    winner_num = int(pred_r.get('top2_num', 0) or 0)
                elif int(rr.get('top3_finish', 0) or 0) == 1:
                    winner_num = int(pred_r.get('top3_num', 0) or 0)

            for j, uma in enumerate(r['umaban_list']):
                uma_int = int(uma) if pd.notna(uma) else 0
                if uma_int == 0: continue
                p18 = float(r['p18'][j]); p19 = float(r['p19'][j])
                o = odds_map.get(uma_int, {'odds': 0, 'pop': 0})
                ev_t = p18 * o['odds']; ev_f = p19 * o['odds']
                is_win_h = 1 if winner_num is not None and uma_int == winner_num else 0
                is_top3_h = 1 if uma_int in trio_set else 0
                all_horse_rows.append({
                    'date': d, 'race_id': rid, 'umaban': uma_int,
                    'p_tansho': p18, 'p_fukusho': p19,
                    'odds': o['odds'], 'pop_rank': o['pop'],
                    'ev_tansho': ev_t, 'ev_fukusho': ev_f,
                    'is_win': is_win_h, 'is_top3': is_top3_h,
                    'winner_known': 1 if winner_num is not None else 0,
                })

            elapsed = time.time() - t0
            print(f"  [{i+1}/{len(races_d)}] {rid} t={elapsed:.0f}s p18_max={max(r['p18']):.3f} winner={winner_num}", flush=True)

    print(f"\nTotal time: {(time.time() - t_start) / 60:.1f}min")
    print(f"Total horse rows: {len(all_horse_rows)}")
    df = pd.DataFrame(all_horse_rows)
    df.to_csv(OUT_PREDICTIONS, index=False, encoding='utf-8-sig')
    print(f"Saved {OUT_PREDICTIONS}")


if __name__ == '__main__':
    main()
