"""V18/V19 sib_exp 統合 LIVE retro (Session #41 D).

既存 tools/v18_v19_retro_no_sib.py 流用、 model dir を v18v19_sib_exp_w5/ に切替 +
sib_*_exp 4 features を build_features 後に 追加注入。

approach:
  1. predict_core.build_features で 188 features の race_df 構築 (no_sib と同じ)
  2. race_df に sib_*_exp 4 columns を追加 (sib_expanding csv from horse_id lookup)
     - same horse の latest expanding values を使用 (LIVE retro なので 5/2 時点で
       使えた値の近似、 微妙にリーク risk あり)
  3. v18_lgb_sib_exp_w5 model で predict (192 features)

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
SIB_EXP_DIR = 'data/v18/v18v19_sib_exp_w5'
OUT_PREDICTIONS = f'{SIB_EXP_DIR}/sib_exp_w5_retro_5_2_5_3_predictions.csv'

SIB_EXP_NEW = ['sib_top3_rate_exp_w5', 'sib_shinba_wr_exp_w5']


def load_sib_exp_models():
    v18 = lgb.Booster(model_file=f'{SIB_EXP_DIR}/v18_lgb_sib_exp_w5.txt')
    v19 = lgb.Booster(model_file=f'{SIB_EXP_DIR}/v19_lgb_sib_exp_w5.txt')
    return v18, v19


def load_sib_exp_lookup():
    """horse_id 単位で latest sib_*_exp 値を取得 (簡易版).

    LIVE retro では 5/2-5/3 時点で利用できる値を使うべきだが、 csv に date 列がないため
    horse_id 単位で 直近の expanding 値 (= sib_total_races_exp 最大の row) を使用。
    sib_expanding csv は jra_races_full.csv 由来で 5/3 までの全 race を含むため、
    各 horse の最新値は ~5/3 時点 + α (数日後の race も若干含む) になる。
    """
    print("[sib_exp_retro] loading sib_expanding_w5 csv ...")
    sib = pd.read_csv("data/netkeiba_siblings_expanding_w5.csv",
                      dtype={'race_id': str, 'horse_id': str})
    print(f"  rows: {len(sib):,}")
    # horse_id 単位 で latest (race_id の数値順、 後ろ ほど 新しい)
    sib_sorted = sib.sort_values(['horse_id', 'race_id'], ascending=[True, False])
    latest = sib_sorted.drop_duplicates(subset=['horse_id'], keep='first')
    print(f"  unique horse_id: {len(latest):,}")
    return latest.set_index('horse_id')[SIB_EXP_NEW]


def predict_one_race(race_id, model_data, v18_lgb, v19_lgb, sib_lookup):
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

            # sib_*_exp 4 columns 追加 (horse_id lookup)
            # netkeiba 出馬表 horse_id_val (10 chars: '2023101394') を
            # blood_full.csv horse_id (8 chars: '23101394') 形式に変換
            def _hid_to_blood_id(hid_val):
                s = str(hid_val).strip()
                if len(s) == 10 and s[:2] == '20':
                    return s[2:]  # '2023101394' → '23101394'
                return s

            if 'horse_id_val' in race_df.columns:
                race_df['_hid_lookup'] = race_df['horse_id_val'].apply(_hid_to_blood_id)
                for col in SIB_EXP_NEW:
                    race_df[col] = race_df['_hid_lookup'].map(sib_lookup[col]).fillna(0)
            elif 'horse_id' in race_df.columns:
                race_df['_hid_lookup'] = race_df['horse_id'].astype(str)
                for col in SIB_EXP_NEW:
                    race_df[col] = race_df['_hid_lookup'].map(sib_lookup[col]).fillna(0)
            else:
                for col in SIB_EXP_NEW:
                    race_df[col] = 0.0

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
            'sib_t3_match': float((race_df[SIB_EXP_NEW[0]] > 0).mean()),
        }
    except Exception as e:
        return {'race_id': race_id, 'error': str(e)[:120]}


def main():
    print(f"=== V18/V19 sib_exp LIVE retro START {datetime.now()} ===")
    md = load_models()
    v18_lgb, v19_lgb = load_sib_exp_models()
    print(f"v18 features: {len(v18_lgb.feature_name())}, v19: {len(v19_lgb.feature_name())}")

    sib_in_v18 = [f for f in v18_lgb.feature_name() if f.startswith('sib_')]
    print(f"v18 sib_* features: {sib_in_v18}")
    sib_exp_in_v18 = [f for f in v18_lgb.feature_name() if f in SIB_EXP_NEW]
    assert len(sib_exp_in_v18) == 2, f"expected 2 sib_*_exp_w5 features, got {sib_exp_in_v18}"
    print("[OK] sib_*_exp_w5 2 features 確認")

    sib_lookup = load_sib_exp_lookup()
    print(f"sib_lookup horse_id n: {len(sib_lookup):,}")

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
            r = predict_one_race(rid, md, v18_lgb, v19_lgb, sib_lookup)
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
            sib_match = r.get('sib_t3_match', 0) * 100
            print(f"  [{i+1}/{len(races_d)}] {rid} t={elapsed:.0f}s p18_max={max(r['p18']):.3f} "
                  f"sib_t3_match={sib_match:.0f}% winner={winner_num}", flush=True)

    print(f"\nTotal time: {(time.time() - t_start) / 60:.1f}min")
    print(f"Total horse rows: {len(all_horse_rows)}")
    df = pd.DataFrame(all_horse_rows)
    df.to_csv(OUT_PREDICTIONS, index=False, encoding='utf-8-sig')
    print(f"Saved {OUT_PREDICTIONS}")


if __name__ == '__main__':
    main()
