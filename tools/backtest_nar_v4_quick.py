"""NAR v4 quick AUC 再現 (data/nar_all_races.csv 利用).

note:
  archive/nar/backtest_nar_leakfree.py は data/chihou_races_2020_2025.csv 依存 (本リポに無し)。
  本 script は data/nar_all_races.csv (54k rows, 2024-2025 部分) で 22 features を encode し、
  v4 model の OOS AUC を計測する簡易版。
"""
import os, sys, pickle, math
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

BASE = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
os.chdir(BASE)

CSV = 'data/nar_all_races.csv'
MODEL = 'data/nar/models/keiba_model_nar_v4.pkl'

COURSE_MAP = {
    '門別':30, '盛岡':35, '水沢':36, '浦和':42, '船橋':43,
    '大井':44, '川崎':45, '金沢':46, '笠松':47, '名古屋':48,
    '園田':50, '姫路':51, '高知':54, '佐賀':55, '帯広':65,
}

CONDITION_MAP = {'良':0, '稍重':1, '重':2, '不良':3}


def encode_row(r, jockey_stats, nh_lookup, default_hw=480.0):
    sex_age = str(r.get('sex_age',''))
    age = 4
    sex_enc = 0
    for ch in '牡牝セ騙':
        if sex_age.startswith(ch):
            sex_enc = {'牡':0,'牝':1,'セ':2,'騙':2}[ch]
            try: age = int(sex_age[len(ch):])
            except: pass
            break
    hn = int(r.get('horse_num', 0) or 0)
    if hn == 0: return None
    rid = str(r.get('race_id',''))
    nh = nh_lookup.get(rid, 13)
    if nh < 2: return None

    if nh <= 8: bracket = hn
    else: bracket = min(8, math.ceil(hn * 8 / nh))
    bracket_pos = 0 if bracket <= 3 else (1 if bracket <= 5 else 2)

    odds_raw = r.get('odds', None)
    try: odds = float(odds_raw)
    except: odds = None
    odds_log = math.log(odds) if (odds and odds > 0) else 0.0
    pr_raw = r.get('pop_rank', 99)
    try:
        pop_rank = int(pr_raw) if pd.notna(pr_raw) else 99
    except Exception:
        pop_rank = 99

    j = str(r.get('jockey_name','')).strip()
    js = jockey_stats.get(j) or {'wr':0.05, 'place_rate':0.18}

    weight_carry = float(r.get('weight_carry', 56.0) or 56.0)
    hw = r.get('horse_weight', None)
    try: horse_weight = float(hw) if hw not in (None,'','nan') else default_hw
    except: horse_weight = default_hw

    course = str(r.get('course','')).strip()
    course_enc = COURSE_MAP.get(course, 44)

    surface = str(r.get('surface',''))
    surface_enc = 0 if '芝' in surface else 1

    cond = str(r.get('condition','良')).strip()
    condition_enc = CONDITION_MAP.get(cond, 0)

    distance = int(r.get('distance', 1600) or 1600)
    if distance <= 1200: dist_cat = 0
    elif distance <= 1400: dist_cat = 1
    elif distance <= 1800: dist_cat = 2
    elif distance <= 2200: dist_cat = 3
    else: dist_cat = 4

    if horse_weight < 440: weight_cat = 0
    elif horse_weight < 480: weight_cat = 1
    elif horse_weight < 520: weight_cat = 2
    else: weight_cat = 3

    age_group = max(0, min(age - 3, 5))

    return {
        'odds_log': odds_log, 'num_horses': nh, 'distance': distance,
        'surface_enc': surface_enc, 'condition_enc': condition_enc, 'course_enc': course_enc,
        'horse_weight': horse_weight, 'weight_carry': weight_carry, 'age': age, 'sex_enc': sex_enc,
        'horse_num': hn, 'bracket': bracket, 'horse_num_ratio': hn/nh,
        'bracket_pos': bracket_pos, 'carry_diff': 0.0,
        'dist_cat': dist_cat, 'weight_cat': weight_cat, 'age_group': age_group,
        'jockey_wr': js['wr'], 'jockey_place_rate': js['place_rate'],
        'pop_rank': pop_rank, 'is_nar': 1,
    }


def main():
    print("=== NAR v4 quick AUC 再現 ===\n")
    m = pickle.load(open(MODEL,'rb'))
    print(f"Model AUC reported: {m['auc']:.4f} (LGB {m['lgb_auc']:.4f}, XGB {m['xgb_auc']:.4f})")
    print(f"trained_at: {m.get('trained_at')}")
    print(f"n_races: {m.get('n_races')}, n_rows: {m.get('n_rows')}")
    features = m['features']
    jockey_stats = m.get('jockey_stats', {})

    df = pd.read_csv(CSV, dtype={'race_id':str})
    print(f"\nCSV: {len(df)} rows, race_id unique: {df['race_id'].nunique()}")
    df['race_date'] = pd.to_datetime(df['race_date'].astype(str), format='%Y%m%d', errors='coerce')
    print(f"date range: {df['race_date'].min()} → {df['race_date'].max()}")

    # 学習 vs OOS 切り分け: 学習は 2020-2024 (model trained_at 参照)
    df_oos = df[df['race_date'] >= '2025-01-01'].copy()
    print(f"OOS (2025-): {len(df_oos)} rows, {df_oos['race_id'].nunique()} races")

    # encode
    nh_map = df_oos.groupby('race_id').size().to_dict()
    rows, ys = [], []
    for _, r in df_oos.iterrows():
        e = encode_row(r, jockey_stats, nh_map)
        if e is None: continue
        rows.append(e)
        try:
            ys.append(1 if int(r['finish']) == 1 else 0)
        except Exception:
            ys.append(0)
    if not rows:
        print("[ERROR] no encodable rows"); return
    X = pd.DataFrame(rows)[features]
    y = np.array(ys)
    print(f"encoded: {len(X)} rows, win rate: {y.mean():.4f}")

    p_lgb = m['model'].predict(X.values)
    import xgboost as xgb
    dmat = xgb.DMatrix(X.values, feature_names=features)
    try:
        p_xgb = m['xgb_model'].predict(dmat)
    except Exception:
        p_xgb = m['xgb_model'].predict(xgb.DMatrix(X.values))
    p_ens = m['ensemble_weights']['lgb'] * p_lgb + m['ensemble_weights']['xgb'] * p_xgb

    auc_lgb = roc_auc_score(y, p_lgb)
    auc_xgb = roc_auc_score(y, p_xgb)
    auc_ens = roc_auc_score(y, p_ens)

    print(f"\n=== OOS AUC (data/nar_all_races.csv 2025-) ===")
    print(f"  LGB     : {auc_lgb:.4f} (vs reported {m['lgb_auc']:.4f}, diff {auc_lgb-m['lgb_auc']:+.4f})")
    print(f"  XGB     : {auc_xgb:.4f} (vs reported {m['xgb_auc']:.4f}, diff {auc_xgb-m['xgb_auc']:+.4f})")
    print(f"  ensemble: {auc_ens:.4f} (vs reported {m['auc']:.4f}, diff {auc_ens-m['auc']:+.4f})")

    # condition 別 AUC
    print(f"\n=== 条件別 AUC ===")
    df_oos2 = df_oos.iloc[:len(X)].copy()
    df_oos2['p_ens'] = p_ens
    df_oos2['y'] = y
    # classify
    def cls(r):
        nh = int(r.get('num_horses', 0) or 0)
        d = int(r.get('distance',0) or 0)
        c = str(r.get('condition','良'))
        heavy = any(x in c for x in ['重','不'])
        if nh <= 7: return 'E'
        if d <= 1400: return 'D'
        if 8 <= nh <= 14 and d >= 1600 and not heavy: return 'A'
        if 8 <= nh <= 14 and d >= 1600 and heavy: return 'B'
        if nh >= 15 and d >= 1600 and not heavy: return 'C'
        return 'X'
    df_oos2['cond'] = df_oos2.apply(cls, axis=1)
    for k in ['A','B','C','D','E','X']:
        sub = df_oos2[df_oos2['cond']==k]
        if len(sub) < 50: print(f"  {k}: n={len(sub)} (skip)"); continue
        try:
            a = roc_auc_score(sub['y'], sub['p_ens'])
            print(f"  {k}: n={len(sub)} AUC={a:.4f} win_rate={sub['y'].mean():.4f}")
        except Exception as e:
            print(f"  {k}: AUC err {e}")


if __name__ == '__main__':
    main()
