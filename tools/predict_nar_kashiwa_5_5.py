"""5/5 かしわ記念 (船橋11R Jpn1) NAR v4 model 予測.

Pattern B (odds_log/horse_weight/pop_rank/condition_enc 含む 22 features).

22 features:
  odds_log, num_horses, distance, surface_enc, condition_enc,
  course_enc, horse_weight, weight_carry, age, sex_enc,
  horse_num, bracket, horse_num_ratio, bracket_pos, carry_diff,
  dist_cat, weight_cat, age_group, jockey_wr, jockey_place_rate,
  pop_rank, is_nar
"""
import os, sys, pickle, math
import pandas as pd
import numpy as np

BASE_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
os.chdir(BASE_DIR)
MODEL_PATH = 'data/nar/models/keiba_model_nar_v4.pkl'
HORSES_CSV = 'data/results/20260505_kashiwa_kinen_horses.csv'

# === 騎手名 マッピング (柏記念 → NAR jockey_stats) ===
# JRA騎手は jockey_stats に不在の可能性大、defaults で代替
JOCKEY_OVERRIDE = {
    # JRA elite (NAR 不在の可能性) — 引用値は JRA 全国データ
    'ルメール': {'wr': 0.226, 'place_rate': 0.564, 'runs': 600},
    '川田将雅': {'wr': 0.234, 'place_rate': 0.541, 'runs': 700},
    '横山武史': {'wr': 0.155, 'place_rate': 0.401, 'runs': 600},
    'レーン': {'wr': 0.182, 'place_rate': 0.444, 'runs': 80},
    '横山和生': {'wr': 0.103, 'place_rate': 0.305, 'runs': 700},
    '川須栄一': {'wr': 0.043, 'place_rate': 0.180, 'runs': 700},
}


def encode(row, jockey_stats):
    horse_num = int(row['horse_num'])
    nh = 13
    age = int(str(row['sex_age']).replace('牡','').replace('牝','').replace('セ',''))
    sex_enc = 0  # all 牡

    # bracket: NAR 8枠制 (13頭立て)
    if horse_num <= 4: bracket = horse_num
    elif horse_num <= 6: bracket = 5
    elif horse_num <= 8: bracket = 6
    elif horse_num <= 10: bracket = 7
    else: bracket = 8

    bracket_pos = 0 if bracket <= 3 else (1 if bracket <= 5 else 2)
    horse_num_ratio = horse_num / nh

    odds = float(row['odds'])
    odds_log = math.log(odds)
    pop_rank = int(row['pop_rank'])

    # 騎手 stats
    j = str(row['jockey']).strip()
    js = JOCKEY_OVERRIDE.get(j) or jockey_stats.get(j) or {'wr': 0.05, 'place_rate': 0.18, 'runs': 100}
    j_wr = js['wr']
    j_pr = js['place_rate']

    # dist_cat: pd.cut bins=[0, 1200, 1400, 1800, 2200, 9999] → 1600m = label 2
    dist_cat = 2

    weight_cat = 1  # 馬体重不明 → mid default
    age_group = max(0, min(age - 3, 5))

    feat = {
        'odds_log': odds_log,
        'num_horses': nh,
        'distance': 1600,
        'surface_enc': 1,         # dirt
        'condition_enc': 0,       # 良
        'course_enc': 43,         # 船橋
        'horse_weight': 480.0,    # mean fill (no data)
        'weight_carry': 57.0,
        'age': age,
        'sex_enc': sex_enc,
        'horse_num': horse_num,
        'bracket': bracket,
        'horse_num_ratio': horse_num_ratio,
        'bracket_pos': bracket_pos,
        'carry_diff': 0.0,        # 全頭57kg
        'dist_cat': dist_cat,
        'weight_cat': weight_cat,
        'age_group': age_group,
        'jockey_wr': j_wr,
        'jockey_place_rate': j_pr,
        'pop_rank': pop_rank,
        'is_nar': 1,
    }
    return feat


def main():
    print("=" * 60)
    print("5/5 かしわ記念 (船橋11R Jpn1) NAR v4 予測")
    print("=" * 60)

    m = pickle.load(open(MODEL_PATH, 'rb'))
    lgb_model = m['model']
    xgb_model = m['xgb_model']
    features = m['features']
    weights = m['ensemble_weights']
    jockey_stats = m.get('jockey_stats', {})
    print(f"  AUC: {m['auc']:.4f} (lgb={m['lgb_auc']:.4f}, xgb={m['xgb_auc']:.4f})")
    print(f"  features={len(features)}, weights LGB={weights['lgb']:.3f} XGB={weights['xgb']:.3f}")
    print(f"  jockey_stats: {len(jockey_stats)} 騎手")

    df = pd.read_csv(HORSES_CSV)
    rows = [encode(r, jockey_stats) for _, r in df.iterrows()]
    X = pd.DataFrame(rows)[features]

    p_lgb = lgb_model.predict(X.values)
    import xgboost as xgb
    dmat = xgb.DMatrix(X.values, feature_names=features)
    try:
        p_xgb = xgb_model.predict(dmat)
    except Exception as e:
        print(f"  XGB predict failed: {e}, trying alt...")
        p_xgb = xgb_model.predict(xgb.DMatrix(X.values))

    p_ens = weights['lgb'] * p_lgb + weights['xgb'] * p_xgb

    df['p_lgb'] = p_lgb
    df['p_xgb'] = p_xgb
    df['p_ens'] = p_ens
    df = df.sort_values('p_ens', ascending=False).reset_index(drop=True)
    df['rank'] = range(1, len(df)+1)

    print("\n=== 全頭ランキング ===")
    print(df[['rank','horse_num','horse_name','jockey','location','odds','pop_rank','p_lgb','p_xgb','p_ens']].to_string(index=False))

    out_csv = 'data/results/20260505_kashiwa_kinen_nar_v4.csv'
    df.to_csv(out_csv, index=False, encoding='utf-8-sig')
    print(f"\n  ✓ Saved {out_csv}")

    # === 三連複 7点 (V13.5b式 1×2×5) ===
    top1 = int(df.iloc[0]['horse_num'])
    top2 = int(df.iloc[1]['horse_num'])
    top3 = int(df.iloc[2]['horse_num'])
    top456 = [int(df.iloc[i]['horse_num']) for i in [3,4,5]]
    print("\n=== 三連複 7点 1x2x5 ===")
    print(f"  軸: {top1} ({df.iloc[0]['horse_name']})")
    print(f"  2列目: {top2}, {top3}")
    print(f"  3列目: {top2}, {top3}, {top456[0]}, {top456[1]}, {top456[2]}")
    combos = []
    seen = set()
    for a in [top2, top3]:
        for b in [top2, top3] + top456:
            if b == a or b == top1: continue
            combo = tuple(sorted([top1, a, b]))
            if combo not in seen:
                seen.add(combo)
                combos.append(combo)
    combos = combos[:7]
    print("\n  7点組合せ:")
    for i, c in enumerate(combos, 1):
        print(f"    {i}. {c[0]:>2}-{c[1]:>2}-{c[2]:>2}")


if __name__ == '__main__':
    main()
