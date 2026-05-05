"""NAR v4 汎用 predict.

usage:
  # 単一レース (race_id 指定)
  python tools/predict_nar.py --race-id 202604301204

  # 当日全 NAR レース (data/nar_today_shutuba.csv が必要、scrape_nar_today で生成想定)
  python tools/predict_nar.py --date 20260512

  # 出馬表 CSV を直接指定 (柏記念互換)
  python tools/predict_nar.py --shutuba-csv data/results/20260505_kashiwa_kinen_horses.csv \
      --num-horses 13 --distance 1600 --course-enc 43

note:
  柏記念 ad-hoc script (predict_nar_kashiwa_5_5.py) を base に汎用化。
  fixed param (num_horses, distance, course_enc, condition_enc) は CLI 引数 or shutuba CSV 列で指定。

config: tools/nar_predict_config.json
model:  data/nar/models/keiba_model_nar_v4.pkl
"""
from __future__ import annotations

import os, sys, argparse, json, math, pickle
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
os.chdir(BASE_DIR)

CONFIG_PATH = 'tools/nar_predict_config.json'

# JRA 騎手 NAR jockey_stats override (NAR open race の JRA 騎手対応)
JOCKEY_OVERRIDE_JRA = {
    'ルメール':   {'wr': 0.226, 'place_rate': 0.564, 'runs': 600},
    '川田将雅':   {'wr': 0.234, 'place_rate': 0.541, 'runs': 700},
    '横山武史':   {'wr': 0.155, 'place_rate': 0.401, 'runs': 600},
    'レーン':     {'wr': 0.182, 'place_rate': 0.444, 'runs': 80},
    '横山和生':   {'wr': 0.103, 'place_rate': 0.305, 'runs': 700},
    '川須栄一':   {'wr': 0.043, 'place_rate': 0.180, 'runs': 700},
    '武豊':       {'wr': 0.130, 'place_rate': 0.358, 'runs': 700},
    '池添謙一':   {'wr': 0.092, 'place_rate': 0.291, 'runs': 600},
    '坂井瑠星':   {'wr': 0.118, 'place_rate': 0.330, 'runs': 600},
}


def load_config():
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_model(path):
    with open(path, 'rb') as f:
        m = pickle.load(f)
    return m


def parse_age(sex_age_str):
    s = str(sex_age_str).strip()
    for ch in '牡牝セ騙':
        s = s.replace(ch, '')
    try:
        return int(s)
    except Exception:
        return 4  # default


def parse_sex(sex_age_str):
    s = str(sex_age_str)
    if s.startswith('牡'): return 0
    if s.startswith('牝'): return 1
    if s.startswith('セ') or s.startswith('騙'): return 2
    return 0


def encode_one(row, *, num_horses, distance, surface_enc, condition_enc, course_enc,
               jockey_stats, default_horse_weight=480.0):
    horse_num = int(row['horse_num'])
    nh = int(num_horses)

    # 性別/年齢
    sex_age = row.get('sex_age', '牡4')
    age = parse_age(sex_age)
    sex_enc = parse_sex(sex_age)

    # bracket: NAR 8 枠制 (頭数別の枠割り、近似)
    if nh <= 8:
        bracket = horse_num
    else:
        # 9-18 頭で 8 枠に分散
        bracket = min(8, math.ceil(horse_num * 8 / nh))
    bracket_pos = 0 if bracket <= 3 else (1 if bracket <= 5 else 2)
    horse_num_ratio = horse_num / nh

    # オッズ
    odds_raw = row.get('odds', None)
    try:
        odds = float(odds_raw)
    except Exception:
        odds = None
    odds_log = math.log(odds) if (odds is not None and odds > 0) else 0.0
    pop_rank = int(row.get('pop_rank', 99) or 99)

    # 騎手
    j = str(row.get('jockey', '')).strip()
    js = JOCKEY_OVERRIDE_JRA.get(j) or jockey_stats.get(j) or {'wr': 0.05, 'place_rate': 0.18, 'runs': 100}
    j_wr = js['wr']
    j_pr = js['place_rate']

    # weight_carry (NAR は基本 同 kg or 別斤量)
    weight_carry = float(row.get('weight_carry', 56.0) or 56.0)

    # horse_weight (なければ default fill)
    hw_raw = row.get('horse_weight', None)
    try:
        horse_weight = float(hw_raw) if hw_raw not in (None, '', 'nan') else default_horse_weight
    except Exception:
        horse_weight = default_horse_weight

    # carry_diff: race 平均 carry が分からない場合 0.0 で fill
    carry_diff = 0.0  # race 全体集計が必要、ここでは省略

    # dist_cat: pd.cut bins=[0,1200,1400,1800,2200,9999] → label 0..4
    if distance <= 1200: dist_cat = 0
    elif distance <= 1400: dist_cat = 1
    elif distance <= 1800: dist_cat = 2
    elif distance <= 2200: dist_cat = 3
    else: dist_cat = 4

    # weight_cat (馬体重カテゴリ)
    if horse_weight < 440: weight_cat = 0
    elif horse_weight < 480: weight_cat = 1
    elif horse_weight < 520: weight_cat = 2
    else: weight_cat = 3

    age_group = max(0, min(age - 3, 5))

    return {
        'odds_log': odds_log,
        'num_horses': nh,
        'distance': distance,
        'surface_enc': surface_enc,
        'condition_enc': condition_enc,
        'course_enc': course_enc,
        'horse_weight': horse_weight,
        'weight_carry': weight_carry,
        'age': age,
        'sex_enc': sex_enc,
        'horse_num': horse_num,
        'bracket': bracket,
        'horse_num_ratio': horse_num_ratio,
        'bracket_pos': bracket_pos,
        'carry_diff': carry_diff,
        'dist_cat': dist_cat,
        'weight_cat': weight_cat,
        'age_group': age_group,
        'jockey_wr': j_wr,
        'jockey_place_rate': j_pr,
        'pop_rank': pop_rank,
        'is_nar': 1,
    }


def predict_one_race(df, race_meta, model_data, config):
    """df: 出馬表 (horse_num, horse_name, jockey, odds, pop_rank, sex_age, weight_carry [, horse_weight])
    race_meta: {'num_horses', 'distance', 'surface_enc', 'condition_enc', 'course_enc'}
    """
    features = model_data['features']
    weights = model_data['ensemble_weights']
    jockey_stats = model_data.get('jockey_stats', {})
    default_hw = config.get('default_horse_weight', 480.0)

    rows = []
    for _, r in df.iterrows():
        feat = encode_one(r, **race_meta, jockey_stats=jockey_stats, default_horse_weight=default_hw)
        rows.append(feat)
    X = pd.DataFrame(rows)[features]

    p_lgb = model_data['model'].predict(X.values)
    import xgboost as xgb
    dmat = xgb.DMatrix(X.values, feature_names=features)
    try:
        p_xgb = model_data['xgb_model'].predict(dmat)
    except Exception:
        p_xgb = model_data['xgb_model'].predict(xgb.DMatrix(X.values))

    p_ens = weights['lgb'] * p_lgb + weights['xgb'] * p_xgb

    out = df.copy()
    out['p_lgb'] = p_lgb
    out['p_xgb'] = p_xgb
    out['p_ens'] = p_ens
    out = out.sort_values('p_ens', ascending=False).reset_index(drop=True)
    out['rank'] = range(1, len(out) + 1)
    return out


def classify_condition(num_horses, distance, condition_enc, conditions_cfg):
    heavy = condition_enc >= 2
    nh = num_horses
    if nh <= 7: return 'E'
    if distance <= 1500: return 'D'
    if 8 <= nh <= 14 and distance >= 1600 and not heavy: return 'A'
    if 8 <= nh <= 14 and distance >= 1600 and heavy: return 'B'
    if nh >= 15 and distance >= 1600 and not heavy: return 'C'
    return 'X'


def build_buy_combos(ranked_df, condition):
    """JRA と同型: 三連複 7点 (1×2×5) or 馬連 2 点"""
    top_nums = [int(ranked_df.iloc[i]['horse_num']) for i in range(min(6, len(ranked_df)))]
    if condition == 'E':
        # 馬連 2 点
        return {
            'bet_type': 'umaren',
            'combos': [
                tuple(sorted([top_nums[0], top_nums[1]])),
                tuple(sorted([top_nums[0], top_nums[2]])),
            ]
        }
    # 三連複 7 点
    seen = set()
    combos = []
    top1, top2, top3 = top_nums[0], top_nums[1], top_nums[2]
    rest = top_nums[3:6]
    for a in [top2, top3]:
        for b in [top2, top3] + rest:
            if b == a or b == top1: continue
            c = tuple(sorted([top1, a, b]))
            if c not in seen:
                seen.add(c)
                combos.append(c)
            if len(combos) >= 7: break
        if len(combos) >= 7: break
    return {'bet_type': 'trio', 'combos': combos[:7]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--race-id', help='netkeiba race_id (将来 scrape from netkeiba.nar)')
    parser.add_argument('--date', help='YYYYMMDD — data/nar_today_shutuba_DATE.csv を読む')
    parser.add_argument('--shutuba-csv', help='出馬表 CSV を直接指定 (race_id/date 不要)')
    parser.add_argument('--num-horses', type=int, default=None)
    parser.add_argument('--distance', type=int, default=None)
    parser.add_argument('--surface-enc', type=int, default=1, help='芝=0, ダート=1 (NAR 基本 1)')
    parser.add_argument('--condition-enc', type=int, default=0, help='良=0, 稍=1, 重=2, 不良=3')
    parser.add_argument('--course-enc', type=int, default=43, help='場 code (船橋=43)')
    parser.add_argument('--output-csv', default=None, help='予測結果 CSV')
    args = parser.parse_args()

    config = load_config()
    model = load_model(config['model_path'])
    print(f"[NAR v4] AUC={model['auc']:.4f} features={len(model['features'])}", flush=True)

    if not (args.race_id or args.date or args.shutuba_csv):
        parser.error("--race-id / --date / --shutuba-csv のいずれか必要")

    # --- shutuba 入力源を選択 ---
    if args.shutuba_csv:
        df = pd.read_csv(args.shutuba_csv)
        nh = args.num_horses or len(df)
        dist = args.distance or 1600
        meta = {
            'num_horses': nh, 'distance': dist,
            'surface_enc': args.surface_enc, 'condition_enc': args.condition_enc,
            'course_enc': args.course_enc,
        }
        ranked = predict_one_race(df, meta, model, config)
        cond = classify_condition(nh, dist, args.condition_enc, config['conditions'])
        print(f"  num_horses={nh} distance={dist} → 条件 {cond}")
        print('\n=== 全頭ランキング ===')
        cols = [c for c in ['rank','horse_num','horse_name','jockey','odds','pop_rank','p_lgb','p_xgb','p_ens'] if c in ranked.columns]
        print(ranked[cols].to_string(index=False))
        buy = build_buy_combos(ranked, cond)
        print(f'\n=== {buy["bet_type"]} {len(buy["combos"])} 点 ===')
        for i, c in enumerate(buy['combos'], 1):
            print(f"  {i}. {'-'.join(str(x) for x in c)}")
        if args.output_csv:
            out_dir = os.path.dirname(args.output_csv)
            if out_dir and not os.path.exists(out_dir): os.makedirs(out_dir)
            ranked.to_csv(args.output_csv, index=False, encoding='utf-8-sig')
            print(f"\n  [OK] Saved {args.output_csv}")
        return

    # --- date 指定: 当日全 NAR (scrape_nar_today.py 経由で生成された CSV を読む) ---
    if args.date:
        shutuba_path = f'data/nar_today_shutuba_{args.date}.csv'
        if not os.path.exists(shutuba_path):
            print(f"[WARN] {shutuba_path} not found.")
            print(f"       先に scrape_nar_today.py --date {args.date} を実行するか、")
            print(f"       --shutuba-csv で出馬表 CSV を直接指定してください。")
            if args.output_csv:
                out_dir = os.path.dirname(args.output_csv)
                if out_dir and not os.path.exists(out_dir): os.makedirs(out_dir)
                pd.DataFrame(columns=['race_id','horse_num','p_ens','rank']).to_csv(args.output_csv, index=False, encoding='utf-8-sig')
                print(f"       [empty] {args.output_csv} 生成 (pipeline 次 step エラー回避)")
            return
        # 実装: race_id 単位で group 化して predict
        all_results = []
        df_all = pd.read_csv(shutuba_path)
        for rid, sub in df_all.groupby('race_id'):
            nh = len(sub)
            # race_meta は CSV に列が必要 (distance/condition_enc/course_enc/surface_enc)
            r0 = sub.iloc[0]
            meta = {
                'num_horses': nh,
                'distance': int(r0.get('distance', 1600)),
                'surface_enc': int(r0.get('surface_enc', 1)),
                'condition_enc': int(r0.get('condition_enc', 0)),
                'course_enc': int(r0.get('course_enc', 43)),
            }
            ranked = predict_one_race(sub, meta, model, config)
            ranked['race_id'] = rid
            cond = classify_condition(meta['num_horses'], meta['distance'], meta['condition_enc'], config['conditions'])
            ranked['condition'] = cond
            all_results.append(ranked)
        if all_results:
            res = pd.concat(all_results, ignore_index=True)
            if args.output_csv:
                out_dir = os.path.dirname(args.output_csv)
                if out_dir and not os.path.exists(out_dir): os.makedirs(out_dir)
                res.to_csv(args.output_csv, index=False, encoding='utf-8-sig')
                print(f"  [OK] Saved {args.output_csv} ({len(res)} rows, {len(all_results)} races)")
        return

    # --- race-id 指定: 単一レース (将来 netkeiba.nar から fetch) ---
    if args.race_id:
        print(f"[NOT IMPLEMENTED] race_id={args.race_id}: netkeiba.nar fetch は未実装。")
        print(f"  暫定: --shutuba-csv で出馬表を渡してください。")
        sys.exit(2)


if __name__ == '__main__':
    main()
