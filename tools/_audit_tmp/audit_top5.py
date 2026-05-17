"""Top5 coverage audit for 5/16 — read-only."""
import json
import pandas as pd

PRED_FP = 'C:/Users/takum/keiba-ai/data/daily_predictions_full/20260516.csv'
RES_FP  = 'C:/Users/takum/keiba-ai/data/daily_results/20260516.csv'

pred = pd.read_csv(PRED_FP, dtype={'race_id': str}, encoding='utf-8-sig')
res  = pd.read_csv(RES_FP,  dtype={'race_id': str}, encoding='utf-8-sig')


def classify_tier(name):
    n = str(name)
    if '新潟大賞典' in n:
        return 'G3'
    if '鞍馬S' in n or '六社S' in n:
        return 'L'
    if 'G1' in n:
        return 'G1'
    if 'G2' in n:
        return 'G2'
    if 'G3' in n:
        return 'G3'
    if '新馬' in n:
        return '新馬'
    if '未勝利' in n:
        return '未勝利'
    if '3歳1勝' in n:
        return '3-1勝C'
    if '4歳以上1勝' in n or '1勝クラス' in n:
        return '1勝C'
    if '4歳以上2勝' in n or '2勝クラス' in n:
        return '2勝C'
    if '3勝' in n:
        return '3勝C'
    if n.strip().endswith('S') or 'ステークス' in n:
        return 'OP/L'
    if '特別' in n or n.strip().endswith('賞'):
        return 'OP特'
    return '?'


race_ids = sorted(set(res['race_id']) & set(pred['race_id']))

meta = pred[['race_id', 'course', 'race_num', 'race_name', 'num_horses',
             'distance', 'surface', 'condition']].drop_duplicates('race_id').set_index('race_id')

rows = []
all_horse_rows = []
for rid in race_ids:
    p = pred[pred['race_id'] == rid].sort_values('rank_in_race')
    top5_nums = p[p['rank_in_race'] <= 5]['horse_num'].astype(int).tolist()
    top3_nums = p[p['rank_in_race'] <= 3]['horse_num'].astype(int).tolist()
    rr = res[res['race_id'] == rid].iloc[0]
    trio_res = str(rr['trio_result'])
    finish_nums = [int(x) for x in trio_res.split('-')]

    cov5 = len(set(top5_nums) & set(finish_nums))
    cov3 = len(set(top3_nums) & set(finish_nums))

    m = meta.loc[rid]
    tier = classify_tier(m['race_name'])
    rows.append({
        'race_id': rid,
        'course': m['course'],
        'race_num': int(m['race_num']),
        'race_name': m['race_name'],
        'tier': tier,
        'condition_band': rr['condition'],
        'num_horses': int(m['num_horses']),
        'distance': int(m['distance']),
        'surface': m['surface'],
        'top5_nums': top5_nums,
        'top3_nums': top3_nums,
        'finish_nums': finish_nums,
        'cov5': cov5,
        'cov3': cov3,
        'trio_hit_7pt': int(rr['trio_hit']),
        'trio_payout_7pt': int(rr['trio_payout']) if pd.notna(rr['trio_payout']) else 0,
        'umaren_hit': int(rr['umaren_hit']) if pd.notna(rr['umaren_hit']) else 0,
        'umaren_payout': int(rr['umaren_payout']) if pd.notna(rr['umaren_payout']) else 0,
    })

    for _, hr in p.iterrows():
        all_horse_rows.append({
            'race_id': rid,
            'course': m['course'],
            'R': int(m['race_num']),
            'rank': int(hr['rank_in_race']),
            'horse_num': int(hr['horse_num']),
            'horse_name': hr['horse_name'],
            'V15_score': float(hr['V15_score']),
            'odds': hr['odds'] if pd.notna(hr['odds']) else '',
            'in_finish': '*' if int(hr['horse_num']) in set(finish_nums) else '',
        })


df = pd.DataFrame(rows)
hdf = pd.DataFrame(all_horse_rows)
df.to_csv('C:/Users/takum/keiba-ai/tools/_audit_tmp/coverage_5_16_full.csv',
          index=False, encoding='utf-8-sig')
hdf.to_csv('C:/Users/takum/keiba-ai/tools/_audit_tmp/all_horses_5_16.csv',
           index=False, encoding='utf-8-sig')

N = len(df)
agg = {
    'N_total': N,
    'cov5_dist': {int(k): int((df['cov5'] == k).sum()) for k in [0, 1, 2, 3]},
    'cov3_dist': {int(k): int((df['cov3'] == k).sum()) for k in [0, 1, 2, 3]},
    'mean_cov5': float(df['cov5'].mean()),
    'mean_cov3': float(df['cov3'].mean()),
    'top5_complete_R': int((df['cov5'] == 3).sum()),
    'top5_complete_pct': float((df['cov5'] == 3).mean() * 100),
    'top3_complete_R': int((df['cov3'] == 3).sum()),
    'top3_complete_pct': float((df['cov3'] == 3).mean() * 100),
    'zero_R': int((df['cov5'] == 0).sum()),
}

tier_agg = df.groupby('tier').agg(
    N=('race_id', 'count'),
    cov5_complete=('cov5', lambda s: int((s == 3).sum())),
    mean_cov5=('cov5', 'mean'),
).reset_index()
tier_agg['complete_pct'] = tier_agg['cov5_complete'] / tier_agg['N'] * 100
agg['tier'] = tier_agg.to_dict('records')


def strat7_exclude(row):
    if row['course'] == '京都':
        return True
    if row['condition_band'] in ('B', 'E'):
        return True
    nm = row['race_name']
    if (nm.endswith('特別') or nm.endswith('賞')) and row['tier'] in ('OP特',):
        return True
    return False


df['strat7_excluded'] = df.apply(strat7_exclude, axis=1)
sub = df[~df['strat7_excluded']]
agg['strat7_remain_N'] = int(len(sub))
agg['strat7_complete_R'] = int((sub['cov5'] == 3).sum())
agg['strat7_complete_pct'] = float((sub['cov5'] == 3).mean() * 100) if len(sub) > 0 else 0.0
agg['strat7_excluded_R'] = int(df['strat7_excluded'].sum())

df['box_hit'] = (df['cov5'] == 3).astype(int)
df['box_payout_known'] = df.apply(
    lambda r: int(r['trio_payout_7pt']) if r['box_hit'] == 1 and r['trio_hit_7pt'] == 1 else 0, axis=1)
df['box_payout_unknown'] = df.apply(
    lambda r: 1 if r['box_hit'] == 1 and r['trio_hit_7pt'] == 0 else 0, axis=1)

box_invest = N * 1000
box_payout_known = int(df['box_payout_known'].sum())
agg['box_top5_invest'] = box_invest
agg['box_top5_known_payout'] = box_payout_known
agg['box_top5_unknown_hit_R'] = int(df['box_payout_unknown'].sum())
agg['box_top5_lower_roi_pct'] = box_payout_known / box_invest * 100
agg['box_top5_hit_R'] = int(df['box_hit'].sum())
agg['box_top5_hit_pct'] = float(df['box_hit'].mean() * 100)

sub_box_inv = (~df['strat7_excluded']).sum() * 1000
sub_box_pay = int(df[~df['strat7_excluded']]['box_payout_known'].sum())
agg['strat7_box_invest'] = int(sub_box_inv)
agg['strat7_box_known_payout'] = sub_box_pay
agg['strat7_box_lower_roi_pct'] = (sub_box_pay / sub_box_inv * 100) if sub_box_inv > 0 else 0
agg['strat7_box_hit_R'] = int(df[~df['strat7_excluded']]['box_hit'].sum())
agg['strat7_box_unknown_R'] = int(df[~df['strat7_excluded']]['box_payout_unknown'].sum())

agg['cur_7pt_invest'] = N * 700
agg['cur_7pt_payout_sum'] = int(df['trio_payout_7pt'].sum())
agg['cur_7pt_hit_R'] = int(df['trio_hit_7pt'].sum())
agg['cur_7pt_roi_pct'] = agg['cur_7pt_payout_sum'] / agg['cur_7pt_invest'] * 100

# strat7 applied current 7pt
agg['strat7_cur_7pt_invest'] = int((~df['strat7_excluded']).sum() * 700)
agg['strat7_cur_7pt_payout'] = int(df[~df['strat7_excluded']]['trio_payout_7pt'].sum())
agg['strat7_cur_7pt_hit_R'] = int(df[~df['strat7_excluded']]['trio_hit_7pt'].sum())
agg['strat7_cur_7pt_roi_pct'] = (agg['strat7_cur_7pt_payout'] / agg['strat7_cur_7pt_invest'] * 100) if agg['strat7_cur_7pt_invest'] > 0 else 0

# 重賞 verdict
key_races = ['202604010511', '202605020711', '202608030711']  # 新潟大賞典, 六社S, 鞍馬S
key = df[df['race_id'].isin(key_races)][
    ['race_id', 'course', 'race_num', 'race_name', 'tier',
     'top5_nums', 'finish_nums', 'cov5', 'trio_hit_7pt', 'trio_payout_7pt']
].to_dict('records')
agg['key_races'] = key

with open('C:/Users/takum/keiba-ai/tools/_audit_tmp/agg_5_16.json', 'w', encoding='utf-8') as f:
    json.dump(agg, f, ensure_ascii=False, indent=2, default=str)

print(json.dumps(agg, ensure_ascii=False, indent=2, default=str))
