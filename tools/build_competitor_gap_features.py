#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Agent 1 で発見した 高 priority 4 features を 即実装.

1. prev_race_disadvantage_score: race_review.csv (277K) から 不利スコア (NLP scoring)
2. corner_position_delta: prev_pass1 - prev_pass4 (動いた量)
3. jockey_trainer_combo_wr: jockey × trainer pair expanding wr
4. bagu_change_flag: 馬具 (ブリンカー) 装着 変更 (JRDB 既存 - 簡易判定)

【V15 投資保護】 derivative features のみ、 V15 model 不変
"""
import argparse
import os
import sys

try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# race_review remarks → score (Agent 1 提案、 重大度別)
REMARK_SEVERITY = {
    # 重不利 (-2)
    '降着': -2.0, '降下': -2.0, '失格': -2.0,
    # 中不利 (-1.5)
    'スタート不利': -1.5, '出遅れ': -1.5,
    # 軽不利 (-1)
    '不利': -1.0, '接触': -1.0, '挟まれる': -1.0,
    'ヨレる': -0.8, '膨れる': -0.7, '馬群': -0.5,
    # 注意 (-0.3)
    '展開不向き': -0.3, 'スローペース': -0.3, 'ハイペース': -0.3,
}


def score_remark(remark):
    if not isinstance(remark, str) or not remark.strip():
        return 0.0
    score = 0.0
    for kw, sc in REMARK_SEVERITY.items():
        if kw in remark:
            score += sc
    return score


def main():
    import pandas as pd

    # 1. prev_race_disadvantage_score
    print('=== 1. prev_race_disadvantage_score ===')
    rmk = pd.read_csv(os.path.join(BASE_DIR, 'data', 'netkeiba_race_review.csv'),
                       encoding='utf-8', dtype={'race_id': str})
    rmk['disadv_score'] = rmk['remarks'].apply(score_remark)
    rmk_with = rmk[rmk['disadv_score'] != 0]
    print(f'  {len(rmk):,} rows、 disadv > 0: {len(rmk_with):,}')
    print(f'  score range: [{rmk["disadv_score"].min():.2f}, {rmk["disadv_score"].max():.2f}]')

    # 2. corner_position_delta (jra_races_full から)
    print('\n=== 2. corner_position_delta ===')
    df = pd.read_csv(os.path.join(BASE_DIR, 'data', 'jra_races_full.csv'),
                      encoding='utf-8', low_memory=False,
                      usecols=['race_id', 'horse_id', 'finish', 'year',
                               'pass1', 'pass2', 'pass3', 'pass4',
                               'jockey_id', 'trainer_id'])
    df = df[df['year'] >= 18]
    df = df[df['finish'] > 0]
    df['top3'] = (df['finish'] <= 3).astype(int)
    df['race_id'] = df['race_id'].astype(str)
    df['horse_id'] = df['horse_id'].astype(str)

    for c in ['pass1', 'pass2', 'pass3', 'pass4']:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # 当該 race の corner data は POST-RACE (LEAK)、 expanding prev で構築
    df = df.sort_values(['horse_id', 'race_id']).reset_index(drop=True)
    gb = df.groupby('horse_id')
    df['prev_pass1'] = gb['pass1'].shift(1)
    df['prev_pass4'] = gb['pass4'].shift(1)
    df['corner_position_delta'] = df['prev_pass1'] - df['prev_pass4']  # +なら前進

    valid = df.dropna(subset=['corner_position_delta'])
    if len(valid) > 0:
        # signal verify
        q = pd.qcut(valid['corner_position_delta'].rank(method='first'),
                     5, labels=[1, 2, 3, 4, 5])
        valid['_q'] = q
        for qi in [1, 5]:
            tr = valid[valid['_q'] == qi]['top3'].mean()
            print(f'  Q{qi}: top3 rate {tr:.3f}')

    # 3. jockey_trainer_combo_wr (expanding)
    print('\n=== 3. jockey_trainer_combo_wr ===')
    df['jockey_id_str'] = df['jockey_id'].astype(str)
    df['trainer_id_str'] = df['trainer_id'].astype(str)
    df['j_t_combo'] = df['jockey_id_str'] + '_' + df['trainer_id_str']
    # expanding: 過去 combo で の top3 rate
    df_sorted = df.sort_values(['j_t_combo', 'race_id']).reset_index(drop=True)
    gb_jt = df_sorted.groupby('j_t_combo')
    df_sorted['jockey_trainer_combo_top3_exp'] = (
        gb_jt['top3'].apply(lambda s: s.shift(1).expanding().mean()).reset_index(level=0, drop=True)
    )
    # merge back
    df = df.merge(df_sorted[['race_id', 'horse_id', 'jockey_trainer_combo_top3_exp']]
                    .drop_duplicates(['race_id', 'horse_id']),
                    on=['race_id', 'horse_id'], how='left')
    valid_jt = df.dropna(subset=['jockey_trainer_combo_top3_exp'])
    if len(valid_jt) > 0:
        q = pd.qcut(valid_jt['jockey_trainer_combo_top3_exp'].rank(method='first'),
                     5, labels=[1, 2, 3, 4, 5])
        valid_jt['_q'] = q
        for qi in [1, 5]:
            tr = valid_jt[valid_jt['_q'] == qi]['top3'].mean()
            print(f'  Q{qi}: top3 rate {tr:.3f}')
        delta = (valid_jt[valid_jt['_q'] == 5]['top3'].mean() -
                  valid_jt[valid_jt['_q'] == 1]['top3'].mean())
        print(f'  Q5-Q1 delta: {delta:+.3f}')

    # 4. bagu_change_flag (JRDB SKB 等 から推定、 簡易版)
    print('\n=== 4. bagu_change_flag (skeleton、 JRDB SKB 詳細 解析 必要) ===')
    skb_path = os.path.join(BASE_DIR, 'data', 'jrdb_skb.csv')
    if os.path.exists(skb_path):
        skb = pd.read_csv(skb_path, encoding='utf-8', low_memory=False, nrows=10000)
        bagu_cols = [c for c in skb.columns if any(k in c.lower() for k in
                                                       ['blinker', 'bagu', 'ブリンカー', 'shozoku'])]
        print(f'  JRDB SKB columns: {len(skb.columns)} 個')
        print(f'  bagu 関連: {bagu_cols}')
        print(f'  詳細実装: bagu 装着 vs 前走 比較 で flag 生成')
    else:
        print('  JRDB SKB 未取得')

    # Save features
    out = df[['race_id', 'horse_id', 'corner_position_delta',
              'jockey_trainer_combo_top3_exp']].copy()
    out_path = os.path.join(BASE_DIR, 'data', 'competitor_gap_features.csv')
    out.to_csv(out_path, index=False)
    print(f'\n[OK] saved: {out_path}')

    # disadv score も merge できるよう保存
    rmk[['race_id', 'umaban', 'disadv_score']].to_csv(
        os.path.join(BASE_DIR, 'data', 'prev_race_disadv_score.csv'), index=False)
    print(f'[OK] saved: data/prev_race_disadv_score.csv')
    return 0


if __name__ == '__main__':
    sys.exit(main())
