#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""C2-C5: Jockey / Trainer / Class / Equipment change effect features.

V15 既存 features に 以下 4 種を追加 計算:
- C2 jockey_change_flag: 騎手 乗り替わり (前走 != 今走)
- C3 trainer_change_flag: 厩舎 移籍
- C4 class_change_flag: 升級 / 降級
- C5 equipment_change_flag: 装鞍変更 (JRDB 装鞍 data 利用、 ない場合は flag のみ)

加えて 各 change の expanding window 効果 (過去 全 race で change のとき top3 率) を計算。

【V15 投資保護】 train/ V15 関連 file 触らず、 新規 features csv 生成のみ。

Usage:
    python tools/build_event_effect_features.py
    python tools/build_event_effect_features.py --out data/event_effect_features.csv
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
INPUT_PATH = os.path.join(BASE_DIR, 'data', 'jra_races_full.csv')


def main():
    ap = argparse.ArgumentParser(description='Jockey/Trainer/Class/Equipment change features (C2-C5)')
    ap.add_argument('--out', default=os.path.join(BASE_DIR, 'data', 'event_effect_features.csv'))
    args = ap.parse_args()

    import pandas as pd
    print(f'[INFO] reading: {INPUT_PATH}')
    df = pd.read_csv(INPUT_PATH, encoding='utf-8', low_memory=False)
    print(f'[INFO] {len(df)} rows, {len(df.columns)} cols')

    # 必須 col 確認
    must = ['race_id', 'horse_id', 'jockey_id', 'trainer_id', 'class_code', 'finish']
    missing = [c for c in must if c not in df.columns]
    if missing:
        print(f'[ERROR] missing columns: {missing}')
        return 1

    # date or race_id でソート (race_id 形式 = YYYYxxxxxx でほぼ時間順)
    df = df.sort_values(['horse_id', 'race_id']).reset_index(drop=True)

    # 各馬の前走 entry を shift で取得
    gb = df.groupby('horse_id')
    df['prev_jockey_id'] = gb['jockey_id'].shift(1)
    df['prev_trainer_id'] = gb['trainer_id'].shift(1)
    df['prev_class_code'] = gb['class_code'].shift(1)

    # change flags
    df['jockey_change'] = ((df['jockey_id'].notna()) &
                            (df['prev_jockey_id'].notna()) &
                            (df['jockey_id'] != df['prev_jockey_id'])).astype(int)
    df['trainer_change'] = ((df['trainer_id'].notna()) &
                             (df['prev_trainer_id'].notna()) &
                             (df['trainer_id'] != df['prev_trainer_id'])).astype(int)
    # class change: code 大きい = 高 class (一般的に)、 +1 = 昇級、 -1 = 降級
    df['class_change'] = ((df['class_code'].notna()) &
                           (df['prev_class_code'].notna()) &
                           (df['class_code'] != df['prev_class_code'])).astype(int)
    df['class_up'] = ((df['class_code'].notna()) &
                       (df['prev_class_code'].notna()) &
                       (df['class_code'] > df['prev_class_code'])).astype(int)
    df['class_down'] = ((df['class_code'].notna()) &
                         (df['prev_class_code'].notna()) &
                         (df['class_code'] < df['prev_class_code'])).astype(int)

    # equipment_change: JRDB 装鞍 data なしの場合 flag だけ 0 で生成 (拡張用 placeholder)
    df['equipment_change'] = 0  # TODO: JRDB 装鞍 csv あれば差分計算

    # top3 indicator (target)
    df['top3'] = (df['finish'] <= 3).astype(int)

    # expanding effect: 過去 jockey_change=1 だった race の top3 率 (累計)
    for col in ['jockey_change', 'trainer_change', 'class_up', 'class_down']:
        # global mean (data leak 防止のため shift で過去のみ集計)
        sub = df[df[col] == 1].copy()
        sub_sorted = sub.sort_values('race_id').reset_index(drop=True)
        cumsum = sub_sorted['top3'].cumsum().shift(1).fillna(0)
        count = pd.Series(range(len(sub_sorted)))
        eff_col = f'{col}_top3_rate_exp'
        sub_sorted[eff_col] = (cumsum / count.where(count > 0, 1)).fillna(0.33)
        # merge back
        df = df.merge(sub_sorted[['race_id', 'horse_id', eff_col]],
                       on=['race_id', 'horse_id'], how='left')
        df[eff_col] = df[eff_col].fillna(0.33)  # default = baseline top3 rate
        print(f'  {col}: n={sub[col].sum()}, top3 rate={sub["top3"].mean():.4f}, '
              f'exp rate end={sub_sorted[eff_col].iloc[-1]:.4f}')

    keep_cols = ['race_id', 'horse_id', 'finish', 'top3',
                 'jockey_change', 'trainer_change', 'class_change',
                 'class_up', 'class_down', 'equipment_change',
                 'jockey_change_top3_rate_exp', 'trainer_change_top3_rate_exp',
                 'class_up_top3_rate_exp', 'class_down_top3_rate_exp']
    out_df = df[keep_cols].copy()
    out_df.to_csv(args.out, index=False, encoding='utf-8')
    print(f'[OK] saved: {args.out}')
    print(f'[OK] shape: {out_df.shape}, cols: {list(out_df.columns)}')

    # 統計
    for col in ['jockey_change', 'trainer_change', 'class_change', 'class_up', 'class_down']:
        n = out_df[col].sum()
        rate = out_df[col].mean()
        if n > 0:
            sub_top3 = out_df[out_df[col] == 1]['top3'].mean()
            base_top3 = out_df[out_df[col] == 0]['top3'].mean()
            print(f'  {col}: n={n}, rate={rate*100:.1f}%, top3 when 1: {sub_top3:.3f}, when 0: {base_top3:.3f}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
