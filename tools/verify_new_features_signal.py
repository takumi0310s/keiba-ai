#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Phase 22-23 で追加した新 features の signal 強度を verify.

event_effect_features.csv + race_review_features.csv で LGB 学習し、
AUC + feature importance を計測。 真に V20/V21 features 候補として 有意か。

【V15 投資保護】 train/ 不変、 検証のみ、 production 関連 file 一切 触らず

Usage:
    python tools/verify_new_features_signal.py
    python tools/verify_new_features_signal.py --features events,remarks
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


def main():
    ap = argparse.ArgumentParser(description='Phase 22-23 features signal verify')
    ap.add_argument('--features', default='events,remarks',
                    help='検証 feature set comma sep (events / remarks)')
    ap.add_argument('--sample', type=int, default=None, help='sample 数制限 (test 用)')
    args = ap.parse_args()

    import pandas as pd
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split

    features_to_use = [f.strip() for f in args.features.split(',')]
    print(f'[INFO] verify features: {features_to_use}')

    # Load base
    base_path = os.path.join(BASE_DIR, 'data', 'jra_races_full.csv')
    df = pd.read_csv(base_path, encoding='utf-8', low_memory=False,
                      usecols=['race_id', 'horse_id', 'umaban', 'finish', 'year', 'class_code',
                               'distance', 'surface', 'condition', 'num_horses', 'age',
                               'horse_weight', 'tansho_odds'])
    print(f'[INFO] base loaded: {df.shape}')

    df['race_id'] = df['race_id'].astype(str)
    df['horse_id'] = df['horse_id'].astype(str)
    df = df[df['finish'] > 0]
    df['top3'] = (df['finish'] <= 3).astype(int)

    # Merge events
    if 'events' in features_to_use:
        evt = pd.read_csv(os.path.join(BASE_DIR, 'data', 'event_effect_features.csv'),
                           encoding='utf-8')
        evt['race_id'] = evt['race_id'].astype(str)
        evt['horse_id'] = evt['horse_id'].astype(str)
        evt_cols = ['race_id', 'horse_id'] + [c for c in evt.columns
                                                 if any(k in c for k in ['change', '_up', '_down', '_rate_exp'])]
        evt = evt[evt_cols].drop_duplicates(['race_id', 'horse_id'])
        df = df.merge(evt, on=['race_id', 'horse_id'], how='left')
        print(f'[INFO] +events features: {df.shape}')

    # Merge remarks (by umaban)
    if 'remarks' in features_to_use:
        rmk = pd.read_csv(os.path.join(BASE_DIR, 'data', 'race_review_features.csv'),
                           encoding='utf-8')
        rmk['race_id'] = rmk['race_id'].astype(str)
        rmk_cols = ['race_id', 'umaban'] + [c for c in rmk.columns if c.startswith('rmk_')]
        rmk = rmk[rmk_cols].drop_duplicates(['race_id', 'umaban'])
        df['umaban'] = pd.to_numeric(df['umaban'], errors='coerce')
        rmk['umaban'] = pd.to_numeric(rmk['umaban'], errors='coerce')
        df = df.merge(rmk, on=['race_id', 'umaban'], how='left')
        print(f'[INFO] +remarks features: {df.shape}')

    # Encode categoricals
    for c in ['surface', 'condition', 'class_code']:
        if c in df.columns:
            df[c] = df[c].astype('category').cat.codes

    if args.sample:
        df = df.sample(n=args.sample, random_state=42).reset_index(drop=True)

    # Target / features split
    target_col = 'top3'
    drop_cols = {'race_id', 'horse_id', 'finish', 'top3'}
    feature_cols = [c for c in df.columns if c not in drop_cols]

    df = df.dropna(subset=[target_col])
    if len(df) < 1000:
        print('[ERROR] not enough samples')
        return 1
    print(f'[INFO] training set: {df.shape}, features: {len(feature_cols)}')

    # year ベース WF (簡易): 24 train, 25 test
    if 25 in df['year'].values and 24 in df['year'].values:
        train_df = df[df['year'] < 25]
        test_df = df[df['year'] == 25]
    else:
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    print(f'[INFO] train: {len(train_df)}, test: {len(test_df)}')

    X_tr, y_tr = train_df[feature_cols].fillna(-1), train_df[target_col]
    X_te, y_te = test_df[feature_cols].fillna(-1), test_df[target_col]

    # 2 model 比較: baseline (event/remarks 除外) vs full
    new_feature_keys = [c for c in feature_cols
                        if c.startswith(('rmk_', 'jockey_change', 'trainer_change',
                                          'class_change', 'class_up', 'class_down',
                                          'equipment_change'))
                        or '_rate_exp' in c]
    base_features = [c for c in feature_cols if c not in new_feature_keys]
    print(f'[INFO] base features: {len(base_features)}, new features: {len(new_feature_keys)}')

    params = {
        'objective': 'binary',
        'metric': 'auc',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'verbose': -1,
        'seed': 42,
    }

    print('\n[1/2] BASELINE (新features なし)')
    m_base = lgb.train(params, lgb.Dataset(X_tr[base_features], y_tr),
                        num_boost_round=200,
                        valid_sets=[lgb.Dataset(X_te[base_features], y_te)],
                        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(50)])
    pred_base = m_base.predict(X_te[base_features])
    auc_base = roc_auc_score(y_te, pred_base)

    print('\n[2/2] FULL (新features 込み)')
    m_full = lgb.train(params, lgb.Dataset(X_tr, y_tr),
                        num_boost_round=200,
                        valid_sets=[lgb.Dataset(X_te, y_te)],
                        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(50)])
    pred_full = m_full.predict(X_te)
    auc_full = roc_auc_score(y_te, pred_full)

    print('\n=== RESULT ===')
    print(f'  AUC baseline:  {auc_base:.4f}')
    print(f'  AUC + new:     {auc_full:.4f}')
    print(f'  Delta:         {auc_full - auc_base:+.4f}')

    # Feature importance
    print('\n[Top 15 features (full model)]')
    imps = m_full.feature_importance(importance_type='gain')
    fnames = m_full.feature_name()
    ranked = sorted(zip(fnames, imps), key=lambda x: -x[1])[:15]
    for name, imp in ranked:
        marker = ' ★' if name in new_feature_keys else '  '
        print(f'  {imp:>12.1f}  {marker} {name}')

    # New features specific importance
    print('\n[New features 自身 の importance]')
    new_imps = [(n, i) for n, i in zip(fnames, imps) if n in new_feature_keys]
    for name, imp in sorted(new_imps, key=lambda x: -x[1]):
        print(f'  {imp:>12.1f}  {name}')

    return 0


if __name__ == '__main__':
    sys.exit(main())
