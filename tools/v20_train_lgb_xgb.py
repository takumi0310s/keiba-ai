#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""V20 simplified training (LGB + XGB 2-ensemble、 5/24+ 投入 用 候補).

v20_training_data_full.csv (190K rows × 101 cols) を 入力に、
LGB + XGB 2-ensemble の WF 6-fold backtest を 実行。
V135b の 4-ensemble (LGB+XGB+FT+IntraRace) は 重いので 簡易版として LGB+XGB のみ。

【V15 投資保護】 V15 model file 一切 不変、 V20 model は **別 file** に保存:
- data/v20_lgb_xgb_models.pkl.gz
- data/v20_lgb_xgb_results.json

【V20 目標】
- WF AUC ≥ 0.85 (V15 0.8939 比 やや低 想定、 4-ensemble なし のため)
- 6 fold 全年 (2020-2025) で 平均 AUC 計測
- 新 features (class_down / hot_streak 等) の effect verify

Usage:
    python tools/v20_train_lgb_xgb.py
    python tools/v20_train_lgb_xgb.py --quick  # 単発 fold のみ
"""
import argparse
import gzip
import json
import os
import pickle
import sys
import time
from datetime import datetime

try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_data():
    import pandas as pd
    print('[INFO] loading v20_training_data_full.csv...')
    path = os.path.join(BASE_DIR, 'data', 'v20_training_data_full.csv')
    df = pd.read_csv(path, encoding='utf-8', low_memory=False)
    df = df[df['finish'] > 0]
    df['top3'] = (df['finish'] <= 3).astype(int)
    print(f'  shape: {df.shape}, top3 rate: {df["top3"].mean():.3f}')
    return df


def prepare_features(df):
    """LGB/XGB 学習用 features 抽出 (LEAK 除外)."""
    import pandas as pd
    drop_cols = {
        'race_id', 'horse_id', 'horse_name', 'jockey', 'trainer',
        'owner', 'breeder', 'finish', 'finish2', 'abnormal_code',
        'time_margin', 'run_time', 'run_time_x10', 'empty',
        'pass1', 'pass2', 'pass3', 'pass4', 'agari_3f',  # POST-RACE LEAK
        'birthday', 'mark1', 'mark2',
        'top3', 'prize',  # target / POST-RACE
        'race_date', 'prev_race_date', '_year_full', '_idx',
        'race_name',
        # pace features の POST-RACE 原値 (expanding 版 のみ使う)
        'final_burst', 'pos_change_1to4', 'pos_change_4tofin',
        'pos_relative_4corner', 'pos_relative_1corner',
        'agari_3f_relative', 'pass4_relative', 'early_pace_diff',
        'pace_avg_pass1', 'pace_std_pass1', 'pace_avg_pass4',
        'pace_std_pass4', 'pace_avg_agari', 'pace_std_agari',
    }
    # categorical encode
    for c in ['surface', 'condition', 'course', 'class_code', 'father',
                'bms', 'sex', 'coat_color', 'location']:
        if c in df.columns:
            df[c] = df[c].astype('category').cat.codes

    # umaban / horse_num: 数値で残す
    if 'horse_num' in df.columns:
        df['horse_num'] = pd.to_numeric(df['horse_num'], errors='coerce')
    if 'umaban' in df.columns:
        df['umaban'] = pd.to_numeric(df['umaban'], errors='coerce')

    feature_cols = []
    for c in df.columns:
        if c in drop_cols:
            continue
        if df[c].dtype in ('int64', 'float64', 'int32', 'float32', 'int8', 'int16'):
            feature_cols.append(c)

    print(f'[INFO] feature count: {len(feature_cols)}')
    return feature_cols


def wf_train_one_fold(df, feature_cols, test_year, quick=False):
    """1 fold (year=test_year に対して) 学習 + 評価."""
    import pandas as pd
    import lightgbm as lgb
    import xgboost as xgb
    from sklearn.metrics import roc_auc_score

    ty = test_year - 2000
    train_df = df[df['year'] < ty]
    test_df = df[df['year'] == ty]
    if len(test_df) < 1000:
        print(f'  [SKIP] test set too small: {len(test_df)}')
        return None
    if len(train_df) < 5000:
        print(f'  [SKIP] train set too small: {len(train_df)} (年 {test_year} 以前 data なし)')
        return None

    X_tr = train_df[feature_cols].fillna(-1)
    y_tr = train_df['top3']
    X_te = test_df[feature_cols].fillna(-1)
    y_te = test_df['top3']

    print(f'\n[fold {test_year}] train={len(X_tr):,}, test={len(X_te):,}')

    # LGB
    lgb_params = {
        'objective': 'binary', 'metric': 'auc',
        'num_leaves': 63, 'learning_rate': 0.05,
        'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 5,
        'min_child_samples': 50, 'reg_alpha': 0.1, 'reg_lambda': 0.1,
        'verbose': -1, 'seed': 42,
    }
    t0 = time.time()
    lgb_model = lgb.train(lgb_params, lgb.Dataset(X_tr, y_tr),
                           num_boost_round=1000 if not quick else 200,
                           valid_sets=[lgb.Dataset(X_te, y_te)],
                           callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)])
    lgb_pred = lgb_model.predict(X_te)
    lgb_auc = roc_auc_score(y_te, lgb_pred)
    print(f'  LGB AUC: {lgb_auc:.4f} ({time.time()-t0:.1f}s)')

    # XGB
    t1 = time.time()
    xgb_params = {
        'objective': 'binary:logistic', 'eval_metric': 'auc',
        'max_depth': 6, 'learning_rate': 0.05,
        'subsample': 0.8, 'colsample_bytree': 0.8,
        'min_child_weight': 50, 'reg_alpha': 0.1, 'reg_lambda': 0.1,
        'seed': 42, 'tree_method': 'hist',
    }
    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dtest = xgb.DMatrix(X_te, label=y_te)
    xgb_model = xgb.train(xgb_params, dtrain,
                          num_boost_round=1000 if not quick else 200,
                          evals=[(dtest, 'eval')],
                          early_stopping_rounds=50,
                          verbose_eval=100)
    xgb_pred = xgb_model.predict(dtest)
    xgb_auc = roc_auc_score(y_te, xgb_pred)
    print(f'  XGB AUC: {xgb_auc:.4f} ({time.time()-t1:.1f}s)')

    # Simple weighted ensemble
    ens_pred = 0.5 * lgb_pred + 0.5 * xgb_pred
    ens_auc = roc_auc_score(y_te, ens_pred)
    print(f'  ENSEMBLE AUC: {ens_auc:.4f}')

    return {
        'test_year': test_year,
        'lgb_auc': float(lgb_auc),
        'xgb_auc': float(xgb_auc),
        'ensemble_auc': float(ens_auc),
        'n_train': int(len(X_tr)),
        'n_test': int(len(X_te)),
        'lgb_model': lgb_model,
        'xgb_model': xgb_model,
        'feature_count': len(feature_cols),
    }


def main():
    ap = argparse.ArgumentParser(description='V20 LGB+XGB training (simplified)')
    ap.add_argument('--quick', action='store_true', help='1 fold (test=25) + 200 rounds')
    args = ap.parse_args()

    df = load_data()
    feature_cols = prepare_features(df)

    if args.quick:
        years = [2025]
    else:
        years = [2020, 2021, 2022, 2023, 2024, 2025]

    print(f'[INFO] WF folds: {years}')

    results = []
    models = {}
    for ty in years:
        r = wf_train_one_fold(df, feature_cols, ty, quick=args.quick)
        if r is not None:
            results.append({k: v for k, v in r.items() if k not in ['lgb_model', 'xgb_model']})
            models[f'fold_{ty}'] = {
                'lgb': r['lgb_model'],
                'xgb': r['xgb_model'],
            }

    # Summary
    print('\n=== WF SUMMARY ===')
    print(f'{"year":<6} {"LGB":>8} {"XGB":>8} {"Ensemble":>10}')
    for r in results:
        print(f'  {r["test_year"]:<6} {r["lgb_auc"]:>8.4f} '
              f'{r["xgb_auc"]:>8.4f} {r["ensemble_auc"]:>10.4f}')

    lgb_aucs = [r['lgb_auc'] for r in results]
    xgb_aucs = [r['xgb_auc'] for r in results]
    ens_aucs = [r['ensemble_auc'] for r in results]
    print(f'\n  MEAN: LGB={sum(lgb_aucs)/len(lgb_aucs):.4f}, '
          f'XGB={sum(xgb_aucs)/len(xgb_aucs):.4f}, '
          f'Ensemble={sum(ens_aucs)/len(ens_aucs):.4f}')

    print(f'\n  V15 baseline: 0.8939 (4-ensemble + 124 features)')
    print(f'  V20 (simpler 2-ensemble + ~70 features): {sum(ens_aucs)/len(ens_aucs):.4f}')

    # Save results JSON
    out_json = os.path.join(BASE_DIR, 'data', 'v20_lgb_xgb_results.json')
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump({
            'results': results,
            'mean_lgb_auc': sum(lgb_aucs) / len(lgb_aucs),
            'mean_xgb_auc': sum(xgb_aucs) / len(xgb_aucs),
            'mean_ensemble_auc': sum(ens_aucs) / len(ens_aucs),
            'feature_count': len(feature_cols),
            'features_used': feature_cols,
            'trained_at': datetime.now().isoformat(),
            'note': 'V20 simplified LGB+XGB 2-ensemble。 V15 4-ensemble (LGB+XGB+FT+IntraRace) より AUC 低い想定。',
        }, f, indent=2, ensure_ascii=False)
    print(f'\n[OK] results saved: {out_json}')

    # Save models (final fold only、 全 fold 保存は重い)
    if models:
        last_year = max(models.keys())
        last_models = models[last_year]
        out_pkl = os.path.join(BASE_DIR, 'data', 'v20_lgb_xgb_models.pkl.gz')
        with gzip.open(out_pkl, 'wb') as f:
            pickle.dump({
                'lgb': last_models['lgb'],
                'xgb': last_models['xgb'],
                'feature_cols': feature_cols,
                'trained_year': last_year,
            }, f)
        print(f'[OK] models saved: {out_pkl}')

    return 0


if __name__ == '__main__':
    sys.exit(main())
