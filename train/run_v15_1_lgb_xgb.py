"""V15.1 LGB+XGB 2-model 互換確認 (Session #37 B).

目的:
- V15.1 既存 LGB single (AUC 0.9426 on 200k subsample) に XGB 追加
- WF 検証: V15 baseline / V15.1 LGB single / V15.1 LGB+XGB ensemble
- 4-model (FT-Transformer + IntraRace Attention) は Phase 3 で

出力:
- data/v15.1/v15_1_xgb.json
- data/v15.1/v15_1_lgb_xgb_results.json
- data/v15.1/v15_1_wf_results.csv

注意:
- V15 production model (keiba_model_v15_central_live.pkl.gz) 完全不変
- predict_core.py / daily_predict.py 完全不変
- subsample 上限 200000 (既存 v15_1_results.json と整合)
"""
from __future__ import annotations

import os, sys, argparse, pickle, gzip, json, time
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import roc_auc_score, log_loss

BASE = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
os.chdir(BASE)
sys.path.insert(0, os.path.join(BASE, 'train'))

from v15_1_features import merge_v15_1_features, V15_1_NEW_FEATURES

CACHE_PATH = 'data/_v15_optuna_df_cache.pkl.gz'
OUT_DIR = 'data/v15.1'
os.makedirs(OUT_DIR, exist_ok=True)

LGB_PARAMS = {
    'objective': 'binary', 'metric': 'auc',
    'boosting_type': 'gbdt', 'num_leaves': 63, 'learning_rate': 0.05,
    'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 5,
    'min_child_samples': 50, 'reg_alpha': 0.1, 'reg_lambda': 0.1,
    'verbose': -1, 'seed': 42,
}
XGB_PARAMS = {
    'objective': 'binary:logistic', 'eval_metric': 'auc',
    'max_depth': 6, 'learning_rate': 0.05,
    'subsample': 0.8, 'colsample_bytree': 0.8,
    'min_child_weight': 50, 'reg_alpha': 0.1, 'reg_lambda': 0.1,
    'seed': 42, 'tree_method': 'hist', 'verbosity': 0,
}


def load_cache():
    print(f"Loading {CACHE_PATH}...")
    t0 = time.time()
    d = pickle.load(gzip.open(CACHE_PATH, 'rb'))
    df, feats = d['df'], d['features']
    print(f"  loaded {len(df):,} rows / {len(feats)} features in {time.time()-t0:.1f}s")
    return df, feats


def split_by_year(df, train_until=2024):
    if 'year_full' in df.columns:
        yr = pd.to_numeric(df['year_full'], errors='coerce')
    elif 'year' in df.columns:
        yr = pd.to_numeric(df['year'], errors='coerce')
        if yr.max() < 100: yr = yr + 2000
    elif 'race_date' in df.columns:
        yr = pd.to_datetime(df['race_date'], errors='coerce').dt.year
    else:
        yy = pd.to_numeric(df['race_id'].astype(str).str[2:4], errors='coerce')
        yr = yy + 2000
    return yr


def train_eval_one(X_tr, y_tr, X_va, y_va, name, max_rounds=1000, max_xgb_rounds=600):
    """LGB + XGB を学習、ensemble の AUC/logloss を返す."""
    print(f"  [{name}] LGB train...")
    t0 = time.time()
    booster_lgb = lgb.train(
        LGB_PARAMS,
        lgb.Dataset(X_tr, y_tr),
        num_boost_round=max_rounds,
        valid_sets=[lgb.Dataset(X_va, y_va)],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
    )
    p_lgb = booster_lgb.predict(X_va)
    auc_lgb = roc_auc_score(y_va, p_lgb)
    ll_lgb = log_loss(y_va, np.clip(p_lgb, 1e-7, 1-1e-7))
    t_lgb = time.time() - t0

    print(f"  [{name}] XGB train...")
    t0 = time.time()
    dtr = xgb.DMatrix(X_tr, label=y_tr)
    dva = xgb.DMatrix(X_va, label=y_va)
    booster_xgb = xgb.train(
        XGB_PARAMS, dtr, num_boost_round=max_xgb_rounds,
        evals=[(dva, 'val')], early_stopping_rounds=50, verbose_eval=0,
    )
    p_xgb = booster_xgb.predict(dva)
    auc_xgb = roc_auc_score(y_va, p_xgb)
    ll_xgb = log_loss(y_va, np.clip(p_xgb, 1e-7, 1-1e-7))
    t_xgb = time.time() - t0

    p_ens = (p_lgb + p_xgb) / 2
    auc_ens = roc_auc_score(y_va, p_ens)
    ll_ens = log_loss(y_va, np.clip(p_ens, 1e-7, 1-1e-7))

    return {
        'auc_lgb': float(auc_lgb), 'auc_xgb': float(auc_xgb), 'auc_ens': float(auc_ens),
        'logloss_lgb': float(ll_lgb), 'logloss_xgb': float(ll_xgb), 'logloss_ens': float(ll_ens),
        'time_lgb_sec': float(t_lgb), 'time_xgb_sec': float(t_xgb),
    }, booster_lgb, booster_xgb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-rows', type=int, default=200000, help='train subsample cap')
    parser.add_argument('--label', default='is_win')
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--wf-years', nargs='+', type=int, default=[2024, 2025], help='WF test years')
    args = parser.parse_args()

    if args.quick:
        LGB_PARAMS['learning_rate'] = 0.1

    print("=" * 60)
    print("V15.1 LGB+XGB 互換確認 (Session #37 B)")
    print(f"max_rows: {args.max_rows}, label: {args.label}, WF: {args.wf_years}")
    start = time.time()

    df, base_features = load_cache()
    df = merge_v15_1_features(df)
    print(f"After merge: shape={df.shape}")

    label_col = args.label
    if label_col not in df.columns:
        if 'finish' in df.columns:
            df[label_col] = (pd.to_numeric(df['finish'], errors='coerce') <= (1 if label_col == 'is_win' else 3)).astype(int)
        else:
            print("[ERROR] no label"); sys.exit(1)

    yr = split_by_year(df)
    df['_y'] = yr

    extended_features = base_features + V15_1_NEW_FEATURES
    print(f"V15 baseline features: {len(base_features)}")
    print(f"V15.1 extended features: {len(extended_features)} (+{len(V15_1_NEW_FEATURES)})")

    # WF: 各テスト年について train (それ以前) → test (該当年)
    wf_results = []
    saved = False
    for test_year in args.wf_years:
        train_mask = (df['_y'] >= 2015) & (df['_y'] < test_year)
        test_mask = df['_y'] == test_year
        n_tr = int(train_mask.sum()); n_te = int(test_mask.sum())
        if n_te < 1000:
            print(f"  [skip] test year {test_year} too small (n={n_te})"); continue

        # subsample train
        train_idx = df.index[train_mask]
        if args.max_rows and len(train_idx) > args.max_rows:
            np.random.seed(42)
            train_idx = np.random.choice(train_idx.values, size=args.max_rows, replace=False)
        test_idx = df.index[test_mask]

        X_v15 = df.loc[train_idx, base_features].apply(pd.to_numeric, errors='coerce').fillna(0)
        X_v15_te = df.loc[test_idx, base_features].apply(pd.to_numeric, errors='coerce').fillna(0)
        X_v151 = df.loc[train_idx, extended_features].apply(pd.to_numeric, errors='coerce').fillna(0)
        X_v151_te = df.loc[test_idx, extended_features].apply(pd.to_numeric, errors='coerce').fillna(0)
        y_tr = df.loc[train_idx, label_col].astype(int)
        y_te = df.loc[test_idx, label_col].astype(int)

        print(f"\n=== WF test_year={test_year} (n_tr={len(train_idx)}, n_te={n_te}) ===")

        # V15 baseline LGB only (XGB は省略、parity に集中)
        print(f"  [V15 baseline LGB only]")
        t0 = time.time()
        m15 = lgb.train(LGB_PARAMS, lgb.Dataset(X_v15.values, y_tr.values),
                        num_boost_round=1000,
                        valid_sets=[lgb.Dataset(X_v15_te.values, y_te.values)],
                        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
        p15 = m15.predict(X_v15_te.values)
        auc_v15 = roc_auc_score(y_te, p15)
        ll_v15 = log_loss(y_te, np.clip(p15, 1e-7, 1-1e-7))
        print(f"    V15 LGB: AUC={auc_v15:.4f} logloss={ll_v15:.4f} t={time.time()-t0:.1f}s")

        # V15.1 LGB + XGB
        m151_results, b151_lgb, b151_xgb = train_eval_one(
            X_v151.values, y_tr.values, X_v151_te.values, y_te.values, "V15.1"
        )
        print(f"    V15.1 LGB: AUC={m151_results['auc_lgb']:.4f}")
        print(f"    V15.1 XGB: AUC={m151_results['auc_xgb']:.4f}")
        print(f"    V15.1 Ens: AUC={m151_results['auc_ens']:.4f}")
        print(f"    Δ (Ens - V15): {(m151_results['auc_ens']-auc_v15)*10000:+.1f}bp")

        wf_results.append({
            'test_year': test_year,
            'n_train_subsample': len(train_idx),
            'n_test': n_te,
            'auc_v15_lgb': float(auc_v15),
            'logloss_v15_lgb': float(ll_v15),
            **{f'v151_{k}': v for k, v in m151_results.items()},
            'delta_auc_ens_vs_v15': float(m151_results['auc_ens'] - auc_v15),
            'delta_auc_lgb_vs_v15': float(m151_results['auc_lgb'] - auc_v15),
        })

        # 最後の test year で model 保存
        if test_year == args.wf_years[-1] and not saved:
            b151_lgb.save_model(f"{OUT_DIR}/v15_1_lgb_v37.txt")
            b151_xgb.save_model(f"{OUT_DIR}/v15_1_xgb.json")
            print(f"  [SAVED] {OUT_DIR}/v15_1_lgb_v37.txt + v15_1_xgb.json")
            saved = True

    total_min = (time.time() - start) / 60
    summary = {
        'session': '37_B',
        'purpose': 'V15.1 LGB+XGB 2-model 互換確認',
        'wf_years': args.wf_years,
        'max_rows': args.max_rows,
        'label': label_col,
        'wf_results': wf_results,
        'mean_auc_v15': float(np.mean([r['auc_v15_lgb'] for r in wf_results])) if wf_results else None,
        'mean_auc_v151_lgb': float(np.mean([r['v151_auc_lgb'] for r in wf_results])) if wf_results else None,
        'mean_auc_v151_ens': float(np.mean([r['v151_auc_ens'] for r in wf_results])) if wf_results else None,
        'elapsed_min': float(total_min),
    }
    if wf_results:
        summary['mean_delta_ens_vs_v15'] = summary['mean_auc_v151_ens'] - summary['mean_auc_v15']
        summary['mean_delta_lgb_vs_v15'] = summary['mean_auc_v151_lgb'] - summary['mean_auc_v15']

    with open(f"{OUT_DIR}/v15_1_lgb_xgb_results.json", 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, default=str)
    pd.DataFrame(wf_results).to_csv(f"{OUT_DIR}/v15_1_wf_results.csv", index=False, encoding='utf-8-sig')

    print(f"\n[OK] Saved {OUT_DIR}/v15_1_lgb_xgb_results.json + WF csv")
    print(f"Total time: {total_min:.1f}min")
    if wf_results:
        print(f"Mean V15 LGB:    {summary['mean_auc_v15']:.4f}")
        print(f"Mean V15.1 LGB:  {summary['mean_auc_v151_lgb']:.4f}")
        print(f"Mean V15.1 Ens:  {summary['mean_auc_v151_ens']:.4f}")
        print(f"Δ Ens vs V15:    {summary['mean_delta_ens_vs_v15']*10000:+.1f}bp")


if __name__ == '__main__':
    main()
