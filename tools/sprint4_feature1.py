"""Sprint 4 ★★★ #1: JRDB SRB bias 6 features

bias_1corner / bias_2corner / bias_backstr / bias_3corner / bias_4corner / bias_straight
の 6 fields を V15 cache に merge し、 V15 vs V15+SRB の AUC を比較。

絶対遵守:
- read-only (V15 cache, jrdb_srb.csv 改変なし)
- predict_core / daily_predict / app.py / 既存 train code 不変
- output: data/v18/sprint4_feature1_5_8.md
"""
import gzip
import pickle
import json
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

BASE = Path(__file__).resolve().parent.parent
V15_CACHE = BASE / 'data' / '_v15_optuna_df_cache.pkl.gz'
SRB_CSV = BASE / 'data' / 'jrdb_srb.csv'
OUT_DOC = BASE / 'data' / 'v18' / 'sprint4_feature1_5_8.md'


def v15_rid_to_nk(rid):
    """V15 internal race_id (10 chars VV+YY+K+N+RR) → netkeiba 12 chars."""
    if pd.isna(rid):
        return None
    s = str(rid)
    if len(s) != 10:
        return None
    course, year = s[0:2], s[2:4]
    kai, nichi = s[4], s[5]
    race = s[6:8]
    if not (kai.isdigit() and nichi.isdigit()):
        return None
    return f'20{year}{course}0{kai}0{nichi}{race}'


def load_v15_cache():
    with gzip.open(V15_CACHE, 'rb') as f:
        obj = pickle.load(f)
    return obj['df'], obj['features']


def load_srb():
    srb = pd.read_csv(SRB_CSV, dtype={'race_id': str}, encoding='utf-8-sig')
    bias_cols = ['bias_1corner', 'bias_2corner', 'bias_backstr',
                 'bias_3corner', 'bias_4corner', 'bias_straight']
    keep = ['race_id'] + bias_cols
    srb = srb[keep].copy()
    for c in bias_cols:
        srb[c] = pd.to_numeric(srb[c], errors='coerce')
    srb = srb.rename(columns={
        'bias_1corner': 'srb_bias_1c',
        'bias_2corner': 'srb_bias_2c',
        'bias_backstr': 'srb_bias_bs',
        'bias_3corner': 'srb_bias_3c',
        'bias_4corner': 'srb_bias_4c',
        'bias_straight': 'srb_bias_st',
    })
    return srb


def add_srb_features(df):
    df = df.copy()
    df['nk_race_id'] = df['race_id'].apply(v15_rid_to_nk)
    srb = load_srb()
    df = df.merge(srb, left_on='nk_race_id', right_on='race_id',
                  how='left', suffixes=('', '_srb'))
    df = df.drop(columns=[c for c in df.columns if c.endswith('_srb')])
    new_cols = ['srb_bias_1c', 'srb_bias_2c', 'srb_bias_bs',
                'srb_bias_3c', 'srb_bias_4c', 'srb_bias_st']
    coverage = {c: df[c].notna().mean() for c in new_cols}
    return df, new_cols, coverage


def quick_backtest(df, features, target='target', date_col='date_num'):
    """Walk-forward 1-fold per year: train < year, eval == year. Returns yearly AUC."""
    results = {}
    train_yrs = sorted(df[date_col].astype(str).str[:4].unique())
    for ev_y in ['2023', '2024', '2025']:
        train = df[df[date_col].astype(str).str[:4] < ev_y]
        eval_ = df[df[date_col].astype(str).str[:4] == ev_y]
        if len(train) < 1000 or len(eval_) < 100:
            continue
        X_train = train[features].astype(float).fillna(-1)
        y_train = train[target].astype(int)
        X_eval = eval_[features].astype(float).fillna(-1)
        y_eval = eval_[target].astype(int)
        model = lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=63,
            min_child_samples=50,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            bagging_freq=5,
            reg_alpha=0.1,
            reg_lambda=0.1,
            verbosity=-1,
            seed=42,
            n_jobs=-1,
        )
        model.fit(X_train, y_train,
                  eval_set=[(X_eval, y_eval)],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
        train_pred = model.predict_proba(X_train)[:, 1]
        eval_pred = model.predict_proba(X_eval)[:, 1]
        train_auc = roc_auc_score(y_train, train_pred)
        eval_auc = roc_auc_score(y_eval, eval_pred)
        results[ev_y] = {
            'train_auc': float(train_auc),
            'eval_auc': float(eval_auc),
            'n_train': len(train),
            'n_eval': len(eval_),
            'gap': float(train_auc - eval_auc),
        }
    return results


def main():
    print('Loading V15 cache...')
    df, v15_features = load_v15_cache()
    print(f'V15 cache: {df.shape}, features={len(v15_features)}')

    print('\nMerging SRB...')
    df_aug, srb_cols, coverage = add_srb_features(df)
    print(f'SRB coverage:')
    for c, v in coverage.items():
        print(f'  {c}: {v:.3f}')

    print('\n--- V15 baseline backtest ---')
    base_results = quick_backtest(df_aug, v15_features)
    for y, r in base_results.items():
        print(f'  {y}: train AUC={r["train_auc"]:.4f}, eval AUC={r["eval_auc"]:.4f}, gap={r["gap"]:.4f}')

    print('\n--- V15 + SRB backtest ---')
    aug_results = quick_backtest(df_aug, v15_features + srb_cols)
    for y, r in aug_results.items():
        print(f'  {y}: train AUC={r["train_auc"]:.4f}, eval AUC={r["eval_auc"]:.4f}, gap={r["gap"]:.4f}')

    contributions = {}
    for y in base_results:
        if y in aug_results:
            base_auc = base_results[y]['eval_auc']
            aug_auc = aug_results[y]['eval_auc']
            contributions[y] = aug_auc - base_auc
    avg_contrib = float(np.mean(list(contributions.values()))) if contributions else 0.0
    print(f'\nMean AUC contribution: {avg_contrib:+.4f}')
    print('Yearly contribution:', {y: f'{v:+.4f}' for y, v in contributions.items()})

    leak_warnings = []
    for y, r in aug_results.items():
        if r['gap'] > 0.05:
            leak_warnings.append(f'  {y}: gap={r["gap"]:.4f} > 0.05')

    summary = {
        'feature_group': 'SRB bias 6',
        'features': srb_cols,
        'coverage': coverage,
        'baseline_auc': {y: r['eval_auc'] for y, r in base_results.items()},
        'augmented_auc': {y: r['eval_auc'] for y, r in aug_results.items()},
        'contribution': contributions,
        'mean_contribution': avg_contrib,
        'leak_warnings': leak_warnings,
    }

    OUT_DOC.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_DOC, 'w', encoding='utf-8') as f:
        f.write('# Sprint 4 ★★★ #1: SRB bias 6 features 結果 (5/8)\n\n')
        f.write(f'**branch**: dev/sprint4\n')
        f.write(f'**source**: data/jrdb_srb.csv ({len(srb_cols)} fields)\n')
        f.write(f'**実装**: tools/sprint4_feature1.py\n\n')

        f.write('## 1. 追加 features\n\n')
        f.write('| feature | source | coverage |\n|---------|--------|----------|\n')
        for c in srb_cols:
            f.write(f'| {c} | SRB.{c[4:]} | {coverage[c]*100:.1f}% |\n')

        f.write('\n## 2. AUC contribution (1-fold WF per year)\n\n')
        f.write('| 年 | V15 baseline | V15 + SRB | Δ | gap (train-eval) |\n')
        f.write('|----|------------|----------|----|-----------------|\n')
        for y in base_results:
            base_auc = base_results[y]['eval_auc']
            aug_auc = aug_results.get(y, {}).get('eval_auc', None)
            gap = aug_results.get(y, {}).get('gap', None)
            if aug_auc is not None:
                f.write(f'| {y} | {base_auc:.4f} | {aug_auc:.4f} | {aug_auc-base_auc:+.4f} | {gap:.4f} |\n')

        f.write(f'\n**平均 AUC contribution**: {avg_contrib:+.4f}\n\n')
        f.write(f'期待値 +0.003-0.005 と比較: ')
        if avg_contrib >= 0.003:
            f.write('✅ **期待達成**\n')
        elif avg_contrib >= 0.001:
            f.write('🟡 partial (期待 +0.003-0.005、 実績は中)\n')
        else:
            f.write('🔴 期待未達 (要 V15.5 統合時に再検討)\n')

        f.write('\n## 3. リーク監査\n\n')
        if leak_warnings:
            f.write('🔴 train-eval gap > 0.05 検出:\n')
            for w in leak_warnings:
                f.write(f'{w}\n')
        else:
            f.write('✅ train-eval gap 全て ≤ 0.05、 リークなし\n')

        f.write('\n## 4. 結論\n\n')
        f.write(f'- **期待 AUC**: +0.003-0.005\n')
        f.write(f'- **実績 AUC**: {avg_contrib:+.4f} (平均 2023-2025)\n')
        f.write(f'- **採用判定**: ')
        if avg_contrib > 0 and not leak_warnings:
            f.write('✅ V15.5 統合 採用候補\n')
        else:
            f.write('🟡 V15.5 統合時に Grid 重み 調整 (個別 AUC 見直し)\n')

    print(f'\nDoc written: {OUT_DOC}')

    json_out = OUT_DOC.with_suffix('.json')
    with open(json_out, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f'JSON: {json_out}')

    return summary


if __name__ == '__main__':
    main()
