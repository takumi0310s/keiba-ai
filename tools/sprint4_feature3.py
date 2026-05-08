"""Sprint 4 ★★★ #3: JRDB JO cid_idx / ls_idx

cid_idx (西田指数系) / ls_idx (ライディング指数系)
を V15 cache に merge (race_id × umaban)、 V15 vs V15+JO の AUC 比較。

リーク risk: JO は朝段階 確定 (pre-race)。 直接 numeric として使用。

絶対遵守: read-only / 既存 code 不変
output: data/v18/sprint4_feature3_5_8.md
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
JO_CSV = BASE / 'data' / 'jrdb_jo.csv'
OUT_DOC = BASE / 'data' / 'v18' / 'sprint4_feature3_5_8.md'

NEW_COLS = ['jo_cid_idx', 'jo_ls_idx']


def v15_rid_to_nk(rid):
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


def add_jo_features(df_v15):
    df = df_v15.copy()
    df['nk_race_id'] = df['race_id'].apply(v15_rid_to_nk)
    df['umaban'] = df['umaban'].astype(int)

    jo = pd.read_csv(JO_CSV, dtype={'race_id': str}, encoding='utf-8-sig')
    jo = jo[['race_id', 'umaban', 'cid_idx', 'ls_idx']].copy()
    jo['umaban'] = pd.to_numeric(jo['umaban'], errors='coerce').astype('Int64')
    jo['cid_idx'] = pd.to_numeric(jo['cid_idx'], errors='coerce')
    jo['ls_idx'] = pd.to_numeric(jo['ls_idx'], errors='coerce')
    jo = jo.rename(columns={'cid_idx': 'jo_cid_idx', 'ls_idx': 'jo_ls_idx'})
    jo = jo.dropna(subset=['umaban'])
    jo['umaban'] = jo['umaban'].astype(int)

    merged = df.merge(jo, left_on=['nk_race_id', 'umaban'],
                      right_on=['race_id', 'umaban'],
                      how='left', suffixes=('', '_jo'))
    merged = merged.drop(columns=['race_id_jo'], errors='ignore')

    coverage = {c: merged[c].notna().mean() for c in NEW_COLS}
    return merged, coverage


def quick_backtest(df, features, target='target', date_col='date_num'):
    results = {}
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
            n_estimators=300, learning_rate=0.05, num_leaves=63,
            min_child_samples=50, feature_fraction=0.8, bagging_fraction=0.8,
            bagging_freq=5, reg_alpha=0.1, reg_lambda=0.1,
            verbosity=-1, seed=42, n_jobs=-1,
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

    print('\nMerging JO...')
    df_aug, coverage = add_jo_features(df)
    print(f'JO coverage:')
    for c, v in coverage.items():
        print(f'  {c}: {v:.3f}')

    print('\n--- V15 baseline backtest ---')
    base_results = quick_backtest(df_aug, v15_features)
    for y, r in base_results.items():
        print(f'  {y}: train={r["train_auc"]:.4f}, eval={r["eval_auc"]:.4f}, gap={r["gap"]:.4f}')

    print('\n--- V15 + JO backtest ---')
    aug_results = quick_backtest(df_aug, v15_features + NEW_COLS)
    for y, r in aug_results.items():
        print(f'  {y}: train={r["train_auc"]:.4f}, eval={r["eval_auc"]:.4f}, gap={r["gap"]:.4f}')

    contributions = {}
    for y in base_results:
        if y in aug_results:
            contributions[y] = aug_results[y]['eval_auc'] - base_results[y]['eval_auc']
    avg_contrib = float(np.mean(list(contributions.values()))) if contributions else 0.0
    print(f'\nMean AUC contribution: {avg_contrib:+.4f}')

    leak_warnings = []
    for y, r in aug_results.items():
        if r['gap'] > 0.05:
            leak_warnings.append(f'  {y}: gap={r["gap"]:.4f} > 0.05')

    summary = {
        'feature_group': 'JO cid_idx / ls_idx',
        'features': NEW_COLS,
        'coverage': coverage,
        'baseline_auc': {y: r['eval_auc'] for y, r in base_results.items()},
        'augmented_auc': {y: r['eval_auc'] for y, r in aug_results.items()},
        'contribution': contributions,
        'mean_contribution': avg_contrib,
        'leak_warnings': leak_warnings,
    }

    OUT_DOC.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_DOC, 'w', encoding='utf-8') as f:
        f.write('# Sprint 4 ★★★ #3: JRDB JO cid_idx / ls_idx 結果 (5/8)\n\n')
        f.write(f'**branch**: dev/sprint4\n')
        f.write(f'**source**: data/jrdb_jo.csv ({len(NEW_COLS)} fields)\n')
        f.write(f'**実装**: tools/sprint4_feature3.py\n')
        f.write(f'**リーク risk**: pre-race (JO は朝段階 確定)\n\n')

        f.write('## 1. 追加 features\n\n')
        f.write('| feature | source | coverage |\n|---------|--------|----------|\n')
        for c in NEW_COLS:
            f.write(f'| {c} | JO.{c[3:]} | {coverage[c]*100:.1f}% |\n')

        f.write('\n## 2. AUC contribution (1-fold WF per year)\n\n')
        f.write('| 年 | V15 baseline | V15 + JO | Δ | gap |\n')
        f.write('|----|------------|---------|----|----|\n')
        for y in base_results:
            ba = base_results[y]['eval_auc']
            aa = aug_results.get(y, {}).get('eval_auc')
            ga = aug_results.get(y, {}).get('gap')
            if aa is not None:
                f.write(f'| {y} | {ba:.4f} | {aa:.4f} | {aa-ba:+.4f} | {ga:.4f} |\n')

        f.write(f'\n**平均 AUC contribution**: {avg_contrib:+.4f}\n\n')
        f.write(f'期待値 +0.002-0.003 と比較: ')
        if avg_contrib >= 0.002:
            f.write('✅ **期待達成**\n')
        elif avg_contrib >= 0.001:
            f.write('🟡 partial\n')
        else:
            f.write('🔴 期待未達\n')

        f.write('\n## 3. リーク監査\n\n')
        if leak_warnings:
            f.write('🔴 train-eval gap > 0.05 検出:\n')
            for w in leak_warnings:
                f.write(f'{w}\n')
        else:
            f.write('✅ train-eval gap 全て ≤ 0.05、 リークなし\n')

        f.write('\n## 4. 結論\n\n')
        f.write(f'- **期待 AUC**: +0.002-0.003\n')
        f.write(f'- **実績 AUC**: {avg_contrib:+.4f} (平均 2023-2025)\n')
        f.write(f'- **採用判定**: ')
        if avg_contrib > 0 and not leak_warnings:
            f.write('✅ V15.5 統合 採用候補\n')
        else:
            f.write('🟡 要再検討\n')

    print(f'\nDoc written: {OUT_DOC}')

    json_out = OUT_DOC.with_suffix('.json')
    with open(json_out, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f'JSON: {json_out}')

    return summary


if __name__ == '__main__':
    main()
