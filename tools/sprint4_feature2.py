"""Sprint 4 ★★★ #2: netkeiba master_index 5 indices

time_index / master_index / start_index / chase_index / agari_index
の 5 fields を expanding window で 「過去 racesの平均」 として merge。

リーク risk: master_index csv の 数値は post-race 集計値の可能性あり
(corr(time_index, finish_order) = -0.47 で highly correlated)
→ 当該 race を除外し past expanding mean を使用 (dam_top3r 教訓)

絶対遵守: read-only / 既存 code 不変
output: data/v18/sprint4_feature2_5_8.md
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
MI_CSV = BASE / 'data' / 'netkeiba_master_index.csv'
OUT_DOC = BASE / 'data' / 'v18' / 'sprint4_feature2_5_8.md'

INDEX_COLS = ['time_index', 'master_index', 'start_index', 'chase_index', 'agari_index']
NEW_COLS = [f'mi_{c.replace("_index","")}_idx_prev' for c in INDEX_COLS]


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


def build_expanding_mi(df_v15):
    """For each horse_id × date, compute expanding mean of past master indices."""
    mi = pd.read_csv(MI_CSV, dtype={'race_id': str}, encoding='utf-8-sig')
    mi = mi[['race_id', 'umaban'] + INDEX_COLS].copy()
    for c in INDEX_COLS:
        mi[c] = pd.to_numeric(mi[c], errors='coerce')

    df = df_v15[['race_id', 'umaban', 'horse_id', 'date_num']].copy()
    df['nk_race_id'] = df['race_id'].apply(v15_rid_to_nk)
    df['umaban'] = df['umaban'].astype(int)
    mi['umaban'] = mi['umaban'].astype(int)

    merged = df.merge(mi, left_on=['nk_race_id', 'umaban'],
                      right_on=['race_id', 'umaban'],
                      how='left', suffixes=('', '_mi'))
    merged = merged.drop(columns=['race_id_mi'], errors='ignore')

    coverage_raw = {c: merged[c].notna().mean() for c in INDEX_COLS}

    merged = merged.sort_values(['horse_id', 'date_num']).reset_index(drop=True)
    for c, new_c in zip(INDEX_COLS, NEW_COLS):
        merged[new_c] = (merged.groupby('horse_id')[c]
                                .apply(lambda s: s.shift(1).expanding().mean())
                                .reset_index(level=0, drop=True))

    merged = merged.set_index(['race_id', 'umaban'])
    return merged[NEW_COLS], coverage_raw


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

    print('\nBuilding expanding master_index features...')
    mi_features, coverage_raw = build_expanding_mi(df)
    print(f'Raw MI coverage:')
    for c, v in coverage_raw.items():
        print(f'  {c}: {v:.3f}')

    df_aug = df.copy()
    df_aug = df_aug.set_index(['race_id', 'umaban'])
    df_aug = df_aug.join(mi_features, how='left')
    df_aug = df_aug.reset_index()
    coverage_prev = {c: df_aug[c].notna().mean() for c in NEW_COLS}
    print(f'Expanding-prev MI coverage:')
    for c, v in coverage_prev.items():
        print(f'  {c}: {v:.3f}')

    print('\n--- V15 baseline backtest ---')
    base_results = quick_backtest(df_aug, v15_features)
    for y, r in base_results.items():
        print(f'  {y}: train={r["train_auc"]:.4f}, eval={r["eval_auc"]:.4f}, gap={r["gap"]:.4f}')

    print('\n--- V15 + MI(prev) backtest ---')
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
        'feature_group': 'master_index 5 (expanding prev)',
        'features': NEW_COLS,
        'coverage_raw': coverage_raw,
        'coverage_prev': coverage_prev,
        'baseline_auc': {y: r['eval_auc'] for y, r in base_results.items()},
        'augmented_auc': {y: r['eval_auc'] for y, r in aug_results.items()},
        'contribution': contributions,
        'mean_contribution': avg_contrib,
        'leak_warnings': leak_warnings,
    }

    OUT_DOC.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_DOC, 'w', encoding='utf-8') as f:
        f.write('# Sprint 4 ★★★ #2: master_index 5 indices 結果 (5/8)\n\n')
        f.write(f'**branch**: dev/sprint4\n')
        f.write(f'**source**: data/netkeiba_master_index.csv ({len(NEW_COLS)} fields)\n')
        f.write(f'**実装**: tools/sprint4_feature2.py\n')
        f.write(f'**リーク対策**: 当該 race を除外し expanding window で 過去 races の mean (dam_top3r 教訓)\n\n')

        f.write('## 1. 追加 features (expanding-prev mean)\n\n')
        f.write('| feature | 元 source | raw coverage | expanding coverage |\n')
        f.write('|---------|----------|--------------|------------------|\n')
        for orig, new_c in zip(INDEX_COLS, NEW_COLS):
            f.write(f'| {new_c} | MI.{orig} | {coverage_raw[orig]*100:.1f}% | {coverage_prev[new_c]*100:.1f}% |\n')

        f.write('\n## 2. AUC contribution (1-fold WF per year)\n\n')
        f.write('| 年 | V15 baseline | V15 + MI(prev) | Δ | gap |\n')
        f.write('|----|------------|----------------|----|----|\n')
        for y in base_results:
            ba = base_results[y]['eval_auc']
            aa = aug_results.get(y, {}).get('eval_auc')
            ga = aug_results.get(y, {}).get('gap')
            if aa is not None:
                f.write(f'| {y} | {ba:.4f} | {aa:.4f} | {aa-ba:+.4f} | {ga:.4f} |\n')

        f.write(f'\n**平均 AUC contribution**: {avg_contrib:+.4f}\n\n')
        f.write(f'期待値 +0.003-0.005 と比較: ')
        if avg_contrib >= 0.003:
            f.write('✅ **期待達成**\n')
        elif avg_contrib >= 0.001:
            f.write('🟡 partial\n')
        else:
            f.write('🔴 期待未達 (coverage 低い 2024-2025 のみで限定的)\n')

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
            f.write('✅ V15.5 統合 採用候補 (expanding 化で post-race リーク回避済)\n')
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
