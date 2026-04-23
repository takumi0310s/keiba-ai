"""v15 特徴量重要度分析

Usage:
    python tools/feature_importance_analysis.py

出力: report/feature_importance_20260423.md
"""
import os, sys, gzip, pickle, json
from datetime import datetime
import pandas as pd
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = os.path.join(BASE, 'keiba_model_v15_central_live.pkl.gz')
OUT = os.path.join(BASE, 'report', 'feature_importance_20260423.md')


def categorize(feat):
    f = feat.lower()
    if f.startswith('jrdb_paddock') or f.startswith('jrdb_odds') or f.startswith('jrdb_live') \
       or f.startswith('jrdb_body') or f.startswith('jrdb_demeanor'):
        return 'jrdb_tyb (当日)'
    if f.startswith('jrdb_prev'):
        return 'jrdb_sed (前走)'
    if f.startswith('jrdb_'):
        return 'jrdb_kyi (基本)'
    if 'odds' in f or 'pop_' in f:
        return 'odds'
    if 'jockey' in f:
        return 'jockey'
    if 'sire' in f or 'bms' in f or 'father' in f:
        return 'pedigree'
    if 'training' in f or 'wood' in f or 'sakaro' in f or 'oikiri' in f:
        return 'training'
    if 'prev_' in f or 'prev2_' in f or 'prev3_' in f or 'last3f' in f or 'agari' in f:
        return 'prev_race'
    if 'horse_' in f or 'career' in f:
        return 'horse_career'
    if 'weight' in f or 'condition' in f or 'cushion' in f or 'moisture' in f:
        return 'condition'
    if 'paci' in f or 'pace' in f:
        return 'pace_paci'
    if 'transport' in f or 'gaisha' in f or 'course_renovated' in f:
        return 'logistics'
    return 'basic'


def main():
    print(f'Loading {MODEL}...')
    m = pickle.load(gzip.open(MODEL, 'rb'))
    feats = m['features']
    print(f'  {len(feats)} features')

    # LGB importance
    lgb_imp = {}
    if 'model' in m:
        lgb_model = m['model']
        try:
            imp = lgb_model.feature_importance(importance_type='gain')
            lgb_imp = dict(zip(feats, imp))
        except Exception as e:
            print(f'LGB importance fail: {e}')

    # XGB importance
    xgb_imp = {}
    if 'xgb_model' in m and m['xgb_model'] is not None:
        xgb_model = m['xgb_model']
        try:
            booster = xgb_model.get_booster() if hasattr(xgb_model, 'get_booster') else xgb_model
            score = booster.get_score(importance_type='gain')
            # XGB の feature名は f0,f1,... の場合がある
            xgb_imp_raw = {}
            for k, v in score.items():
                if k.startswith('f') and k[1:].isdigit():
                    idx = int(k[1:])
                    if idx < len(feats):
                        xgb_imp_raw[feats[idx]] = v
                else:
                    xgb_imp_raw[k] = v
            xgb_imp = xgb_imp_raw
        except Exception as e:
            print(f'XGB importance fail: {e}')

    # 結合 (rank-based)
    df = pd.DataFrame({'feature': feats})
    df['lgb_imp'] = df['feature'].map(lgb_imp).fillna(0)
    df['xgb_imp'] = df['feature'].map(xgb_imp).fillna(0)
    df['lgb_rank'] = df['lgb_imp'].rank(ascending=False, method='min')
    df['xgb_rank'] = df['xgb_imp'].rank(ascending=False, method='min')
    df['avg_rank'] = (df['lgb_rank'] + df['xgb_rank']) / 2
    df['category'] = df['feature'].apply(categorize)

    # ソート (avg_rank 昇順 = 重要度高)
    df_sorted = df.sort_values('avg_rank')

    # カテゴリ別集計
    cat_stats = df.groupby('category').agg(
        count=('feature', 'count'),
        lgb_total=('lgb_imp', 'sum'),
        lgb_mean_rank=('lgb_rank', 'mean'),
        xgb_total=('xgb_imp', 'sum'),
    ).sort_values('lgb_total', ascending=False)

    L = []
    L.append(f'# v15 特徴量重要度分析\n\n')
    L.append(f'生成日時: {datetime.now().strftime("%Y-%m-%d %H:%M")}\n')
    L.append(f'モデル: {os.path.basename(MODEL)}\n')
    L.append(f'特徴量数: {len(feats)}\n\n')

    L.append(f'## カテゴリ別集計\n\n')
    L.append(f'| カテゴリ | n | LGB total gain | XGB total gain | LGB平均順位 |\n')
    L.append(f'|----------|---|---------------|---------------|------------|\n')
    for cat, r in cat_stats.iterrows():
        L.append(f'| {cat} | {int(r["count"])} | {r["lgb_total"]:.0f} | '
                 f'{r["xgb_total"]:.0f} | {r["lgb_mean_rank"]:.1f} |\n')
    L.append(f'\n')

    L.append(f'## TOP30 (LGB+XGB 平均順位ベース)\n\n')
    L.append(f'| Rank | feature | category | lgb_imp | xgb_imp |\n')
    L.append(f'|------|---------|----------|---------|---------|\n')
    for i, r in df_sorted.head(30).reset_index(drop=True).iterrows():
        L.append(f'| {i+1} | `{r["feature"]}` | {r["category"]} | '
                 f'{r["lgb_imp"]:.0f} | {r["xgb_imp"]:.0f} |\n')
    L.append(f'\n')

    L.append(f'## 下位30 (削除候補、LGB+XGB ともに寄与小)\n\n')
    L.append(f'| Rank | feature | category | lgb_imp | xgb_imp |\n')
    L.append(f'|------|---------|----------|---------|---------|\n')
    for i, r in df_sorted.tail(30).reset_index(drop=True).iterrows():
        L.append(f'| {len(feats)-29+i} | `{r["feature"]}` | {r["category"]} | '
                 f'{r["lgb_imp"]:.0f} | {r["xgb_imp"]:.0f} |\n')
    L.append(f'\n')

    # 4/23 修正特徴量の追跡 (jrdb_prev_idm 等)
    L.append(f'## 4/23 SED merge 修正特徴量の重要度\n\n')
    target_feats = [f for f in feats if 'prev' in f.lower() and ('idm' in f.lower() or 'sed' in f.lower())]
    if target_feats:
        L.append(f'| feature | LGB rank | XGB rank | LGB imp | XGB imp |\n')
        L.append(f'|---------|----------|----------|---------|---------|\n')
        for f in target_feats:
            row = df[df['feature'] == f].iloc[0]
            L.append(f'| `{f}` | {int(row["lgb_rank"])} | {int(row["xgb_rank"])} | '
                     f'{row["lgb_imp"]:.0f} | {row["xgb_imp"]:.0f} |\n')
    else:
        L.append(f'(該当特徴量名なし — `prev_*` カテゴリ全体は prev_race セクション参照)\n')

    L.append(f'\n## 改善方向性\n\n')
    L.append(f'1. **削除候補**: 下位30の中で複数モデルともに 0 ベースの特徴量は次回再学習で削除検討\n')
    L.append(f'2. **強化候補**: TOP30 の中で派生・組み合わせ可能な特徴量は新規生成検討\n')
    L.append(f'3. **カテゴリ偏り**: jrdb_* が下位にあれば SED merge カバレッジ修正の効果未反映の可能性\n')

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w', encoding='utf-8') as f:
        f.write(''.join(L))
    print(f'Report: {OUT}')


if __name__ == '__main__':
    main()
