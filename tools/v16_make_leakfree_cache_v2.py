#!/usr/bin/env python3
"""大規模①: leak-free cache 高カバレッジ化(v2)。
ze4特徴を ★NFKC正規化 horse_name★ で expanding 再計算 → カバレッジ 85.8%→90.4%(=上限)。
残9.6%は過去ZEDが無い真の初出走馬(default正当・100%は構造的に不可能)。
当該/同日/未来ZEDは厳密除外(リーク再発防止)。 元cache・本番・jrdb_features.py 不変。
出力: data/_v15_optuna_df_cache_leakfree_v2.pkl.gz
"""
from __future__ import annotations
import os, sys, gzip, pickle, time, unicodedata
if sys.platform == "win32": sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd, lightgbm as lgb, xgboost as xgb
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr
sys.path.insert(0, os.path.abspath('tools'))
from v16_anaba_s3_roi import load_payouts
DATA = 'data'
ZE = ['jrdb_ze_idm_avg', 'jrdb_ze_ten_avg', 'jrdb_ze_agari_avg', 'jrdb_ze_furi_count']
DEFAULTS = {'jrdb_ze_idm_avg': 37.0, 'jrdb_ze_ten_avg': -15.0, 'jrdb_ze_agari_avg': -12.0, 'jrdb_ze_furi_count': 0.0}


def norm(s):
    return unicodedata.normalize('NFKC', str(s)).replace(' ', '').replace('　', '').strip()


def recompute_ze_v2(df):
    ze = pd.read_csv(os.path.join(DATA, 'jrdb_ze.csv'), encoding='utf-8-sig', dtype=str,
                     usecols=['horse_name', 'yyyymmdd', 'idm', 'ten_idx', 'agari_idx', 'furi'])
    ze['hn'] = ze['horse_name'].map(norm)
    ze['d'] = pd.to_numeric(ze['yyyymmdd'], errors='coerce')
    for c, s in [('idm', 'idm'), ('ten_idx', 'ten'), ('agari_idx', 'agari')]:
        ze[s] = pd.to_numeric(ze[c], errors='coerce')
    ze['furibin'] = (pd.to_numeric(ze['furi'], errors='coerce').fillna(0) > 0).astype(int)
    ze = ze.dropna(subset=['d']).sort_values(['hn', 'd'])
    g = ze.groupby('hn')
    # ★NaN無視の累積平均(cumsumのNaN伝播を回避=回収余地4.7ptを取り戻す)★
    def run_mean(col):
        valid = ze[col].notna()
        num = g[col].apply(lambda s: s.fillna(0).cumsum()).reset_index(level=0, drop=True)
        cnt = valid.groupby(ze['hn']).cumsum()
        return num / cnt.replace(0, np.nan)
    ze['c_idm'] = run_mean('idm')
    ze['c_ten'] = run_mean('ten')
    ze['c_agari'] = run_mean('agari')
    ze['c_furi'] = g['furibin'].cumsum()
    zr = ze[['hn', 'd', 'c_idm', 'c_ten', 'c_agari', 'c_furi']].sort_values('d')
    left = pd.DataFrame({'hn': df['horse_name'].map(norm),
                         'd': pd.to_numeric(df['date_num'], errors='coerce')})
    left['_row'] = np.arange(len(left)); left = left.sort_values('d')
    m = pd.merge_asof(left, zr, on='d', by='hn', direction='backward', allow_exact_matches=False)
    m = m.sort_values('_row')
    out = pd.DataFrame(index=df.index)
    out['jrdb_ze_idm_avg'] = m['c_idm'].values
    out['jrdb_ze_ten_avg'] = m['c_ten'].values
    out['jrdb_ze_agari_avg'] = m['c_agari'].values
    out['jrdb_ze_furi_count'] = m['c_furi'].values
    matched = out['jrdb_ze_idm_avg'].notna().mean()
    for c in ZE: out[c] = out[c].fillna(DEFAULTS[c])
    return out, matched


def override_auc(df, feats, label):
    df = df.copy()
    df['_rk'] = [f"{int(a)}_{b}_{int(c)}_{int(e)}_{int(f)}" for a, b, c, e, f in
                 zip(df['date_num'], df['course'], df['kai'], df['nichi'], df['race_num'])]
    df['pop'] = pd.to_numeric(df['oz_base_pop_rank'], errors='coerce').fillna(df['num_horses_val'])
    y = pd.to_numeric(df['year'], errors='coerce')
    for f in feats: df[f] = pd.to_numeric(df[f], errors='coerce').fillna(0)
    P = {'objective': 'binary', 'metric': 'auc', 'num_leaves': 63, 'learning_rate': 0.05,
         'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 5, 'min_child_samples': 50, 'verbose': -1, 'seed': 42}
    XP = {'objective': 'binary:logistic', 'max_depth': 6, 'learning_rate': 0.05, 'subsample': 0.8,
          'colsample_bytree': 0.8, 'min_child_weight': 50, 'seed': 42, 'tree_method': 'hist', 'verbosity': 0}
    rows = []; aucs = []
    for ty in [23, 24, 25]:
        tr = y < ty; te = y == ty
        ml = lgb.train(P, lgb.Dataset(df.loc[tr, feats].values, label=df.loc[tr, 'target'].values), num_boost_round=400)
        mx = xgb.train(XP, xgb.DMatrix(df.loc[tr, feats].values, label=df.loc[tr, 'target'].values), num_boost_round=400)
        p = 0.5 * ml.predict(df.loc[te, feats].values) + 0.5 * mx.predict(xgb.DMatrix(df.loc[te, feats].values))
        aucs.append(roc_auc_score(df.loc[te, 'target'].values, p))
        s = df.loc[te, ['_rk', 'finish', 'pop', 'horse_num']].copy(); s['s'] = p; rows.append(s)
    oof = pd.concat(rows); pay = load_payouts()
    ow = on = tw = tn = tr_ = 0
    for k, g in oof.groupby('_rk'):
        if k not in pay or len(g) < 4: continue
        t1 = g.sort_values('s', ascending=False).iloc[0]; fav = g.sort_values('pop').iloc[0]
        tn += 1; w = int(t1['finish']) == 1; tw += w; tr_ += pay[k]['tansho_pay'] if w else 0
        if int(t1['horse_num']) != int(fav['horse_num']): on += 1; ow += w
    print(f"  {label}: AUC={np.mean(aucs):.4f} top1勝率{tw/tn*100:.1f}% override{ow/on*100:.1f}% 単勝ROI{tr_/(100*tn)*100:.1f}%")
    return np.mean(aucs)


def main():
    t0 = time.time()
    obj = pickle.load(gzip.open(os.path.join(DATA, '_v15_optuna_df_cache.pkl.gz'), 'rb'))
    df = obj['df']; feats = obj['features']
    if 'target' not in df.columns: df['target'] = (df['finish'] <= 3).astype(int)
    print("=== ze4 expanding 再計算 (NFKC正規化名・当該日前のみ) ===")
    new_ze, matched = recompute_ze_v2(df)
    print(f"  カバレッジ(過去ZEDマッチ): {matched*100:.1f}% (上限90.4%・残りは初出走default)")
    win = df['finish'] == 1; fin = pd.to_numeric(df['finish'], errors='coerce')
    oldv = pd.to_numeric(df['jrdb_ze_idm_avg'], errors='coerce')  # 元cache(リーク版)
    m = oldv.notna() & fin.notna()
    print(f"  ze_idm_avg 勝馬平均: 元cache(leak)={oldv[win].mean():.1f} → v2={new_ze['jrdb_ze_idm_avg'][win].mean():.1f}")
    print(f"  corr(ze_idm_avg,finish): 元cache(leak)={pearsonr(oldv[m],fin[m])[0]:+.3f} → v2={pearsonr(new_ze['jrdb_ze_idm_avg'][m],fin[m])[0]:+.3f} (~-0.20ならleak無)")
    df_v2 = df.copy()
    for c in ZE: df_v2[c] = new_ze[c].values
    out = {'df': df_v2, 'features': feats}
    for k in obj:
        if k not in ('df', 'features'): out[k] = obj[k]
    path = os.path.join(DATA, '_v15_optuna_df_cache_leakfree_v2.pkl.gz')
    with gzip.open(path, 'wb') as f: pickle.dump(out, f, protocol=4)
    print(f"  保存: {path} (元cache・本番不変)")

    print("\n=== リーク再発チェック (v2 で override・対照RAW=34.8%/市場本命31.7%) ===")
    override_auc(df_v2, feats, 'V15(v2 100%cov)')
    print(f"\nDONE in {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
