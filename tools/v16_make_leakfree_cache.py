#!/usr/bin/env python3
"""ステップ5-2: jrdb_ze_* 4特徴を expanding(当該race日付より前のみ)で再計算し、
leak-free eval cache を新ファイルに生成 + override正常化検証。
元cache(_v15_optuna_df_cache.pkl.gz)・本番V15/V16・jrdb_features.py は不変。

リーク機構(5-1): jrdb_features.py が ZED(過去走成績)を全期間mean(日付カットオフ無)→当該/未来混入。
修正: 各 (horse, race日付) で yyyymmdd < 当該日 の ZED だけで mean(merge_asof backward strict)。
join key = horse_name(cacheにblood_num無し・両者JRDB由来で共通)。
"""
from __future__ import annotations
import os, sys, gzip, pickle, time
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd, lightgbm as lgb, xgboost as xgb
sys.path.insert(0, os.path.abspath('tools'))
from v16_anaba_s3_roi import load_payouts
DATA = 'data'
ZE = ['jrdb_ze_idm_avg', 'jrdb_ze_ten_avg', 'jrdb_ze_agari_avg', 'jrdb_ze_furi_count']
DEFAULTS = {'jrdb_ze_idm_avg': 37.0, 'jrdb_ze_ten_avg': -15.0, 'jrdb_ze_agari_avg': -12.0, 'jrdb_ze_furi_count': 0.0}


def recompute_ze(df):
    """各cache行に対し、当該race日付より前のZEDのみで ze 4特徴を再計算。"""
    ze = pd.read_csv(os.path.join(DATA, 'jrdb_ze.csv'), encoding='utf-8-sig', dtype=str,
                     usecols=['horse_name', 'yyyymmdd', 'idm', 'ten_idx', 'agari_idx', 'furi'])
    ze['hn'] = ze['horse_name'].astype(str).str.strip()
    ze['d'] = pd.to_numeric(ze['yyyymmdd'], errors='coerce')
    for c, s in [('idm', 'idm'), ('ten_idx', 'ten'), ('agari_idx', 'agari')]:
        ze[s] = pd.to_numeric(ze[c], errors='coerce')
    ze['furibin'] = (pd.to_numeric(ze['furi'], errors='coerce').fillna(0) > 0).astype(int)
    ze = ze.dropna(subset=['d']).sort_values(['hn', 'd'])
    # 累積(inclusive)平均/件数 → merge_asof backward-strict で「当該日より前」を拾う
    g = ze.groupby('hn')
    n = g.cumcount() + 1
    ze['c_idm'] = g['idm'].cumsum() / n
    ze['c_ten'] = g['ten'].cumsum() / n
    ze['c_agari'] = g['agari'].cumsum() / n
    ze['c_furi'] = g['furibin'].cumsum()
    zr = ze[['hn', 'd', 'c_idm', 'c_ten', 'c_agari', 'c_furi']].sort_values('d')

    left = df[['horse_name', 'date_num']].copy()
    left['hn'] = left['horse_name'].astype(str).str.strip()
    left['d'] = pd.to_numeric(left['date_num'], errors='coerce')
    left['_row'] = np.arange(len(left))
    left = left.sort_values('d')
    merged = pd.merge_asof(left, zr, on='d', by='hn', direction='backward', allow_exact_matches=False)
    merged = merged.sort_values('_row')
    out = pd.DataFrame(index=df.index)
    out['jrdb_ze_idm_avg'] = merged['c_idm'].values
    out['jrdb_ze_ten_avg'] = merged['c_ten'].values
    out['jrdb_ze_agari_avg'] = merged['c_agari'].values
    out['jrdb_ze_furi_count'] = merged['c_furi'].values
    matched = out['jrdb_ze_idm_avg'].notna().mean()
    for c in ZE:
        out[c] = out[c].fillna(DEFAULTS[c])
    return out, matched


def override_test(df, feats, label):
    df = df.copy()
    df['_rk'] = [f"{int(a)}_{b}_{int(c)}_{int(e)}_{int(f)}" for a, b, c, e, f in
                 zip(df['date_num'], df['course'], df['kai'], df['nichi'], df['race_num'])]
    df['pop'] = pd.to_numeric(df['oz_base_pop_rank'], errors='coerce').fillna(df['num_horses_val'])
    y = pd.to_numeric(df['year'], errors='coerce')
    for f in feats: df[f] = pd.to_numeric(df[f], errors='coerce').fillna(0)
    P = {'objective': 'binary', 'metric': 'auc', 'num_leaves': 63, 'learning_rate': 0.05,
         'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 5, 'min_child_samples': 50,
         'verbose': -1, 'seed': 42}
    XP = {'objective': 'binary:logistic', 'max_depth': 6, 'learning_rate': 0.05, 'subsample': 0.8,
          'colsample_bytree': 0.8, 'min_child_weight': 50, 'seed': 42, 'tree_method': 'hist', 'verbosity': 0}
    from sklearn.metrics import roc_auc_score
    pay = load_payouts()
    rows = []; aucs = []
    for ty in [23, 24, 25]:
        tr = y < ty; te = y == ty
        ml = lgb.train(P, lgb.Dataset(df.loc[tr, feats].values, label=df.loc[tr, 'target'].values), num_boost_round=400)
        mx = xgb.train(XP, xgb.DMatrix(df.loc[tr, feats].values, label=df.loc[tr, 'target'].values), num_boost_round=400)
        p = 0.5 * ml.predict(df.loc[te, feats].values) + 0.5 * mx.predict(xgb.DMatrix(df.loc[te, feats].values))
        aucs.append(roc_auc_score(df.loc[te, 'target'].values, p))
        s = df.loc[te, ['_rk', 'finish', 'pop', 'horse_num']].copy(); s['s'] = p; rows.append(s)
    oof = pd.concat(rows)
    ow = on = tw = tn = tr_ = 0
    for k, g in oof.groupby('_rk'):
        if k not in pay or len(g) < 4: continue
        t1 = g.sort_values('s', ascending=False).iloc[0]; fav = g.sort_values('pop').iloc[0]
        tn += 1; w = int(t1['finish']) == 1; tw += w; tr_ += pay[k]['tansho_pay'] if w else 0
        if int(t1['horse_num']) != int(fav['horse_num']): on += 1; ow += w
    print(f"  {label:24s} AUC={np.mean(aucs):.4f} top1勝率{tw/tn*100:4.1f}% override{ow/on*100:4.1f}% 単勝ROI{tr_/(100*tn)*100:6.1f}%")
    return {'auc': float(np.mean(aucs)), 'top1': tw/tn, 'override': ow/on, 'roi': tr_/(100*tn)}


def main():
    t0 = time.time()
    obj = pickle.load(gzip.open(os.path.join(DATA, '_v15_optuna_df_cache.pkl.gz'), 'rb'))
    df = obj['df']; feats = obj['features']
    if 'target' not in df.columns: df['target'] = (df['finish'] <= 3).astype(int)

    print("=== ze 4特徴 expanding 再計算 (当該race日付より前のZEDのみ) ===")
    new_ze, matched = recompute_ze(df)
    print(f"  ZED過去走でマッチした行: {matched*100:.1f}% (残りは初出走等=default)")
    # 旧 vs 新 比較(リーク除去の確認: 勝ち馬の ze_idm_avg は新の方が低いはず=当該勝走を除外)
    win = df['finish'] == 1
    old = pd.to_numeric(df['jrdb_ze_idm_avg'], errors='coerce')
    print(f"  jrdb_ze_idm_avg 勝ち馬平均: 旧={old[win].mean():.2f} → 新={new_ze['jrdb_ze_idm_avg'][win].mean():.2f} (低下=当該勝走を除外できた)")
    print(f"  jrdb_ze_idm_avg 負け馬平均: 旧={old[~win].mean():.2f} → 新={new_ze['jrdb_ze_idm_avg'][~win].mean():.2f}")
    from scipy.stats import pearsonr
    fin = pd.to_numeric(df['finish'], errors='coerce')
    m = old.notna() & fin.notna()
    print(f"  corr(ze_idm_avg, finish): 旧={pearsonr(old[m], fin[m])[0]:+.3f} → 新={pearsonr(new_ze['jrdb_ze_idm_avg'][m], fin[m])[0]:+.3f} (0に近づく=リーク減)")

    # leak-free cache 生成(別ファイル)
    df_lf = df.copy()
    for c in ZE: df_lf[c] = new_ze[c].values
    out = {'df': df_lf, 'features': feats}
    for k in obj:
        if k not in ('df', 'features'): out[k] = obj[k]
    path = os.path.join(DATA, '_v15_optuna_df_cache_leakfree.pkl.gz')
    with gzip.open(path, 'wb') as f: pickle.dump(out, f, protocol=4)
    print(f"\n  leak-free cache 保存: {path} (元cache・本番不変)")

    print("\n=== override正常化検証 (V15相当 145特徴) ===")
    print("  [リーク時参照] V15_full(元cache) top1 45.1% / override 44.3% / ROI 156.5%、対照RAW override~35%")
    r_lf = override_test(df_lf, feats, 'V15(leak-free cache)')
    r_old = override_test(df, feats, 'V15(元cache=リーク)')
    print("\n=== 判定 ===")
    print(f"  override: 元{r_old['override']*100:.1f}% → leak-free {r_lf['override']*100:.1f}% (対照RAW~35%・市場本命31.7%)")
    print(f"  単勝ROI : 元{r_old['roi']*100:.1f}% → leak-free {r_lf['roi']*100:.1f}%")
    print(f"  top1勝率: 元{r_old['top1']*100:.1f}% → leak-free {r_lf['top1']*100:.1f}%")
    print(f"  WF AUC  : 元{r_old['auc']:.4f} → leak-free {r_lf['auc']:.4f}")
    import json
    json.dump({'matched': matched, 'leakfree': r_lf, 'orig': r_old},
              open(os.path.join(DATA, 'v16_leakfree_validation.json'), 'w'), ensure_ascii=False, indent=2)
    print(f"\nDONE in {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
