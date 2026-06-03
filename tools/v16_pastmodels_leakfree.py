#!/usr/bin/env python3
"""過去モデル(V24/V24b/+best-effort V25) を leak-free cache で V15/s2b と横並び再評価。
leak-free cache・WF OOF・JRA公式払戻でROI。本番不変・投票未使用。
V20(NAR/sib_exp)/V21(動画)/V22(track_lap=post-raceリーク)/V23(外部lap parquet)は対象外(外部特徴/別リーク)。
"""
from __future__ import annotations
import os, sys, gzip, pickle, json, time
if sys.platform == "win32": sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd, lightgbm as lgb, xgboost as xgb
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.abspath('tools'))
from v16_anaba_s2_eval import build_features, ODDS_REMOVE, PROXY_FAMILY, RAW_REPLACE, NEW
from v16_leakfree_roi_grid import load_payouts, LGB_P, XGB_P, S_tan, S_trio4, S_umaren_t3box, S_fuku1
EXTRA = ['paci_goal_rank', 'paci_goal_diff', 'paci_dochu_rank']; DATA = 'data'
# V24/V24b 削除特徴
V24_REMOVE = ['prev_race_last3f','prev_race_first3f','prev_race_pace_diff','prev_odds_log','pci',
              'sire_shinba_top3r','has_training','odds_sharp_drop','is_nar','gaisha_rank',
              'course_renovated','jrdb_prev_interference','stable_comment_score','has_wood_training','jrdb_ls_idx']
V24B_REMOVE = ['prev_race_last3f','prev_race_first3f','prev_race_pace_diff','prev_odds_log','pci',
               'sire_shinba_top3r','has_training','odds_sharp_drop','is_nar','gaisha_rank']
TYB_NEW = ['tyb_ashimoto','tyb_batai_code','tyb_kehai_code','tyb_padock_idx']


def merge_tyb(df):
    """V25用: jrdb_tyb.csv の直前データ(発走前)を race_id+umaban で結合。失敗時 None。"""
    try:
        tyb = pd.read_csv(os.path.join(DATA,'jrdb_tyb.csv'), dtype={'race_id':str},
                          usecols=['race_id','umaban','ashimoto','batai_code','kehai_code','padock_idx'])
        tyb['race_id']=tyb['race_id'].astype(str); tyb['umaban']=pd.to_numeric(tyb['umaban'],errors='coerce')
        key_df = df['race_id'].astype(str)
        cov = key_df.isin(set(tyb['race_id'])).mean()
        if cov < 0.5:
            return None, cov
        tyb=tyb.rename(columns={'ashimoto':'tyb_ashimoto','batai_code':'tyb_batai_code','kehai_code':'tyb_kehai_code','padock_idx':'tyb_padock_idx'})
        m = df[['race_id','horse_num']].copy(); m['race_id']=m['race_id'].astype(str); m['umaban']=pd.to_numeric(m['horse_num'],errors='coerce')
        m=m.merge(tyb[['race_id','umaban']+TYB_NEW],on=['race_id','umaban'],how='left')
        for c in TYB_NEW: df[c]=pd.to_numeric(m[c],errors='coerce').fillna(0).values
        return df, cov
    except Exception as e:
        print("  TYB merge err:",e); return None, 0


def oof(df,feats,y):
    s=pd.Series(index=df.index,dtype=float); aucs=[]
    for ty in [23,24,25]:
        tr=y<ty; te=y==ty
        ml=lgb.train(LGB_P,lgb.Dataset(df.loc[tr,feats].values,label=df.loc[tr,'target'].values),num_boost_round=420)
        mx=xgb.train(XGB_P,xgb.DMatrix(df.loc[tr,feats].values,label=df.loc[tr,'target'].values),num_boost_round=420)
        p=0.5*ml.predict(df.loc[te,feats].values)+0.5*mx.predict(xgb.DMatrix(df.loc[te,feats].values))
        aucs.append(roc_auc_score(df.loc[te,'target'].values,p)); s.loc[te]=p
    return s,float(np.mean(aucs))


def roi_of(df,scol,pay,fn,pts_div):
    ret=stake=hit=n=0
    for k,g in df.groupby('_rk'):
        if k not in pay or len(g)<5: continue
        o=[int(x) for x in g.sort_values(scol,ascending=False)['horse_num'].tolist()]
        r,pts=fn(o,pay[k]); n+=1; ret+=r; stake+=100*pts; hit+=(r>0)
    return ret/stake if stake else 0, hit/n if n else 0, n


def main():
    t0=time.time()
    obj=pickle.load(gzip.open(os.path.join(DATA,'_v15_optuna_df_cache_leakfree.pkl.gz'),'rb')); df=obj['df']; v15=obj['features']
    if 'target' not in df: df['target']=(df['finish']<=3).astype(int)
    df=build_features(df)
    df['pop']=pd.to_numeric(df['oz_base_pop_rank'],errors='coerce').fillna(df['num_horses_val'])
    df['_rk']=[f"{int(a)}_{b}_{int(c)}_{int(e)}_{int(f)}" for a,b,c,e,f in zip(df['date_num'],df['course'],df['kai'],df['nichi'],df['race_num'])]
    y=pd.to_numeric(df['year'],errors='coerce')
    v16=[f for f in v15 if f not in ODDS_REMOVE]
    s2b=[f for f in v16 if f not in (PROXY_FAMILY+EXTRA+RAW_REPLACE)]+NEW
    V24=[f for f in v15 if f not in V24_REMOVE]
    V24b=[f for f in v15 if f not in V24B_REMOVE]
    models={'V15':v15,'s2b':s2b,'V24':V24,'V24b':V24b}
    # V25 best-effort (TYB)
    df,cov=merge_tyb(df) if True else (df,0)
    if df is not None and cov>=0.5:
        V25=[f for f in v15 if f not in ['has_wood_training']]+TYB_NEW  # V15 + TYB直前
        models['V25(+TYB)']=V25
        print(f"V25 TYB merge OK (race_idカバレッジ {cov*100:.0f}%)")
    else:
        # merge_tyb が None を返すと df が None になるので元cache再ロード
        obj=pickle.load(gzip.open(os.path.join(DATA,'_v15_optuna_df_cache_leakfree.pkl.gz'),'rb')); df=obj['df']
        df=build_features(df); df['pop']=pd.to_numeric(df['oz_base_pop_rank'],errors='coerce').fillna(df['num_horses_val'])
        df['_rk']=[f"{int(a)}_{b}_{int(c)}_{int(e)}_{int(f)}" for a,b,c,e,f in zip(df['date_num'],df['course'],df['kai'],df['nichi'],df['race_num'])]
        print(f"V25 TYB merge SKIP (race_idカバレッジ {cov*100:.0f}% <50%、cache race_id と jrdb_tyb race_id 形式不一致)")
    for f in set().union(*models.values()):
        if f in df.columns: df[f]=pd.to_numeric(df[f],errors='coerce').fillna(0)

    pay=load_payouts()
    print("\nWF学習(leak-free)...",flush=True)
    res={}
    for name,feats in models.items():
        feats=[f for f in feats if f in df.columns]
        s,auc=oof(df,feats,y); df['_s']=s
        ev=df[y>=23]
        tan=roi_of(ev,'_s',pay,S_tan,1); t4=roi_of(ev,'_s',pay,S_trio4,4); um=roi_of(ev,'_s',pay,S_umaren_t3box,3); fk=roi_of(ev,'_s',pay,S_fuku1,1)
        res[name]={'n_feat':len(feats),'auc':auc,'tan':tan,'t4':t4,'um':um,'fk':fk}
        print(f"  {name:10s}({len(feats):3d}feat) AUC={auc:.4f} 単勝{tan[0]*100:5.1f}% 複勝top1{fk[0]*100:5.1f}% 馬連top3box{um[0]*100:5.1f}% 三連複top4box{t4[0]*100:5.1f}%(N={t4[2]})",flush=True)

    print("\n=== 横並び比較 (leak-free, 2023-25 WF) ===")
    print(f"{'model':12s}{'feat':>5s}{'AUC':>8s}{'単勝':>8s}{'複勝t1':>8s}{'馬連box':>9s}{'三連複t4':>9s}")
    for n,r in res.items():
        print(f"{n:12s}{r['n_feat']:5d}{r['auc']:8.4f}{r['tan'][0]*100:7.1f}%{r['fk'][0]*100:7.1f}%{r['um'][0]*100:8.1f}%{r['t4'][0]*100:8.1f}%")
    # s2b超えチェック
    sb=res['s2b']
    print(f"\n=== s2b(単勝{sb['tan'][0]*100:.1f}%/三連複t4 {sb['t4'][0]*100:.1f}%/AUC{sb['auc']:.4f}) を超える過去モデル ===")
    beat=[n for n,r in res.items() if n not in('s2b','V15') and (r['t4'][0]>sb['t4'][0] or r['tan'][0]>sb['tan'][0] or r['auc']>sb['auc'])]
    print("  超過:",beat if beat else "なし(s2bが最良)")
    json.dump({n:{'auc':r['auc'],'tan':r['tan'][0],'t4':r['t4'][0],'um':r['um'][0],'n':r['t4'][2]} for n,r in res.items()},
              open(os.path.join(DATA,'v16_pastmodels_leakfree.json'),'w'),ensure_ascii=False,indent=2)
    print(f"\nDONE in {time.time()-t0:.0f}s")


if __name__=='__main__':
    main()
