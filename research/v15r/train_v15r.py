# -*- coding: utf-8 -*-
"""v15r WF学習+ablation (PREREG_V15R_TRAINING.md 準拠・基準変更禁止)。"""
import json, os, sys, gzip, pickle
BASE=os.path.abspath(os.path.join(os.path.dirname(__file__),'..','..'))
HERE=os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,os.path.join(BASE,'tools'))
if sys.platform=='win32': sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, lightgbm as lgb, xgboost as xgb
from sklearn.metrics import roc_auc_score

LGB_P=dict(objective='binary',metric='auc',num_leaves=63,learning_rate=0.05,
 feature_fraction=0.8,bagging_fraction=0.8,bagging_freq=5,min_child_samples=50,
 reg_alpha=0.1,reg_lambda=0.1,verbose=-1,seed=42)
XGB_P=dict(objective='binary:logistic',eval_metric='auc',max_depth=6,learning_rate=0.05,
 subsample=0.8,colsample_bytree=0.8,min_child_weight=50,reg_alpha=0.1,reg_lambda=0.1,
 seed=42,tree_method='hist')

def wf_auc(df, feats, label):
    aucs=[]
    for Y in [21,22,23,24,25]:
        tr=df[df['year']<Y]; va=df[df['year']==Y-1]; te=df[df['year']==Y]
        ytr=(tr['finish']<=3).astype(int); yva=(va['finish']<=3).astype(int); yte=(te['finish']<=3).astype(int)
        Xtr=tr[feats].apply(pd.to_numeric,errors='coerce'); Xva=va[feats].apply(pd.to_numeric,errors='coerce'); Xte=te[feats].apply(pd.to_numeric,errors='coerce')
        lm=lgb.train(LGB_P,lgb.Dataset(Xtr,ytr),num_boost_round=1000,
            valid_sets=[lgb.Dataset(Xva,yva)],callbacks=[lgb.early_stopping(50,verbose=False)])
        xm=xgb.train(XGB_P,xgb.DMatrix(Xtr,label=ytr),num_boost_round=1000,
            evals=[(xgb.DMatrix(Xva,label=yva),'v')],early_stopping_rounds=50,verbose_eval=False)
        p=0.5*lm.predict(Xte)+0.5*xm.predict(xgb.DMatrix(Xte))
        a=roc_auc_score(yte,p); aucs.append(a)
        print(f"    fold20{Y}: AUC={a:.4f}",flush=True)
    return float(np.mean(aucs))

def main():
    df=pd.read_parquet(os.path.join(HERE,'v15r_train.parquet'))
    df['year']=pd.to_numeric(df['year'])
    fj=json.load(open(os.path.join(HERE,'v15r_features.json'),encoding='utf-8'))
    base,jv,srb,kka=fj['base_from_v15'],fj['jv'],fj['srb'],fj['kka']
    full=base+jv+srb+kka
    # (d) V15参照: cacheの145をそのまま
    d=pickle.load(gzip.open(os.path.join(BASE,'data','_v15_optuna_df_cache_leakfree_v2.pkl.gz'),'rb'))
    v15df=d['df'].copy(); v15df['year']=pd.to_numeric(v15df['year'])
    res={}
    print("=== (d) V15参照 145 (同一fold) ==="); res['d_v15']=wf_auc(v15df,d['features'],'finish')
    print("=== (a) full 140 ==="); res['a_full']=wf_auc(df,full,'finish')
    print("=== (b) -JV 132 ==="); res['b_nojv']=wf_auc(df,base+srb+kka,'finish')
    print("=== (c) -第2弾 130 ==="); res['c_noblk2']=wf_auc(df,base+jv,'finish')
    res['contrib_jv']=round(res['a_full']-res['b_nojv'],5)
    res['contrib_blk2']=round(res['a_full']-res['c_noblk2'],5)
    json.dump(res,open(os.path.join(HERE,'v15r_wf_results.json'),'w'),indent=1)
    print("\n=== 結果 ===")
    for k,v in res.items(): print(f"  {k}: {v:.4f}" if isinstance(v,float) else f"  {k}: {v}")
    # 事前登録判定
    thr=0.860 if res['d_v15']>=0.865 else res['d_v15']-0.005
    print(f"\n合格基準 AUC≥{thr:.4f}: {'★合格★' if res['a_full']>=thr else '★不合格★'} (a={res['a_full']:.4f})")
    print(f"第2弾寄与={res['contrib_blk2']:+.4f} → {'保持' if res['contrib_blk2']>0 else '★落とす★'}")
    print(f"JV寄与={res['contrib_jv']:+.4f} (供給代替・正負で落とさない/-0.002未満で再設計)")
if __name__=='__main__': main()
