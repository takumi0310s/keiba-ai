# -*- coding: utf-8 -*-
"""(e) V15現行145を「ライブ実態」(premium16+lap4=default定数化, S1は生存) で同一fold測定。
主従宣言のための公平比較軸 (prereg基準の変更ではない・補助診断)。"""
import os,sys,gzip,pickle,json
BASE=os.path.abspath(os.path.join(os.path.dirname(__file__),'..','..'))
if sys.platform=='win32': sys.stdout.reconfigure(encoding='utf-8')
import numpy as np,pandas as pd,lightgbm as lgb,xgboost as xgb
from sklearn.metrics import roc_auc_score
DEAD=['index_max_filled','index_run1_filled','index_avg5_filled','stable_comment_score',
 'wood_best_4f_filled','sakaro_best_4f_filled','sakaro_best_3f_filled','time_1f_last_filled',
 'training_intensity_enc','wood_count_2w','total_training_count','training_per_dist',
 'has_training','has_wood_training','has_sakaro_training','training_time_filled',
 'prev_race_first3f','prev_race_last3f','prev_race_pace_diff','prev_agari_relative']
LGB_P=dict(objective='binary',metric='auc',num_leaves=63,learning_rate=0.05,feature_fraction=0.8,
 bagging_fraction=0.8,bagging_freq=5,min_child_samples=50,reg_alpha=0.1,reg_lambda=0.1,verbose=-1,seed=42)
XGB_P=dict(objective='binary:logistic',eval_metric='auc',max_depth=6,learning_rate=0.05,subsample=0.8,
 colsample_bytree=0.8,min_child_weight=50,reg_alpha=0.1,reg_lambda=0.1,seed=42,tree_method='hist')
d=pickle.load(gzip.open(os.path.join(BASE,'data','_v15_optuna_df_cache_leakfree_v2.pkl.gz'),'rb'))
df=d['df'].copy(); feats=d['features']; df['year']=pd.to_numeric(df['year'])
med={c:pd.to_numeric(df[df['year']<21][c],errors='coerce').median() for c in DEAD}
for c in DEAD: df[c]=med[c]   # ライブ実態: 死特徴=定数 (学習/テスト両方)
aucs=[]
for Y in [21,22,23,24,25]:
    tr=df[df['year']<Y]; va=df[df['year']==Y-1]; te=df[df['year']==Y]
    ytr=(tr['finish']<=3).astype(int); yva=(va['finish']<=3).astype(int); yte=(te['finish']<=3).astype(int)
    X=lambda s: s[feats].apply(pd.to_numeric,errors='coerce')
    lm=lgb.train(LGB_P,lgb.Dataset(X(tr),ytr),num_boost_round=1000,
        valid_sets=[lgb.Dataset(X(va),yva)],callbacks=[lgb.early_stopping(50,verbose=False)])
    xm=xgb.train(XGB_P,xgb.DMatrix(X(tr),label=ytr),num_boost_round=1000,
        evals=[(xgb.DMatrix(X(va),label=yva),'v')],early_stopping_rounds=50,verbose_eval=False)
    p=0.5*lm.predict(X(te))+0.5*xm.predict(xgb.DMatrix(X(te)))
    a=roc_auc_score(yte,p); aucs.append(a); print(f"fold20{Y}: {a:.4f}",flush=True)
print(f"(e) V15ライブ実態 mean AUC = {np.mean(aucs):.4f}")
json.dump({'e_v15_degraded':float(np.mean(aucs)),'folds':[float(x) for x in aucs]},
 open(os.path.join(os.path.dirname(__file__),'v15r_e_result.json'),'w'))
