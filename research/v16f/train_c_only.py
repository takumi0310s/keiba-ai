# -*- coding: utf-8 -*-
"""
A. c-only 再学習 + 事前登録評価（PRE_REGISTRATION.md に従う）。
c(純能力125)のみで LGB+XGB を V15同一WF fold(2021-25)学習→OOF→乖離最上位デシルの
複勝回収CI下限 vs 市場ベースライン で GO/NO-GO。data/ は読み取りのみ。
"""
import sys, os, gzip, pickle
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd, lightgbm as lgb, xgboost as xgb

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
A_ODDS = {'prev_odds_log','oz_tansho_base_log','oz_fukusho_base_log','oz_base_pop_rank',
          'odds_change_rate','pop_rank_change','odds_sharp_drop'}
B_MKT = {'jrdb_training_idx','jrdb_cid_idx','paci_manken_idx','paci_goal_rank','paci_dochu_rank',
         'paci_goal_diff','paci_jockey_exp_wr','paci_jockey_exp_3rd','paci_ninki_idx',
         'paci_sogo_mark','paci_idm_mark','paci_jockey_mark','paci_train_mark'}
LGB_P = dict(objective='binary', metric='auc', boosting_type='gbdt', num_leaves=63,
             learning_rate=0.05, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
             min_child_samples=50, reg_alpha=0.1, reg_lambda=0.1, verbose=-1, seed=42)
XGB_P = dict(objective='binary:logistic', eval_metric='auc', max_depth=6, learning_rate=0.05,
             subsample=0.8, colsample_bytree=0.8, min_child_weight=50, reg_alpha=0.1,
             reg_lambda=0.1, seed=42, tree_method='hist')


def main():
    with gzip.open(os.path.join(ROOT, "data", "_v15_optuna_df_cache_leakfree_v2.pkl.gz"), "rb") as f:
        d = pickle.load(f)
    df = d["df"]; feats = d["features"]
    c_feats = [f for f in feats if f not in A_ODDS and f not in B_MKT]
    print(f"c-only 特徴: {len(c_feats)} (145 - a7 - b13)")
    df = df.copy()
    df["yr"] = pd.to_numeric(df["year"], errors="coerce")  # 18-25
    df["rid"] = df["race_id_str"].astype(str)
    df["target"] = (df["finish"] <= 3).astype(int)
    from sklearn.metrics import roc_auc_score

    oof = []
    for Y in [21, 22, 23, 24, 25]:
        tr = df[df["yr"] < Y]; va = df[df["yr"] == (Y - 1)]; te = df[df["yr"] == Y]
        if len(te) == 0: continue
        Xtr, ytr = tr[c_feats], tr["target"]; Xva, yva = va[c_feats], va["target"]
        # LGB
        dtr = lgb.Dataset(Xtr, ytr); dva = lgb.Dataset(Xva, yva)
        lm = lgb.train(LGB_P, dtr, num_boost_round=1000, valid_sets=[dva],
                       callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
        lp = lm.predict(te[c_feats])
        # XGB
        dtrx = xgb.DMatrix(Xtr, label=ytr); dvax = xgb.DMatrix(Xva, label=yva)
        xm = xgb.train(XGB_P, dtrx, num_boost_round=1000, evals=[(dvax, "v")],
                       early_stopping_rounds=50, verbose_eval=False)
        xp = xm.predict(xgb.DMatrix(te[c_feats]))
        pred = 0.5 * lp + 0.5 * xp
        auc = roc_auc_score(te["target"], pred)
        o = te[["rid", "target", "finish", "oz_tansho_base_log", "oz_fukusho_base_log", "yr"]].copy()
        o["pred"] = pred
        oof.append(o)
        print(f"  fold {2000+Y}: n={len(te)} AUC={auc:.4f}")
    O = pd.concat(oof, ignore_index=True)
    print(f"\nOOF 総数: {len(O)}  レース: {O['rid'].nunique()}")

    # レース内正規化 → divergence
    O["tan_odds"] = np.exp(O["oz_tansho_base_log"])
    O["mkt_raw"] = 1.0 / O["tan_odds"]
    O["model_norm"] = O.groupby("rid")["pred"].transform(lambda s: s / s.sum())
    O["mkt_norm"] = O.groupby("rid")["mkt_raw"].transform(lambda s: s / s.sum())
    O["divergence"] = O["model_norm"] - O["mkt_norm"]
    # 複勝return proxy
    O["fuku_odds"] = np.exp(O["oz_fukusho_base_log"])
    O["fuku_ret"] = np.where(O["finish"] <= 3, O["fuku_odds"], 0.0)
    O["tan_ret"] = np.where(O["finish"] == 1, O["tan_odds"], 0.0)

    O["dec"] = pd.qcut(O["divergence"].rank(method="first"), 10, labels=False)
    base_roi = O["fuku_ret"].mean()  # 市場ベースライン(全馬複勝回収)

    def boot_ci(sub, col, n=2000, seed=42):
        rng = np.random.RandomState(seed)
        groups = [g[col].values for _, g in sub.groupby("rid")]
        if not groups: return (np.nan, np.nan)
        ms = []
        for _ in range(n):
            idx = rng.randint(0, len(groups), len(groups))
            ms.append(np.concatenate([groups[i] for i in idx]).mean())
        return np.percentile(ms, 2.5), np.percentile(ms, 97.5)

    top = O[O["dec"] == 9]  # 最上位デシル(divergence最大)
    lo, hi = boot_ci(top, "fuku_ret")
    print("\n" + "=" * 60)
    print(f"市場ベースライン(全馬複勝回収): {base_roi*100:.1f}%")
    print(f"★最上位divergenceデシル★ n={len(top)} 複勝率={ (top['finish']<=3).mean()*100:.1f}%")
    print(f"  複勝回収 ROI = {top['fuku_ret'].mean()*100:.1f}%  95%CI[{lo*100:.1f}, {hi*100:.1f}]")
    print(f"  単勝回収 ROI = {top['tan_ret'].mean()*100:.1f}% (参考)")
    verdict = "GO" if lo > base_roi else "NO-GO"
    print(f"\n  事前基準: CI下限({lo*100:.1f}%) > 市場ベースライン({base_roi*100:.1f}%) ? → ★{verdict}★")

    print("\n--- 参考: 全デシル(判定に使わない) ---")
    g = O.groupby("dec").agg(n=("target","size"), 複勝率=("target","mean"),
        複勝回収=("fuku_ret","mean"), 平均div=("divergence","mean"), 平均人気odds=("tan_odds","median"))
    g["複勝率"]=(g["複勝率"]*100).round(1); g["複勝回収"]=(g["複勝回収"]*100).round(1)
    g["平均div"]=g["平均div"].round(3); g["平均人気odds"]=g["平均人気odds"].round(1)
    print(g.to_string())


if __name__ == "__main__":
    main()
