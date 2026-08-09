# -*- coding: utf-8 -*-
"""
item2: c-only を 640R の★確定オッズ★で本判定（PRE_REGISTRATION.md 基準・変更禁止）。
- c-only は cache 2020-2025 全体で学習（2026は学習外＝out-of-time, リークなし）。
- 予測は data/v15_feat_dump の 125 c特徴。★JRDB健全日のみ★(JRDB定数≤30) を対象。
- 市場prob・複勝払戻・着順は research/ruiji/raw_results（★確定オッズ・確定配当★）。
- 乖離最上位デシルの複勝回収CI下限 > 市場ベースライン で GO。

実行: python item2_final_odds_eval.py
"""
import sys, os, glob, json, gzip, pickle
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd, lightgbm as lgb, xgboost as xgb

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DUMP = os.path.join(ROOT, "data", "v15_feat_dump")
RES = os.path.join(ROOT, "research", "ruiji", "raw_results")
A_ODDS = {'prev_odds_log','oz_tansho_base_log','oz_fukusho_base_log','oz_base_pop_rank',
          'odds_change_rate','pop_rank_change','odds_sharp_drop'}
B_MKT = {'jrdb_training_idx','jrdb_cid_idx','paci_manken_idx','paci_goal_rank','paci_dochu_rank',
         'paci_goal_diff','paci_jockey_exp_wr','paci_jockey_exp_3rd','paci_ninki_idx',
         'paci_sogo_mark','paci_idm_mark','paci_jockey_mark','paci_train_mark'}
LGB_P = dict(objective='binary', metric='auc', num_leaves=63, learning_rate=0.05,
             feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5, min_child_samples=50,
             reg_alpha=0.1, reg_lambda=0.1, verbose=-1, seed=42)
XGB_P = dict(objective='binary:logistic', eval_metric='auc', max_depth=6, learning_rate=0.05,
             subsample=0.8, colsample_bytree=0.8, min_child_weight=50, reg_alpha=0.1,
             reg_lambda=0.1, seed=42, tree_method='hist')
JRDB_HEALTH_MAX = 30  # JRDB定数特徴が これ以下 の日のみ採用


def train_c_only():
    with gzip.open(os.path.join(ROOT, "data", "_v15_optuna_df_cache_leakfree_v2.pkl.gz"), "rb") as f:
        d = pickle.load(f)
    df = d["df"]; feats = d["features"]
    c = [f for f in feats if f not in A_ODDS and f not in B_MKT]
    y = (df["finish"] <= 3).astype(int)
    va = df[pd.to_numeric(df["year"], errors="coerce") == 25]
    tr = df[pd.to_numeric(df["year"], errors="coerce") < 25]
    lm = lgb.train(LGB_P, lgb.Dataset(tr[c], (tr["finish"]<=3).astype(int)), num_boost_round=1000,
                   valid_sets=[lgb.Dataset(va[c], (va["finish"]<=3).astype(int))],
                   callbacks=[lgb.early_stopping(50, verbose=False)])
    xm = xgb.train(XGB_P, xgb.DMatrix(tr[c], label=(tr["finish"]<=3).astype(int)), num_boost_round=1000,
                   evals=[(xgb.DMatrix(va[c], label=(va["finish"]<=3).astype(int)), "v")],
                   early_stopping_rounds=50, verbose_eval=False)
    return lm, xm, c


def load_results():
    R = {}
    for fp in glob.glob(os.path.join(RES, "*.json")):
        for r in json.load(open(fp, encoding="utf-8")):
            fo = {int(k): int(v) for k, v in (r.get("finish_order") or {}).items()}
            od = {int(k): float(v) for k, v in (r.get("odds") or {}).items()}
            fk = {int(k): int(v) for k, v in ((r.get("payouts") or {}).get("fukusho") or {}).items()}
            R[str(r["race_id"])] = {"fin": fo, "odds": od, "fuku": fk}
    return R


def jrdb_dead(df, cfeats):
    jr = [c for c in cfeats if c.startswith("jrdb_")]
    return int((df[jr].nunique() <= 2).sum())


def main():
    lm, xm, cfeats = train_c_only()
    jr = [c for c in cfeats if c.startswith("jrdb_")]
    print(f"c-only 学習完了 (c={len(cfeats)}, JRDB={len(jr)})")
    R = load_results()
    rows = []; healthy_dates = set(); dead_dates = set()
    for datedir in sorted(glob.glob(os.path.join(DUMP, "2026*"))):
        dt = os.path.basename(datedir)
        parts = []
        for fp in glob.glob(os.path.join(datedir, "*.parquet")):
            if os.path.getsize(fp) == 0: continue
            try: p = pd.read_parquet(fp)
            except Exception: continue
            if len(p) < 3 or "race_id" not in p.columns: continue
            parts.append(p)
        if not parts: continue
        big = pd.concat(parts, ignore_index=True)
        dead = jrdb_dead(big, cfeats)
        if dead > JRDB_HEALTH_MAX:
            dead_dates.add(dt); continue
        healthy_dates.add(dt)
        for p in parts:
            rid = str(p["race_id"].iloc[0])
            if rid not in R: continue
            pred = 0.5*lm.predict(p[cfeats]) + 0.5*xm.predict(xgb.DMatrix(p[cfeats]))
            pp = p.copy(); pp["pred"] = pred
            pp["umaban"] = pd.to_numeric(pp["馬番"], errors="coerce")
            pp = pp.dropna(subset=["umaban"]); pp["umaban"] = pp["umaban"].astype(int)
            rr = R[rid]
            for _, h in pp.iterrows():
                ub = int(h["umaban"]); fin = rr["fin"].get(ub); od = rr["odds"].get(ub)
                if fin is None or od is None or od <= 0: continue
                rows.append(dict(rid=rid, date=dt, umaban=ub, pred=h["pred"],
                                 finish=fin, odds=od, fuku=rr["fuku"].get(ub, 0)))
    O = pd.DataFrame(rows)
    print(f"対象: 健全日{len(healthy_dates)} (除外JRDB死{len(dead_dates)})  馬={len(O)} レース={O['rid'].nunique()}")
    if len(O) < 500:
        print("※ n小。健全日のみの先行判定。");
    O["mkt_raw"] = 1.0 / O["odds"]
    O["model_norm"] = O.groupby("rid")["pred"].transform(lambda s: s/s.sum())
    O["mkt_norm"] = O.groupby("rid")["mkt_raw"].transform(lambda s: s/s.sum())
    O["divergence"] = O["model_norm"] - O["mkt_norm"]
    O["place"] = (O["finish"] <= 3).astype(int)
    O["fuku_ret"] = O["fuku"] / 100.0          # ★確定複勝払戻★
    O["dec"] = pd.qcut(O["divergence"].rank(method="first"), 10, labels=False)
    base = O["fuku_ret"].mean()

    def ci(sub, col, n=2000, seed=42):
        rng = np.random.RandomState(seed)
        g = [x[col].values for _, x in sub.groupby("rid")]
        if not g: return (np.nan, np.nan)
        m = [np.concatenate([g[i] for i in rng.randint(0,len(g),len(g))]).mean() for _ in range(n)]
        return np.percentile(m,2.5), np.percentile(m,97.5)

    top = O[O["dec"] == 9]
    lo, hi = ci(top, "fuku_ret")
    print("\n"+"="*60)
    print(f"市場ベースライン(全馬 確定複勝回収)= {base*100:.1f}%")
    print(f"★最上位divergenceデシル★ n={len(top)} 複勝率={top['place'].mean()*100:.1f}% 中央odds={top['odds'].median():.1f}")
    print(f"  確定複勝回収 = {top['fuku_ret'].mean()*100:.1f}%  95%CI[{lo*100:.1f}, {hi*100:.1f}]")
    verdict = "GO" if lo > base else "NO-GO"
    print(f"  事前基準: CI下限({lo*100:.1f}%) > 市場ベースライン({base*100:.1f}%)? → ★{verdict}★")
    print("\n--- 全デシル(参考) ---")
    g = O.groupby("dec").agg(n=("place","size"),複勝率=("place","mean"),確定複勝回収=("fuku_ret","mean"),中央odds=("odds","median"))
    g["複勝率"]=(g["複勝率"]*100).round(1); g["確定複勝回収"]=(g["確定複勝回収"]*100).round(1)
    print(g.to_string())
    print(f"\n健全日: {sorted(healthy_dates)}")


if __name__ == "__main__":
    main()
