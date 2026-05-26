"""V15 + sib_*_exp 単一-fold AUC / winner_top1 検証 (Phase 3 前半).

V17 cache が存在しないため、V15 cache を直接使用。
sib_top3_rate_exp / sib_shinba_wr_exp を追加し、V15 baseline と比較。

入力:
  data/_v15_train_df_cache.pkl         (875 MB)
  data/netkeiba_siblings_expanding.csv  (41 MB)

出力:
  data/v20/sib_exp_v15_singlefold_metrics.json
"""
import sys, os, json, time, pickle, gzip
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ===== 1. Load V15 cache =====
print("Loading V15 cache...")
t0 = time.time()
with open(os.path.join(BASE, "data/_v15_train_df_cache.pkl"), "rb") as f:
    d = pickle.load(f)
df = d["df"].copy()
v15_features = list(d["v15_features"])
print(f"  shape={df.shape}, v15_features={len(v15_features)}, elapsed={time.time()-t0:.0f}s")
print(f"  year range: {df['year'].min()}-{df['year'].max()}")

# Year (2-digit) → 4-digit
df["_y"] = df["year"].apply(lambda y: 2000 + int(y) if int(y) <= 30 else 1900 + int(y))

# ===== 2. Normalize horse_id for merge =====
# V15 cache: '04101502' (8-char with leading zeros)
# sib CSV: 15110062 (int, no leading zero)
# Normalize: str.zfill(8)
df["_horse_id_norm"] = df["horse_id"].astype(str).str.strip().apply(
    lambda x: str(int(float(x))).zfill(8) if x.replace(".", "").isdigit() else x
)
df["_race_id_norm"] = df["race_id"].astype(str).str.strip()

# ===== 3. Load sib_expanding CSV =====
print("\nLoading sib_expanding.csv...")
t0 = time.time()
SIB_EXP_COLS = ["sib_top3_rate_exp", "sib_shinba_wr_exp",
                "sib_total_races_exp", "sib_total_offspring_exp"]
sib = pd.read_csv(
    os.path.join(BASE, "data/netkeiba_siblings_expanding.csv"),
    dtype={"race_id": str, "horse_id": str},
    usecols=["race_id", "horse_id"] + SIB_EXP_COLS,
)
sib["race_id"] = sib["race_id"].str.strip()
sib["horse_id"] = sib["horse_id"].astype(str).apply(
    lambda x: str(int(float(x))).zfill(8) if x.replace(".", "").isdigit() else x
)
print(f"  sib rows={len(sib):,}, elapsed={time.time()-t0:.1f}s")
print(f"  sib horse_id sample: {sib['horse_id'].iloc[:3].tolist()}")
print(f"  df horse_id sample:  {df['_horse_id_norm'].iloc[:3].tolist()}")

# ===== 4. Merge =====
print("\nMerging sib_exp into df...")
n_before = len(df)
df_m = df.merge(
    sib.rename(columns={"race_id": "_race_id_norm", "horse_id": "_horse_id_norm"}),
    on=["_race_id_norm", "_horse_id_norm"],
    how="left",
)
matched = df_m[SIB_EXP_COLS[0]].notna().sum()
print(f"  matched: {matched:,}/{n_before:,} = {matched/n_before*100:.1f}%")
if matched < n_before * 0.3:
    print("  [WARN] match rate < 30% — horse_id format mismatch likely")
    print("  sib sample:", sib[["race_id","horse_id"]].head(3).to_dict("records"))
    print("  df sample:",  df[["_race_id_norm","_horse_id_norm"]].head(3).to_dict("records"))

# Fill NaN with 0 (mother 不明馬)
for c in SIB_EXP_COLS:
    df_m[c] = pd.to_numeric(df_m[c], errors="coerce").fillna(0)
    nz = int((df_m[c] > 0).sum())
    print(f"  {c}: mean={df_m[c].mean():.4f}, nonzero={nz:,}")

df = df_m

# ===== 5. Feature sets =====
v15_avail = [f for f in v15_features if f in df.columns]
sib_avail  = [f for f in SIB_EXP_COLS if f in df.columns]

print(f"\nv15_features available: {len(v15_avail)}/{len(v15_features)}")
print(f"sib_exp available:       {len(sib_avail)}")

FEATS_BASE = v15_avail
FEATS_SIB  = v15_avail + sib_avail

# ===== 6. Single-fold split =====
train_mask = (df["_y"] >= 2015) & (df["_y"] <= 2024)
test_mask  = df["_y"] == 2025
print(f"\nTrain: {train_mask.sum():,}, Test: {test_mask.sum():,}")

y_tr = (df.loc[train_mask, "finish"] <= 3).astype(int)
y_te = (df.loc[test_mask,  "finish"] <= 3).astype(int)
# Use race_id_str (per-race grouping key) not race_id (per-horse unique) — see V15-audit session note
race_te = df.loc[test_mask, "race_id_str"].astype(str) if "race_id_str" in df.columns else df.loc[test_mask, "race_id"].astype(str)
print(f"  Train positive: {y_tr.mean():.4f}, Test positive: {y_te.mean():.4f}")
print(f"  Unique races in test: {race_te.nunique()}")

LGB_PARAMS = {
    "objective": "binary", "metric": "auc",
    "learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 50,
    "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 5,
    "lambda_l1": 0.1, "lambda_l2": 0.1,
    "verbose": -1, "seed": 42,
}

def run_fold(feats, label):
    X_tr = df.loc[train_mask, feats]
    X_te = df.loc[test_mask,  feats]
    t = time.time()
    m = lgb.train(
        LGB_PARAMS,
        lgb.Dataset(X_tr, y_tr),
        num_boost_round=500,
        valid_sets=[lgb.Dataset(X_te, y_te)],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
    )
    p = m.predict(X_te)
    auc = roc_auc_score(y_te, p)
    # winner_top1 (top-scored horse in each race, count actual winner)
    tmp = pd.DataFrame({"race_id": race_te.values, "y": y_te.values, "p": p})
    top1 = tmp.loc[tmp.groupby("race_id")["p"].idxmax()]
    w1 = top1["y"].mean()
    elapsed = time.time() - t
    print(f"  [{label}] AUC={auc:.4f} winner_top1={w1:.4f} ({w1*100:.2f}%) n_feats={len(feats)} time={elapsed:.0f}s")
    return {"auc": float(auc), "winner_top1": float(w1), "n_feats": len(feats)}

print("\n=== Baseline (V15 only) ===")
res_base = run_fold(FEATS_BASE, "V15_base")

print("\n=== V15 + sib_*_exp ===")
res_sib = run_fold(FEATS_SIB, "V15+sib_exp")

# ===== 7. Summary =====
delta_auc = res_sib["auc"] - res_base["auc"]
delta_w1  = res_sib["winner_top1"] - res_base["winner_top1"]
print(f"\n=== Delta ===")
print(f"  ΔAUC:          {delta_auc:+.4f}")
print(f"  Δwinner_top1:  {delta_w1:+.4f} ({delta_w1*100:+.2f}%)")

results = {
    "base": res_base,
    "sib_exp": res_sib,
    "delta_auc": float(delta_auc),
    "delta_winner_top1": float(delta_w1),
    "sib_match_rate": float(matched / n_before),
    "sib_cols": sib_avail,
}

os.makedirs(os.path.join(BASE, "data/v20"), exist_ok=True)
out = os.path.join(BASE, "data/v20/sib_exp_v15_singlefold_metrics.json")
with open(out, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
print(f"\n[SAVED] {out}")

verdict = "GO" if delta_w1 >= 0.02 else ("MARGINAL" if delta_w1 >= 0 else "NO-GO")
print(f"\n=== VERDICT: {verdict} ===")
print(f"  基準: Δwinner_top1 ≥ +2pt → GO (V20 に sib_*_exp 追加)")
