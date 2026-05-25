"""V22 vs V15 retroactive comparison on 2025 test data.

Metrics:
- AUC (top-3 prediction)
- winner_top1: % races where predicted #1 was actual winner (finish=1)
- top3_recall: % races where top3 predicted included winner
- trio_rate: % races where any of top6 predicted are all 3 actual top-3 horses
"""
import sys, os, gzip, pickle, json, time
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

print("Loading V15 train cache...")
t0 = time.time()
with open(os.path.join(BASE, "data/_v15_train_df_cache.pkl"), "rb") as f:
    d = pickle.load(f)
df = d["df"].copy()
v15_features = list(d.get("v15_features", d.get("features", [])))
print(f"  shape={df.shape}, features={len(v15_features)}, elapsed={time.time()-t0:.0f}s")

# Year normalization
df["_y"] = df["year"].apply(lambda y: 2000 + int(y) if int(y) <= 30 else 1900 + int(y))

# Load V22 model
print("\nLoading V22 model...")
with gzip.open(os.path.join(BASE, "keiba_model_v22_top100_central.pkl.gz"), "rb") as f:
    v22 = pickle.load(f)
v22_features = v22["features"]
lgb_v22 = v22["lgb_model"]
xgb_v22 = v22["xgb_model"]
print(f"  V22 features: {len(v22_features)}")

# Check feature availability
v22_avail = [f for f in v22_features if f in df.columns]
v22_missing = [f for f in v22_features if f not in df.columns]
print(f"  V22 available: {len(v22_avail)}/{len(v22_features)} (missing: {v22_missing[:5]})")

v15_avail = [f for f in v15_features if f in df.columns]
print(f"  V15 available: {len(v15_avail)}/{len(v15_features)}")

# Target variable
if "finish" in df.columns:
    df["target"] = (df["finish"] <= 3).astype(int)
elif "top3" in df.columns:
    df["target"] = df["top3"]
else:
    raise ValueError("No finish/top3 column")

# 2025 test split
train_mask = (df["_y"] >= 2015) & (df["_y"] <= 2024)
test_mask  = df["_y"] == 2025
print(f"\nTrain: {train_mask.sum():,}, Test: {test_mask.sum():,}")

from lightgbm import Booster
import xgboost as xgb

y_te = df.loc[test_mask, "target"].values
race_te = df.loc[test_mask, "race_id_str"].astype(str).values if "race_id_str" in df.columns else (df.loc[test_mask, "race_id"].astype(str).values if "race_id" in df.columns else None)
finish_te = df.loc[test_mask, "finish"].values if "finish" in df.columns else None

# --- V15 prediction ---
print("\n=== V15 prediction ===")
import lightgbm as lgb

# Load V15 production model
v15_path = os.path.join(BASE, "keiba_model_v15_central.pkl.gz")
with gzip.open(v15_path, "rb") as f:
    v15_model = pickle.load(f)

lgb_v15 = v15_model.get("model") or v15_model.get("lgb_model") or v15_model.get("lgb")
xgb_v15 = v15_model.get("xgb_model") or v15_model.get("xgb")
v15_prod_feats = v15_model.get("features") or v15_model.get("feature_names") or v15_avail

v15_feats_avail = [f for f in v15_prod_feats if f in df.columns]
X_te_v15 = df.loc[test_mask, v15_feats_avail].fillna(0).values.astype(np.float32)

if hasattr(lgb_v15, "predict"):
    p_lgb_v15 = lgb_v15.predict(X_te_v15)
    p_xgb_v15 = xgb_v15.predict(xgb.DMatrix(X_te_v15))
    p_v15 = 0.5 * p_lgb_v15 + 0.5 * p_xgb_v15
else:
    p_v15 = np.zeros(len(X_te_v15))

# --- V22 prediction ---
print("\n=== V22 prediction ===")
# Fill missing V22 features with 0
for f in v22_missing:
    df[f] = 0.0

X_te_v22 = df.loc[test_mask, v22_features].fillna(0).values.astype(np.float32)
p_lgb_v22 = lgb_v22.predict(X_te_v22)
p_xgb_v22 = xgb_v22.predict(xgb.DMatrix(X_te_v22))
p_v22 = 0.5 * p_lgb_v22 + 0.5 * p_xgb_v22


# --- Metrics ---
def compute_metrics(preds, targets, races, finish):
    auc = roc_auc_score(targets, preds)

    tmp = pd.DataFrame({
        "race_id": races,
        "y": targets,
        "p": preds,
        "finish": finish if finish is not None else 0,
    })

    # winner_top1: % where top predicted horse is actual winner (finish=1)
    if finish is not None:
        top1 = tmp.loc[tmp.groupby("race_id")["p"].idxmax()]
        w1 = (top1["finish"] == 1).mean()
    else:
        top1 = tmp.loc[tmp.groupby("race_id")["p"].idxmax()]
        w1 = top1["y"].mean()

    # trio_rate: top6 predicted includes all 3 actual top-3
    def trio_hit(grp):
        top6 = set(grp.nlargest(6, "p").index)
        actual_top3 = set(grp[grp["finish"] <= 3].index) if finish is not None else set(grp[grp["y"] == 1].index)
        return len(actual_top3) >= 3 and actual_top3.issubset(top6)

    trio = tmp.groupby("race_id").apply(trio_hit).mean() if finish is not None else 0.0

    return {"auc": float(auc), "winner_top1": float(w1), "trio_rate": float(trio)}


print("\nComputing metrics...")
race_ids = race_te if race_te is not None else np.arange(len(y_te)).astype(str)
finish_vals = finish_te if finish_te is not None else None

met_v15 = compute_metrics(p_v15, y_te, race_ids, finish_vals)
met_v22 = compute_metrics(p_v22, y_te, race_ids, finish_vals)

print(f"\n{'Metric':<20} {'V15':>10} {'V22':>10} {'Delta':>10}")
print("-" * 52)
for k in ["auc", "winner_top1", "trio_rate"]:
    v = met_v15[k]
    v2 = met_v22[k]
    d = v2 - v
    print(f"  {k:<18} {v:>10.4f} {v2:>10.4f} {d:>+10.4f}")

results = {
    "v15": met_v15, "v22": met_v22,
    "delta_auc": met_v22["auc"] - met_v15["auc"],
    "delta_winner_top1": met_v22["winner_top1"] - met_v15["winner_top1"],
    "delta_trio_rate": met_v22["trio_rate"] - met_v15["trio_rate"],
    "v22_missing_feats": v22_missing,
    "v22_available": len(v22_avail),
}

out = os.path.join(BASE, "data/v20/v22_vs_v15_2025_retro.json")
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
print(f"\n[SAVED] {out}")
