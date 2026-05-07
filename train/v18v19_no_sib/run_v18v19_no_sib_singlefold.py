"""V18/V19 sib抜き 単一-fold 再学習 (Session #37 A).

目的:
- v162_added の sib_top3_rate / sib_shinba_wr を除外して再学習
- 単一 fold (train 2015-2024, test 2025) で LGB-only
- 既存 V18/V19 (sib 含む) との AUC / logloss / winner_top1 比較

出力:
- data/v18/v18v19_retraining/v18_lgb_no_sib_v1.txt
- data/v18/v18v19_retraining/v19_lgb_no_sib_v1.txt
- data/v18/v18v19_retraining/v18_no_sib_oos_2025.csv
- data/v18/v18v19_retraining/v19_no_sib_oos_2025.csv
- data/v18/v18v19_retraining/no_sib_metrics.json

注意:
- LGB-only (XGB は Session #38 で追加検討)
- 既存 V18/V19 model files (data/v18/models/) は触らない
- V15 daily_predict / predict_core / schtasks 完全不変
"""
import sys, os, io, json, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

import pandas as pd
import numpy as np
import pickle
import gzip
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, log_loss
from datetime import datetime

# === LEAK 除外 (v17_morning, v18, v19 と同じ) ===
TYB_EXCLUDE_MORNING = [
    "tyb_idm", "tyb_horse_weight", "tyb_weight_diff",
    "tyb_kehai_code", "tyb_batai_code", "tyb_bagu_change",
]
LEAK_OLD = [
    "tyb_jockey_idx", "jo_cid_soten_idx", "tyb_odds_idx", "tyb_info_idx",
    "jo_cid_idx", "jo_cid_sara_idx", "tyb_padock_idx", "jo_ls_idx",
    "tyb_sogo_mark", "tyb_sogo_idx", "tyb_odds_mark", "tyb_padock_mark",
]
LEAK_NEW_INDIVIDUAL = ["zk_prev_kishi_avg", "skb_kishi_avg"]
LEAK_TIME_INVARIANT = [
    "zk_prev_anshin", "skb_anshin", "skb_aisho", "skb_heavy_apt",
    "zk_prev_heavy_apt", "zk_prev_kyaku_left",
    "kka_bms_rensho_max", "tyb_ashimoto", "kka_bms_rensho_min",
]
# === Session #37 NEW: sib リーク疑い 削除 ===
SIB_LEAK_SUSPECT = ["sib_top3_rate", "sib_shinba_wr"]

ALL_LEAK = LEAK_OLD + LEAK_NEW_INDIVIDUAL + LEAK_TIME_INVARIANT + TYB_EXCLUDE_MORNING + SIB_LEAK_SUSPECT

OUT_DIR = "data/v18/v18v19_retraining"
os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 60)
print("V18/V19 sib抜き single-fold 再学習 (Session #37 A)")
print(f"Start: {datetime.now()}")
print(f"除外 features: {len(ALL_LEAK)} (sib 抜き 2 つ追加)")
print(f"sib 除外: {SIB_LEAK_SUSPECT}")
start_total = time.time()

# === データロード ===
print("\nLoading v17 cache (1.2GB)...")
ts = time.time()
with open("data/v17/_v17_train_df_cache.pkl", "rb") as f:
    d = pickle.load(f)
df = d["df"]
df["_y"] = 2000 + df["year"]
print(f"  Loaded in {(time.time()-ts):.0f}s, {len(df):,} rows")

with gzip.open("keiba_model_v15_central_live.pkl.gz", "rb") as f:
    m15 = pickle.load(f)
v15_features = list(m15["features"])

v162_added = d.get("v162_added_features", [])
v17_added = d.get("v17_added_features", [])

v15_avail = [f for f in v15_features if f in df.columns]
v162_avail = [f for f in v162_added if f in df.columns and f not in v15_avail and f not in ALL_LEAK]
v17_avail = [f for f in v17_added if f in df.columns and f not in v15_avail and f not in v162_avail and f not in ALL_LEAK]
all_avail = v15_avail + v162_avail + v17_avail

print(f"\nv15 features: {len(v15_avail)}")
print(f"v162 added (no sib): {len(v162_avail)}")
print(f"v17 added (no leak): {len(v17_avail)}")
print(f"Total features (no sib): {len(all_avail)}")

# 確認: sib_ が含まれていないこと
sib_in_features = [f for f in all_avail if f.startswith("sib_")]
assert not sib_in_features, f"sib_ features 残存: {sib_in_features}"
print(f"[OK] sib_* features 完全除外確認")

# === 単一 fold (train 2015-2024, test 2025) ===
train_mask = (df["_y"] >= 2015) & (df["_y"] <= 2024)
test_mask = df["_y"] == 2025
n_tr = int(train_mask.sum())
n_te = int(test_mask.sum())
print(f"\nTrain rows: {n_tr:,}, Test rows: {n_te:,}")

X_tr = df.loc[train_mask, all_avail]
X_te = df.loc[test_mask, all_avail]

# race_id / horse_num は両 model で共通保存
race_id_te = df.loc[test_mask, "race_id"].astype(str).values if "race_id" in df.columns else np.arange(n_te)
horse_num_te = df.loc[test_mask, "horse_num"].values if "horse_num" in df.columns else np.arange(n_te)

LGB_PARAMS = {
    "objective": "binary", "metric": "auc",
    "learning_rate": 0.03, "num_leaves": 255, "min_data_in_leaf": 50,
    "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 5,
    "lambda_l1": 0.1, "lambda_l2": 0.1,
    "verbose": -1, "seed": 42,
}

results = {}

# === V18 (TANSHO, is_win) ===
print("\n" + "=" * 60)
print("V18 TANSHO (is_win) sib抜き 学習")
print("=" * 60)
target_col = "is_win"
y_tr = df.loc[train_mask, target_col].astype(int)
y_te = df.loc[test_mask, target_col].astype(int)
print(f"Positive rate: {y_tr.mean():.4f}")

ts = time.time()
m_v18 = lgb.train(
    LGB_PARAMS,
    lgb.Dataset(X_tr, y_tr),
    num_boost_round=2000,
    valid_sets=[lgb.Dataset(X_te, y_te)],
    callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)],
)
p_v18 = m_v18.predict(X_te)
auc_v18 = roc_auc_score(y_te, p_v18)
ll_v18 = log_loss(y_te, np.clip(p_v18, 1e-7, 1 - 1e-7))
print(f"V18 LGB no-sib: AUC={auc_v18:.4f} logloss={ll_v18:.4f} time={(time.time()-ts)/60:.1f}min")

# winner_top1: race ごと top1 が is_win=1 の比率
v18_oos = pd.DataFrame({
    "race_id": race_id_te, "umaban": horse_num_te, "year": df.loc[test_mask, "_y"].values,
    "is_win": y_te.values, "p_v18_no_sib": p_v18,
})
top1_v18 = v18_oos.loc[v18_oos.groupby("race_id")["p_v18_no_sib"].idxmax()]
winner_top1_v18 = top1_v18["is_win"].mean()
mean_p18 = p_v18.mean()
max_p18 = p_v18.max()
print(f"V18 winner_top1: {winner_top1_v18:.4f} ({winner_top1_v18*100:.2f}%)")
print(f"V18 mean p18: {mean_p18:.4f}, max p18: {max_p18:.4f}")

m_v18.save_model(f"{OUT_DIR}/v18_lgb_no_sib_v1.txt")
v18_oos.to_csv(f"{OUT_DIR}/v18_no_sib_oos_2025.csv", index=False, encoding="utf-8-sig")
print(f"[SAVED] {OUT_DIR}/v18_lgb_no_sib_v1.txt + OOS")

results["v18_no_sib"] = {
    "auc": float(auc_v18),
    "logloss": float(ll_v18),
    "winner_top1": float(winner_top1_v18),
    "mean_p18": float(mean_p18),
    "max_p18": float(max_p18),
    "feature_count": len(all_avail),
}

# === V19 (FUKUSHO, is_top3) ===
print("\n" + "=" * 60)
print("V19 FUKUSHO (is_top3) sib抜き 学習")
print("=" * 60)
target_col = "is_top3"
y_tr = df.loc[train_mask, target_col].astype(int)
y_te = df.loc[test_mask, target_col].astype(int)
print(f"Positive rate: {y_tr.mean():.4f}")

ts = time.time()
m_v19 = lgb.train(
    LGB_PARAMS,
    lgb.Dataset(X_tr, y_tr),
    num_boost_round=2000,
    valid_sets=[lgb.Dataset(X_te, y_te)],
    callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)],
)
p_v19 = m_v19.predict(X_te)
auc_v19 = roc_auc_score(y_te, p_v19)
ll_v19 = log_loss(y_te, np.clip(p_v19, 1e-7, 1 - 1e-7))
print(f"V19 LGB no-sib: AUC={auc_v19:.4f} logloss={ll_v19:.4f} time={(time.time()-ts)/60:.1f}min")

v19_oos = pd.DataFrame({
    "race_id": race_id_te, "umaban": horse_num_te, "year": df.loc[test_mask, "_y"].values,
    "is_top3": y_te.values, "p_v19_no_sib": p_v19,
})
mean_p19 = p_v19.mean()
max_p19 = p_v19.max()

m_v19.save_model(f"{OUT_DIR}/v19_lgb_no_sib_v1.txt")
v19_oos.to_csv(f"{OUT_DIR}/v19_no_sib_oos_2025.csv", index=False, encoding="utf-8-sig")
print(f"[SAVED] {OUT_DIR}/v19_lgb_no_sib_v1.txt + OOS")

results["v19_no_sib"] = {
    "auc": float(auc_v19),
    "logloss": float(ll_v19),
    "mean_p19": float(mean_p19),
    "max_p19": float(max_p19),
    "feature_count": len(all_avail),
}

# === 既存 V18 / V19 (sib含む) との比較 ===
print("\n" + "=" * 60)
print("既存 V18/V19 (sib 含む) との比較")
print("=" * 60)

# 既存 OOS 2025
existing_v18_oos_path = "data/v18/v18_tansho_oos_2025.csv"
existing_v19_oos_path = "data/v18/v19_fukusho_oos_2025.csv"

if os.path.exists(existing_v18_oos_path):
    ex_v18 = pd.read_csv(existing_v18_oos_path)
    ex_auc_v18 = roc_auc_score(ex_v18["is_win"], ex_v18["p_ens"])
    ex_top1 = ex_v18.loc[ex_v18.groupby("race_id")["p_ens"].idxmax()]
    ex_winner_top1_v18 = ex_top1["is_win"].mean()
    print(f"V18 既存 (LGB+XGB ens, 含 sib): AUC={ex_auc_v18:.4f} winner_top1={ex_winner_top1_v18:.4f}")
    print(f"V18 sib抜き (LGB only):           AUC={auc_v18:.4f} winner_top1={winner_top1_v18:.4f}")
    print(f"  Δ AUC: {(auc_v18-ex_auc_v18):+.4f}")
    print(f"  Δ winner_top1: {(winner_top1_v18-ex_winner_top1_v18):+.4f}")
    results["v18_compare"] = {
        "existing_auc_ens": float(ex_auc_v18),
        "existing_winner_top1": float(ex_winner_top1_v18),
        "delta_auc": float(auc_v18 - ex_auc_v18),
        "delta_winner_top1": float(winner_top1_v18 - ex_winner_top1_v18),
    }

if os.path.exists(existing_v19_oos_path):
    ex_v19 = pd.read_csv(existing_v19_oos_path)
    ex_auc_v19 = roc_auc_score(ex_v19["is_top3"], ex_v19["p_ens"])
    print(f"V19 既存 (LGB+XGB ens, 含 sib): AUC={ex_auc_v19:.4f}")
    print(f"V19 sib抜き (LGB only):           AUC={auc_v19:.4f}")
    print(f"  Δ AUC: {(auc_v19-ex_auc_v19):+.4f}")
    results["v19_compare"] = {
        "existing_auc_ens": float(ex_auc_v19),
        "delta_auc": float(auc_v19 - ex_auc_v19),
    }

# === 結果保存 ===
total_min = (time.time() - start_total) / 60
results["meta"] = {
    "timestamp": str(datetime.now()),
    "session": "37_A",
    "purpose": "V18/V19 sib抜き 単一-fold 再学習",
    "fold": "train 2015-2024, test 2025",
    "leak_excluded_count": len(ALL_LEAK),
    "sib_excluded": SIB_LEAK_SUSPECT,
    "feature_count": len(all_avail),
    "elapsed_min": float(total_min),
    "n_train": n_tr,
    "n_test": n_te,
}

with open(f"{OUT_DIR}/no_sib_metrics.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, default=str)
print(f"\n[OK] {OUT_DIR}/no_sib_metrics.json saved")
print(f"Total time: {total_min:.0f}min")
print("=" * 60)
