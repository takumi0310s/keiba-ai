"""V18/V19 sib_exp w5 統合 単一-fold 再学習 (Session #43 C).

Session #41 D (full expanding) → Session #42 F (window=5 corr +0.032) → 本 Session
window=5 採用版 で再学習。

期待:
- BT 2025 winner_top1 (race_part 修正後): 45.88% → 46.0-46.5% (+0.1-0.6pt)
- LIVE 5/2-5/3 winner_top1: 31.03% → 32-34% (+1-3pt)
- shift_factor: 1.48x → 1.45x (微改善)

input:
- data/v17/_v17_train_df_cache.pkl (1.2 GB、 既存)
- data/netkeiba_siblings_expanding_w5.csv (Session #42 F 出力)
- keiba_model_v15_central_live.pkl.gz (V15 features 参照のみ、 不変)

output:
- data/v18/v18v19_sib_exp_w5/v18_lgb_sib_exp_w5.txt
- data/v18/v18v19_sib_exp_w5/v19_lgb_sib_exp_w5.txt
- data/v18/v18v19_sib_exp_w5/v18_sib_exp_w5_oos_2025.csv
- data/v18/v18v19_sib_exp_w5/v19_sib_exp_w5_oos_2025.csv
- data/v18/v18v19_sib_exp_w5/sib_exp_w5_metrics.json

V15 production 完全不変 (新規 dir + 新規 model file)。
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

# === LEAK 除外 (Session #41 D と同) ===
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
SIB_OLD_EXCLUDED = ["sib_top3_rate", "sib_shinba_wr"]

# Session #41 D の sib_*_exp 4 features → Session #43 C の w5 版 2 features に変更
SIB_W5_NEW = ["sib_top3_rate_exp_w5", "sib_shinba_wr_exp_w5"]

ALL_LEAK = LEAK_OLD + LEAK_NEW_INDIVIDUAL + LEAK_TIME_INVARIANT + TYB_EXCLUDE_MORNING + SIB_OLD_EXCLUDED

OUT_DIR = "data/v18/v18v19_sib_exp_w5"
os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 60)
print("V18/V19 sib_exp w5 統合 single-fold 再学習 (Session #43 C)")
print(f"Start: {datetime.now()}")
print(f"sib 旧版 削除: {SIB_OLD_EXCLUDED}")
print(f"sib w5 追加: {SIB_W5_NEW}")
start_total = time.time()

# === データロード ===
print("\nLoading v17 cache (1.2GB)...")
ts = time.time()
with open("data/v17/_v17_train_df_cache.pkl", "rb") as f:
    d = pickle.load(f)
df = d["df"]
df["_y"] = 2000 + df["year"]
print(f"  Loaded in {(time.time()-ts):.0f}s, {len(df):,} rows")

# === sib w5 を merge ===
print("\nLoading sib_expanding_w5 csv...")
ts = time.time()
sib_w5 = pd.read_csv("data/netkeiba_siblings_expanding_w5.csv",
                     dtype={'race_id': str, 'horse_id': str})
print(f"  sib_w5: {len(sib_w5):,} rows in {time.time()-ts:.1f}s")

if 'race_id' in df.columns:
    df['race_id'] = df['race_id'].astype(str)
if 'horse_id' in df.columns:
    df['horse_id'] = df['horse_id'].astype(str)

n_before = len(df)
df = df.merge(sib_w5[['race_id', 'horse_id'] + SIB_W5_NEW],
              on=['race_id', 'horse_id'], how='left')
matched = df[SIB_W5_NEW[0]].notna().sum()
print(f"  merge: matched {matched:,}/{n_before:,} = {matched/n_before*100:.1f}%")

# fill NaN with 0
for col in SIB_W5_NEW:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    print(f"  {col}: mean={df[col].mean():.4f}, nonzero={int((df[col] > 0).sum()):,}")

# === V15 / v162 / v17 features list ===
with gzip.open("keiba_model_v15_central_live.pkl.gz", "rb") as f:
    m15 = pickle.load(f)
v15_features = list(m15["features"])

v162_added = d.get("v162_added_features", [])
v17_added = d.get("v17_added_features", [])

v15_avail = [f for f in v15_features if f in df.columns]
v162_avail = [f for f in v162_added if f in df.columns and f not in v15_avail and f not in ALL_LEAK]
v17_avail = [f for f in v17_added if f in df.columns and f not in v15_avail and f not in v162_avail and f not in ALL_LEAK]
sib_w5_avail = [f for f in SIB_W5_NEW if f in df.columns]
all_avail = v15_avail + v162_avail + v17_avail + sib_w5_avail

print(f"\nv15: {len(v15_avail)} / v162: {len(v162_avail)} / v17: {len(v17_avail)} / sib_w5: {len(sib_w5_avail)}")
print(f"Total features: {len(all_avail)}")

# 確認
sib_old_in = [f for f in all_avail if f in SIB_OLD_EXCLUDED]
assert not sib_old_in, f"sib 旧版 残存: {sib_old_in}"
print(f"[OK] sib 旧版 完全除外 + sib_w5 統合 確認")

# === 単一 fold ===
train_mask = (df["_y"] >= 2015) & (df["_y"] <= 2024)
test_mask = df["_y"] == 2025
n_tr = int(train_mask.sum())
n_te = int(test_mask.sum())
print(f"\nTrain: {n_tr:,}, Test: {n_te:,}")

X_tr = df.loc[train_mask, all_avail]
X_te = df.loc[test_mask, all_avail]

race_id_te = df.loc[test_mask, "race_id"].astype(str).values
horse_num_te = df.loc[test_mask, "horse_num"].values if "horse_num" in df.columns else np.arange(n_te)

LGB_PARAMS = {
    "objective": "binary", "metric": "auc",
    "learning_rate": 0.03, "num_leaves": 255, "min_data_in_leaf": 50,
    "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 5,
    "lambda_l1": 0.1, "lambda_l2": 0.1,
    "verbose": -1, "seed": 42,
}

results = {}

# === V18 ===
print("\n" + "=" * 60)
print("V18 TANSHO (is_win) sib_exp w5 学習")
print("=" * 60)
y_tr = df.loc[train_mask, "is_win"].astype(int)
y_te = df.loc[test_mask, "is_win"].astype(int)
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
ll_v18 = log_loss(y_te, np.clip(p_v18, 1e-7, 1-1e-7))
print(f"V18 LGB sib_w5: AUC={auc_v18:.4f}, logloss={ll_v18:.4f}, time={(time.time()-ts)/60:.1f}min")

# winner_top1 (race_part = race_id[:-2])
v18_oos = pd.DataFrame({
    "race_id": race_id_te, "umaban": horse_num_te, "year": df.loc[test_mask, "_y"].values,
    "is_win": y_te.values, "p_v18_w5": p_v18,
})
v18_oos['race_part'] = v18_oos['race_id'].astype(str).str[:-2]
top1 = v18_oos.loc[v18_oos.groupby('race_part')['p_v18_w5'].idxmax()]
winner_top1 = top1['is_win'].mean()
print(f"V18 winner_top1 (race_part): {winner_top1:.4f} ({winner_top1*100:.2f}%)")

m_v18.save_model(f"{OUT_DIR}/v18_lgb_sib_exp_w5.txt")
v18_oos.to_csv(f"{OUT_DIR}/v18_sib_exp_w5_oos_2025.csv", index=False, encoding="utf-8-sig")

results["v18_w5"] = {
    "auc": float(auc_v18),
    "logloss": float(ll_v18),
    "winner_top1": float(winner_top1),
    "feature_count": len(all_avail),
}

# === V19 ===
print("\n" + "=" * 60)
print("V19 FUKUSHO (is_top3) sib_exp w5 学習")
print("=" * 60)
y_tr = df.loc[train_mask, "is_top3"].astype(int)
y_te = df.loc[test_mask, "is_top3"].astype(int)
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
ll_v19 = log_loss(y_te, np.clip(p_v19, 1e-7, 1-1e-7))
print(f"V19 LGB sib_w5: AUC={auc_v19:.4f}, logloss={ll_v19:.4f}, time={(time.time()-ts)/60:.1f}min")

v19_oos = pd.DataFrame({
    "race_id": race_id_te, "umaban": horse_num_te, "year": df.loc[test_mask, "_y"].values,
    "is_top3": y_te.values, "p_v19_w5": p_v19,
})
m_v19.save_model(f"{OUT_DIR}/v19_lgb_sib_exp_w5.txt")
v19_oos.to_csv(f"{OUT_DIR}/v19_sib_exp_w5_oos_2025.csv", index=False, encoding="utf-8-sig")

results["v19_w5"] = {
    "auc": float(auc_v19),
    "logloss": float(ll_v19),
    "feature_count": len(all_avail),
}

# === 比較 (Session #41 D の sib_exp v1 と) ===
print("\n" + "=" * 60)
print("Session #41 D (sib_exp v1) との比較")
print("=" * 60)
sib_exp_v1_metrics_path = "data/v18/v18v19_sib_exp_v1/sib_exp_metrics.json"
if os.path.exists(sib_exp_v1_metrics_path):
    with open(sib_exp_v1_metrics_path, "r", encoding="utf-8") as f:
        v1 = json.load(f)
    v1_v18 = v1.get("v18_sib_exp", {})
    if v1_v18:
        v1_auc = v1_v18.get("auc")
        v1_top1 = v1_v18.get("winner_top1")  # 訂正前 race_id ベース (誤)
        print(f"V18 sib_exp v1 (Session #41 D): AUC={v1_auc:.4f}")
        print(f"V18 sib_exp w5 (本 Session):     AUC={auc_v18:.4f}")
        print(f"  ΔAUC: {auc_v18-v1_auc:+.4f}")
        results["v18_compare_v1"] = {
            "v1_auc": float(v1_auc),
            "delta_auc": float(auc_v18 - v1_auc),
        }

# === 結果保存 ===
total_min = (time.time() - start_total) / 60
results["meta"] = {
    "timestamp": str(datetime.now()),
    "session": "43_C",
    "purpose": "V18/V19 sib_exp w5 単一-fold 再学習",
    "fold": "train 2015-2024, test 2025",
    "leak_excluded_count": len(ALL_LEAK),
    "sib_old_excluded": SIB_OLD_EXCLUDED,
    "sib_w5_added": sib_w5_avail,
    "feature_count": len(all_avail),
    "elapsed_min": float(total_min),
    "n_train": n_tr,
    "n_test": n_te,
}

with open(f"{OUT_DIR}/sib_exp_w5_metrics.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, default=str, ensure_ascii=False)
print(f"\n[OK] {OUT_DIR}/sib_exp_w5_metrics.json saved")
print(f"Total time: {total_min:.0f}min")
print("=" * 60)
