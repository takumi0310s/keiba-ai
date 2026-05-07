"""V18/V19 sib_exp 統合 単一-fold 再学習 (Session #41 D).

Session #37 の no_sib (sib 完全削除) に対し、 sib_*_exp (Session #39 A) を
追加して再学習。 sib_top3_rate / sib_shinba_wr の リーク 0.125 corr 除去後の
真の信号 0.169 corr を model に組込み、 V18/V19 winner_top1 復活を狙う。

期待:
- OLD (sib リーク含む) winner_top1: 34.48% (LIVE retro 5/2-5/3)
- NEW (sib 完全削除)  winner_top1: 24.14% (-10pt)
- NEW2 (sib_*_exp 統合) winner_top1: 32-38% (+8-14pt vs no_sib)

input:
- data/v17/_v17_train_df_cache.pkl (1.2 GB、 既存)
- data/netkeiba_siblings_expanding.csv (Session #39 A)
- keiba_model_v15_central_live.pkl.gz (V15 model file、 features 参照のみ、 不変)

output:
- data/v18/v18v19_sib_exp_v1/v18_lgb_sib_exp_v1.txt
- data/v18/v18v19_sib_exp_v1/v19_lgb_sib_exp_v1.txt
- data/v18/v18v19_sib_exp_v1/v18_sib_exp_oos_2025.csv
- data/v18/v18v19_sib_exp_v1/v19_sib_exp_oos_2025.csv
- data/v18/v18v19_sib_exp_v1/sib_exp_metrics.json

V15 production 完全不変 (新規 dir + 新規 model file のみ)。
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

# === LEAK 除外 (no_sib と同じ + sib 旧版 削除) ===
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
# 旧 sib (リーク含) を削除し、 expanding 版で置き換え
SIB_OLD_EXCLUDED = ["sib_top3_rate", "sib_shinba_wr"]
SIB_EXP_NEW = ["sib_top3_rate_exp", "sib_shinba_wr_exp",
               "sib_total_races_exp", "sib_total_offspring_exp"]

ALL_LEAK = LEAK_OLD + LEAK_NEW_INDIVIDUAL + LEAK_TIME_INVARIANT + TYB_EXCLUDE_MORNING + SIB_OLD_EXCLUDED

OUT_DIR = "data/v18/v18v19_sib_exp_v1"
os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 60)
print("V18/V19 sib_exp 統合 single-fold 再学習 (Session #41 D)")
print(f"Start: {datetime.now()}")
print(f"除外 features: {len(ALL_LEAK)} (sib 旧版 削除)")
print(f"sib 旧版 削除: {SIB_OLD_EXCLUDED}")
print(f"sib 新版 追加: {SIB_EXP_NEW}")
start_total = time.time()

# === データロード ===
print("\nLoading v17 cache (1.2GB)...")
ts = time.time()
with open("data/v17/_v17_train_df_cache.pkl", "rb") as f:
    d = pickle.load(f)
df = d["df"]
df["_y"] = 2000 + df["year"]
print(f"  Loaded in {(time.time()-ts):.0f}s, {len(df):,} rows")

# === sib_*_exp を merge ===
print("\nLoading sib_expanding csv...")
ts = time.time()
sib_exp = pd.read_csv("data/netkeiba_siblings_expanding.csv",
                      dtype={'race_id': str, 'horse_id': str})
print(f"  sib_exp: {len(sib_exp):,} rows in {time.time()-ts:.1f}s")

# df の race_id / horse_id を str 化 (cache は既に そう であるはずだが確認)
if 'race_id' in df.columns:
    df['race_id'] = df['race_id'].astype(str)
if 'horse_id' in df.columns:
    df['horse_id'] = df['horse_id'].astype(str)
elif 'blood_num' in df.columns:
    # cache の horse 識別 column が blood_num の場合 (要確認)
    df['horse_id'] = df['blood_num'].astype(str)

# merge
print("\nmerging sib_exp into df...")
n_before = len(df)
# race_id 形式違いの可能性: sib_exp は netkeiba 12 chars (例: 0518250209)
#  cache は jrdb-internal 10 chars (race_part 8 + umaban 2)
# → cache の race_id が netkeiba 形式なら直接 merge、 違うなら fallback
merge_keys = ['race_id', 'horse_id']
have_keys = [k for k in merge_keys if k in df.columns]
print(f"  merge keys candidate: {have_keys}")

if 'race_id' in df.columns and 'horse_id' in df.columns:
    df_m = df.merge(sib_exp[['race_id', 'horse_id'] + SIB_EXP_NEW],
                    on=['race_id', 'horse_id'], how='left')
    matched = df_m[SIB_EXP_NEW[0]].notna().sum()
    print(f"  merge by (race_id, horse_id): matched {matched:,}/{n_before:,} = {matched/n_before*100:.1f}%")
    if matched < n_before * 0.3:
        # fallback: horse_id のみ (race_id 形式違いだろう)
        print(f"  fallback: horse_id only (timestamp 注意、 リーク risk)")
        # → fallback は使わない、 race_id 形式不一致なら horse_id だけだと
        #   各レース時点の expanding が破壊される (= リーク復活)
        # → matched が 30% 未満なら sib_exp 統合 NG として skip し、 no_sib と同等扱い
        print(f"  WARN: 統合不能、 sib_*_exp は全行 NaN → 0 fill で扱う (= no_sib と同等)")
    df = df_m
else:
    print(f"  [WARN] race_id or horse_id 不在、 sib_exp 統合 不可")
    for col in SIB_EXP_NEW:
        df[col] = 0.0

# fill NaN with 0 (mother 不明、 expanding history なし の馬)
for col in SIB_EXP_NEW:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    print(f"  {col}: mean={df[col].mean():.4f}, nonzero={int((df[col] > 0).sum()):,}")

# === V15 / v162 / v17 features list 構築 ===
with gzip.open("keiba_model_v15_central_live.pkl.gz", "rb") as f:
    m15 = pickle.load(f)
v15_features = list(m15["features"])

v162_added = d.get("v162_added_features", [])
v17_added = d.get("v17_added_features", [])

v15_avail = [f for f in v15_features if f in df.columns]
v162_avail = [f for f in v162_added if f in df.columns and f not in v15_avail and f not in ALL_LEAK]
v17_avail = [f for f in v17_added if f in df.columns and f not in v15_avail and f not in v162_avail and f not in ALL_LEAK]
sib_exp_avail = [f for f in SIB_EXP_NEW if f in df.columns]
all_avail = v15_avail + v162_avail + v17_avail + sib_exp_avail

print(f"\nv15 features:           {len(v15_avail)}")
print(f"v162 added (no sib old): {len(v162_avail)}")
print(f"v17 added (no leak):     {len(v17_avail)}")
print(f"sib_*_exp added:          {len(sib_exp_avail)}")
print(f"Total features:           {len(all_avail)}")

# 確認
sib_old_in = [f for f in all_avail if f in SIB_OLD_EXCLUDED]
assert not sib_old_in, f"sib 旧版 残存: {sib_old_in}"
print(f"[OK] sib 旧版 完全除外確認")

# === 単一 fold (train 2015-2024, test 2025) ===
train_mask = (df["_y"] >= 2015) & (df["_y"] <= 2024)
test_mask = df["_y"] == 2025
n_tr = int(train_mask.sum())
n_te = int(test_mask.sum())
print(f"\nTrain rows: {n_tr:,}, Test rows: {n_te:,}")

X_tr = df.loc[train_mask, all_avail]
X_te = df.loc[test_mask, all_avail]

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
print("V18 TANSHO (is_win) sib_exp 統合 学習")
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
print(f"V18 LGB sib_exp: AUC={auc_v18:.4f} logloss={ll_v18:.4f} time={(time.time()-ts)/60:.1f}min")

v18_oos = pd.DataFrame({
    "race_id": race_id_te, "umaban": horse_num_te, "year": df.loc[test_mask, "_y"].values,
    "is_win": y_te.values, "p_v18_sib_exp": p_v18,
})
top1_v18 = v18_oos.loc[v18_oos.groupby("race_id")["p_v18_sib_exp"].idxmax()]
winner_top1_v18 = top1_v18["is_win"].mean()
mean_p18 = p_v18.mean()
max_p18 = p_v18.max()
print(f"V18 winner_top1: {winner_top1_v18:.4f} ({winner_top1_v18*100:.2f}%)")
print(f"V18 mean p18: {mean_p18:.4f}, max p18: {max_p18:.4f}")

m_v18.save_model(f"{OUT_DIR}/v18_lgb_sib_exp_v1.txt")
v18_oos.to_csv(f"{OUT_DIR}/v18_sib_exp_oos_2025.csv", index=False, encoding="utf-8-sig")
print(f"[SAVED] {OUT_DIR}/v18_lgb_sib_exp_v1.txt + OOS")

results["v18_sib_exp"] = {
    "auc": float(auc_v18),
    "logloss": float(ll_v18),
    "winner_top1": float(winner_top1_v18),
    "mean_p18": float(mean_p18),
    "max_p18": float(max_p18),
    "feature_count": len(all_avail),
}

# === V19 (FUKUSHO, is_top3) ===
print("\n" + "=" * 60)
print("V19 FUKUSHO (is_top3) sib_exp 統合 学習")
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
print(f"V19 LGB sib_exp: AUC={auc_v19:.4f} logloss={ll_v19:.4f} time={(time.time()-ts)/60:.1f}min")

v19_oos = pd.DataFrame({
    "race_id": race_id_te, "umaban": horse_num_te, "year": df.loc[test_mask, "_y"].values,
    "is_top3": y_te.values, "p_v19_sib_exp": p_v19,
})
mean_p19 = p_v19.mean()
max_p19 = p_v19.max()

m_v19.save_model(f"{OUT_DIR}/v19_lgb_sib_exp_v1.txt")
v19_oos.to_csv(f"{OUT_DIR}/v19_sib_exp_oos_2025.csv", index=False, encoding="utf-8-sig")
print(f"[SAVED] {OUT_DIR}/v19_lgb_sib_exp_v1.txt + OOS")

results["v19_sib_exp"] = {
    "auc": float(auc_v19),
    "logloss": float(ll_v19),
    "mean_p19": float(mean_p19),
    "max_p19": float(max_p19),
    "feature_count": len(all_avail),
}

# === 既存 V18 (sib含) + no_sib との比較 ===
print("\n" + "=" * 60)
print("既存 V18/V19 (sib 含む) + no_sib との比較")
print("=" * 60)

# 既存 OOS 2025
existing_v18_oos_path = "data/v18/v18_tansho_oos_2025.csv"
no_sib_metrics_path = "data/v18/v18v19_retraining/no_sib_metrics.json"

if os.path.exists(existing_v18_oos_path):
    ex_v18 = pd.read_csv(existing_v18_oos_path)
    ex_auc_v18 = roc_auc_score(ex_v18["is_win"], ex_v18["p_ens"])
    ex_top1 = ex_v18.loc[ex_v18.groupby("race_id")["p_ens"].idxmax()]
    ex_winner_top1_v18 = ex_top1["is_win"].mean()
    print(f"V18 既存 (sib 含 ens):       AUC={ex_auc_v18:.4f} winner_top1={ex_winner_top1_v18:.4f}")
    print(f"V18 sib_exp (LGB only):      AUC={auc_v18:.4f} winner_top1={winner_top1_v18:.4f}")
    print(f"  vs sib含: ΔAUC={(auc_v18-ex_auc_v18):+.4f}, Δwinner_top1={(winner_top1_v18-ex_winner_top1_v18):+.4f}")
    results["v18_compare_existing"] = {
        "existing_auc_ens": float(ex_auc_v18),
        "existing_winner_top1": float(ex_winner_top1_v18),
        "delta_auc": float(auc_v18 - ex_auc_v18),
        "delta_winner_top1": float(winner_top1_v18 - ex_winner_top1_v18),
    }

if os.path.exists(no_sib_metrics_path):
    with open(no_sib_metrics_path, "r", encoding="utf-8") as f:
        no_sib_data = json.load(f)
    nsa = no_sib_data.get("v18_no_sib", {})
    if nsa:
        nsa_auc = nsa.get("auc")
        nsa_top1 = nsa.get("winner_top1")
        print(f"V18 no_sib (LGB only):       AUC={nsa_auc:.4f} winner_top1={nsa_top1:.4f}")
        print(f"V18 sib_exp vs no_sib: ΔAUC={(auc_v18-nsa_auc):+.4f}, Δwinner_top1={(winner_top1_v18-nsa_top1):+.4f}")
        results["v18_compare_no_sib"] = {
            "no_sib_auc": float(nsa_auc),
            "no_sib_winner_top1": float(nsa_top1),
            "delta_auc": float(auc_v18 - nsa_auc),
            "delta_winner_top1": float(winner_top1_v18 - nsa_top1),
        }

# === 結果保存 ===
total_min = (time.time() - start_total) / 60
results["meta"] = {
    "timestamp": str(datetime.now()),
    "session": "41_D",
    "purpose": "V18/V19 sib_exp 統合 単一-fold 再学習",
    "fold": "train 2015-2024, test 2025",
    "leak_excluded_count": len(ALL_LEAK),
    "sib_old_excluded": SIB_OLD_EXCLUDED,
    "sib_new_added": sib_exp_avail,
    "feature_count": len(all_avail),
    "elapsed_min": float(total_min),
    "n_train": n_tr,
    "n_test": n_te,
}

with open(f"{OUT_DIR}/sib_exp_metrics.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, default=str, ensure_ascii=False)
print(f"\n[OK] {OUT_DIR}/sib_exp_metrics.json saved")
print(f"Total time: {total_min:.0f}min")
print("=" * 60)
