"""Session #54 C: NAR V5 学習 + backtest.

V4 22 features + 15 新規 = 37 features
LGB + XGB ensemble (V4 と同様)
時系列 split: 80% train / 20% test

Usage:
    python tools/train_nar_v5.py
"""

from __future__ import annotations

import json
import pickle
import warnings
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
SCRAPED_CSV = ROOT / "data" / "nar_all_races.csv"
V5_MODEL = ROOT / "data" / "nar" / "models" / "keiba_model_nar_v5.pkl"
V5_RESULTS = ROOT / "data" / "v18" / "session_54_nar_v5_metrics.json"

# Place codes (rakuten NAR)
COURSE_MAP = {
    "札幌": 0, "盛岡": 1, "水沢": 2, "浦和": 10, "船橋": 11, "大井": 12, "川崎": 13,
    "金沢": 20, "笠松": 21, "名古屋": 22, "園田": 30, "姫路": 31, "高知": 22,
    "佐賀": 23, "帯広": 24,
}
SURFACE_MAP = {"芝": 0, "ダ": 1, "ダート": 1, "障": 2}
COND_MAP = {"良": 0, "稍": 1, "稍重": 1, "重": 2, "不": 3, "不良": 3}
SEX_MAP = {"牡": 0, "牝": 1, "セ": 2, "騸": 2}


# V4 22 features
NAR_V4_FEATURES = [
    "odds_log", "num_horses", "distance", "surface_enc", "condition_enc",
    "course_enc", "horse_weight", "weight_carry", "age", "sex_enc",
    "horse_num", "bracket", "horse_num_ratio", "bracket_pos",
    "carry_diff", "dist_cat", "weight_cat", "age_group",
    "jockey_wr", "jockey_place_rate", "pop_rank", "is_nar",
]

# V5 新規 15 features
NAR_V5_NEW = [
    # expanding 7
    "horse_dist_top3r", "horse_surface_top3r",
    "jockey_course_wr", "frame_course_dist_wr",
    "horse_career_races", "horse_career_wr", "horse_career_top3r",
    # 既取得 4
    "horse_weight_change", "horse_weight_change_abs",
    "last3f_filled", "trainer_wr",
    # NAR 独自 4
    "course_dist_wr", "weight_cat_dist",
    "nar_class_enc", "rest_days_filled",
]
NAR_V5_FEATURES = NAR_V4_FEATURES + NAR_V5_NEW


def load_and_prepare() -> pd.DataFrame:
    """Load nar_all_races.csv + V4 base preprocessing."""
    print(f"Loading {SCRAPED_CSV}", flush=True)
    df = pd.read_csv(SCRAPED_CSV, encoding="utf-8")
    print(f"  Loaded: {len(df)} rows, {df['race_id'].nunique()} races", flush=True)

    # Filter
    df["finish"] = pd.to_numeric(df["finish"], errors="coerce")
    df = df.dropna(subset=["finish"]).copy()
    df["finish"] = df["finish"].astype(int)
    df = df[df["finish"] > 0].copy()

    df["target"] = (df["finish"] <= 3).astype(int)

    # Sort by race_date for expanding (date_int)
    df["race_date_int"] = pd.to_numeric(df["race_date"], errors="coerce").fillna(0).astype(int)
    df = df.sort_values(["race_date_int", "race_id", "horse_num"]).reset_index(drop=True)

    # Parse sex/age
    df["sex"] = df["sex_age"].astype(str).str[0]
    df["age"] = pd.to_numeric(df["sex_age"].astype(str).str[1:], errors="coerce").fillna(4)

    # Numeric coerce
    df["odds"] = pd.to_numeric(df["odds"], errors="coerce").fillna(30)
    df["odds_log"] = np.log1p(df["odds"].clip(1, 999))
    df["num_horses"] = pd.to_numeric(df["num_horses"], errors="coerce").fillna(10)
    df["distance"] = pd.to_numeric(df["distance"], errors="coerce").fillna(1600)
    df["horse_weight"] = pd.to_numeric(df["horse_weight"], errors="coerce").fillna(470)
    df["horse_weight_change"] = pd.to_numeric(df["horse_weight_change"], errors="coerce").fillna(0)
    df["horse_weight_change_abs"] = df["horse_weight_change"].abs()
    df["weight_carry"] = pd.to_numeric(df["weight_carry"], errors="coerce").fillna(55)
    df["horse_num"] = pd.to_numeric(df["horse_num"], errors="coerce").fillna(5)
    df["bracket"] = pd.to_numeric(df["bracket"], errors="coerce").fillna(4)
    df["pop_rank"] = pd.to_numeric(df["pop_rank"], errors="coerce").fillna(5)
    df["last3f"] = pd.to_numeric(df["last3f"], errors="coerce")

    # Encodings
    df["surface_enc"] = df["surface"].map(SURFACE_MAP).fillna(1)
    df["condition_enc"] = df["condition"].astype(str).str[0].map(COND_MAP).fillna(0)
    df["course_enc"] = df["course"].map(COURSE_MAP).fillna(10)
    df["sex_enc"] = df["sex"].map(SEX_MAP).fillna(0)

    # Categorical
    df["dist_cat"] = pd.cut(df["distance"], bins=[0, 1200, 1400, 1800, 2200, 9999],
                             labels=[0, 1, 2, 3, 4]).astype(float).fillna(2)
    df["weight_cat"] = pd.cut(df["horse_weight"], bins=[0, 440, 480, 520, 9999],
                               labels=[0, 1, 2, 3]).astype(float).fillna(1)
    df["age_group"] = df["age"].clip(2, 7)
    df["horse_num_ratio"] = df["horse_num"] / df["num_horses"].clip(1)
    df["bracket_pos"] = pd.cut(df["bracket"], bins=[0, 3, 6, 8],
                                labels=[0, 1, 2]).astype(float).fillna(1)
    df["carry_diff"] = df["weight_carry"] - df["weight_carry"].mean()
    df["weight_cat_dist"] = df["weight_cat"] * 10 + df["dist_cat"]
    df["is_nar"] = 1

    # NAR class encoding (text → int)
    df["nar_class_enc"] = pd.Categorical(df["class_info"]).codes.astype(float)
    df.loc[df["nar_class_enc"] < 0, "nar_class_enc"] = -1

    return df


def compute_jockey_trainer_stats(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Compute jockey/trainer expanding stats (V4 same logic)."""
    print("Computing jockey/trainer expanding...", flush=True)
    jockey_runs, jockey_wins, jockey_top3 = {}, {}, {}
    trainer_runs, trainer_wins = {}, {}

    jwr_list, jpr_list, twr_list = [], [], []

    for j, t, fin in zip(df["jockey_name"], df["trainer_name"], df["finish"]):
        # jockey
        runs_j = jockey_runs.get(j, 0)
        wins_j = jockey_wins.get(j, 0)
        t3_j = jockey_top3.get(j, 0)
        jwr_list.append(wins_j / runs_j if runs_j >= 10 else 0.08)
        jpr_list.append(t3_j / runs_j if runs_j >= 10 else 0.25)
        jockey_runs[j] = runs_j + 1
        if fin == 1:
            jockey_wins[j] = wins_j + 1
        if fin <= 3:
            jockey_top3[j] = t3_j + 1
        # trainer
        runs_t = trainer_runs.get(t, 0)
        wins_t = trainer_wins.get(t, 0)
        twr_list.append(wins_t / runs_t if runs_t >= 10 else 0.08)
        trainer_runs[t] = runs_t + 1
        if fin == 1:
            trainer_wins[t] = wins_t + 1

    df["jockey_wr"] = jwr_list
    df["jockey_place_rate"] = jpr_list
    df["trainer_wr"] = twr_list

    jockey_stats = {}
    for j in jockey_runs:
        if jockey_runs[j] >= 5:
            jockey_stats[j] = {
                "wr": round(jockey_wins.get(j, 0) / jockey_runs[j], 4),
                "place_rate": round(jockey_top3.get(j, 0) / jockey_runs[j], 4),
            }
    return df, jockey_stats


def compute_horse_expanding(df: pd.DataFrame) -> pd.DataFrame:
    """Compute horse-level expanding statistics."""
    print("Computing horse expanding...", flush=True)
    df["is_win"] = (df["finish"] == 1).astype(int)
    df["is_top3"] = (df["finish"] <= 3).astype(int)
    global_wr = df["is_win"].mean()
    global_t3 = df["is_top3"].mean()
    alpha = 5

    # horse_id proxy = horse_name (NAR data has no horse_id)
    df["hid"] = df["horse_name"].astype(str)

    # horse_dist_top3r
    df["hd_r"] = df.groupby(["hid", "dist_cat"]).cumcount()
    df["hd_t3"] = df.groupby(["hid", "dist_cat"])["is_top3"].cumsum() - df["is_top3"]
    df["horse_dist_top3r"] = (df["hd_t3"] + alpha * global_t3) / (df["hd_r"] + alpha)

    # horse_surface_top3r
    df["hs_r"] = df.groupby(["hid", "surface_enc"]).cumcount()
    df["hs_t3"] = df.groupby(["hid", "surface_enc"])["is_top3"].cumsum() - df["is_top3"]
    df["horse_surface_top3r"] = (df["hs_t3"] + alpha * global_t3) / (df["hs_r"] + alpha)

    # jockey × course
    alpha_jc = 10
    df["jc_r"] = df.groupby(["jockey_name", "course_enc"]).cumcount()
    df["jc_w"] = df.groupby(["jockey_name", "course_enc"])["is_win"].cumsum() - df["is_win"]
    df["jockey_course_wr"] = (df["jc_w"] + alpha_jc * global_wr) / (df["jc_r"] + alpha_jc)

    # frame × course × dist
    alpha_frm = 50
    df["fk"] = df["course_enc"].astype(str) + "_" + df["dist_cat"].astype(str) + "_" + df["bracket"].astype(str)
    df["fr_r"] = df.groupby("fk").cumcount()
    df["fr_w"] = df.groupby("fk")["is_win"].cumsum() - df["is_win"]
    df["frame_course_dist_wr"] = (df["fr_w"] + alpha_frm * global_wr) / (df["fr_r"] + alpha_frm)

    # horse career
    df["hc_r"] = df.groupby("hid").cumcount()
    df["hc_w"] = df.groupby("hid")["is_win"].cumsum() - df["is_win"]
    df["hc_t3"] = df.groupby("hid")["is_top3"].cumsum() - df["is_top3"]
    df["horse_career_races"] = df["hc_r"]
    df["horse_career_wr"] = (df["hc_w"] + alpha * global_wr) / (df["hc_r"] + alpha)
    df["horse_career_top3r"] = (df["hc_t3"] + alpha * global_t3) / (df["hc_r"] + alpha)

    # course × dist 別 win rate
    alpha_cd = 30
    df["cd"] = df["course_enc"].astype(str) + "_" + df["dist_cat"].astype(str)
    df["cd_r"] = df.groupby("cd").cumcount()
    df["cd_w"] = df.groupby("cd")["is_win"].cumsum() - df["is_win"]
    df["course_dist_wr"] = (df["cd_w"] + alpha_cd * global_wr) / (df["cd_r"] + alpha_cd)

    # last3f は post-race。 horse 単位で shift(1) して 前走 last3f を 使う (leak-free)
    df["prev_last3f"] = df.groupby("hid")["last3f"].shift(1)
    last3f_mean = df["prev_last3f"].mean()
    df["last3f_filled"] = df["prev_last3f"].fillna(last3f_mean)

    # rest_days: race_date diff per horse (前回出走 から)
    df["race_date"] = pd.to_datetime(df["race_date"].astype(str), format="%Y%m%d", errors="coerce")
    df["prev_race_date"] = df.groupby("hid")["race_date"].shift(1)
    df["rest_days_raw"] = (df["race_date"] - df["prev_race_date"]).dt.days
    df["rest_days_filled"] = df["rest_days_raw"].fillna(30).clip(1, 365)

    # cleanup
    drop = [c for c in df.columns if any(c.startswith(p) for p in [
        "hd_", "hs_", "jc_", "fr_", "hc_", "cd_", "fk", "cd"
    ]) and c not in NAR_V5_FEATURES + ["course_dist_wr"]]
    drop += ["is_win", "is_top3", "hid", "prev_race_date", "rest_days_raw", "prev_last3f"]
    df = df.drop(columns=[c for c in drop if c in df.columns], errors="ignore")
    return df


def train_lgb_xgb(df: pd.DataFrame, features: list[str], label: str) -> dict:
    """Train LGB+XGB ensemble + time-based split."""
    print(f"\n{'='*60}\n  {label}\n  Features: {len(features)}\n{'='*60}")

    for f in features:
        if f not in df.columns:
            df[f] = 0
        df[f] = pd.to_numeric(df[f], errors="coerce").fillna(0)

    X = df[features].values
    y = df["target"].values

    # Time-based 80/20 split
    n = len(df)
    split = int(n * 0.8)
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]
    print(f"  Train: {len(X_tr)}, Test: {len(X_te)}")
    print(f"  Target rate: train={y_tr.mean():.3f}, test={y_te.mean():.3f}")

    # LGB
    params_lgb = {
        "objective": "binary", "metric": "auc", "boosting_type": "gbdt",
        "num_leaves": 31, "learning_rate": 0.04, "feature_fraction": 0.8,
        "bagging_fraction": 0.8, "bagging_freq": 5, "min_child_samples": 20,
        "reg_alpha": 0.3, "reg_lambda": 0.3, "verbose": -1,
        "n_jobs": -1, "seed": 42,
    }
    dtrain = lgb.Dataset(X_tr, label=y_tr, feature_name=features)
    dtest = lgb.Dataset(X_te, label=y_te, feature_name=features, reference=dtrain)
    lgb_model = lgb.train(
        params_lgb, dtrain, num_boost_round=2000,
        valid_sets=[dtest],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
    )
    lgb_pred = lgb_model.predict(X_te)
    lgb_auc = roc_auc_score(y_te, lgb_pred)
    print(f"  LightGBM AUC: {lgb_auc:.4f}")

    # XGB
    import xgboost as xgb_lib
    dtrain_xgb = xgb_lib.DMatrix(X_tr, label=y_tr)
    dtest_xgb = xgb_lib.DMatrix(X_te, label=y_te)
    xgb_params = {
        "objective": "binary:logistic", "eval_metric": "auc",
        "max_depth": 5, "learning_rate": 0.03, "subsample": 0.8,
        "colsample_bytree": 0.8, "min_child_weight": 15,
        "reg_alpha": 0.3, "reg_lambda": 0.3, "seed": 42,
        "tree_method": "hist", "verbosity": 0,
    }
    xgb_model = xgb_lib.train(
        xgb_params, dtrain_xgb, num_boost_round=2000,
        evals=[(dtest_xgb, "valid")],
        early_stopping_rounds=50, verbose_eval=0,
    )
    xgb_pred = xgb_model.predict(dtest_xgb)
    xgb_auc = roc_auc_score(y_te, xgb_pred)
    print(f"  XGBoost AUC: {xgb_auc:.4f}")

    total = lgb_auc + xgb_auc
    w_lgb = lgb_auc / total
    w_xgb = xgb_auc / total
    ens_pred = lgb_pred * w_lgb + xgb_pred * w_xgb
    ens_auc = roc_auc_score(y_te, ens_pred)
    print(f"  Ensemble AUC: {ens_auc:.4f} (LGB w={w_lgb:.3f}, XGB w={w_xgb:.3f})")

    # Feature importance
    importance = lgb_model.feature_importance(importance_type="gain")
    fi = sorted(zip(features, importance), key=lambda x: x[1], reverse=True)
    print(f"\n  Top 15 importance:")
    for name, imp in fi[:15]:
        print(f"    {name:28s} {imp:10.1f}")

    return {
        "lgb_model": lgb_model,
        "xgb_model": xgb_model,
        "lgb_auc": lgb_auc,
        "xgb_auc": xgb_auc,
        "ens_auc": ens_auc,
        "w_lgb": w_lgb,
        "w_xgb": w_xgb,
        "feature_importance": fi[:30],
    }


def main() -> None:
    df = load_and_prepare()
    df, jockey_stats = compute_jockey_trainer_stats(df)
    df = compute_horse_expanding(df)

    print(f"\nTotal rows after preprocessing: {len(df)}")
    print(f"Unique races: {df['race_id'].nunique()}")
    print(f"Date range: {df['race_date_int'].min()} ~ {df['race_date_int'].max()}")

    # V4 baseline (再現)
    res_v4 = train_lgb_xgb(df, NAR_V4_FEATURES, "V4 baseline (22 features)")

    # V5 (37 features)
    res_v5 = train_lgb_xgb(df, NAR_V5_FEATURES, "V5 candidate (37 features)")

    delta = res_v5["ens_auc"] - res_v4["ens_auc"]
    print(f"\n{'='*60}\n  V4 → V5 AUC delta: {delta:+.5f}\n{'='*60}")

    # Save V5 model
    model_data = {
        "version": "nar_v5",
        "model": res_v5["lgb_model"],
        "xgb_model": res_v5["xgb_model"],
        "features": NAR_V5_FEATURES,
        "ensemble_weights": {"lgb": res_v5["w_lgb"], "xgb": res_v5["w_xgb"]},
        "jockey_stats": jockey_stats,
        "auc": res_v5["ens_auc"],
        "lgb_auc": res_v5["lgb_auc"],
        "xgb_auc": res_v5["xgb_auc"],
        "n_races": int(df["race_id"].nunique()),
        "n_rows": int(len(df)),
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "v4_baseline_auc": res_v4["ens_auc"],
        "delta_vs_v4": delta,
    }
    V5_MODEL.parent.mkdir(parents=True, exist_ok=True)
    with open(V5_MODEL, "wb") as f:
        pickle.dump(model_data, f)
    print(f"\n[OK] Model saved: {V5_MODEL}")

    # Save metrics JSON
    metrics = {
        "v4": {
            "n_features": len(NAR_V4_FEATURES),
            "lgb_auc": res_v4["lgb_auc"],
            "xgb_auc": res_v4["xgb_auc"],
            "ens_auc": res_v4["ens_auc"],
        },
        "v5": {
            "n_features": len(NAR_V5_FEATURES),
            "lgb_auc": res_v5["lgb_auc"],
            "xgb_auc": res_v5["xgb_auc"],
            "ens_auc": res_v5["ens_auc"],
            "ens_w_lgb": res_v5["w_lgb"],
            "ens_w_xgb": res_v5["w_xgb"],
            "feature_importance_top30": [(n, float(i)) for n, i in res_v5["feature_importance"][:30]],
        },
        "delta": delta,
        "n_rows": int(len(df)),
        "n_races": int(df["race_id"].nunique()),
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    V5_RESULTS.parent.mkdir(parents=True, exist_ok=True)
    V5_RESULTS.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[OK] Metrics saved: {V5_RESULTS}")


if __name__ == "__main__":
    main()
