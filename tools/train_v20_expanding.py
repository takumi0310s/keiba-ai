"""V20 base + expanding features LGB 学習 (Session #55 C).

baseline (V20 base feature set) と V20 + expanding を比較。
expanding 6 features の AUC contribution を計測。

cross-validation: 年度別 3-fold (2023, 2024, 2025 を valid)
LGB single model (FT/IR は Session #56+ で評価)

usage:
    python tools/train_v20_expanding.py
    -> data/v20/models/v20_expanding_v1.pkl
       data/v18/session_55_v20_expanding_training_raw.json
"""
from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SRC = DATA / "jra_races_full.csv"
EXP = DATA / "v20" / "expanding_features_v1.parquet"
MODEL_DIR = DATA / "v20" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUT_RESULT = DATA / "v18" / "session_55_v20_expanding_training_raw.json"

BASE_FEATURES = [
    "weight_carry",
    "age",
    "distance",
    "course_enc",
    "surface_enc",
    "sex_enc",
    "num_horses",
    "umaban",
    "horse_num_ratio",
    "prev_finish",
    "prev_pass4",
    "rest_days",
    "horse_career_wr_lifetime",
    "jockey_wr_lifetime",
    "trainer_top3_lifetime",
    "popularity",
]
EXPANDING_FEATURES = [
    "jockey_wr_w30",
    "jockey_top3_w30",
    "trainer_top3_w90",
    "horse_career_top3_w5",
    "horse_career_wr_w5",
    "horse_career_top3_w10",
]

LGB_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "num_leaves": 63,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_child_samples": 50,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "verbose": -1,
    "seed": 42,
}


def _norm_year4(y: int) -> int:
    return 2000 + y if y < 80 else 1900 + y


def _to_int_safe(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(-1).astype(int)


def _build_dataset() -> pd.DataFrame:
    print("[C] read jra_races_full")
    cols = [
        "year",
        "month",
        "day",
        "race_id",
        "horse_id",
        "jockey_id",
        "trainer_id",
        "umaban",
        "weight_carry",
        "age",
        "distance",
        "course",
        "surface",
        "sex",
        "num_horses",
        "finish",
        "abnormal_code",
        "popularity",
    ]
    df = pd.read_csv(SRC, usecols=cols, low_memory=False)
    df["year_4digit"] = df["year"].apply(_norm_year4)
    df = df[(df["year_4digit"] >= 2018) & (df["year_4digit"] <= 2025)].copy()
    df["finish_num"] = pd.to_numeric(df["finish"], errors="coerce")
    valid = (df["abnormal_code"].fillna(0) == 0) & df["finish_num"].notna()
    df = df.loc[valid].reset_index(drop=True)
    df["target"] = (df["finish_num"].between(1, 3)).astype(int)
    print(f"[C] base rows (2018-2025, valid): {len(df):,}")

    # encode (簡易ルールベース)
    df["course_enc"] = _to_int_safe(df["course"])
    df["surface_enc"] = df["surface"].map({"芝": 0, "ダ": 1, "障": 2}).fillna(-1).astype(int)
    df["sex_enc"] = df["sex"].map({"牡": 0, "牝": 1, "セ": 2}).fillna(-1).astype(int)
    df["weight_carry"] = pd.to_numeric(df["weight_carry"], errors="coerce")
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["distance"] = pd.to_numeric(df["distance"], errors="coerce")
    df["num_horses"] = pd.to_numeric(df["num_horses"], errors="coerce")
    df["umaban"] = pd.to_numeric(df["umaban"], errors="coerce")
    df["popularity"] = pd.to_numeric(df["popularity"], errors="coerce").fillna(99)
    df["horse_num_ratio"] = df["umaban"] / df["num_horses"].replace(0, np.nan)

    # prev features (前走系)
    df = df.sort_values(["horse_id", "year_4digit", "month", "day", "race_id"]).reset_index(
        drop=True
    )
    g = df.groupby("horse_id", sort=False)
    df["prev_finish"] = g["finish_num"].shift(1)
    df["race_dt"] = pd.to_datetime(
        df["year_4digit"].astype(str)
        + "-"
        + df["month"].astype(str).str.zfill(2)
        + "-"
        + df["day"].astype(str).str.zfill(2),
        errors="coerce",
    )
    df["prev_dt"] = g["race_dt"].shift(1)
    df["rest_days"] = (df["race_dt"] - df["prev_dt"]).dt.days.clip(0, 365)
    df["prev_pass4"] = np.nan  # 簡易省略

    # lifetime expanding (V15 の jockey_wr_calc / horse_career 等を簡易に再現)
    df = df.sort_values(["horse_id", "race_dt", "race_id"]).reset_index(drop=True)
    h = df.groupby("horse_id", sort=False)
    horse_top3_cum = h["target"].shift(1).expanding().sum().reset_index(level=0, drop=True)
    horse_cnt_cum = h["target"].shift(1).expanding().count().reset_index(level=0, drop=True)
    df["horse_career_wr_lifetime"] = (horse_top3_cum + 1.0) / (horse_cnt_cum + 5.0)

    df = df.sort_values(["jockey_id", "race_dt", "race_id"]).reset_index(drop=True)
    j = df.groupby("jockey_id", sort=False)
    j_win = (df["finish_num"] == 1).astype(int)
    j_win_cum = j_win.groupby(df["jockey_id"]).shift(1).groupby(
        df["jockey_id"]
    ).cumsum().reset_index(level=0, drop=True)
    j_cnt_cum = j["target"].shift(1).expanding().count().reset_index(level=0, drop=True)
    df["jockey_wr_lifetime"] = (j_win_cum + 1.0) / (j_cnt_cum + 10.0)

    df = df.sort_values(["trainer_id", "race_dt", "race_id"]).reset_index(drop=True)
    t = df.groupby("trainer_id", sort=False)
    t_top3_cum = t["target"].shift(1).expanding().sum().reset_index(level=0, drop=True)
    t_cnt_cum = t["target"].shift(1).expanding().count().reset_index(level=0, drop=True)
    df["trainer_top3_lifetime"] = (t_top3_cum + 3.0) / (t_cnt_cum + 10.0)

    print("[C] read expanding parquet")
    exp = pd.read_parquet(EXP)
    exp = exp.drop(columns=[c for c in ("year_4digit",) if c in exp.columns])
    df = df.merge(exp, on=["race_id", "horse_id", "umaban"], how="left")

    print(f"[C] merged rows: {len(df):,}")
    print(
        f"[C] expanding feature coverage after merge:"
        f" jockey_top3_w30={df['jockey_top3_w30'].notna().mean()*100:.1f}%"
    )

    return df


def _train_eval(
    df: pd.DataFrame, features: list[str], valid_year: int, label: str
) -> tuple[float, lgb.Booster]:
    train_df = df[df["year_4digit"] < valid_year]
    valid_df = df[df["year_4digit"] == valid_year]
    X_tr = train_df[features].astype(float)
    y_tr = train_df["target"].astype(int)
    X_va = valid_df[features].astype(float)
    y_va = valid_df["target"].astype(int)

    dtrain = lgb.Dataset(X_tr, label=y_tr)
    dvalid = lgb.Dataset(X_va, label=y_va, reference=dtrain)
    booster = lgb.train(
        LGB_PARAMS,
        dtrain,
        num_boost_round=1000,
        valid_sets=[dvalid],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )
    pred = booster.predict(X_va, num_iteration=booster.best_iteration)
    auc = roc_auc_score(y_va, pred)
    print(f"  [{label}] valid={valid_year} AUC={auc:.4f} n_train={len(train_df):,} n_val={len(valid_df):,}")
    return auc, booster


def main() -> None:
    t0 = time.time()
    df = _build_dataset()

    # 訓練可能なベース features のみ抽出
    base_avail = [c for c in BASE_FEATURES if c in df.columns]
    full_avail = base_avail + EXPANDING_FEATURES
    print(f"[C] base features: {len(base_avail)} avail")
    print(f"[C] full features: {len(full_avail)} avail")

    # NaN 補正
    for c in base_avail + EXPANDING_FEATURES:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(df[c].median() if df[c].dtype.kind in "fc" else 0)

    valid_years = [2023, 2024, 2025]
    results = {"base": {}, "expanding": {}}

    print()
    print("=" * 60)
    print("[C] Cross-validation: V20 base (no expanding)")
    print("=" * 60)
    for vy in valid_years:
        auc, _ = _train_eval(df, base_avail, vy, "base")
        results["base"][str(vy)] = auc

    print()
    print("=" * 60)
    print("[C] Cross-validation: V20 base + expanding (6 features)")
    print("=" * 60)
    last_booster = None
    for vy in valid_years:
        auc, b = _train_eval(df, full_avail, vy, "exp ")
        results["expanding"][str(vy)] = auc
        last_booster = b

    base_avg = float(np.mean(list(results["base"].values())))
    exp_avg = float(np.mean(list(results["expanding"].values())))
    delta = exp_avg - base_avg

    print()
    print("=" * 60)
    print(f"[C] SUMMARY: base avg AUC = {base_avg:.4f}")
    print(f"[C] SUMMARY: exp  avg AUC = {exp_avg:.4f}")
    print(f"[C] SUMMARY: delta        = {delta:+.4f}")
    print("=" * 60)

    # feature importance (last booster, full features)
    if last_booster is not None:
        imp_gain = last_booster.feature_importance(importance_type="gain")
        imp_pairs = sorted(zip(full_avail, imp_gain), key=lambda x: -x[1])
        print("[C] Top 30 importance (gain, full model):")
        for name, score in imp_pairs[:30]:
            mark = " ★" if name in EXPANDING_FEATURES else ""
            print(f"  {score:>10.0f}  {name}{mark}")
        results["importance_top30"] = [(n, float(s)) for n, s in imp_pairs[:30]]
        results["expanding_importance"] = {
            n: float(s) for n, s in imp_pairs if n in EXPANDING_FEATURES
        }

    # save
    model_path = MODEL_DIR / "v20_expanding_v1.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({"booster": last_booster, "features": full_avail}, f)
    print(f"[C] saved {model_path}")

    OUT_RESULT.parent.mkdir(parents=True, exist_ok=True)
    results["features_base"] = base_avail
    results["features_expanding"] = EXPANDING_FEATURES
    results["base_avg_auc"] = base_avg
    results["expanding_avg_auc"] = exp_avg
    results["delta_auc"] = delta
    results["elapsed_s"] = time.time() - t0
    OUT_RESULT.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[C] saved {OUT_RESULT}")
    print(f"[C] elapsed {results['elapsed_s']:.1f}s")


if __name__ == "__main__":
    main()
