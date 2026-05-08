"""V20 + expanding クラス別 / 距離別 / 馬場別 AUC 評価 (Session #55 D).

C 領域の最終 fold (valid=2025) で:
- クラス別 (新馬 / 未勝利 / 1勝 / 2勝 / 3勝 / OP / 重賞)
- 距離別 (短 / マイル / 中 / 長)
- 馬場別 (芝 / ダート)

base vs base+expanding の AUC を比較。
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SRC = DATA / "jra_races_full.csv"
EXP = DATA / "v20" / "expanding_features_v1.parquet"
OUT_RESULT = DATA / "v18" / "session_55_v20_expanding_eval_raw.json"


def _norm_year4(y: int) -> int:
    return 2000 + y if y < 80 else 1900 + y


CLASS_MAP = {
    "11": "新馬",
    "12": "未勝利",
    "13": "1勝",
    "14": "2勝",
    "15": "3勝",
    "16": "OP",
    "17": "重賞",
    "18": "重賞",  # 一部 G2
    "19": "重賞",  # G1
}


def _classify_dist(d: float) -> str:
    if d < 1400:
        return "短距離"
    if d < 1800:
        return "マイル"
    if d < 2200:
        return "中距離"
    return "長距離"


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


def main() -> None:
    print("[D] read jra_races_full")
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
        "class_code",
    ]
    df = pd.read_csv(SRC, usecols=cols, low_memory=False)
    df["year_4digit"] = df["year"].apply(_norm_year4)
    df = df[(df["year_4digit"] >= 2018) & (df["year_4digit"] <= 2025)].copy()
    df["finish_num"] = pd.to_numeric(df["finish"], errors="coerce")
    valid = (df["abnormal_code"].fillna(0) == 0) & df["finish_num"].notna()
    df = df.loc[valid].reset_index(drop=True)
    df["target"] = (df["finish_num"].between(1, 3)).astype(int)

    df["course_enc"] = pd.to_numeric(df["course"], errors="coerce").fillna(-1).astype(int)
    df["surface_enc"] = df["surface"].map({"芝": 0, "ダ": 1, "障": 2}).fillna(-1).astype(int)
    df["sex_enc"] = df["sex"].map({"牡": 0, "牝": 1, "セ": 2}).fillna(-1).astype(int)
    for c in ("weight_carry", "age", "distance", "num_horses", "umaban", "popularity"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["popularity"] = df["popularity"].fillna(99)
    df["horse_num_ratio"] = df["umaban"] / df["num_horses"].replace(0, np.nan)
    df["race_dt"] = pd.to_datetime(
        df["year_4digit"].astype(str)
        + "-"
        + df["month"].astype(str).str.zfill(2)
        + "-"
        + df["day"].astype(str).str.zfill(2),
        errors="coerce",
    )

    df = df.sort_values(["horse_id", "race_dt", "race_id"]).reset_index(drop=True)
    g = df.groupby("horse_id", sort=False)
    df["prev_finish"] = g["finish_num"].shift(1)
    df["prev_dt"] = g["race_dt"].shift(1)
    df["rest_days"] = (df["race_dt"] - df["prev_dt"]).dt.days.clip(0, 365)

    h_top3 = g["target"].shift(1).expanding().sum().reset_index(level=0, drop=True)
    h_cnt = g["target"].shift(1).expanding().count().reset_index(level=0, drop=True)
    df["horse_career_wr_lifetime"] = (h_top3 + 1.0) / (h_cnt + 5.0)

    df = df.sort_values(["jockey_id", "race_dt", "race_id"]).reset_index(drop=True)
    j = df.groupby("jockey_id", sort=False)
    j_win = (df["finish_num"] == 1).astype(int)
    j_win_cum = j_win.groupby(df["jockey_id"]).shift(1).groupby(df["jockey_id"]).cumsum().reset_index(level=0, drop=True)
    j_cnt = j["target"].shift(1).expanding().count().reset_index(level=0, drop=True)
    df["jockey_wr_lifetime"] = (j_win_cum + 1.0) / (j_cnt + 10.0)

    df = df.sort_values(["trainer_id", "race_dt", "race_id"]).reset_index(drop=True)
    t = df.groupby("trainer_id", sort=False)
    t_top3 = t["target"].shift(1).expanding().sum().reset_index(level=0, drop=True)
    t_cnt = t["target"].shift(1).expanding().count().reset_index(level=0, drop=True)
    df["trainer_top3_lifetime"] = (t_top3 + 3.0) / (t_cnt + 10.0)

    print("[D] read expanding parquet")
    exp = pd.read_parquet(EXP)
    exp = exp.drop(columns=[c for c in ("year_4digit",) if c in exp.columns])
    df = df.merge(exp, on=["race_id", "horse_id", "umaban"], how="left")

    base_avail = [c for c in BASE_FEATURES if c in df.columns]
    full_avail = base_avail + EXPANDING_FEATURES
    for c in base_avail + EXPANDING_FEATURES:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        med = df[c].median()
        df[c] = df[c].fillna(med if pd.notna(med) else 0)

    # train: <2025, valid: 2025
    train_df = df[df["year_4digit"] < 2025]
    valid_df = df[df["year_4digit"] == 2025].copy()

    print(f"[D] train={len(train_df):,} valid_2025={len(valid_df):,}")
    print("[D] train base / full models...")

    def _train(features: list[str]) -> lgb.Booster:
        X_tr = train_df[features].astype(float)
        y_tr = train_df["target"].astype(int)
        X_va = valid_df[features].astype(float)
        y_va = valid_df["target"].astype(int)
        d_tr = lgb.Dataset(X_tr, label=y_tr)
        d_va = lgb.Dataset(X_va, label=y_va, reference=d_tr)
        return lgb.train(
            LGB_PARAMS,
            d_tr,
            num_boost_round=1000,
            valid_sets=[d_va],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
        )

    booster_base = _train(base_avail)
    booster_full = _train(full_avail)
    valid_df["pred_base"] = booster_base.predict(
        valid_df[base_avail].astype(float), num_iteration=booster_base.best_iteration
    )
    valid_df["pred_full"] = booster_full.predict(
        valid_df[full_avail].astype(float), num_iteration=booster_full.best_iteration
    )

    # クラス別
    valid_df["class_code_str"] = valid_df["class_code"].astype(str).str[:2]
    valid_df["class_label"] = valid_df["class_code_str"].map(CLASS_MAP).fillna("その他")
    valid_df["dist_label"] = valid_df["distance"].apply(_classify_dist)
    valid_df["surface_label"] = valid_df["surface_enc"].map({0: "芝", 1: "ダート", 2: "障害"}).fillna("不明")

    out: dict = {"valid_year": 2025, "n_valid": len(valid_df), "breakdowns": {}}

    def _safe_auc(y, p):
        if len(y) < 50 or len(set(y)) < 2:
            return None
        return float(roc_auc_score(y, p))

    for label_col, head in (
        ("class_label", "by_class"),
        ("dist_label", "by_distance"),
        ("surface_label", "by_surface"),
    ):
        out["breakdowns"][head] = {}
        print()
        print(f"=== {head} ===")
        for grp, sub in valid_df.groupby(label_col):
            auc_b = _safe_auc(sub["target"].values, sub["pred_base"].values)
            auc_f = _safe_auc(sub["target"].values, sub["pred_full"].values)
            delta = (auc_f - auc_b) if (auc_b and auc_f) else None
            n = len(sub)
            n_pos = int(sub["target"].sum())
            out["breakdowns"][head][str(grp)] = {
                "n": n,
                "n_pos": n_pos,
                "auc_base": auc_b,
                "auc_full": auc_f,
                "delta": delta,
            }
            ab = f"{auc_b:.4f}" if auc_b is not None else "  N/A"
            af = f"{auc_f:.4f}" if auc_f is not None else "  N/A"
            ds = f"{delta:+.4f}" if delta is not None else "  N/A"
            print(f"  {grp:<10} n={n:>6,}  base={ab}  full={af}  delta={ds}")

    OUT_RESULT.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print()
    print(f"[D] saved {OUT_RESULT}")


if __name__ == "__main__":
    main()
