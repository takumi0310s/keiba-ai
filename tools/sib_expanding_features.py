"""sib expanding window features (Session #39 A、PoC).

V18/V19 sib リーク (Session #38) の修正版。

旧 sib (data/netkeiba_siblings.csv):
  - mother 単位で全期間集計 → 全年データを学習時に利用 → リーク
  - sib_top3_rate / sib_shinba_wr の corr_target が高い (0.29)

修正版 (本 module):
  - 各レース時点で「過去のみ」のデータで集計 (expanding window)
  - cumulative count / cumulative top3 rate を「当該レース直前まで」で算出
  - 当該 horse 自身の race row は除外 (cumsum - current)
  - Bayesian smoothing (alpha=10) で低 sample 時の過学習防止

Output schema:
  race_id, horse_id, mother,
  sib_top3_rate_exp, sib_shinba_wr_exp, sib_total_races_exp, sib_total_offspring_exp

usage:
  python tools/sib_expanding_features.py --out data/netkeiba_siblings_expanding.csv
  # then merge into V18/V19 / V20 training pipeline (replace static sib_*)
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

BASE = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))


def build_expanding_sib(
    blood_path: str = "data/blood_full.csv",
    races_path: str = "data/jra_races_full.csv",
    alpha: float = 10.0,
    smoothing_top3_prior: float = 0.30,
    smoothing_shinba_prior: float = 0.10,
) -> pd.DataFrame:
    """各 (race_id, horse_id) に expanding window で sib stats を付与."""
    t0 = time.time()

    print(f"[sib_exp] loading blood ...")
    blood = pd.read_csv(
        os.path.join(BASE, blood_path),
        encoding="utf-8",
        usecols=["horse_id", "mother"],
    )
    blood["horse_id"] = blood["horse_id"].astype(str).str.strip()
    blood["mother"] = blood["mother"].fillna("").astype(str).str.strip()
    blood = blood.dropna(subset=["horse_id"])
    blood = blood[blood["mother"] != ""]
    print(f"  blood: {len(blood):,} (unique mothers={blood['mother'].nunique():,})")

    print(f"[sib_exp] loading races ...")
    races = pd.read_csv(
        os.path.join(BASE, races_path),
        encoding="utf-8",
        usecols=[
            "year", "month", "day", "horse_id", "race_id",
            "finish", "class_code",
        ],
        low_memory=False,
    )
    # races horse_id は float 形式 (例: '17105533.0') → integer 化して string
    def _norm_id(x):
        try:
            return str(int(float(str(x).strip())))
        except Exception:
            return str(x).strip()
    races["horse_id"] = races["horse_id"].apply(_norm_id)
    races["finish"] = pd.to_numeric(races["finish"], errors="coerce")
    races["class_code"] = pd.to_numeric(races["class_code"], errors="coerce")
    races = races.dropna(subset=["finish"])
    print(f"  races: {len(races):,}")

    # date 構築 (year は 2 桁: '15'→2015、'25'→2025、'26'→2026)
    races["year_int"] = pd.to_numeric(races["year"], errors="coerce")
    races = races.dropna(subset=["year_int"])
    # 00-30 → 20XX, 31-99 → 19XX (race data 2010+)
    races["year_full"] = races["year_int"].apply(lambda y: 2000 + int(y) if int(y) <= 30 else 1900 + int(y))
    races["date"] = pd.to_datetime(
        races["year_full"].astype(int).astype(str) + "-"
        + races["month"].astype(str).str.zfill(2) + "-"
        + races["day"].astype(str).str.zfill(2),
        errors="coerce",
    )
    races = races.dropna(subset=["date"])

    # blood merge
    races = races.merge(blood[["horse_id", "mother"]], on="horse_id", how="left")
    races = races[races["mother"].notna() & (races["mother"] != "")]
    print(f"  races with mother: {len(races):,}")

    # ===== 時系列 expanding 集計 =====
    # mother 単位 → date 順 sort → cumulative count / cumulative top3 / shinba count
    print(f"[sib_exp] sorting by (mother, date) ...")
    races = races.sort_values(["mother", "date", "race_id"]).reset_index(drop=True)

    # is_top3 (1 if finish<=3 else 0)
    races["is_top3"] = (races["finish"] <= 3).astype(int)
    # is_shinba_win (class_code==15 (新馬戦) and finish==1)
    races["is_shinba"] = (races["class_code"] == 15).astype(int)
    races["is_shinba_win"] = ((races["class_code"] == 15) & (races["finish"] == 1)).astype(int)

    grp = races.groupby("mother", sort=False)

    # cumulative count (1-indexed: shift to exclude current)
    races["mother_cum_races"] = grp.cumcount()  # 0-indexed = 過去 count
    races["mother_cum_top3"] = grp["is_top3"].cumsum() - races["is_top3"]
    races["mother_cum_shinba_runs"] = grp["is_shinba"].cumsum() - races["is_shinba"]
    races["mother_cum_shinba_wins"] = grp["is_shinba_win"].cumsum() - races["is_shinba_win"]

    # offspring count: 過去に出走した unique horse_id 数 (mother 単位、当該 horse 除外)
    # → simple な実装: cumcount で nunique は重い。代替: mother_cum_races で代用 (offspring count proxy)
    # 厳密版: groupby + apply (重いので cum_unique) → 簡易: race count 自体 + 当該 horse の race history を
    # mother 単位の unique offspring = 過去 mother レースで出現した horse_id set のサイズ。
    # PoC では mother_cum_races を offspring proxy として残し、 厳密化は後段。
    print(f"[sib_exp] computing offspring count proxy ...")

    # mother+horse の最初の出走 flag
    races["is_first_appearance"] = ~races.duplicated(subset=["mother", "horse_id"], keep="first")
    # mother 単位 cum first-appearance = offspring count
    races["mother_cum_offspring"] = grp["is_first_appearance"].cumsum() - races["is_first_appearance"].astype(int)

    # ===== Bayesian smoothing =====
    races["sib_top3_rate_exp"] = (
        (races["mother_cum_top3"] + alpha * smoothing_top3_prior)
        / (races["mother_cum_races"] + alpha)
    )
    races["sib_shinba_wr_exp"] = (
        (races["mother_cum_shinba_wins"] + alpha * smoothing_shinba_prior)
        / (races["mother_cum_shinba_runs"] + alpha)
    )
    races["sib_total_races_exp"] = races["mother_cum_races"]
    races["sib_total_offspring_exp"] = races["mother_cum_offspring"]

    out = races[[
        "race_id", "horse_id", "mother",
        "sib_top3_rate_exp", "sib_shinba_wr_exp",
        "sib_total_races_exp", "sib_total_offspring_exp",
    ]].copy()

    elapsed = time.time() - t0
    print(f"[sib_exp] done {elapsed:.1f}s, output={len(out):,}")
    print(f"  sib_top3_rate_exp: mean={out['sib_top3_rate_exp'].mean():.4f}, std={out['sib_top3_rate_exp'].std():.4f}")
    print(f"  sib_shinba_wr_exp: mean={out['sib_shinba_wr_exp'].mean():.4f}, std={out['sib_shinba_wr_exp'].std():.4f}")
    print(f"  sib_total_races_exp:    p50={out['sib_total_races_exp'].median():.0f}, max={out['sib_total_races_exp'].max():.0f}")
    print(f"  sib_total_offspring_exp p50={out['sib_total_offspring_exp'].median():.0f}, max={out['sib_total_offspring_exp'].max():.0f}")
    return out


def cli():
    p = argparse.ArgumentParser(description="sib expanding window features (Session #39 A PoC)")
    p.add_argument("--blood", default="data/blood_full.csv")
    p.add_argument("--races", default="data/jra_races_full.csv")
    p.add_argument("--out", default="data/netkeiba_siblings_expanding.csv")
    p.add_argument("--alpha", type=float, default=10.0)
    p.add_argument("--top3-prior", type=float, default=0.30)
    p.add_argument("--shinba-prior", type=float, default=0.10)
    args = p.parse_args()

    out = build_expanding_sib(
        blood_path=args.blood,
        races_path=args.races,
        alpha=args.alpha,
        smoothing_top3_prior=args.top3_prior,
        smoothing_shinba_prior=args.shinba_prior,
    )
    out_path = os.path.join(BASE, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out.to_csv(out_path, index=False, encoding="utf-8")
    print(f"[sib_exp] written: {out_path}  ({os.path.getsize(out_path)/1024/1024:.1f} MB)")


if __name__ == "__main__":
    cli()
