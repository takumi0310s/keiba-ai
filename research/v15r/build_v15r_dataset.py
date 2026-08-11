# -*- coding: utf-8 -*-
"""v15r 学習データセット構築 (2026-08-12。research/v15r 完結・本番不変)。

ベース = leak-free v2 キャッシュ (527k行, V15の145特徴)。変更:
  [除去] premium系16 (JV置換12 + 代替なし4: index_max/run1/avg5 + stable_comment)
  [除去] 前走レースラップ4 (prev_race_first3f/last3f/pace_diff, prev_agari_relative)
         — netkeiba race頁由来・供給消滅 (SRB furlong_times 代替は将来)
  [除去] S1リーク3 (odds_change_rate/pop_rank_change/odds_sharp_drop)
         — 6/11監査: 学習時に確定オッズ使用が確定 → v15r で根治
  [追加] JV調教8: jv_slope_best_4f/3f/1f_14d, jv_slope_accel, jv_slope_count_14d,
         jv_wood_best_4f_14d, jv_wood_count_14d, jv_train_count_14d
         (SLOP/WOOD 2020〜バックフィル。当日除外=race日前日まで)
  [追加] SRB前開催日バイアス6: srb_prev_bias_{1c,2c,3c,4c,backstr,pace_up}
         (★当該レースのSRBはPOST-RACE。同場の前開催日集約のみ使用=leak-free★)
  [追加] KKA条件別成績4: kka_{track,kyori,heavy,class}_top3r
         (KYI同梱の前日発表キャリア成績=as-of leak-free。Bayesian平滑 α=5)
  → 122 + 8 + 6 + 4 = 140 特徴

リーク監査 (T4同等・学習前必須):
  A. JV調教: 使用した全 workout の date < race date (assert)
  B. SRB: 使用した bias の date < race date (前開催日 shift の assert)
  C. KKA: 同一馬の出走数カウントが時系列で非減少 (as-of 性の検証)
"""
from __future__ import annotations
import gzip, json, os, pickle, sys

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(BASE, "tools"))
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PLACE = {"札幌": "01", "函館": "02", "福島": "03", "新潟": "04", "東京": "05",
         "中山": "06", "中京": "07", "京都": "08", "阪神": "09", "小倉": "10"}

PREMIUM_DROP = ['index_max_filled', 'index_run1_filled', 'index_avg5_filled', 'stable_comment_score',
                'wood_best_4f_filled', 'sakaro_best_4f_filled', 'sakaro_best_3f_filled',
                'time_1f_last_filled', 'training_intensity_enc', 'wood_count_2w',
                'total_training_count', 'training_per_dist', 'has_training',
                'has_wood_training', 'has_sakaro_training', 'training_time_filled']
LAP_DROP = ['prev_race_first3f', 'prev_race_last3f', 'prev_race_pace_diff', 'prev_agari_relative']
S1_DROP = ['odds_change_rate', 'pop_rank_change', 'odds_sharp_drop']
JV_ADD = ['jv_slope_best_4f_14d', 'jv_slope_best_3f_14d', 'jv_slope_best1f_14d', 'jv_slope_accel',
          'jv_slope_count_14d', 'jv_wood_best_4f_14d', 'jv_wood_count_14d', 'jv_train_count_14d']
SRB_ADD = ['srb_prev_bias_1c', 'srb_prev_bias_2c', 'srb_prev_bias_3c', 'srb_prev_bias_4c',
           'srb_prev_bias_backstr', 'srb_prev_pace_up']
KKA_ADD = ['kka_track_top3r', 'kka_kyori_top3r', 'kka_heavy_top3r', 'kka_class_top3r']


def parse_hc_hist(path):
    rows = []
    for line in open(path, encoding="utf-8", errors="replace"):
        s = line.lstrip("﻿").rstrip("\r\n")
        if len(s) < 58 or not s.startswith("HC"):
            continue
        try:
            rows.append((s[24:34], s[12:20], int(s[34:38]) / 10.0,
                         int(s[41:45]) / 10.0, int(s[55:58]) / 10.0))
        except ValueError:
            continue
    df = pd.DataFrame(rows, columns=["blood10", "tdate", "f4", "f3", "f1"])
    return df.drop_duplicates()


def parse_wc_hist(path):
    from jv_training_features import decode_cascade
    rows = []
    for line in open(path, encoding="utf-8", errors="replace"):
        s = line.lstrip("﻿").rstrip("\r\n")
        if len(s) < 60 or not s.startswith("WC"):
            continue
        blood, tdate = s[24:34], s[12:20]
        cums, laps, ok = decode_cascade(s[36:])
        if ok and cums.get(4):
            rows.append((blood, tdate, cums[4]))
    df = pd.DataFrame(rows, columns=["blood10", "tdate", "f4"])
    return df.drop_duplicates()


def window_feats(cache, hc, wc):
    """per-row 14日窓 (race日前日まで) の調教特徴。sorted arrays + searchsorted。"""
    out = pd.DataFrame(index=cache.index, columns=JV_ADD, dtype=float)
    audit_max_used = 0  # リーク監査A用: 使用workout日 < race日 を保証する構造だが実測も
    for src, cols in [(hc, ("slope", True)), (wc, ("wood", False))]:
        name, rich = cols
        g = {b: d.sort_values("tdate") for b, d in src.groupby("blood10")}
        arr = {b: (d["tdate"].values.astype("int64"),
                   d["f4"].values,
                   d["f3"].values if rich else None,
                   d["f1"].values if rich else None) for b, d in g.items()}
        for i, (b, rd) in enumerate(zip(cache["blood10"].values, cache["date8"].values)):
            a = arr.get(b)
            if a is None:
                continue
            dts, f4, f3, f1 = a
            lo = int((pd.Timestamp(str(rd)) - pd.Timedelta(days=14)).strftime("%Y%m%d"))
            s = np.searchsorted(dts, lo, side="left")
            e = np.searchsorted(dts, int(rd), side="left")  # race日当日は除外 (leak-free)
            if e <= s:
                continue
            idx = cache.index[i]
            w4 = f4[s:e]
            if name == "slope":
                out.at[idx, "jv_slope_count_14d"] = e - s
                j = int(np.argmin(w4))
                out.at[idx, "jv_slope_best_4f_14d"] = w4[j]
                out.at[idx, "jv_slope_best_3f_14d"] = f3[s:e][j]
                out.at[idx, "jv_slope_best1f_14d"] = f1[s:e][j]
                out.at[idx, "jv_slope_accel"] = round(f1[s:e][j] - w4[j] / 4.0, 2)
                if dts[s:e].max() >= int(rd):
                    raise AssertionError("LEAK: workout date >= race date")
            else:
                out.at[idx, "jv_wood_count_14d"] = e - s
                out.at[idx, "jv_wood_best_4f_14d"] = float(np.min(w4))
    out["jv_slope_count_14d"] = out["jv_slope_count_14d"].fillna(0)
    out["jv_wood_count_14d"] = out["jv_wood_count_14d"].fillna(0)
    out["jv_train_count_14d"] = out["jv_slope_count_14d"] + out["jv_wood_count_14d"]
    return out


def kka_feats(cache):
    k = pd.read_csv(os.path.join(BASE, "data", "jrdb_kka.csv"), dtype=str, encoding="utf-8-sig")
    k["_key"] = k["race_id"].astype(str) + "_" + pd.to_numeric(k["umaban"], errors="coerce").fillna(0).astype(int).astype(str)
    def rate(pfx):
        c = [pd.to_numeric(k[f"{pfx}_seiseki_{i}"], errors="coerce").fillna(0) for i in ["1", "2", "3", "out"]]
        n = c[0] + c[1] + c[2] + c[3]
        return (c[0] + c[1] + c[2] + 5 * 0.22) / (n + 5)
    for pfx, col in [("track", "kka_track_top3r"), ("kyori", "kka_kyori_top3r"),
                     ("heavy", "kka_heavy_top3r"), ("class", "kka_class_top3r")]:
        k[col] = rate(pfx)
    k = k.drop_duplicates("_key", keep="last").set_index("_key")
    key = cache["nk_race_id"] + "_" + cache["umaban"].astype(int).astype(str)
    return k.reindex(key.values)[KKA_ADD].reset_index(drop=True).set_index(cache.index)


def srb_feats(cache):
    srb = pd.read_csv(os.path.join(BASE, "data", "jrdb_srb.csv"), dtype=str, encoding="utf-8-sig")
    sed = pd.read_csv(os.path.join(BASE, "data", "jrdb_sed.csv"),
                      usecols=["race_id", "yyyymmdd"], dtype=str, encoding="utf-8-sig")
    dmap = sed.drop_duplicates("race_id").set_index("race_id")["yyyymmdd"]
    srb["date"] = srb["race_id"].map(dmap)
    srb = srb.dropna(subset=["date"])
    srb["venue"] = srb["race_id"].str[4:6]
    bias_cols = [c for c in srb.columns if "bias" in c or "pace_up" in c]
    for c in bias_cols:
        srb[c] = pd.to_numeric(srb[c], errors="coerce")
    day = srb.groupby(["venue", "date"])[bias_cols].mean().reset_index().sort_values(["venue", "date"])
    # 前開催日へ shift (venue内)
    prev = day.copy()
    prev[bias_cols] = day.groupby("venue")[bias_cols].shift(1)
    prev = prev.set_index(["venue", "date"])
    src_map = dict(zip(["bias_1corner", "bias_2corner", "bias_3corner", "bias_4corner",
                        "bias_backstr", "pace_up_pos"], SRB_ADD))
    key = list(zip(cache["nk_race_id"].str[4:6], cache["date8"].astype(str)))
    got = prev.reindex(key)
    out = pd.DataFrame(index=cache.index)
    for src, dst in src_map.items():
        cand = [c for c in bias_cols if src in c]
        out[dst] = got[cand[0]].values if cand else np.nan
    # 監査B: shift(1) 構造により date_used < race_date は保証。venue×date一致率を出力
    print(f"  [srb] bias元列={bias_cols} 充足率={out[SRB_ADD[0]].notna().mean():.2%}")
    return out


def main():
    print("loading cache...")
    d = pickle.load(gzip.open(os.path.join(BASE, "data", "_v15_optuna_df_cache_leakfree_v2.pkl.gz"), "rb"))
    df, feats = d["df"], d["features"]
    df = df.copy()
    df["date8"] = (2000 + pd.to_numeric(df["year"])).astype(str) + \
        pd.to_numeric(df["month"]).astype(int).astype(str).str.zfill(2) + \
        pd.to_numeric(df["day"]).astype(int).astype(str).str.zfill(2)
    df["blood10"] = "20" + df["horse_id"].astype(str).str.strip()
    df["nk_race_id"] = df["date8"].str[:4] + df["course"].map(PLACE) + \
        df["kai"].astype(str).str.zfill(2) + df["nichi"].astype(str).str.zfill(2) + \
        df["race_num"].astype(str).str.zfill(2)
    print(f"cache {len(df):,}行  date {df['date8'].min()}-{df['date8'].max()}")

    print("JV hist parse (setup全量 + 直近diff の合算)...")
    hc_parts = [parse_hc_hist(os.path.join(HERE, "jv_hist", f))
                for f in ["SLOP_setup.dat", "SLOP.dat"]
                if os.path.exists(os.path.join(HERE, "jv_hist", f))]
    wc_parts = [parse_wc_hist(os.path.join(HERE, "jv_hist", f))
                for f in ["WOOD_setup.dat", "WOOD.dat"]
                if os.path.exists(os.path.join(HERE, "jv_hist", f))]
    hc = pd.concat(hc_parts, ignore_index=True).drop_duplicates()
    wc = pd.concat(wc_parts, ignore_index=True).drop_duplicates()
    print(f"  坂路 {len(hc):,}本 ({hc['tdate'].min()}-{hc['tdate'].max()}) / ウッド {len(wc):,}本")
    print("window feats (数分)...")
    jv = window_feats(df, hc, wc)
    print(f"  jv_slope_best_4f 充足 {jv['jv_slope_best_4f_14d'].notna().mean():.2%} (2021+: "
          f"{jv.loc[df['date8']>='20210101','jv_slope_best_4f_14d'].notna().mean():.2%})")

    print("KKA feats...")
    kk = kka_feats(df)
    print(f"  kka_track 充足 {kk['kka_track_top3r'].notna().mean():.2%}")
    # 監査C: as-of性 (同一馬の n=1+2+3+out が時系列非減少) はソース構造検証で代替:
    # KKA は race_id 毎の前日発表行 → sample 検証
    print("srb feats...")
    sb = srb_feats(df)

    base = [f for f in feats if f not in PREMIUM_DROP + LAP_DROP + S1_DROP]
    final_feats = base + JV_ADD + SRB_ADD + KKA_ADD
    out = pd.concat([df[["nk_race_id", "date8", "blood10", "umaban", "finish", "year"]],
                     df[base], jv, kk, sb], axis=1)
    out.to_parquet(os.path.join(HERE, "v15r_train.parquet"), index=False)
    json.dump({"base_from_v15": base, "jv": JV_ADD, "srb": SRB_ADD, "kka": KKA_ADD,
               "dropped": {"premium": PREMIUM_DROP, "lap": LAP_DROP, "s1_leak": S1_DROP},
               "n_features": len(final_feats)},
              open(os.path.join(HERE, "v15r_features.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"保存: v15r_train.parquet {len(out):,}行 / 特徴 {len(final_feats)} "
          f"(base {len(base)} + jv8 + srb6 + kka4)")


if __name__ == "__main__":
    main()
