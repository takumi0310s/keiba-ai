# -*- coding: utf-8 -*-
"""JV-Link 調教特徴量ビルダー (2026-08-12 第1弾-3。v15r 学習セット用)。

netkeiba premium 死亡12特徴の代替を JV-Link 公式調教 (SLOP=HC坂路 / WOOD=WCウッド) で再定義。

時計ブロックの構造 (2026-08-12 実レコードから加算整合で解読・自己検証):
  HC(58B): [34:58] = 4F計(4) lap(3) 3F計(4) lap(3) 2F計(4) lap(3) 1F(3)   ※全て1/10秒
  WC(103B): 末尾から同型カスケード [... 5F計(4) lap(3) 4F計(4) lap(3) 3F計(4) lap(3) 2F計(4) 1F(3)]
            未走行の長い距離はゼロ埋め。デコーダは「計[i] - lap[i] ≈ 計[i+1]」で自己検証。

premium → JV 代替マッピング (v15r 学習セットの定義):
  training_time_filled   → jv_slope_best_4f_14d (坂路ベスト4F、直近14日)
  sakaro_best_4f_filled  → jv_slope_best_4f_14d (同上)
  sakaro_best_3f_filled  → jv_slope_best_3f_14d
  time_1f_last_filled    → jv_slope_best1f_14d (ベスト本の1F)
  wood_best_4f_filled    → jv_wood_best_4f_14d
  has_training/has_sakaro→ jv_slope_count_14d > 0
  has_wood_training      → jv_wood_count_14d > 0
  wood_count_2w          → jv_wood_count_14d
  total_training_count   → jv_train_count_14d (坂路+ウッド)
  training_per_dist      → jv_train_count_14d / (距離/1000) ※v15r側で合成
  training_intensity_enc → jv_slope_accel (ベスト本の 1F − 平均lap、負=加速仕上げ)
  index_*/stable_comment → 代替なし (v15r では欠損運用)

usage:
  python tools/jv_training_features.py --validate            # デコーダ検証+カバレッジ
  build_training_features(asof_date, blood_nums) → DataFrame # v15r から import
"""
from __future__ import annotations
import argparse, os, sys
from datetime import datetime, timedelta

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import pandas as pd

JV = os.path.join(BASE, "data", "jvlink")


def decode_cascade(block: str):
    """末尾アンカーのカスケードデコード。
    return (cum_times dict {n_furlong: sec}, laps list[sec] 4F区間分, ok)"""
    s = block.strip()
    if len(s) < 10 or not s.isdigit():
        return {}, [], False
    # 末尾: 1F(3)。その前に [計(4) lap(3)] が並ぶ
    try:
        f1 = int(s[-3:]) / 10.0
    except ValueError:
        return {}, [], False
    cums, laps = {1: f1}, [f1]
    pos = len(s) - 3
    nf = 2
    prev_cum = f1
    while pos >= 7:
        cum = int(s[pos - 7:pos - 3]) / 10.0
        lap = int(s[pos - 3:pos]) / 10.0 if nf > 2 else None  # 直前ブロックのlap位置管理
        # 実際の並び: ... 計(4) lap(3) | 計(4) lap(3) | 1F(3)
        cum4 = int(s[pos - 7:pos - 3]) / 10.0
        lap3 = int(s[pos - 3:pos]) / 10.0
        if cum4 == 0:
            break  # 未走行ゾーン
        # 検証: 計 - lap(この区間) == 前の計
        if abs(cum4 - lap3 - (prev_cum if nf == 2 else prev_cum)) > 0.15 and nf == 2:
            # 2F計: cum4 - lap3 == f1
            pass
        if abs(cum4 - lap3 - prev_cum) > 0.15:
            return cums, laps, False
        cums[nf] = cum4
        laps.append(lap3)
        prev_cum = cum4
        nf += 1
        pos -= 7
        if nf > 10:
            break
    return cums, laps, True


_HC = None
_WC = None


def _load():
    global _HC, _WC
    if _HC is None:
        _HC = pd.read_csv(os.path.join(JV, "jv_hc.csv"), dtype=str, encoding="utf-8-sig")
        d = _HC["time_block"].map(decode_cascade)
        _HC["_cums"] = d.map(lambda x: x[0])
        _HC["_ok"] = d.map(lambda x: x[2])
        _HC["f4"] = _HC["_cums"].map(lambda c: c.get(4))
        _HC["f3"] = _HC["_cums"].map(lambda c: c.get(3))
        _HC["f1"] = _HC["_cums"].map(lambda c: c.get(1))
    if _WC is None:
        p = os.path.join(JV, "jv_wc.csv")
        _WC = pd.read_csv(p, dtype=str, encoding="utf-8-sig") if os.path.exists(p) else pd.DataFrame()
        if len(_WC):
            # jvlink_parser.parse_wc: raw_wood = raw[21:] (ヘッダずれ) → 生の再構成:
            # year+month_day+horse_id が [3:21] に相当し blood_num は [24:34] 相当が raw_wood[3:13]
            _WC["blood_num"] = _WC["raw_wood"].str[3:13]
            _WC["train_date"] = _WC["year"].astype(str) + _WC["month_day"].astype(str)
            d = _WC["raw_wood"].map(lambda s: decode_cascade(str(s)[15:]))
            _WC["_cums"] = d.map(lambda x: x[0])
            _WC["_ok"] = d.map(lambda x: x[2])
            _WC["f4"] = _WC["_cums"].map(lambda c: c.get(4))
    return _HC, _WC


def build_training_features(asof_date: str, blood_nums: list[str], window_days: int = 14):
    """asof_date 前 window_days 日の JV 調教から per-blood_num 特徴を構築 (leak-free: 当日除外)。"""
    hc, wc = _load()
    lo = (datetime.strptime(asof_date, "%Y%m%d") - timedelta(days=window_days)).strftime("%Y%m%d")
    rows = []
    h = hc[(hc["train_date"] >= lo) & (hc["train_date"] < asof_date) & hc["_ok"]]
    w = wc[(wc["train_date"] >= lo) & (wc["train_date"] < asof_date)] if len(wc) else pd.DataFrame()
    for bn in blood_nums:
        hh = h[h["blood_num"] == bn]
        ww = w[w["blood_num"] == bn] if len(w) else pd.DataFrame()
        f4s = pd.to_numeric(hh["f4"], errors="coerce").dropna()
        rec = dict(blood_num=bn,
                   jv_slope_count_14d=len(hh),
                   jv_wood_count_14d=len(ww),
                   jv_train_count_14d=len(hh) + len(ww))
        if len(f4s):
            i = f4s.idxmin()
            rec["jv_slope_best_4f_14d"] = float(f4s.min())
            rec["jv_slope_best_3f_14d"] = float(pd.to_numeric(hh.loc[i, "f3"], errors="coerce"))
            rec["jv_slope_best1f_14d"] = float(pd.to_numeric(hh.loc[i, "f1"], errors="coerce"))
            b4, b1 = rec["jv_slope_best_4f_14d"], rec["jv_slope_best1f_14d"]
            rec["jv_slope_accel"] = round(b1 - b4 / 4.0, 2) if b4 and b1 else np.nan
        if len(ww):
            wf4 = pd.to_numeric(ww["f4"], errors="coerce").dropna()
            if len(wf4):
                rec["jv_wood_best_4f_14d"] = float(wf4.min())
        rows.append(rec)
    return pd.DataFrame(rows)


def validate():
    hc, wc = _load()
    print(f"HC(坂路): {len(hc):,}本  デコード成功率 {hc['_ok'].mean()*100:.1f}%  "
          f"4F範囲 {pd.to_numeric(hc['f4'],errors='coerce').min():.1f}-{pd.to_numeric(hc['f4'],errors='coerce').max():.1f}s "
          f"中央 {pd.to_numeric(hc['f4'],errors='coerce').median():.1f}s")
    if len(wc):
        print(f"WC(ウッド): {len(wc):,}本  デコード成功率 {wc['_ok'].mean()*100:.1f}%  "
              f"4F中央 {pd.to_numeric(wc['f4'],errors='coerce').median():.1f}s")
    # カバレッジ: 8/9 出走馬 (KYI) との blood_num join
    kyi = pd.read_csv(os.path.join(BASE, "data", "jrdb_kyi.csv"),
                      usecols=["nk_race_id", "血統登録番号"], dtype=str, encoding="utf-8-sig")
    day = kyi[kyi["nk_race_id"].astype(str).str.startswith("2026") &
              kyi["nk_race_id"].isin(
                  kyi[kyi["nk_race_id"].str[:4] == "2026"]["nk_race_id"])]
    # 8/9 レースの馬
    import glob as _g
    rids = [os.path.basename(f).replace(".parquet", "")
            for f in _g.glob(os.path.join(BASE, "data", "v15_feat_dump", "20260809", "*.parquet"))]
    # ★血統番号の形式差: KYI/JRDB=8桁(YY+連番) / JV-Link=10桁(YYYY+連番)。'20'+8桁 で変換★
    tgt = ("20" + kyi[kyi["nk_race_id"].isin(rids)]["血統登録番号"]
           .dropna().astype(str).str.strip()).unique().tolist()
    ft = build_training_features("20260809", tgt)
    cov = (ft["jv_slope_count_14d"] + ft["jv_wood_count_14d"] > 0).mean()
    b4cov = ft["jv_slope_best_4f_14d"].notna().mean() if "jv_slope_best_4f_14d" in ft.columns else 0
    print(f"\n8/9 出走馬 {len(tgt)}頭 カバレッジ: 調教あり {cov*100:.1f}% / 坂路4Fあり {b4cov*100:.1f}%")
    print(ft[[c for c in ft.columns if c != 'blood_num']].describe().loc[['mean', '50%']].round(2).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    a = ap.parse_args()
    if a.validate:
        validate()
