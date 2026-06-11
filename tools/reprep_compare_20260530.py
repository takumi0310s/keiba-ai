#!/usr/bin/env python3
"""検証用 一括再予測 + 朝版比較 (2026-05-30、 paci+ZE修復効果の測定)。

★ 検証専用。 今日の投票は朝版で確定済・上書きしない。 V15 model / predict_core /
  daily_predict 不変。 pid2548 無干渉 (別プロセス・read only) ★

公平性 (fairness):
  - 朝オッズを data/odds_base_20260530.csv から固定再利用 (load_odds_base)。
    → 終わったレースで確定オッズ(結果リーク)を使わない。
    → 変えるのは paci+ZE (修復済) のみ → 純粋に修復効果を測定。
  - 馬成績/血統は朝と同一(過去走は不変)。 障害は除外。

出力: data/allscores/20260530_v2_repaired/<race_id>.json (朝版は別dir・無変更)
比較: 朝版 data/allscores/20260530/<race_id>.json との score/順位/買い目差。
"""
from __future__ import annotations
import os, io, sys, json, time, glob
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE); sys.path.insert(0, os.path.join(BASE, "tools"))
import pandas as pd, numpy as np

DATE = "20260530"
MORN_DIR = os.path.join(BASE, "data", "allscores", DATE)
OUT_DIR = os.path.join(BASE, "data", "allscores", f"{DATE}_v2_repaired")
os.makedirs(OUT_DIR, exist_ok=True)

from predict_core import (load_models, parse_shutuba, build_features, predict_race,
                          get_horse_stats, apply_horse_stats, set_horse_defaults,
                          fetch_jra_and_weather, load_odds_base)
from jrdb_features import merge_jrdb_once

import gzip, pickle
_V16 = None
def v16_scores(df):
    global _V16
    if _V16 is None:
        mp = os.path.join(BASE, "models", "v16_ability_candidate.pkl.gz")
        with gzip.open(mp, "rb") as f:
            _V16 = pickle.load(f)
    feats = _V16.get("features", [])
    if not feats:
        return {}
    X = df.reindex(columns=feats).fillna(0.0).values.astype("float32")
    p = _V16["model"].predict(X)
    w = _V16.get("ensemble_weights", {"lgb": 0.5, "xgb": 0.5})
    if _V16.get("xgb_model") is not None:
        import xgboost as xgb
        px = _V16["xgb_model"].predict(xgb.DMatrix(X))
        p = w.get("lgb", 0.5) * p + w.get("xgb", 0.5) * px
    return {str(int(df.iloc[i]["馬番"])): float(p[i]) for i in range(len(df))}


def repredict(race_id, model_data, jw_cache):
    race_name, horses, horse_ids, rinfo = parse_shutuba(race_id)
    if not horses or rinfo.get("surface") == "障":
        return None
    # ★ 朝オッズ固定 (結果リーク防止) ★
    ob = load_odds_base(race_id, DATE)
    odds_dict = {u: v["odds"] for u, v in ob.items()}
    for h in horses:
        u = h.get("馬番", 0)
        if u in ob:
            h["単勝オッズ"] = ob[u]["odds"]; h["人気順位"] = ob[u]["pop_rank"]
    course = rinfo.get("course", "")
    if model_data.get("is_live") and course:
        if course not in jw_cache:
            try: jw_cache[course] = fetch_jra_and_weather(course)
            except Exception: jw_cache[course] = ({}, {})
        jra, wx = jw_cache[course]
    else:
        jra, wx = {}, {}
    for i, (h, hid) in enumerate(zip(horses, horse_ids)):
        try:
            if hid:
                apply_horse_stats(h, get_horse_stats(hid, rinfo["distance"], rinfo["surface"], course), rinfo)
            else:
                set_horse_defaults(h)
        except Exception:
            set_horse_defaults(h)
        if i < len(horses) - 1:
            time.sleep(0.3)
    df = build_features(horses, rinfo, model_data, race_id=race_id, odds_dict=odds_dict,
                        jra_track_info=jra, weather_info=wx)
    try:
        # ★二重マージ禁止: build_features 内で適用済み (6/11 Fable sweep)。歴史的出力(5/30)は二重マージで生成された点に注意★
        df = merge_jrdb_once(df, race_id)  # healthy paci+ZE
    except Exception as e:
        print(f"    [JRDB merge] {e}")
    df = predict_race(df, model_data, bool(odds_dict), race_info=rinfo)
    df = df.sort_values("スコア", ascending=False).reset_index(drop=True)
    v15 = {str(int(r["馬番"])): float(r["スコア"]) for _, r in df.iterrows()}
    try: v16 = v16_scores(df)
    except Exception as e: print(f"    [V16] {e}"); v16 = {}
    return {"race_id": race_id, "race_name": race_name, "course": course,
            "v15_scores": v15, "v16_scores": v16}


def main():
    print(f"[{time.strftime('%H:%M:%S')}] 検証用 一括再予測 (朝オッズ固定・healthy paci+ZE)")
    md = load_models()
    if md.get("model") is None:
        print("model load fail"); return 1
    morn_files = sorted(glob.glob(os.path.join(MORN_DIR, "*.json")))
    print(f"  朝版レース: {len(morn_files)}")
    jw = {}; done = 0
    for mf in morn_files:
        rid = os.path.splitext(os.path.basename(mf))[0]
        try:
            res = repredict(rid, md, jw)
            if res:
                with open(os.path.join(OUT_DIR, f"{rid}.json"), "w", encoding="utf-8") as f:
                    json.dump(res, f, ensure_ascii=False)
                done += 1
                print(f"  [{done}] {res['course']} {rid} OK ({len(res['v15_scores'])}頭)")
        except Exception as e:
            print(f"  {rid} ERROR: {e}")
    print(f"[{time.strftime('%H:%M:%S')}] 完了 {done}R → {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
