# -*- coding: utf-8 -*-
"""V15 健全データ再判定 (PRE_REGISTRATION_V15_REJUDGE.md 準拠・基準変更禁止)。

フロー:
  Phase1 healing   : 対象6日の feat_dump を research 側にコピーし、JRDB系特徴のみ
                     再構築済CSVから本番関数 merge_jrdb_predict_features で再計算・上書き。
                     (本番 dump は不変。merged NaN は劣化時デフォルト値でフォールバック=本番と同義)
  Phase2 検証      : healed の JRDB定数 ≤10/日 + live pkl 予測 range ≥0.3 (0802)。NG=exit 2。
  Phase3 校正ゲート: アームD買い目 vs 台帳 8/2 settled の trio_bets_str 一致率 ≥90%。未達=exit 3 (判定に進まない)。
  Phase4 判定      : D(劣化実際スコア) vs H(live pkl×healed特徴) の paper ROI + race-cluster
                     bootstrap CI + ペア差CI + 感度(最高配当1本除外) + 日別内訳。

data/ へは一切書き込まない (healed は research/v15r/feat_dump_healed/)。
"""
from __future__ import annotations
import glob, gzip, json, os, pickle, sys

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(BASE, "tools"))
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import pandas as pd

_keep = [sys.stdout, sys.stderr]
import jrdb_features  # noqa: E402
_keep += [sys.stdout, sys.stderr]
from strategy_filters import evaluate_bet_decision  # noqa: E402
_keep += [sys.stdout, sys.stderr]

DATES = ["20260627", "20260628", "20260711", "20260802", "20260808", "20260809"]
DUMP = os.path.join(BASE, "data", "v15_feat_dump")
HEAL = os.path.join(BASE, "research", "v15r", "feat_dump_healed")
RES = os.path.join(BASE, "research", "ruiji", "raw_results")
PLACE = {"01": "札幌", "02": "函館", "03": "福島", "04": "新潟", "05": "東京",
         "06": "中山", "07": "中京", "08": "京都", "09": "阪神", "10": "小倉"}

# ---- read_csv キャッシュ (merge_jrdb_predict_features は毎回 ~500MB を再読するため) ----
_orig_read = jrdb_features.pd.read_csv
_cache = {}


def _cached_read(path, *a, **k):
    ap = os.path.abspath(str(path))
    if ap not in _cache:
        _cache[ap] = _orig_read(ap, encoding="utf-8-sig", dtype=str, low_memory=False)
    df = _cache[ap]
    uc = k.get("usecols")
    if uc is not None:
        cols = [c for c in df.columns if uc(c)] if callable(uc) else [c for c in uc if c in df.columns]
        return df[cols].copy()
    return df  # 呼び出し側の変異は idempotent (rename if-missing / 同値列追加) を確認済


jrdb_features.pd.read_csv = _cached_read


def load_live_model():
    with gzip.open(os.path.join(BASE, "keiba_model_v15_central_live.pkl.gz"), "rb") as f:
        m = pickle.load(f)
    return m


def load_results():
    R = {}
    for fp in glob.glob(os.path.join(RES, "*.json")):
        for r in json.load(open(fp, encoding="utf-8")):
            fo = {int(k): int(v) for k, v in (r.get("finish_order") or {}).items()}
            pay = r.get("payouts") or {}
            R[str(r["race_id"])] = {
                "fin": fo,
                "trio": int(pay.get("trio", 0) or 0),
                "umaren": int(pay.get("umaren", 0) or 0),
            }
    return R


def classify_condition(nh, dist, cond_enc):
    heavy = (cond_enc is not None) and (cond_enc >= 2)
    if nh <= 7:
        return "E"
    if dist <= 1400:
        return "D"
    if 8 <= nh <= 14 and dist >= 1600 and not heavy:
        return "A"
    if 8 <= nh <= 14 and dist >= 1600 and heavy:
        return "B"
    if nh >= 15 and dist >= 1600 and not heavy:
        return "C"
    return "X"


def trio_bets(nums6):
    n1 = nums6[0]
    bets = set()
    for s in nums6[1:3]:
        for t in nums6[1:6]:
            c = tuple(sorted({n1, s, t}))
            if len(c) == 3:
                bets.add(c)
    return sorted(bets)  # 7点 (apply_c3=False, 台帳と同形)


def bets_str(bets):
    return "; ".join("-".join(str(x) for x in b) for b in bets)


def heal_all():
    os.makedirs(HEAL, exist_ok=True)
    stats = []
    with gzip.open(os.path.join(BASE, "keiba_model_v15_central.pkl.gz"), "rb") as f:
        featsA = pickle.load(f)["features"]
    jrdbA = [c for c in featsA if c.startswith("jrdb_")]
    for dt in DATES:
        od = os.path.join(HEAL, dt)
        os.makedirs(od, exist_ok=True)
        dead_b = dead_a = nrace = 0
        day_frames_b, day_frames_a = [], []
        for fp in sorted(glob.glob(os.path.join(DUMP, dt, "*.parquet"))):
            if os.path.getsize(fp) == 0:
                continue
            try:
                df = pd.read_parquet(fp)
            except Exception:
                continue
            if len(df) < 2 or "race_id" not in df.columns or "馬番" not in df.columns:
                continue
            rid = str(df["race_id"].iloc[0])
            hdf = pd.DataFrame({
                "horse_num": pd.to_numeric(df["馬番"], errors="coerce").fillna(0).astype(int),
                "馬名": df["馬名"].astype(str) if "馬名" in df.columns else "",
            })
            try:
                merged = jrdb_features.merge_jrdb_predict_features(hdf.copy(), rid)
            except Exception as e:
                print(f"  [WARN] merge失敗 {rid}: {e}")
                merged = None
            out = df.copy()
            if merged is not None and len(merged) == len(df):
                # heal範囲 = JRDB系の全供給列 (jrdb_* + paci_* + oz_*)。
                # ★v1 は jrdb_* のみで paci(gain~49%)/oz が死んだままの分布外組合せとなり
                #   Hアームが崩壊した (2026-08-10 診断)。★
                for c in [c for c in merged.columns
                          if c.startswith(("jrdb_", "paci_", "oz_"))]:
                    if c in out.columns:
                        newv = pd.to_numeric(merged[c], errors="coerce")
                        oldv = pd.to_numeric(out[c], errors="coerce")
                        out[c] = newv.fillna(oldv).values  # NaN=劣化時デフォルトへフォールバック(本番同義)
            out.to_parquet(os.path.join(od, os.path.basename(fp)), index=False)
            nrace += 1
            day_frames_b.append(df)
            day_frames_a.append(out)
        if day_frames_b:
            b = pd.concat(day_frames_b, ignore_index=True)
            a = pd.concat(day_frames_a, ignore_index=True)
            dead_b = int((b[jrdbA].nunique() <= 2).sum())
            dead_a = int((a[jrdbA].nunique() <= 2).sum())
            allf_b = int((b[featsA].nunique() <= 2).sum())
            allf_a = int((a[featsA].nunique() <= 2).sum())
        stats.append((dt, nrace, dead_b, dead_a))
        print(f"[heal] {dt}: races={nrace}  JRDB定数 {dead_b}/40 → {dead_a}/40"
              f"  全145定数 {allf_b} → {allf_a}")
    return stats


def predict_live(m, df):
    import xgboost as xgbm
    feats = m["features"]
    # V15-audit-1: live の features list は 150 だが booster は先頭145で学習 (truncate仕様)
    nfeat = m["model"].num_feature()
    feats = feats[:nfeat]
    missing = [c for c in feats if c not in df.columns]
    if missing:
        raise RuntimeError(f"live特徴欠落: {missing[:8]}")
    X = df[feats].apply(pd.to_numeric, errors="coerce")
    w = m.get("ensemble_weights") or {"lgb": 0.5, "xgb": 0.5}
    pred = w.get("lgb", 0.5) * m["model"].predict(X)
    if m.get("xgb_model") is not None:
        pred = pred + w.get("xgb", 0.5) * m["xgb_model"].predict(xgbm.DMatrix(X))
    return pred


def race_meta(df, rid):
    dist = int(pd.to_numeric(df.get("距離(m)", df.get("distance")).iloc[0]))
    ce = pd.to_numeric(df["condition_enc"].iloc[0], errors="coerce") if "condition_enc" in df.columns else None
    ce = None if pd.isna(ce) else int(ce)
    rn = str(df["race_name"].iloc[0]) if "race_name" in df.columns else ""
    course = PLACE.get(rid[4:6], "")
    nh = len(df)
    return rn, course, dist, ce, nh


def is_jump(rn, df):
    if any(t in rn for t in ("障害", "ジャンプ", "J・", "JS")):
        return True
    if "surface" in df.columns and str(df["surface"].iloc[0]) == "障":
        return True
    return False


def build_bets_for(df, score_col):
    s = df.copy()
    s["_sc"] = pd.to_numeric(s[score_col], errors="coerce")
    s = s.dropna(subset=["_sc"])
    if len(s) < 6:
        return None
    if score_col == "スコア" and "AI順位" in s.columns and s["AI順位"].notna().all():
        s = s.sort_values(pd.to_numeric(s["AI順位"], errors="coerce").name and "AI順位",
                          key=lambda x: pd.to_numeric(x, errors="coerce"))
    else:
        s = s.sort_values("_sc", ascending=False)
    nums = pd.to_numeric(s["馬番"], errors="coerce").dropna().astype(int).tolist()[:6]
    if len(nums) < 6:
        return None
    return trio_bets(nums)


def main():
    print("=" * 62)
    print("V15 健全データ再判定 (事前登録準拠)")
    print("=" * 62)

    # ---- Phase 1: healing ----
    print("\n--- Phase 1: feat_dump healing (research側コピー) ---")
    stats = heal_all()

    # ---- Phase 2: 検証 ----
    print("\n--- Phase 2: 検証 (JRDB定数≤10 / 予測range≥0.3) ---")
    ng = [s for s in stats if s[3] > 10]
    if ng:
        print(f"★NG★ JRDB定数>10 の日: {ng} → ロールバック対象 (exit 2)")
        return 2
    m = load_live_model()
    frames = [pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(HEAL, "20260802", "*.parquet")))]
    big = pd.concat(frames, ignore_index=True)
    pred = predict_live(m, big)
    rng = float(pred.max() - pred.min())
    print(f"live pkl 予測range(0802 healed) = {rng:.3f}  (劣化時~0.10 / 健全~0.8)")
    if rng < 0.3:
        print("★NG★ 予測range<0.3 → healing不十分 (exit 2)")
        return 2
    # H経路妥当性: 健全日0620に同一heal適用 → 本番スコアと corr ≥0.95 を要求
    vb, va = [], []
    for fp in sorted(glob.glob(os.path.join(DUMP, "20260620", "*.parquet"))):
        if os.path.getsize(fp) == 0:
            continue
        d0 = pd.read_parquet(fp)
        if len(d0) < 2 or "馬番" not in d0.columns:
            continue
        rid0 = str(d0["race_id"].iloc[0])
        h0 = pd.DataFrame({"horse_num": pd.to_numeric(d0["馬番"], errors="coerce").fillna(0).astype(int),
                           "馬名": d0["馬名"].astype(str) if "馬名" in d0.columns else ""})
        try:
            m0 = jrdb_features.merge_jrdb_predict_features(h0.copy(), rid0)
        except Exception:
            m0 = None
        o0 = d0.copy()
        if m0 is not None and len(m0) == len(d0):
            for c in [c for c in m0.columns if c.startswith(("jrdb_", "paci_", "oz_"))]:
                if c in o0.columns:
                    o0[c] = pd.to_numeric(m0[c], errors="coerce").fillna(
                        pd.to_numeric(o0[c], errors="coerce")).values
        vb.append(d0); va.append(o0)
    vb = pd.concat(vb, ignore_index=True); va = pd.concat(va, ignore_index=True)
    cv = float(np.corrcoef(predict_live(m, va), pd.to_numeric(vb["スコア"], errors="coerce"))[0, 1])
    print(f"H経路妥当性 (0620 heal適用予測 vs 本番スコア corr) = {cv:.4f}  (要≥0.95)")
    if cv < 0.95:
        print("★NG★ H経路が健全日で本番を再現できない → 実装バグ (exit 2)")
        return 2
    print("Phase 2 PASS")

    # ---- Phase 3: 校正ゲート ----
    print("\n--- Phase 3: 校正ゲート (アームD vs 台帳 8/2 settled) ---")
    led = pd.read_csv(os.path.join(BASE, "data", "cumulative_results.csv"), dtype=str)
    led = led[(led["date"] == "20260802") & (led["status"] == "settled")]
    led = led[led["bet_type"] == "trio"]
    match = tot = 0
    mism = []
    dumps02 = {}
    for fp in sorted(glob.glob(os.path.join(DUMP, "20260802", "*.parquet"))):
        if os.path.getsize(fp) == 0:
            continue
        try:
            d = pd.read_parquet(fp)
        except Exception:
            continue
        if len(d) and "race_id" in d.columns:
            dumps02[str(d["race_id"].iloc[0])] = d
    for _, row in led.iterrows():
        rid = str(row["race_id"])
        if rid not in dumps02:
            continue
        d = dumps02[rid]
        if "スコア" not in d.columns:
            continue
        bets = build_bets_for(d, "スコア")
        if bets is None:
            continue
        tot += 1
        mine = bets_str(bets)
        theirs = str(row["trio_bets_str"]).replace(";", "; ").replace("  ", " ").strip()
        theirs = "; ".join(x.strip() for x in theirs.split(";") if x.strip())
        if mine == theirs:
            match += 1
        else:
            mism.append((rid, mine, theirs))
    rate = match / tot * 100 if tot else 0.0
    print(f"一致 {match}/{tot} = {rate:.1f}%  (ゲート≥90%)")
    for rid, a, b in mism[:5]:
        print(f"  mismatch {rid}\n    mine  : {a}\n    ledger: {b}")
    if rate < 90:
        print("★校正ゲート未達 → 判定に進まず停止 (exit 3)★")
        return 3
    print("校正ゲート PASS")

    # ---- Phase 4: 判定 ----
    print("\n--- Phase 4: 再判定 (D=劣化実際 vs H=健全化) ---")
    R = load_results()
    rows = []
    for dt in DATES:
        for fp in sorted(glob.glob(os.path.join(HEAL, dt, "*.parquet"))):
            try:
                h = pd.read_parquet(fp)
            except Exception:
                continue
            if len(h) < 6 or "race_id" not in h.columns:
                continue
            rid = str(h["race_id"].iloc[0])
            if rid not in R or "スコア" not in h.columns:
                continue
            rn, course, dist, ce, nh = race_meta(h, rid)
            if is_jump(rn, h):
                continue
            cond = classify_condition(nh, dist, ce)
            should, reason = evaluate_bet_decision(rn, course, dist, cond)
            if not should:
                continue
            # D: 当時の実スコア
            bets_d = build_bets_for(h, "スコア")
            # H: live pkl × healed特徴
            h = h.copy()
            h["_predH"] = predict_live(m, h)
            bets_h = build_bets_for(h, "_predH")
            if bets_d is None or bets_h is None:
                continue
            r = R[rid]
            top3 = tuple(sorted([u for u, f in r["fin"].items() if f <= 3]))
            if len(top3) != 3:
                continue
            stake = 700
            ret_d = r["trio"] if top3 in set(map(tuple, bets_d)) else 0
            ret_h = r["trio"] if top3 in set(map(tuple, bets_h)) else 0
            rows.append(dict(date=dt, rid=rid, cond=cond, stake=stake,
                             ret_d=ret_d, ret_h=ret_h,
                             hit_d=int(ret_d > 0), hit_h=int(ret_h > 0)))
    b = pd.DataFrame(rows)
    print(f"賭け対象: {len(b)}R (6日, skip/障害/結果欠落を除く)")
    S = b["stake"].sum()
    roi_d = b["ret_d"].sum() / S * 100
    roi_h = b["ret_h"].sum() / S * 100

    rng_ = np.random.RandomState(42)
    n = len(b)
    dd, hh, diff = [], [], []
    for _ in range(5000):
        idx = rng_.randint(0, n, n)
        s = b["stake"].values[idx].sum()
        d_ = b["ret_d"].values[idx].sum() / s * 100
        h_ = b["ret_h"].values[idx].sum() / s * 100
        dd.append(d_); hh.append(h_); diff.append(h_ - d_)
    ci = lambda a: (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)))
    cid, cih, cidiff = ci(dd), ci(hh), ci(diff)

    print("\n" + "=" * 62)
    print(f"ROI_D (劣化・実際)   = {roi_d:6.1f}%  95%CI[{cid[0]:.1f}, {cid[1]:.1f}]  的中 {b['hit_d'].sum()}/{len(b)} ({b['hit_d'].mean()*100:.1f}%)")
    print(f"ROI_H (健全化)       = {roi_h:6.1f}%  95%CI[{cih[0]:.1f}, {cih[1]:.1f}]  的中 {b['hit_h'].sum()}/{len(b)} ({b['hit_h'].mean()*100:.1f}%)")
    print(f"ペア差 (H−D)         = {roi_h-roi_d:+6.1f}pt 95%CI[{cidiff[0]:+.1f}, {cidiff[1]:+.1f}]")

    # 感度: 最高配当1本除外
    def roi_excl_max(col):
        i = b[col].idxmax()
        bb = b.drop(index=i)
        return bb[col.replace("ret", "ret")].sum() / bb["stake"].sum() * 100, b.loc[i, col]
    rh_ex, mx_h = roi_excl_max("ret_h")
    rd_ex, mx_d = roi_excl_max("ret_d")
    print(f"\n感度 (最高配当1本除外): ROI_H={rh_ex:.1f}% (除外額{mx_h:,}円) / ROI_D={rd_ex:.1f}% (除外額{mx_d:,}円)")

    print("\n日別内訳:")
    g = b.groupby("date").agg(n=("stake", "size"), 投資=("stake", "sum"),
                              払戻D=("ret_d", "sum"), 払戻H=("ret_h", "sum"))
    g["ROI_D"] = (g["払戻D"] / g["投資"] * 100).round(1)
    g["ROI_H"] = (g["払戻H"] / g["投資"] * 100).round(1)
    print(g.to_string())

    # 事前登録判定
    print("\n" + "=" * 62)
    if roi_h >= 100 and cidiff[0] > 0:
        v = "「データ死が主因」支持 → V15復旧+T1v2で再開路線 (paper 2-4週は必須)"
    elif roi_h < 80:
        v = "「市場適応」初実証 → v15r+別施策路線"
    else:
        v = "非決定 → paper延長 (再開はいずれにせよ paper 経由)"
    print(f"事前登録判定: ★{v}★")
    b.to_csv(os.path.join(BASE, "research", "v15r", "rejudge_races.csv"),
             index=False, encoding="utf-8-sig")
    print("保存: research/v15r/rejudge_races.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
