# -*- coding: utf-8 -*-
"""再開ゲート レポート (docs/RESUMPTION_PLAYBOOK_2026.md の月次判定会用)。

事前登録 (docs/PRE_REGISTRATION_RESUMPTION_GATE.md) のメインゲート/副判定を
cumulative_results.csv (paper期間) から機械的に算出する。判定基準の変更は禁止。

usage:
  python tools/resumption_gate_report.py --start 20260906            # 本判定期間の開始日から
  python tools/resumption_gate_report.py --start 20260815 --ref     # 夏の参考記録
"""
import argparse, os, sys
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
CENTRAL4 = ["中山", "阪神", "東京", "京都"]


def boot_ci(inv, ret, n=5000, seed=42):
    rng = np.random.RandomState(seed)
    inv = np.asarray(inv, dtype=float); ret = np.asarray(ret, dtype=float)
    k = len(inv)
    if k == 0:
        return (np.nan, np.nan)
    rois = []
    for _ in range(n):
        idx = rng.randint(0, k, k)
        s = inv[idx].sum()
        rois.append(ret[idx].sum() / s * 100 if s else np.nan)
    return float(np.nanpercentile(rois, 2.5)), float(np.nanpercentile(rois, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", default="29991231")
    ap.add_argument("--ref", action="store_true", help="参考記録モード(判定文言を出さない)")
    a = ap.parse_args()
    cr = pd.read_csv(os.path.join(BASE, "data", "cumulative_results.csv"),
                     dtype=str, encoding="utf-8-sig")
    w = cr[(cr["status"] == "settled") & (cr["date"].between(a.start, a.end))].copy()
    for c in ["investment", "actual_payout", "race_num"]:
        w[c] = pd.to_numeric(w[c], errors="coerce")
    w = w[w["investment"] > 0].copy()
    w["hit"] = (w["actual_payout"] > 0).astype(int)
    # course が NaN の行 (6/20+ 台帳) は race_id の場コードから導出
    _PLACE = {"01": "札幌", "02": "函館", "03": "福島", "04": "新潟", "05": "東京",
              "06": "中山", "07": "中京", "08": "京都", "09": "阪神", "10": "小倉"}
    _fb = w["race_id"].astype(str).str[4:6].map(_PLACE)
    w["course"] = w["course"].where(w["course"].notna() & (w["course"] != "nan"), _fb)
    # メインゲート母集団 = 中央4場 (戦略⑦は台帳時点で適用済み)
    m = w[w["course"].isin(CENTRAL4)].copy()
    n = len(m)
    if n == 0:
        # 夏開催(札幌/新潟/中京)等 中央4場なし → 参考として全場のみ表示
        print(f"期間 {a.start}-{a.end}: 中央4場 0R (本判定対象なし)。全場 {len(w)}R の参考内訳:")
        if len(w):
            g = w.groupby("course").agg(n=("investment", "size"), 投資=("investment", "sum"),
                                        払戻=("actual_payout", "sum"))
            g["ROI%"] = (g["払戻"] / g["投資"] * 100).round(1)
            print(g[["n", "ROI%"]].to_string())
        return 0
    inv, ret = m["investment"].values, m["actual_payout"].values
    roi = ret.sum() / inv.sum() * 100
    lo, hi = boot_ci(inv, ret)
    # 感度: 最高払戻1本除外
    i = int(np.argmax(ret))
    roi_ex = (ret.sum() - ret[i]) / (inv.sum() - inv[i]) * 100 if n > 1 else np.nan
    print("=" * 60)
    print(f"再開ゲート レポート  期間 {a.start}〜{a.end}  (中央4場・⑦適用後)")
    print("=" * 60)
    print(f"n = {n}R   ROI = {roi:.1f}%   95%CI[{lo:.1f}, {hi:.1f}]   的中 {m['hit'].mean()*100:.1f}%")
    print(f"感度(最高払戻¥{int(ret[i]):,}除外) = {roi_ex:.1f}%")
    print("\n--- 副判定 (参考・単独GO権なし・3-4月仮説の前向き検証) ---")
    hs = m[m["course"] == "阪神"]
    if len(hs):
        print(f"阪神        : n={len(hs):3d}  ROI={hs['actual_payout'].sum()/hs['investment'].sum()*100:6.1f}%  (基準 n≥50 & >110%)")
    r912 = m[m["race_num"] >= 9]
    if len(r912):
        print(f"R9-12(特別/OP): n={len(r912):3d}  ROI={r912['actual_payout'].sum()/r912['investment'].sum()*100:6.1f}%  (基準 n≥40 & >120%)")
    print("\n--- 場別内訳 ---")
    g = w.groupby("course").agg(n=("investment", "size"), 投資=("investment", "sum"),
                                払戻=("actual_payout", "sum"))
    g["ROI%"] = (g["払戻"] / g["投資"] * 100).round(1)
    print(g[["n", "ROI%"]].to_string())
    if not a.ref:
        print("\n--- 事前登録ゲート判定 (PRE_REGISTRATION_RESUMPTION_GATE.md) ---")
        if n < 120:
            print(f"n={n} < 120 → 判定持ち越し (充足待ち)")
        elif roi > 100 and lo > 85 and roi_ex > 90:
            print("★3条件全達 (点>100 / CI下限>85 / 感度点>90) → GO検討会の開催要件を満たす★")
        elif roi < 90:
            print("★点推定<90% → NO-GO確定 (V15系引退、戦略層/s2b路線へ)★")
        elif roi > 100 and lo <= 85:
            print("点>100 だが CI下限≤85 → n≥200 まで延長 (1回のみ)。n≥200でも跨げば NO-GO")
        else:
            print("90≤点≤100 → 判定継続 (月次判定会で評価。前倒しGO禁止)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
