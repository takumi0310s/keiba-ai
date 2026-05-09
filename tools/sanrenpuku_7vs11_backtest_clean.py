"""Session #69: 三連複 7 点 vs 11 点 leak-free backtest.

★ ユーザー指摘 (リーク risk) を完全反映 ★

source:
- data/daily_predictions/{date}.csv (production score、 leak-free)
- data/daily_results/{date}.csv (実 trio_payout、 trio_result)

formation:
- 7 点 (現状 V15 baseline): top1-{top2,3}-{top3-6}
- 11 点 (本 Session): top1 軸 box5 (10) + top2-top3-top4 (1) = 11
  ※ top7-10 は production 未保存 = LEAK risk のため不採用 (user prompt から変更)

leak-free 保証:
- top1-3 は production saved
- top4-6 は trio_bets パターンから出現回数で leak-free 復元
- top7+ は使用しない

usage:
  python tools/sanrenpuku_7vs11_backtest_clean.py
  python tools/sanrenpuku_7vs11_backtest_clean.py --dates 20260411,20260412
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
PRED_DIR = ROOT / "data" / "daily_predictions"
RES_DIR = ROOT / "data" / "daily_results"
PAYOUTS_CSV = ROOT / "data" / "jra_payouts.csv"
OUT_CSV = ROOT / "data" / "v18" / "session_69_per_race.csv"
OUT_JSON = ROOT / "data" / "v18" / "session_69_metrics.json"

# race_id course code → JRA course name (jra_payouts.csv 表記)
COURSE_CODE_NAME = {
    "01": "札幌", "02": "函館", "03": "福島", "04": "新潟",
    "05": "東京", "06": "中山", "07": "中京", "08": "京都",
    "09": "阪神", "10": "小倉",
}


def load_payouts_lookup(dates: list) -> dict:
    """jra_payouts.csv → {race_id: (trio_result_tuple, trio_payout)}.

    race_id format: YYYY + course(2) + kai(2) + nichi(2) + race_num(2)
    """
    p = pd.read_csv(PAYOUTS_CSV, encoding="utf-8-sig", dtype=str)
    p = p[p["race_date"].isin(dates)].copy()
    name2code = {v: k for k, v in COURSE_CODE_NAME.items()}
    out = {}
    for _, r in p.iterrows():
        course = r["course"]
        cc = name2code.get(course)
        if cc is None:
            continue
        year = r["race_date"][:4]
        rid = f"{year}{cc}{r['kai'].zfill(2)}{r['nichi'].zfill(2)}{r['race_num'].zfill(2)}"
        try:
            nums = tuple(sorted(int(x) for x in str(r["trio_nums"]).split("-")))
            pay = float(r["trio_payout"]) if r["trio_payout"] else 0.0
            if len(nums) == 3:
                out[rid] = (nums, pay)
        except Exception:
            continue
    return out

DEFAULT_DATES = [
    "20260314", "20260315", "20260321",
    "20260411", "20260412", "20260418", "20260419", "20260425",
]
BET_PER_PT = 100  # 100 円/点


def parse_trio_str(s: str) -> tuple | None:
    if not isinstance(s, str):
        return None
    try:
        nums = [int(x) for x in s.replace(",", "-").split("-") if x.strip()]
        return tuple(sorted(nums)) if len(nums) == 3 else None
    except Exception:
        return None


def parse_bets(s: str) -> list:
    if not isinstance(s, str):
        return []
    out = []
    for b in s.split(";"):
        try:
            t = tuple(sorted([int(x) for x in b.strip().split("-") if x.strip()]))
            if len(t) == 3:
                out.append(t)
        except Exception:
            pass
    return out


def reconstruct_top16(top1: int, top2: int, top3: int, bets: list) -> list:
    """7 点 trio_bets から top4-6 を出現回数で逆算 (leak-free)."""
    counter: dict = {}
    for tb in bets:
        for n in tb:
            counter[n] = counter.get(n, 0) + 1
    rest = sorted(
        [n for n in counter if n not in (top1, top2, top3)],
        key=lambda x: -counter.get(x, 0),
    )
    while len(rest) < 3:
        rest.append(0)
    return [top1, top2, top3] + rest[:3]


def fmt_v15_baseline_7(top6: list) -> list:
    """V15 既存 7 点: top1-{top2,3}-{top3-6}."""
    n1, n2, n3 = top6[0:3]
    bets = []
    for x in top6[2:6]:
        if x and x not in (n1, n2):
            bets.append(tuple(sorted([n1, n2, x])))
    for x in top6[3:6]:
        if x and x not in (n1, n3):
            bets.append(tuple(sorted([n1, n3, x])))
    return list(set(bets))


def fmt_top16_11(top6: list) -> list:
    """11 点: top1 軸 box5 (10) + top2-top3-top4 (1)."""
    n1 = top6[0]
    others = [n for n in top6[1:6] if n and n != n1]
    bets = []
    for a, b in itertools.combinations(others, 2):
        bets.append(tuple(sorted([n1, a, b])))
    if len(top6) >= 4 and all(top6[i] for i in (1, 2, 3)):
        t2, t3, t4 = top6[1], top6[2], top6[3]
        if len({t2, t3, t4}) == 3:
            bets.append(tuple(sorted([t2, t3, t4])))
    return list(set(bets))


def evaluate(formation_fn, top6: list, win_trio: tuple | None,
             trio_pay_100: float) -> tuple:
    """formation 評価 → (n_bets, invest, payout, hit)."""
    bets = formation_fn(top6)
    n = len(bets)
    inv = n * BET_PER_PT
    if win_trio and win_trio in bets:
        pay = trio_pay_100 * (BET_PER_PT / 100.0)  # 100円換算 → BET_PER_PT 単位
        return n, inv, pay, 1
    return n, inv, 0.0, 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dates", default=",".join(DEFAULT_DATES))
    args = ap.parse_args()
    dates = [d.strip() for d in args.dates.split(",") if d.strip()]

    print(f"[load] jra_payouts.csv (true 三連複 配当 lookup)")
    payout_lookup = load_payouts_lookup(dates)
    print(f"  loaded {len(payout_lookup)} race payouts")

    rows = []
    skip_summary = []
    miss_payout = 0
    for d in dates:
        pp = PRED_DIR / f"{d}.csv"
        rp = RES_DIR / f"{d}.csv"
        if not pp.exists() or not rp.exists():
            skip_summary.append(f"{d}: pred={pp.exists()} res={rp.exists()}")
            continue

        pred = pd.read_csv(pp, dtype={"race_id": str}).drop_duplicates("race_id", keep="last")
        res = pd.read_csv(rp, encoding="utf-8-sig", dtype={"race_id": str})
        res = res.drop_duplicates("race_id", keep="last")
        if "status" in res.columns:
            res = res[res["status"] == "settled"]

        for _, r in res.iterrows():
            rid = r["race_id"]
            p = pred[pred["race_id"] == rid]
            if len(p) == 0:
                continue
            pp_row = p.iloc[0]
            try:
                t1 = int(pp_row["top1_num"])
                t2 = int(pp_row["top2_num"])
                t3 = int(pp_row["top3_num"])
            except Exception:
                continue
            bets_v15 = parse_bets(pp_row.get("trio_bets", ""))
            top6 = reconstruct_top16(t1, t2, t3, bets_v15)

            # 真の trio_result + 配当 (jra_payouts.csv 由来)
            if rid not in payout_lookup:
                miss_payout += 1
                continue
            win, trio_pay = payout_lookup[rid]

            n7, inv7, pay7, hit7 = evaluate(fmt_v15_baseline_7, top6, win, trio_pay)
            n11, inv11, pay11, hit11 = evaluate(fmt_top16_11, top6, win, trio_pay)

            rows.append({
                "date": d, "race_id": rid,
                "course": pp_row.get("course", ""),
                "race_num": pp_row.get("race_num", ""),
                "condition": pp_row.get("condition", ""),
                "num_horses": pp_row.get("num_horses", ""),
                "distance": pp_row.get("distance", ""),
                "surface": pp_row.get("surface", ""),
                "top6": "-".join(str(x) for x in top6),
                "trio_result": "-".join(str(x) for x in win) if win else "",
                "trio_pay_100": trio_pay,
                "n7": n7, "inv7": inv7, "pay7": pay7, "hit7": hit7,
                "n11": n11, "inv11": inv11, "pay11": pay11, "hit11": hit11,
            })

    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    n_r = len(df)
    print(f"\n=== Session #69 7 点 vs 11 点 backtest (leak-free) ===")
    print(f"期間: {dates[0]} - {dates[-1]} ({n_r} R)")
    print(f"skip dates: {skip_summary}")
    print(f"missing payout entries: {miss_payout}")
    print()

    if n_r == 0:
        print("[NO DATA] 結果 0 件、 集計不可")
        return

    def summarize(label, n_col, inv_col, pay_col, hit_col):
        n_bets_avg = df[n_col].mean()
        inv = df[inv_col].sum()
        pay = df[pay_col].sum()
        prof = pay - inv
        roi = pay / inv * 100 if inv else 0
        hits = df[hit_col].sum()
        hit_rate = hits / n_r * 100 if n_r else 0
        avg_pay_hit = pay / hits if hits else 0
        return {
            "label": label, "n_races": n_r, "n_bets_avg": float(n_bets_avg),
            "total_invest": int(inv), "total_payout": int(pay),
            "profit": int(prof), "roi_pct": float(roi),
            "hits": int(hits), "hit_rate_pct": float(hit_rate),
            "avg_payout_per_hit": float(avg_pay_hit),
        }

    s7 = summarize("7点 (V15 baseline)", "n7", "inv7", "pay7", "hit7")
    s11 = summarize("11点 (top1-6 拡張)", "n11", "inv11", "pay11", "hit11")

    for s in (s7, s11):
        print(f"--- {s['label']} ---")
        print(f"  n_R={s['n_races']:4d}  avg_bets={s['n_bets_avg']:.1f}")
        print(f"  invest={s['total_invest']:>9,}円  payout={s['total_payout']:>9,}円")
        print(f"  profit={s['profit']:>+9,}円  ROI={s['roi_pct']:6.2f}%")
        print(f"  hits={s['hits']:>3d}  hit_rate={s['hit_rate_pct']:5.2f}%")
        print(f"  avg_payout_per_hit={s['avg_payout_per_hit']:>7,.0f}円")
        print()

    delta_roi = s11["roi_pct"] - s7["roi_pct"]
    delta_hits = s11["hits"] - s7["hits"]
    delta_profit = s11["profit"] - s7["profit"]
    ev_per_r_7 = s7["profit"] / n_r if n_r else 0
    ev_per_r_11 = s11["profit"] / n_r if n_r else 0
    delta_ev_r = ev_per_r_11 - ev_per_r_7

    print("=== 比較 ===")
    print(f"  hit delta:    {delta_hits:+d} (7→{s7['hits']} / 11→{s11['hits']})")
    print(f"  profit delta: {delta_profit:+,}円")
    print(f"  ROI delta:    {delta_roi:+.2f}pt")
    print(f"  EV/R (7点):   {ev_per_r_7:+.1f}円")
    print(f"  EV/R (11点):  {ev_per_r_11:+.1f}円")
    print(f"  EV/R delta:   {delta_ev_r:+.1f}円/R")

    # bootstrap CI (1000 iter) on ROI delta
    import numpy as np
    rng = np.random.default_rng(42)
    boot = []
    n = len(df)
    for _ in range(2000):
        idx = rng.integers(0, n, n)
        s = df.iloc[idx]
        i7 = s["inv7"].sum()
        p7 = s["pay7"].sum()
        i11 = s["inv11"].sum()
        p11 = s["pay11"].sum()
        r7 = p7 / i7 * 100 if i7 else 0
        r11 = p11 / i11 * 100 if i11 else 0
        boot.append(r11 - r7)
    boot = np.array(boot)
    ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
    p_pos = (boot > 0).mean()

    print()
    print(f"  bootstrap CI95 (Δ ROI): [{ci_lo:+.2f}, {ci_hi:+.2f}] pt  (n=2000)")
    print(f"  P(11点 > 7点 in ROI): {p_pos*100:.1f}%")

    metrics = {
        "period": {"start": dates[0], "end": dates[-1], "dates": dates,
                    "skipped": skip_summary, "n_races": n_r},
        "7pt": s7, "11pt": s11,
        "delta": {
            "hits": int(delta_hits),
            "profit_yen": int(delta_profit),
            "roi_pt": float(delta_roi),
            "ev_per_race_yen": float(delta_ev_r),
            "ci95_roi_pt": [float(ci_lo), float(ci_hi)],
            "p_11pt_better": float(p_pos),
        },
        "leak_free": {
            "source": "data/daily_predictions/ (production saved)",
            "top4_6_recovery": "trio_bets occurrence count (deterministic)",
            "top7_10_used": False,
            "rationale": "user指摘 (リーク risk) 反映、 top1-6 のみで 11 点設計",
        },
    }
    OUT_JSON.write_text(json.dumps(metrics, indent=2, ensure_ascii=False),
                         encoding="utf-8")
    print(f"\nsaved: {OUT_CSV}")
    print(f"saved: {OUT_JSON}")


if __name__ == "__main__":
    main()
