# -*- coding: utf-8 -*-
"""
リーク監査タスク3: rsim.ag3 と実際の上がりの相関を日付別に分解。
狙い: 「事後再計算(古い週ほど高相関)」か「取得時点で全ファイル再生成済(全週一様)」かを切り分け。
★8/9 は 8/8 取得時点で未発走 → 天然の対照(prospective)。ここが低相関なら事後再計算が確定★
入力: research/ruiji/raw_results/<date>.json (agari 込み, fetch_results.py) + raw/ ツール
"""
import sys, os, glob, json, unicodedata
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd
from tool_filters import R_DR, R_AG3

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "raw_results")
PLACE = {'札幌':'01','函館':'02','福島':'03','新潟':'04','東京':'05',
         '中山':'06','中京':'07','京都':'08','阪神':'09','小倉':'10'}
def nz(s): return unicodedata.normalize("NFKC", str(s or "")).strip()


def load_actual_agari():
    m = {}
    for fp in glob.glob(os.path.join(RES, "*.json")):
        date = os.path.basename(fp)[:8]
        for r in json.load(open(fp, encoding="utf-8")):
            rid = str(r["race_id"]); place = rid[4:6]; rno = int(rid[-2:])
            for ub_s, ag in (r.get("agari") or {}).items():
                m[(date, place, rno, int(ub_s))] = ag
    return m


def main():
    act = load_actual_agari()
    print(f"実上がりレコード: {len(act)}")
    rows = []
    for fp in glob.glob(os.path.join(HERE, "raw/*/scatter_*.json")):
        d = json.load(open(fp, encoding="utf-8")); rc = d["race"]
        date = str(rc["date"]); place = PLACE.get(nz(rc["v"])); rno = int(rc["no"])
        ag3 = (d.get("rsim") or {}).get("ag3") or {}
        for h in d.get("horses", []):
            num = str(h.get("num"))
            if not num.isdigit(): continue
            pred = ag3.get(num)
            a = act.get((date, place, rno, int(num)))
            # 過去平均上がり(当該日前)
            pa = [r[R_AG3] for r in h.get("runs", [])
                  if len(r) >= 15 and str(r[R_DR]).isdigit() and int(r[R_DR]) < int(date) and r[R_AG3] is not None]
            if pred is not None and a is not None:
                rows.append(dict(date=date, pred=pred, actual=a,
                                 past_avg=(np.mean(pa) if pa else np.nan)))
    df = pd.DataFrame(rows)
    print(f"pred×actual 突合: {len(df)}  (日数 {df['date'].nunique()})\n")

    print(f"{'日付':>10} {'n':>4} {'corr(pred,act)':>14} {'corr(dev,dev)':>13} {'|pred-act|':>10} {'|past-act|':>10}")
    for dt in sorted(df["date"].unique()):
        s = df[df["date"] == dt]
        c1 = s["pred"].corr(s["actual"])
        sd = s.dropna(subset=["past_avg"])
        if len(sd) > 5:
            cdev = (sd["pred"] - sd["past_avg"]).corr(sd["actual"] - sd["past_avg"])
            pastmae = (sd["past_avg"] - sd["actual"]).abs().mean()
        else:
            cdev, pastmae = np.nan, np.nan
        mae = (s["pred"] - s["actual"]).abs().mean()
        tag = "  ← 8/9=発走前取得(対照)" if dt == "20260809" else ("  ← 8/8" if dt == "20260808" else "")
        print(f"{dt:>10} {len(s):>4} {c1:>14.3f} {cdev:>13.3f} {mae:>10.2f} {pastmae:>10.2f}{tag}")

    print("\n判定指針:")
    print(" - corr(dev,dev)≈0.8級=そのレース固有の上がりズレを予測が知る=リーク")
    print(" - 全週一様に高相関 → 取得時点(8/8-9)で全ファイル再生成済")
    print(" - 8/9(発走前)だけ低相関 → 事後再計算が確定(発走後に結果を注入)")


if __name__ == "__main__":
    main()
