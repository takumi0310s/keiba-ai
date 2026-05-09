"""Session #67 A2: JRDB SED + HJC から 5/9 全 36R 結果 + 払戻取得.

netkeiba は 16:50 以降も HTTP 400 で blocked、 JRDB 経由で代替。

入力:
  data/jrdb/extracted/Sed/SED260509.txt — 着順 (495 行 = 36 R × 出走馬)
  data/jrdb/extracted/Hjc/HJC260509.txt — 払戻 (36 行 = 36 R)

出力:
  data/results/20260509_results.csv

仕様:
  SED 場(1-2)+年(3-4)+回(5)+日(6)+R(7-8)+馬番(9-10)+着順(141-142)+異常(143)
  HJC 同 1-8 + 三連複払戻 (近似: 三連単では複雑なので 三連複セット内払戻位置を抽出)
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

BASE = Path(r"C:/Users/takum/keiba-ai")

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def race_id_to_jrdb_key(race_id: str) -> str:
    """race_id 202604010312 → JRDB key 04261312 (場+年2桁+回+日+R)."""
    rid = str(race_id)
    if len(rid) != 12:
        return ""
    yyyy = rid[0:4]; cc = rid[4:6]; kk = rid[6:8]; dd = rid[8:10]; rr = rid[10:12]
    yy = yyyy[-2:]
    k = kk.lstrip("0") or "0"
    d = dd.lstrip("0") or "0"
    if len(k) > 1 or len(d) > 1:
        return ""
    return f"{cc}{yy}{k}{d}{rr}"


def parse_sed(sed_path: Path) -> dict[str, list[dict]]:
    """SED → {race_key: [{umaban, finish, abnormal}]}."""
    result: dict[str, list[dict]] = {}
    with open(sed_path, "rb") as f:
        for line in f:
            if len(line) < 145:
                continue
            try:
                key = line[0:8].decode("shift_jis", errors="replace")
                umaban = int(line[8:10].decode("shift_jis", errors="replace") or 0)
                finish_b = line[140:142].decode("shift_jis", errors="replace").strip()
                abnormal = line[142:143].decode("shift_jis", errors="replace").strip()
            except Exception:
                continue
            try:
                finish = int(finish_b) if finish_b.isdigit() else 0
            except Exception:
                finish = 0
            result.setdefault(key, []).append({
                "umaban": umaban,
                "finish": finish,
                "abnormal": abnormal,
            })
    return result


def parse_hjc(hjc_path: Path) -> dict[str, dict]:
    """HJC → {race_key: {tansho, fukusho, umaren, wide, sanrenpuku, sanrentan}}.

    HJC レコードは固定長で、 レイアウト:
    場(2)+年(2)+回(1)+日(1)+R(2) = 8 byte
    その後、 各券種の (馬番組合せ + 払戻金) の繰り返し。

    JRDB HJC 仕様 (簡略): 単勝(7byte)→複勝(...)→...→三連複(複数)→三連単
    実フォーマット解析: 数字パターンから 三連複 払戻 を抽出する正規表現方式 (rough)。
    """
    result: dict[str, dict] = {}

    # 仕様 (JRDB v4): 1-8 race key, 9-15 単勝馬番, 16-22 単勝払戻, ...
    # 複雑なので 大まかに位置で取る。 v4 spec:
    # 単勝: 馬番1(2)+払戻金(7) = 9-10 / 11-17
    # 複勝: 馬番1(2)+払戻(7), 馬番2(2)+払戻(7), 馬番3(2)+払戻(7)
    # 馬連: 馬番1(2)+馬番2(2)+払戻(7)
    # ワイド: 3 通り
    # 馬単: 馬番1(2)+馬番2(2)+払戻(7)
    # 三連複: 馬番1(2)+馬番2(2)+馬番3(2)+払戻(7)
    # 三連単: 同様

    # 簡易抽出: 行全体を空白で split せず、 固定オフセットで読む
    # 但し JRDB ハライ戻し file は版/年で format 揺れあり。
    # ここでは 確実な 三連複/三連単 の数字 7 桁 + 末尾 0 (?) パターンを正規表現で取る。

    with open(hjc_path, "rb") as f:
        for line in f:
            if len(line) < 50:
                continue
            try:
                text = line.decode("shift_jis", errors="replace")
            except Exception:
                continue
            key = text[0:8]

            d = {"tansho": 0, "fukusho": 0, "umaren": 0, "wide": 0,
                 "sanrenpuku": 0, "sanrentan": 0,
                 "sanrenpuku_nums": "", "sanrentan_nums": ""}

            # 単勝 払戻: byte 11-17 (0-indexed 10:17) — 7桁 ZZZZZZ9
            try:
                d["tansho"] = int(text[10:17].strip() or 0)
            except Exception:
                pass

            # 三連複 / 三連単 探索: 末尾 0 (filler) 直前の 7 桁数字 を取る
            # 複雑なので: line を分析して 「3 つの 2 桁数字 + 4-7 桁数字」 パターンを捕捉
            # JRDB HJC 三連複部分: 馬番1(2)+馬番2(2)+馬番3(2)+払戻(7)
            # 1 RACE に 1 三連複 (普通) + 三連単 1 ... (special 馬単 etc)

            # 末端 ~50 byte が 三連単 フィールド付近の傾向
            # ここでは regex で 「(\d{2})(\d{2})(\d{2})\s*(\d{4,8})」 を全 match して候補列挙
            tail = text[200:]  # 後半
            cand_3rp = re.findall(r"(\d{2})(\d{2})(\d{2})\s+(\d{4,8})", tail)
            # 三連複: 馬番昇順 + 払戻 4-7 桁
            for c in cand_3rp:
                a, b, cc, p = c
                ua, ub, uc = int(a), int(b), int(cc)
                if ua < ub < uc and 100 <= int(p) <= 99999990:
                    payout = int(p)
                    if payout > d["sanrenpuku"]:
                        d["sanrenpuku"] = payout
                        d["sanrenpuku_nums"] = f"{ua}-{ub}-{uc}"

            # 三連単 候補: 馬番 順番任意 + 払戻
            # 三連単 払戻は通常 三連複 より大、 それで判定
            cand_3rt = re.findall(r"(\d{2})(\d{2})(\d{2})\s+(\d{6,9})", tail)
            for c in cand_3rt:
                a, b, cc, p = c
                ua, ub, uc = int(a), int(b), int(cc)
                # 三連単 = 順番ありの馬番組
                if 1 <= ua <= 18 and 1 <= ub <= 18 and 1 <= uc <= 18 and ua != ub and ub != uc and ua != uc:
                    payout = int(p)
                    if payout > d["sanrentan"] and payout > d["sanrenpuku"]:
                        d["sanrentan"] = payout
                        d["sanrentan_nums"] = f"{ua}-{ub}-{uc}"

            result[key] = d
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--date", default="20260509")
    args = p.parse_args()

    yymmdd = args.date[2:]
    sed_path = BASE / "data" / "jrdb" / "extracted" / "Sed" / f"SED{yymmdd}.txt"
    hjc_path = BASE / "data" / "jrdb" / "extracted" / "Hjc" / f"HJC{yymmdd}.txt"

    if not sed_path.exists():
        print(f"[FAIL] {sed_path} 不在", file=sys.stderr)
        sys.exit(1)

    sed_data = parse_sed(sed_path)
    hjc_data = parse_hjc(hjc_path) if hjc_path.exists() else {}
    print(f"[parse] SED races: {len(sed_data)}, HJC races: {len(hjc_data)}")

    pred_csv = BASE / "data" / "daily_predictions" / f"{args.date}.csv"
    import pandas as pd
    pred_df = pd.read_csv(pred_csv, dtype=str)

    out_dir = BASE / "data" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"{args.date}_results.csv"

    fields = ["race_id", "course", "race_num", "race_name", "num_horses",
              "finish_1", "finish_2", "finish_3",
              "trio_nums", "umaren_nums",
              "payout_tansho", "payout_umaren", "payout_trio", "payout_sanrentan",
              "fetch_status"]
    rows = []
    n_ok = n_fail = 0

    for _, r in pred_df.iterrows():
        rid = str(r.get("race_id", ""))
        key = race_id_to_jrdb_key(rid)
        sed = sed_data.get(key, [])
        hjc = hjc_data.get(key, {})

        row = {f: "" for f in fields}
        row["race_id"] = rid
        row["course"] = str(r.get("course", ""))
        row["race_num"] = str(r.get("race_num", ""))
        row["race_name"] = str(r.get("race_name", ""))
        row["num_horses"] = str(r.get("num_horses", ""))
        row["fetch_status"] = "fail"
        row["payout_tansho"] = 0
        row["payout_umaren"] = 0
        row["payout_trio"] = 0
        row["payout_sanrentan"] = 0

        if sed:
            ranked = sorted([s for s in sed if s["finish"] >= 1], key=lambda x: x["finish"])
            if len(ranked) >= 3:
                row["finish_1"] = ranked[0]["umaban"]
                row["finish_2"] = ranked[1]["umaban"]
                row["finish_3"] = ranked[2]["umaban"]
                row["trio_nums"] = "-".join(str(x["umaban"]) for x in sorted(ranked[:3], key=lambda x: x["umaban"]))
                row["umaren_nums"] = "-".join(str(x["umaban"]) for x in sorted(ranked[:2], key=lambda x: x["umaban"]))
                row["fetch_status"] = "ok"
                if hjc:
                    row["payout_tansho"] = hjc.get("tansho", 0)
                    row["payout_trio"] = hjc.get("sanrenpuku", 0)
                    row["payout_sanrentan"] = hjc.get("sanrentan", 0)
                n_ok += 1
                print(f"  [OK] {row['course']} R{row['race_num']}: {row['trio_nums']} (trio ¥{row['payout_trio']:,})")
            else:
                n_fail += 1
        else:
            n_fail += 1

        rows.append(row)

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print(f"\n=== summary ===")
    print(f"  total: {len(rows)}, ok: {n_ok}, fail: {n_fail}")
    print(f"  out: {out_csv.relative_to(BASE)}")


if __name__ == "__main__":
    main()
