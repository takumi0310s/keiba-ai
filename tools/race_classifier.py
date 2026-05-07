"""race_name → クラス自動分類 (Session #40 A2).

5/8 21:00 出馬表確定後、 各 race_name から自動でクラス判定し、 採用 R を抽出。
戦略⑦ + 案B改 5/9 適用版:
  - 採用: 1勝クラス (12R 戦略)
  - 除外: G1/G2/G3、 オープン特別、 平場特別 (06_特別)、 新馬戦、 未勝利

usage:
  python tools/race_classifier.py --csv data/daily_predict_lite_20260509.csv
  # → 採用 R を Discord 通知 (channel: bets)
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from typing import List, Tuple

import pandas as pd

BASE = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))


# ===== 分類 =====

# クラス pattern (優先順位順、 上位 match で確定)
CLASS_PATTERNS = [
    # 重賞
    (r"\bG1\b|（G1）|\(G1\)|GⅠ", "G1",        "重賞 G1"),
    (r"\bG2\b|（G2）|\(G2\)|GⅡ", "G2",        "重賞 G2"),
    (r"\bG3\b|（G3）|\(G3\)|GⅢ", "G3",        "重賞 G3"),
    (r"\bL\b|（L）|リステッド",     "L",          "リステッド"),
    # 特別
    (r"オープン特別|OP特別|オープン",   "OP",          "オープン特別"),
    (r"特別\(?S\)?|S特別",         "S",           "特別 S"),
    (r"3勝クラス|（3勝クラス）",     "3勝",         "3勝クラス"),
    (r"2勝クラス|（2勝クラス）",     "2勝",         "2勝クラス"),
    (r"1勝クラス|（1勝クラス）",     "1勝",         "1勝クラス"),
    # 平場
    (r"未勝利",                       "未勝利",      "未勝利"),
    (r"新馬",                         "新馬",        "新馬戦"),
    # フォールバック
    (r"特別",                         "06_特別",    "平場特別 (G/L 以外、 06_)"),
]


def classify_race(race_name: str) -> Tuple[str, str]:
    """race_name → (class_code, description)."""
    if not isinstance(race_name, str): return ("UNKNOWN", "不明")
    s = race_name.strip()
    for pat, code, desc in CLASS_PATTERNS:
        if re.search(pat, s):
            return (code, desc)
    return ("UNKNOWN", "不明 (regex 未match)")


# ===== 採用判定 (戦略⑦ + 案B改 5/9) =====

ACCEPT_CLASSES = {"1勝"}  # 案B改: 1勝クラスのみ
EXCLUDE_CLASSES_REASON = {
    "G1": "重賞 G1 (BT サンプル少)",
    "G2": "重賞 G2 (同上)",
    "G3": "重賞 G3 (同上)",
    "L":  "リステッド (BT サンプル少)",
    "OP": "オープン特別 (BT 不安定)",
    "06_特別": "平場特別 (-9,470円損失源、 戦略⑦)",
    "新馬": "新馬戦 (sib_*_exp Phase 3 後検討)",
    "未勝利": "未勝利 (BT 強化未着手)",
    "S":  "特別 S (重賞前哨戦、 ROI 不安定)",
    "3勝": "3勝クラス (案B改 1勝固定の方針)",
    "2勝": "2勝クラス (同上)",
    "UNKNOWN": "分類不能 → 安全側 除外",
}


def decide_accept(class_code: str, course: str = "", num_horses: int = 0) -> Tuple[bool, str]:
    if class_code in ACCEPT_CLASSES:
        # 戦略⑦ filter: 京都 / 条件E / 条件B 除外は分類後 別 layer で適用
        if course == "京都":
            return (False, "戦略⑦: 京都除外 (course_renovated 安定待ち)")
        if 0 < num_horses <= 7:
            return (False, "戦略⑦: 条件E (頭数<=7) 除外")
        return (True, "1勝クラス + 戦略⑦ 通過")
    return (False, EXCLUDE_CLASSES_REASON.get(class_code, "クラス除外"))


# ===== CLI =====

def main():
    p = argparse.ArgumentParser(description="race_name → クラス分類 + 5/9 採用判定")
    p.add_argument('--csv', help='input csv (race_name 列必須)')
    p.add_argument('--name', help='single race_name 判定 (test)')
    p.add_argument('--course', default='', help='開催場 (戦略⑦ 京都除外用)')
    p.add_argument('--num-horses', type=int, default=0, help='頭数 (条件E 除外用)')
    p.add_argument('--out', default=None, help='out csv (省略時 stdout)')
    args = p.parse_args()

    if args.name:
        code, desc = classify_race(args.name)
        accept, reason = decide_accept(code, args.course, args.num_horses)
        print(f"race_name: {args.name!r}")
        print(f"  course={args.course!r} num_horses={args.num_horses}")
        print(f"  → class_code: {code}  ({desc})")
        print(f"  → ACCEPT: {accept}   ({reason})")
        return

    if not args.csv:
        print("[!] --csv または --name を指定してください")
        sys.exit(1)

    df = pd.read_csv(os.path.join(BASE, args.csv))
    df['class_code'], df['class_desc'] = zip(*df['race_name'].apply(classify_race))
    course_col = 'course' if 'course' in df.columns else None
    nh_col = 'num_horses' if 'num_horses' in df.columns else None
    accepts = []
    reasons = []
    for _, rec in df.iterrows():
        c = rec[course_col] if course_col else ''
        nh = int(rec[nh_col]) if (nh_col and pd.notna(rec[nh_col])) else 0
        a, r = decide_accept(rec['class_code'], c, nh)
        accepts.append(a)
        reasons.append(r)
    df['accept_5_9'] = accepts
    df['decide_reason'] = reasons

    n_total = len(df)
    n_accept = sum(accepts)
    print(f"[race_cls] N={n_total}, accept (5/9)={n_accept} ({n_accept/n_total*100:.1f}%)")
    print(df[['race_name', 'class_code', 'accept_5_9', 'decide_reason']].head(20).to_string())

    if args.out:
        out_path = os.path.join(BASE, args.out)
        df.to_csv(out_path, index=False, encoding='utf-8')
        print(f"  written: {out_path}")


if __name__ == '__main__':
    main()
