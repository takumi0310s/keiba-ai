#!/usr/bin/env python
"""JV-Link SE small-scale test + byte offset inspection.

32-bit Python required: C:/Users/takum/jvlink-venv/Scripts/python.exe
(64-bit Python cannot dispatch JV-Link COM objects)

Usage (32-bit venv):
  C:/Users/takum/jvlink-venv/Scripts/python.exe tools/jvlink_se_test.py
  C:/Users/takum/jvlink-venv/Scripts/python.exe tools/jvlink_se_test.py --from 20260401 --max 500
  C:/Users/takum/jvlink-venv/Scripts/python.exe tools/jvlink_se_test.py --dump-bytes
"""
from __future__ import annotations
import argparse
import json
import sys
import os
import csv
from datetime import datetime, timedelta
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# 32-bit Python チェック (エラーメッセージを分かりやすく)
if sys.maxsize > 2**32:
    print("=" * 60)
    print("[ERROR] 64-bit Python で実行されています")
    print("JV-Link COM API は 32-bit Python が必須です。")
    print()
    print("Setup:")
    print("1. Install 32-bit Python from python.org (Windows installer 32-bit)")
    print("2. C:/Python311-32/python.exe -m venv C:/Users/takum/jvlink-venv")
    print("3. C:/Users/takum/jvlink-venv/Scripts/pip install pywin32")
    print("4. C:/Users/takum/jvlink-venv/Scripts/python.exe tools/jvlink_se_test.py")
    print("=" * 60)
    sys.exit(1)


# SE record field definitions (JV-Data 仕様書 v4.7 準拠)
# offset: (start_byte, length, encoding, description)
SE_FIELDS = {
    "record_type":  (0,  2,  "ascii", "レコード種別 'SE'"),
    "data_kbn":     (2,  1,  "ascii", "データ区分"),
    "make_date":    (3,  8,  "ascii", "作成日 YYYYMMDD"),
    "year_holding": (11, 2,  "ascii", "開催年 YY"),
    "month_day":    (13, 4,  "ascii", "開催月日 MMDD"),
    "course_code":  (17, 2,  "ascii", "競馬場コード"),
    "kai":          (19, 2,  "ascii", "開催回"),
    "nichi":        (21, 2,  "ascii", "開催日"),
    "race_num":     (23, 2,  "ascii", "レース番号"),
    "wakuban":      (25, 1,  "ascii", "枠番"),
    "umaban":       (26, 2,  "ascii", "馬番"),
    "horse_id":     (28, 10, "ascii", "血統登録番号"),
    "horse_name":   (38, 36, "sjis",  "馬名"),
    # 以降の offset は仮値 → 実データで確認要 (dump-bytes モード)
    # JV-Data SE 仕様書 PDF で正確な offset を確認すること
    "sex":          (74, 2,  "ascii", "性別コード"),
    "age":          (76, 2,  "ascii", "年齢"),
    # ... (中略: 騎手・調教師・斤量・馬体重等 多数フィールド)
    # ★ 以下は仮 offset (要確認) ★
    "finish_order": (141, 2, "ascii", "着順 (仮 offset)"),
    "finish_time":  (143, 4, "ascii", "走破タイム 10分の1秒 (仮 offset)"),
    "corner1":      (147, 2, "ascii", "第1コーナー通過順 (仮 offset)"),
    "corner2":      (149, 2, "ascii", "第2コーナー通過順 (仮 offset)"),
    "corner3":      (151, 2, "ascii", "第3コーナー通過順 (仮 offset)"),
    "corner4":      (153, 2, "ascii", "第4コーナー通過順 (仮 offset)"),
    "agari3f":      (155, 3, "ascii", "上がり3ハロン 10分の1秒 (仮 offset)"),
}


def _slice(raw: bytes, offset: int, length: int, enc: str) -> str:
    chunk = raw[offset:offset+length]
    if enc == "sjis":
        return chunk.decode("shift_jis", errors="replace").strip()
    return chunk.decode("ascii", errors="replace").strip()


def _parse_time(raw_str: str) -> float | None:
    """走破タイム文字列 (10分の1秒) → 秒 に変換。 例: '1384' → 138.4"""
    try:
        v = int(raw_str.strip())
        if v <= 0:
            return None
        return v / 10.0
    except (ValueError, TypeError):
        return None


def parse_se_record(raw: bytes) -> dict:
    """SE レコード (bytes) → dict。 finish_time / agari3f を含む。"""
    result = {}
    for fname, (offset, length, enc, desc) in SE_FIELDS.items():
        if offset + length > len(raw):
            result[fname] = None
            continue
        result[fname] = _slice(raw, offset, length, enc)

    # 走破タイム → 秒変換
    if result.get("finish_time"):
        result["finish_time_sec"] = _parse_time(result["finish_time"])
    if result.get("agari3f"):
        result["agari3f_sec"] = _parse_time(result["agari3f"])

    result["_raw_len"] = len(raw)
    return result


def dump_byte_structure(raw: bytes, max_display: int = 400):
    """SE raw record の byte 構造を人間可読で表示。 offset 探索用。"""
    print(f"\n=== SE Raw Record (len={len(raw)}) ===")
    print(f"{'Offset':>6}  {'ASCII':30}  {'HEX'}")
    for i in range(0, min(len(raw), max_display), 10):
        chunk = raw[i:i+10]
        asc = chunk.decode("ascii", errors=".")
        hex_s = chunk.hex()
        print(f"  {i:>4}:  {asc:30}  {hex_s}")

    # SJIS 領域 (馬名周辺 offset 38-74) を SJIS decode
    print(f"\n--- SJIS decode (offset 38-74, 馬名推定) ---")
    try:
        sjis_chunk = raw[38:74]
        print(f"  '{sjis_chunk.decode('shift_jis', errors='replace')}'")
    except Exception as e:
        print(f"  SJIS decode error: {e}")


def fetch_se_small(from_date: str, max_records: int = 500) -> list[dict]:
    """SE 小規模取得 (32-bit Python 専用)。"""
    try:
        import win32com.client
    except ImportError:
        print("[ERROR] pywin32 未インストール。pip install pywin32 を実行")
        return []

    fromtime = from_date + "000000"
    print(f"[se_test] JVInit 開始...")

    jv = win32com.client.Dispatch("JVDTLab.JVLink")
    rc = jv.JVInit("UNKNOWN/0.0")
    print(f"[se_test] JVInit rc={rc}")
    if rc != 0:
        print(f"[ERROR] JVInit 失敗 (rc={rc}). JVSetUIProperties で ID/PW を設定してください")
        return []

    # SE は単独 dataspec ではなく RACE パッケージ内の record type
    # JVOpen("RACE", ...) で RA / SE / HR の混合レコードが返る
    # → record 先頭 2 byte で "SE" を filter する
    print(f"[se_test] JVOpen RACE {fromtime} option=4 ...")
    rc2, n_data, n_files, last_time = jv.JVOpen("RACE", fromtime, 1)
    print(f"[se_test] JVOpen rc={rc2}, records={n_data}, files={n_files}")
    if rc2 < 0:
        print(f"[ERROR] JVOpen 失敗 (rc={rc2})")
        jv.JVClose()
        return [], []

    records = []
    raw_samples = []
    n_total = 0
    n_se = 0
    while n_se < max_records:
        rc3, buf, fname = jv.JVRead(2048, "")
        if rc3 == 0:
            break
        if rc3 == -1:
            continue
        if rc3 < 0:
            print(f"[ERROR] JVRead rc={rc3}")
            break
        n_total += 1
        # buf は str (COM BSTR) として返る; bytes に変換
        raw_bytes = buf.encode("cp932", errors="replace") if isinstance(buf, str) else bytes(buf)
        # SE レコードのみフィルタ
        rtype = raw_bytes[:2].decode("ascii", errors="replace")
        if rtype != "SE":
            continue
        records.append(parse_se_record(raw_bytes))
        if len(raw_samples) < 3:
            raw_samples.append(raw_bytes)
        n_se += 1
        if n_se % 100 == 0:
            print(f"  SE: {n_se} / total: {n_total} records...")

    jv.JVClose()
    print(f"[se_test] 完了: SE={len(records)}, total={n_total} records")
    return records, raw_samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="from_date", default="20260401",
                    help="取得開始日 YYYYMMDD (default: 20260401)")
    ap.add_argument("--max", type=int, default=500, help="最大レコード数")
    ap.add_argument("--dump-bytes", action="store_true", help="byte 構造を表示 (offset 探索)")
    ap.add_argument("--out", default="data/jvlink/se_test.json", help="出力 JSON")
    args = ap.parse_args()

    print(f"[se_test] SE 小規模テスト from={args.from_date} max={args.max}")
    result, raw_samples = fetch_se_small(args.from_date, args.max)

    if not result:
        print("[se_test] レコード取得失敗")
        return 1

    # byte 構造表示
    if args.dump_bytes and raw_samples:
        print("\n★ byte オフセット確認用 ★")
        for i, raw in enumerate(raw_samples[:2]):
            print(f"\n--- Sample {i+1} ---")
            dump_byte_structure(raw)

    # agari3f が取れているか確認
    with_agari = sum(1 for r in result if r.get("agari3f_sec") is not None)
    print(f"\n[se_test] agari3f 取得率: {with_agari}/{len(result)} "
          f"({with_agari/len(result)*100:.1f}%)")

    # 全レースの coverage 確認
    race_ids = set()
    for r in result:
        yd = r.get("make_date", "")
        c = r.get("course_code", "")
        rn = r.get("race_num", "")
        if yd and c and rn:
            race_ids.add(f"{yd[:8]}_{c}_{rn}")
    print(f"[se_test] ユニークレース数: {len(race_ids)}")

    # サンプル表示
    print("\n[se_test] サンプル 3 レコード:")
    for r in result[:3]:
        print(f"  horse={r.get('horse_name','?')} "
              f"finish_time={r.get('finish_time_sec')} "
              f"agari3f={r.get('agari3f_sec')} "
              f"corner4={r.get('corner4')}")

    # 保存
    out = BASE_DIR / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "fetch_date": datetime.now().isoformat(),
            "from_date": args.from_date,
            "total_records": len(result),
            "agari3f_coverage": f"{with_agari}/{len(result)}",
            "unique_races": len(race_ids),
            "samples": result[:10],
        }, f, ensure_ascii=False, indent=2)
    print(f"\n[se_test] 保存: {out}")
    print("[se_test] ★ 注意: agari3f offset は仮値 → dump-bytes で確認後に修正が必要 ★")
    return 0


if __name__ == "__main__":
    sys.exit(main())
