"""JV-Link COM API fetcher (Session #39 B、PoC 試作).

JRA-VAN DataLab (月額 2,090円) の JV-Link DLL を Windows COM 経由で呼出し、
公式 race / odds / training / payout データを取得する。

前提:
1. JRA-VAN DataLab 加入完了
2. JV-Link DLL インストール済 (https://jra-van.jp/dlb/)
3. ID/PW 設定済 (JVSetUIProperties で初回 GUI 入力)
4. Python 環境: pywin32 (`pip install pywin32`)

usage:
  python tools/jvlink_fetcher.py --datatype RACE --from 20260507 --to 20260509
  python tools/jvlink_fetcher.py --realtime --datatype O1   # オッズ速報

API 仕様:
  https://jra-van.jp/dlb/sdv/sdk.html
  https://jra-van.jp/dlb/manual/jvlink.html

統合先:
  - data/jvlink/<datatype>/<date>.csv に保存
  - tools/jrdb_features.py / train/v15_master.py から merge
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from typing import List, Optional

# ===== 定数 =====

# JV-Link データ種別
JVLINK_DATATYPES = {
    "RACE": "レース詳細 (RACE_RA = 開催情報)",
    "DIFF": "差分データ (前回取得後の更新分のみ)",
    "BLOD": "血統登録",
    "MING": "馬名イニシャル",
    "SNPN": "騎手・調教師",
    "TCOV": "調教タイム",
    "WOOD": "木曽用調教",
    "RC": "レース短信",
    # オッズ系 (リアルタイム)
    "O1": "単複枠オッズ",
    "O2": "馬連オッズ",
    "O3": "ワイドオッズ",
    "O4": "馬単オッズ",
    "O5": "三連複オッズ",
    "O6": "三連単オッズ",
    # 払戻
    "HR":  "払戻金",
    # 速報
    "JG":  "競走除外/発走時刻変更",
    "DM":  "コメント",
    "WF":  "馬体重情報",
    "CC":  "コース情報",
}


# ===== JV-Link COM 接続 =====

def init_jvlink(sid: str = "UNKNOWN", verbose: bool = True):
    """JV-Link COM オブジェクト作成 + 初期化。

    sid: ソフトウェア識別子 (登録された ID/SOFT.NAME)
    """
    try:
        import win32com.client  # type: ignore
    except ImportError:
        raise RuntimeError("pywin32 未インストール。 `pip install pywin32` を実行")

    jv = win32com.client.Dispatch("JVDTLab.JVLink")
    rc = jv.JVInit(sid)
    if verbose:
        print(f"[jvlink] JVInit('{sid}') → rc={rc}")
    if rc != 0:
        raise RuntimeError(f"JVInit failed rc={rc}。 ID/PW 未設定なら JVSetUIProperties() で GUI 起動")
    return jv


def open_data(jv, dataspec: str, fromtime: str, option: int = 4, verbose: bool = True):
    """JVOpen — データ取得開始。

    dataspec: 'RACE' / 'O1' / 'BLOD' etc.
    fromtime: 'YYYYMMDDhhmmss' (例: '20260507000000')
    option: 1 = 通常, 2 = 今週分, 3 = セットアップ, 4 = 通常 (差分)
    """
    rc, num_data, num_files, last_filetime = jv.JVOpen(dataspec, fromtime, option)
    if verbose:
        print(f"[jvlink] JVOpen({dataspec},{fromtime},opt={option}) → rc={rc}, "
              f"data={num_data}, files={num_files}, last={last_filetime}")
    if rc != 0:
        raise RuntimeError(f"JVOpen failed rc={rc}")
    return num_data, num_files, last_filetime


def read_records(jv, max_records: Optional[int] = None) -> List[str]:
    """JVRead でレコードを順次取得。 max_records を超えたら停止。"""
    out: List[str] = []
    while True:
        rc, buff, filename = jv.JVRead(2048, "")
        if rc == 0:
            break  # 終了
        elif rc == -1:
            # ファイル切替、 メッセージのみ
            continue
        elif rc < 0:
            print(f"[jvlink] JVRead error rc={rc}")
            break
        out.append(buff.rstrip("\r\n"))
        if max_records is not None and len(out) >= max_records:
            break
    return out


def close(jv, verbose: bool = True):
    rc = jv.JVClose()
    if verbose:
        print(f"[jvlink] JVClose → rc={rc}")
    return rc


# ===== 高レベル wrapper =====

def fetch_to_csv(
    dataspec: str,
    fromtime: str,
    out_path: str,
    sid: str = "UNKNOWN",
    option: int = 4,
    max_records: Optional[int] = None,
):
    """指定 dataspec を JV-Link 経由で取得し、 raw レコードを CSV (1 列) に保存。"""
    jv = init_jvlink(sid)
    try:
        n_data, n_files, last = open_data(jv, dataspec, fromtime, option)
        records = read_records(jv, max_records=max_records)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["raw_record"])
            for r in records:
                w.writerow([r])
        print(f"[jvlink] saved {len(records)} records → {out_path}")
        return len(records)
    finally:
        close(jv)


# ===== CLI =====

def cli():
    p = argparse.ArgumentParser(description="JV-Link fetcher (Session #39 B PoC)")
    p.add_argument("--sid", default="UNKNOWN", help="登録された SOFT.ID")
    p.add_argument("--datatype", default="RACE", choices=list(JVLINK_DATATYPES.keys()))
    p.add_argument("--from", dest="from_", default=None, help="YYYYMMDD (時刻は 00:00:00 補完)")
    p.add_argument("--to", default=None, help="reserved (現状未使用)")
    p.add_argument("--option", type=int, default=4, choices=[1, 2, 3, 4])
    p.add_argument("--max-records", type=int, default=None)
    p.add_argument("--realtime", action="store_true", help="速報系 (O1〜O6/HR/DM/WF) で option=2 + 即時")
    p.add_argument("--out-dir", default="data/jvlink")
    p.add_argument("--list-datatypes", action="store_true", help="利用可能 datatype 一覧")
    args = p.parse_args()

    if args.list_datatypes:
        for k, v in JVLINK_DATATYPES.items():
            print(f"  {k:6s} : {v}")
        return

    fromtime = (args.from_ or time.strftime("%Y%m%d")) + "000000"
    if args.realtime:
        args.option = 2

    out_path = os.path.join(args.out_dir, args.datatype, f"{fromtime[:8]}.csv")
    n = fetch_to_csv(
        dataspec=args.datatype,
        fromtime=fromtime,
        out_path=out_path,
        sid=args.sid,
        option=args.option,
        max_records=args.max_records,
    )
    print(f"[jvlink] OK total={n} records")


if __name__ == "__main__":
    cli()
