"""JV-Link 32-bit Python 動作確認 (Session #41 A3).

32-bit Python venv (C:\\Users\\takum\\jvlink-venv\\) から実行する前提。
既存 keiba-ai 用 64-bit Python (3.14) では JVDTLab.JVLink COM 接続不可。

確認項目:
1. pywin32 import 成功
2. JVDTLab.JVLink COM オブジェクト Dispatch
3. JVInit (sid="UNKNOWN") rc=0
4. JVOpen (datatype=RACE, fromtime=20260503000000) rc=0 + num_files >= 1
5. JVRead 1 件 取得 + 内容 print
6. JVClose rc=0

usage:
  python tools\jvlink_test_python32.py
  python tools\jvlink_test_python32.py --check-only       # COM Dispatch のみ
  python tools\jvlink_test_python32.py --date 20260503    # 別日付テスト
  python tools\jvlink_test_python32.py --read-records 5   # 5 件 read

V15 production 完全独立、 既存 keiba-ai 動作不変。
"""
from __future__ import annotations

import argparse
import os
import platform
import sys


def main():
    p = argparse.ArgumentParser(description="JV-Link 32-bit Python 動作確認 (Session #41 A3)")
    p.add_argument("--sid", default="UNKNOWN", help="ソフトウェア ID")
    p.add_argument("--date", default="20260503", help="YYYYMMDD")
    p.add_argument("--datatype", default="RACE")
    p.add_argument("--option", type=int, default=4)
    p.add_argument("--read-records", type=int, default=3, help="JVRead 取得 件数 (試行)")
    p.add_argument("--check-only", action="store_true", help="COM Dispatch + JVInit のみ確認")
    args = p.parse_args()

    # Step 0: arch 確認
    arch = platform.architecture()[0]
    print(f"[Step 0] Python arch: {arch}")
    if "32bit" not in arch:
        print(f"  [ERROR] 32-bit Python ではない。 JV-Link DLL は 32-bit COM のみ提供。")
        print(f"  setup_python32.ps1 で 32-bit venv を作成し、 そこから実行してください。")
        sys.exit(1)
    print(f"  OK: 32-bit Python")

    # Step 1: pywin32 import
    print(f"\n[Step 1] pywin32 import")
    try:
        import win32com.client  # type: ignore
        print(f"  OK: win32com.client import")
    except ImportError as e:
        print(f"  [ERROR] pywin32 未 install: {e}")
        print(f"  pip install pywin32 を実行 + python -m pywin32_postinstall -install")
        sys.exit(2)

    # Step 2: COM Dispatch
    print(f"\n[Step 2] JVDTLab.JVLink COM Dispatch")
    try:
        jv = win32com.client.Dispatch("JVDTLab.JVLink")
        print(f"  OK: COM Dispatch 成功")
    except Exception as e:
        print(f"  [ERROR] Dispatch 失敗: {e}")
        print(f"  JV-Link DLL 未登録 or COM 登録失敗の可能性")
        print(f"  C:\\Windows\\SysWow64\\JVDTLAB\\JVDTLab.dll を regsvr32 で登録")
        sys.exit(3)

    # Step 3: JVInit
    print(f"\n[Step 3] JVInit('{args.sid}')")
    try:
        rc = jv.JVInit(args.sid)
        print(f"  rc={rc}")
        if rc != 0:
            print(f"  [WARN] rc != 0、 ID/PW 未設定の可能性")
            print(f"  jv.JVSetUIProperties() を 別 script で 1 回実行して GUI で設定")
            sys.exit(4)
        print(f"  OK: JVInit 成功")
    except Exception as e:
        print(f"  [ERROR] JVInit error: {e}")
        sys.exit(5)

    if args.check_only:
        print(f"\n--check-only モード: ここで終了")
        return

    # Step 4: JVOpen
    fromtime = args.date + "000000"
    print(f"\n[Step 4] JVOpen('{args.datatype}', '{fromtime}', option={args.option})")
    try:
        # JVOpen 仕様: rc, ReadCount, DownloadCount, LastFileTimestamp = JVOpen(...)
        # pywin32 では tuple return
        ret = jv.JVOpen(args.datatype, fromtime, args.option)
        if isinstance(ret, tuple) and len(ret) >= 3:
            rc, num_data, num_files = ret[0], ret[1], ret[2]
            last_filetime = ret[3] if len(ret) >= 4 else "?"
        else:
            rc = ret
            num_data = num_files = -1
            last_filetime = "?"
        print(f"  rc={rc}, data={num_data}, files={num_files}, last={last_filetime}")
        if rc != 0:
            print(f"  [WARN] rc != 0")
            jv.JVClose()
            sys.exit(6)
        print(f"  OK: JVOpen 成功")
    except Exception as e:
        print(f"  [ERROR] JVOpen error: {e}")
        try: jv.JVClose()
        except: pass
        sys.exit(7)

    # Step 5: JVRead 試行
    print(f"\n[Step 5] JVRead loop ({args.read_records} 件 試行)")
    records = []
    try:
        for i in range(args.read_records):
            ret = jv.JVRead(2048, "")
            if isinstance(ret, tuple) and len(ret) >= 3:
                rc, buff, filename = ret[0], ret[1], ret[2]
            else:
                rc = ret; buff = ""; filename = "?"
            print(f"  [{i+1}] rc={rc}, file={filename}, len(buff)={len(buff) if buff else 0}")
            if rc == 0:
                print(f"  EOF (rc=0)")
                break
            elif rc == -1:
                print(f"  ファイル切替 (rc=-1)")
                continue
            elif rc < 0:
                print(f"  [ERROR] JVRead rc={rc}")
                break
            records.append(buff)
            # 先頭 100 chars print
            print(f"      content: {buff[:100]!r}")
    except Exception as e:
        print(f"  [ERROR] JVRead error: {e}")

    print(f"\n  total records read: {len(records)}")

    # Step 6: JVClose
    print(f"\n[Step 6] JVClose")
    try:
        rc = jv.JVClose()
        print(f"  rc={rc}")
    except Exception as e:
        print(f"  [WARN] JVClose error: {e}")

    print(f"\n=== JV-Link 32-bit 動作確認 完了 ===")
    print(f"次 step:")
    print(f"  python tools\\jvlink_fetcher.py --date {args.date} --datatype {args.datatype}")


if __name__ == "__main__":
    main()
