"""JV-Link parser skeleton (Phase 22 / Session #86).

JV-Link COM 経由で JRA-VAN DataLab API を叩き、 RACE/HR/O1/TCOV/WOOD/BLOD/SE/UM の
8 datatype の raw record を parse する skeleton。

設計 doc:
  - docs/PHASE_3_4_5_INTEGRATED_ROADMAP_v3.md (6/9-6/13 plan)
  - data/v18/jvlink_backfill_2025_04_actual_5_8.md (Session #43 B)

環境前提:
  - JV-Link DLL: C:\\Windows\\SysWow64\\JVDTLAB\\JVDTLab.dll (32-bit only, ver 1.18)
  - ProgID: JVDTLab.JVLink
  - 32-bit Python venv: C:\\Users\\takum\\jvlink-venv\\  (必須、 64-bit は COM 不可)
  - JRA-VAN DataLab 加入済 (5/7+)

★ 64-bit Python では win32com.client.Dispatch() が失敗するため、 本 module は
  32-bit venv での実行を前提。 import チェックのみは 64-bit でも OK (lazy import)。

★ V15 production 完全不変 (新規 file、 既存 model / predict / train 触らず)。

usage (32-bit venv):
  python tools/jvlink_parser.py --datatype RACE --from 20260301 --to 20260507
  python tools/jvlink_parser.py --datatype HR --realtime --raceid 202605070611
  python tools/jvlink_parser.py --test-com         # COM 接続テストのみ

usage (64-bit、 schema 検証のみ):
  python -c "from tools.jvlink_parser import JVLinkParser; p=JVLinkParser(dry_run=True); \
             print(p.list_datatypes())"
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Iterator

BASE = Path(r"C:/Users/takum/keiba-ai")
DEFAULT_DLL = r"C:\Windows\SysWow64\JVDTLAB\JVDTLab.dll"
DEFAULT_PROGID = "JVDTLab.JVLink"


# ===== JV-Link dataspec 一覧 (Phase 22 skeleton で扱う 8 種) =====
# 仕様: JV-Linkインターフェース仕様書 / JV-Data 仕様書PDF
# (open 系) JVOpen + JVRead 用 = 蓄積系
# (RT 系)   JVRTOpen + JVRead 用 = 速報系
DATASPECS = {
    # dataspec_id : (mode, record_type, description)
    "RACE":  {"mode": "open",   "record": "RA", "desc": "レース詳細 (蓄積系)"},
    "SE":    {"mode": "open",   "record": "SE", "desc": "馬毎レース情報"},
    "HR":    {"mode": "open",   "record": "HR", "desc": "払戻"},
    "UM":    {"mode": "open",   "record": "UM", "desc": "馬個体"},
    "BLOD":  {"mode": "open",   "record": "HN/SK/BT", "desc": "血統 (HN/SK/BT 複合)"},
    "WOOD":  {"mode": "open",   "record": "WC",  "desc": "ウッドチップ調教"},
    "TCOV":  {"mode": "open",   "record": "TK",  "desc": "コース情報"},
    "O1":    {"mode": "rt",     "record": "O1",  "desc": "速報単複オッズ"},
}

OPTION_VALUES = {
    # JVOpen() の option 引数
    "EVERY":         1,   # 通常データ取得 (差分)
    "THIS_WEEK":     2,   # 今週 開催 (RT 系で多用)
    "SETUP":         3,   # 初期 (前回 まで全部)
    "DIALOG":        4,   # ダイアログ表示
}


# ===== JVLinkParser class skeleton =====

class JVLinkParser:
    """JV-Link COM wrapper + raw record parser skeleton.

    Args:
      dlpath: JV-Link DLL 絶対 path (default: C:\\Windows\\SysWow64\\JVDTLAB\\JVDTLab.dll)
      progid: COM ProgID  (default: JVDTLab.JVLink)
      sid:    DataLab Service ID (env JRA_VAN_SID で上書き可)
      dry_run: True なら COM dispatch しない (schema 検証のみ、 64-bit OK)
    """

    def __init__(self,
                 dlpath: str = DEFAULT_DLL,
                 progid: str = DEFAULT_PROGID,
                 sid: str | None = None,
                 dry_run: bool = False):
        self.dlpath = dlpath
        self.progid = progid
        self.sid = sid or ""
        self.dry_run = dry_run
        self.jv = None     # COM dispatch handle
        self._initialized = False

    # -------- low-level COM ops --------

    def _ensure_com(self):
        """Lazy COM dispatch (32-bit Python 必須)."""
        if self.dry_run:
            return
        if self.jv is not None:
            return
        try:
            import win32com.client  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                f"pywin32 が見つからない (32-bit venv で `pip install pywin32` が必要): {e}"
            )
        # 32-bit Python check
        if sys.maxsize > 2**32:
            raise RuntimeError(
                f"64-bit Python では JV-Link COM 不可。 "
                f"C:/Users/takum/jvlink-venv/Scripts/python.exe を使う事。 "
                f"(現在 maxsize={sys.maxsize})"
            )
        self.jv = win32com.client.Dispatch(self.progid)

    def initialize(self, sid: str | None = None) -> int:
        """JVInit() 呼び出し. return value: 0=OK、 負=error."""
        if self.dry_run:
            self._initialized = True
            return 0
        self._ensure_com()
        sid_arg = sid or self.sid or "UNKNOWN/0.0"
        rc = self.jv.JVInit(sid_arg)
        self._initialized = (rc == 0)
        return rc

    def open(self, dataspec: str, fromtime: str, option: int = 1) -> dict:
        """JVOpen() 呼び出し (蓄積系).

        Args:
          dataspec: 'RACE', 'SE', 'HR', 'UM', 'BLOD', 'WOOD', 'TCOV' 等
          fromtime: 'YYYYMMDDHHmmss' or 'YYYYMMDD' (option=SETUP は無視)
          option: 1=EVERY, 2=THIS_WEEK, 3=SETUP, 4=DIALOG
        """
        if self.dry_run:
            return {"dry_run": True, "dataspec": dataspec, "fromtime": fromtime,
                    "read_count": 0, "download_count": 0, "last_filetime": ""}
        self._ensure_com()
        # JVOpen(DataSpec, FromTime, Option, ReadCount, DownloadCount, LastFileTimestamp)
        read_count = 0
        download_count = 0
        last_filetime = ""
        rc = self.jv.JVOpen(dataspec, fromtime, option,
                            read_count, download_count, last_filetime)
        return {"rc": rc, "dataspec": dataspec, "fromtime": fromtime,
                "read_count": read_count,
                "download_count": download_count,
                "last_filetime": last_filetime}

    def rt_open(self, dataspec: str, key: str) -> dict:
        """JVRTOpen() 呼び出し (速報系)."""
        if self.dry_run:
            return {"dry_run": True, "dataspec": dataspec, "key": key}
        self._ensure_com()
        rc = self.jv.JVRTOpen(dataspec, key)
        return {"rc": rc, "dataspec": dataspec, "key": key}

    def read(self, max_size: int = 110000) -> tuple[int, bytes, str]:
        """JVRead() 呼び出し. return (rc, raw_bytes, filename).

        rc: > 0 = bytes read、 0 = EOF、 -1 = file change、 < 0 = error
        """
        if self.dry_run:
            return (0, b"", "")
        self._ensure_com()
        buf = bytearray(max_size)
        filename = ""
        rc = self.jv.JVRead(buf, max_size, filename)
        # 実装上は VARIANT 経由で bytes と filename を取得 (32-bit COM 仕様)
        # ↓ skeleton では bytes() 化のみ示す
        return (rc, bytes(buf[:max(0, rc)]), filename)

    def close(self):
        if self.dry_run:
            return
        if self.jv is None:
            return
        try:
            self.jv.JVClose()
        except Exception:
            pass

    # -------- high-level fetch + parse --------

    def fetch(self, dataspec: str, fromtime: str, totime: str | None = None,
              option: int = 1, max_records: int | None = None) -> list[dict]:
        """fetch + parse 高レベル wrapper.

        dataspec の record_type を見て parse_* 関数に dispatch。

        Args:
          dataspec: 'RACE', 'SE', 'HR' 等
          fromtime: 'YYYYMMDDHHmmss'
          totime:   未対応 (JV-Link は from 以降を全取得、 frontend で filter する)
        """
        spec = DATASPECS.get(dataspec)
        if not spec:
            raise ValueError(f"unknown dataspec {dataspec}, "
                             f"known: {sorted(DATASPECS)}")
        records: list[dict] = []
        if spec["mode"] == "open":
            self.open(dataspec, fromtime, option=option)
        else:
            self.rt_open(dataspec, fromtime)

        # parse loop
        n = 0
        while True:
            rc, raw, fname = self.read()
            if rc == 0:
                break  # EOF
            if rc < 0 and rc != -1:
                break  # error
            if rc == -1:
                continue  # file change
            parsed = self.parse(dataspec, raw)
            parsed["_filename"] = fname
            if totime and parsed.get("_event_date", "") > totime:
                # frontend filter (JV-Link 自体は totime 不支持)
                continue
            records.append(parsed)
            n += 1
            if max_records and n >= max_records:
                break
        self.close()
        return records

    def parse(self, dataspec: str, raw: bytes) -> dict:
        """dataspec → record_type に応じて parse_* に dispatch."""
        spec = DATASPECS.get(dataspec)
        if not spec:
            return {"raw_hex": raw.hex()[:100], "error": "unknown dataspec"}
        rtype = spec["record"]
        fn = getattr(self, f"parse_{rtype.lower()}", None)
        if fn is None:
            return {"record_type": rtype, "raw_size": len(raw),
                    "raw_head": raw[:50].decode("ascii", errors="replace")}
        return fn(raw)

    # -------- parser methods (skeleton、 spec 準拠 placeholder) --------

    def parse_ra(self, raw: bytes) -> dict:
        """RA: レース情報. spec: JV-Data 仕様書 RA1/RA2/RA3/RA4 record_type.

        主要 fields: race_id, race_name, course_code, distance, surface,
                    weather, going, race_class, num_horses
        """
        # TODO (6/9-6/13): JV-Data RA spec 仕様 完全準拠 で実装
        return {
            "record_type": "RA",
            "data_kbn":    self._slice_ascii(raw, 2, 1),
            "year":        self._slice_ascii(raw, 3, 4),
            "month_day":   self._slice_ascii(raw, 7, 4),
            "course_code": self._slice_ascii(raw, 19, 2),
            "kai":         self._slice_ascii(raw, 21, 2),
            "nichi":       self._slice_ascii(raw, 23, 2),
            "race_num":    self._slice_ascii(raw, 25, 2),
            "race_name":   self._slice_sjis(raw, 28, 60),
            "_event_date": self._slice_ascii(raw, 3, 8),
        }

    def parse_se(self, raw: bytes) -> dict:
        """SE: 馬毎レース情報. fields: horse_id, name, sex, age, jockey, weight."""
        return {
            "record_type": "SE",
            "year":        self._slice_ascii(raw, 3, 4),
            "month_day":   self._slice_ascii(raw, 7, 4),
            "course_code": self._slice_ascii(raw, 19, 2),
            "race_num":    self._slice_ascii(raw, 25, 2),
            "wakuban":     self._slice_ascii(raw, 27, 1),
            "umaban":      self._slice_ascii(raw, 28, 2),
            "horse_id":    self._slice_ascii(raw, 30, 10),
            "horse_name":  self._slice_sjis(raw, 40, 36),
            "_event_date": self._slice_ascii(raw, 3, 8),
        }

    def parse_hr(self, raw: bytes) -> dict:
        """HR: 払戻. 単/複/枠連/馬連/馬単/ワイド/三連複/三連単 全種別."""
        return {
            "record_type": "HR",
            "year":        self._slice_ascii(raw, 3, 4),
            "month_day":   self._slice_ascii(raw, 7, 4),
            "course_code": self._slice_ascii(raw, 19, 2),
            "race_num":    self._slice_ascii(raw, 25, 2),
            "raw_payouts": raw[27:].decode("ascii", errors="replace"),
            "_event_date": self._slice_ascii(raw, 3, 8),
        }

    def parse_o1(self, raw: bytes) -> dict:
        """O1: 速報 単複オッズ + 票数. RT mode 用."""
        return {
            "record_type": "O1",
            "year":        self._slice_ascii(raw, 3, 4),
            "month_day":   self._slice_ascii(raw, 7, 4),
            "course_code": self._slice_ascii(raw, 19, 2),
            "race_num":    self._slice_ascii(raw, 25, 2),
            "raw_odds":    raw[27:].decode("ascii", errors="replace"),
            "_event_date": self._slice_ascii(raw, 3, 8),
        }

    def parse_tk(self, raw: bytes) -> dict:
        """TCOV (record_type TK): コース情報 (内回り/外回り、 高低差等)."""
        return {
            "record_type": "TK",
            "course_code": self._slice_ascii(raw, 2, 2),
            "course_type": self._slice_ascii(raw, 4, 1),
            "raw_course":  raw[5:].decode("ascii", errors="replace"),
        }

    def parse_wc(self, raw: bytes) -> dict:
        """WOOD (record_type WC): ウッドチップ調教."""
        return {
            "record_type": "WC",
            "year":        self._slice_ascii(raw, 3, 4),
            "month_day":   self._slice_ascii(raw, 7, 4),
            "horse_id":    self._slice_ascii(raw, 11, 10),
            "raw_wood":    raw[21:].decode("ascii", errors="replace"),
            "_event_date": self._slice_ascii(raw, 3, 8),
        }

    def parse_um(self, raw: bytes) -> dict:
        """UM: 馬個体. fields: horse_id, name, sire, dam, bms."""
        return {
            "record_type": "UM",
            "horse_id":    self._slice_ascii(raw, 3, 10),
            "horse_name":  self._slice_sjis(raw, 13, 36),
        }

    def parse_blod(self, raw: bytes) -> dict:
        """BLOD: 血統 (HN/SK/BT 複合 dataspec). record_type で再分岐."""
        rtype = self._slice_ascii(raw, 0, 2)
        return {
            "record_type": rtype,
            "raw_size":    len(raw),
            "raw_head":    raw[:50].decode("ascii", errors="replace"),
        }

    # -------- helpers --------

    @staticmethod
    def _slice_ascii(raw: bytes, start: int, length: int) -> str:
        try:
            return raw[start:start + length].decode("ascii", errors="replace").strip()
        except Exception:
            return ""

    @staticmethod
    def _slice_sjis(raw: bytes, start: int, length: int) -> str:
        try:
            return raw[start:start + length].decode("shift_jis", errors="replace").rstrip("\x00 ")
        except Exception:
            return ""

    # -------- meta --------

    def list_datatypes(self) -> dict:
        """利用可能 datatype 一覧 (dry-run 確認用)."""
        return {k: v for k, v in DATASPECS.items()}


# ===== CLI =====

def _test_com() -> int:
    """COM 接続テスト (32-bit 必須)."""
    print(f"  Python maxsize = {sys.maxsize}  (32-bit なら 2147483647)")
    print(f"  DLL path = {DEFAULT_DLL}")
    if sys.maxsize > 2**32:
        print("  [!] 64-bit Python では COM dispatch 不可。 "
              "C:/Users/takum/jvlink-venv/Scripts/python.exe で実行する事。")
        return 1
    try:
        import win32com.client  # noqa: F401
    except ImportError:
        print("  [!] pywin32 未 install。 `pip install pywin32` 必須。")
        return 2
    parser = JVLinkParser()
    rc = parser.initialize()
    print(f"  JVInit() rc={rc}")
    parser.close()
    return 0 if rc == 0 else 3


def cli():
    p = argparse.ArgumentParser(description="JV-Link parser skeleton (Phase 22)")
    p.add_argument("--datatype", choices=sorted(DATASPECS),
                   help="dataspec id")
    p.add_argument("--from", dest="fromtime", default=None,
                   help="YYYYMMDDHHmmss or YYYYMMDD")
    p.add_argument("--to", dest="totime", default=None,
                   help="YYYYMMDDHHmmss (frontend filter)")
    p.add_argument("--realtime", action="store_true",
                   help="JVRTOpen を使う (O1 等)")
    p.add_argument("--raceid", default=None, help="rt mode の key (race id)")
    p.add_argument("--max", type=int, default=100, help="max records")
    p.add_argument("--out", default=None, help="output JSON path")
    p.add_argument("--dry-run", action="store_true",
                   help="COM 起動せず schema 確認のみ (64-bit OK)")
    p.add_argument("--test-com", action="store_true",
                   help="COM 接続テストのみ")
    p.add_argument("--list", action="store_true",
                   help="dataspec 一覧 表示")
    args = p.parse_args()

    if args.test_com:
        sys.exit(_test_com())

    if args.list:
        for k, v in DATASPECS.items():
            print(f"  {k:6s} mode={v['mode']:5s} record={v['record']:8s} {v['desc']}")
        return

    parser = JVLinkParser(dry_run=args.dry_run)
    if args.dry_run:
        print("  DRY RUN (schema 確認のみ、 COM 未接続)")

    t0 = time.time()
    if not args.datatype:
        print("[!] --datatype 指定必須")
        sys.exit(1)

    if args.realtime:
        if not args.raceid:
            print("[!] --realtime に --raceid 必須")
            sys.exit(1)
        result = parser.fetch(args.datatype, args.raceid,
                              max_records=args.max)
    else:
        if not args.fromtime:
            print("[!] --from YYYYMMDDHHmmss 必須")
            sys.exit(1)
        result = parser.fetch(args.datatype, args.fromtime,
                              totime=args.totime, max_records=args.max)

    print(f"  fetched {len(result)} records in {time.time()-t0:.1f}s")
    if args.out:
        out_path = Path(args.out)
        if not out_path.is_absolute():
            out_path = BASE / args.out
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False, default=str)
        print(f"  out -> {out_path}")
    else:
        for r in result[:3]:
            print(r)


if __name__ == "__main__":
    cli()
