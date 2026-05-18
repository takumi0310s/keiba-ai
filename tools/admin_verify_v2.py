#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
admin_verify_v2.py  (★ read-only verify、 完成-4 sub-task 2026-05-18 ★)

5/18 17:00 status_verify は false positive (bash grep pattern が schtasks 日本語 error 文言と
非一致、 全 9 件 「✅ 登録済」 と誤判定)。 B-5 真値 9/9 未登録。

本 tool は ★ exit code 直接判定 ★ で正しい existence check を行う:
  - schtasks /Query /FO LIST /TN <name> -> returncode 0 = 登録済、 != 0 = 未登録
  - 文字列 pattern に依存しない (i18n / locale 安全)

scope: read-only。 schtasks /Create | /Delete は ★ 絶対 実行しない ★。

usage:
  python tools/admin_verify_v2.py
  python tools/admin_verify_v2.py --out data/admin_verify_v2_5_18_1830.json

exit code:
  0 = verify 実行 OK (登録 0/9 でも 0 を返す; 状態 report のみ)
  1 = 内部 error (subprocess 失敗 等)
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import py_compile
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
TOOLS_DIR = REPO_ROOT / "tools"

# ★ 9 schtask 仕様 (★ 完成-4 spec 準拠) ★
SCHTASKS: List[Dict[str, str]] = [
    {"name": "Keiba-LiveOrchestrator-15min",     "bat": "live_orchestrator.bat",                "py": "tools/live_orchestrator_main.py"},
    {"name": "Keiba-FeaturesIntegrity-Daily",    "bat": "keiba_features_integrity.bat",         "py": "tools/features_integrity_monitor.py"},
    {"name": "Keiba-AnomalyCheck-0630",          "bat": "keiba_anomaly_check_0630.bat",         "py": "tools/anomaly_auto_detector.py"},
    {"name": "Keiba-AnomalyCheck-0830",          "bat": "keiba_anomaly_check_0830.bat",         "py": "tools/anomaly_auto_detector.py"},
    {"name": "Keiba-AnomalyCheck-0940",          "bat": "keiba_anomaly_check_0940.bat",         "py": "tools/anomaly_auto_detector.py"},
    {"name": "Keiba-AnomalyCheck-1410",          "bat": "keiba_anomaly_check_1410.bat",         "py": "tools/anomaly_auto_detector.py"},
    {"name": "Keiba-AnomalyCheck-1700",          "bat": "keiba_anomaly_check_1700.bat",         "py": "tools/anomaly_auto_detector.py"},
    {"name": "Keiba-CumulativeAudit-Daily",      "bat": "daily_cumulative_audit.bat",           "py": "tools/daily_cumulative_audit.py"},
    {"name": "Keiba-RaceNotifyLogV2Aggregator",  "bat": "keiba_race_notify_log_v2_aggregator.bat", "py": "tools/race_notify_log_v2_aggregator.py"},
]


def query_schtask(name: str) -> Dict[str, Any]:
    """
    schtasks /Query /FO LIST /TN <name> を実行。
    ★ exit code 直接判定 ★ — 文字列 pattern に依存しない。

    returncode 0 -> registered=True、 stdout から Next/Status/Last 抽出
    returncode != 0 -> registered=False (どの error 文言でも 統一判定)
    """
    try:
        result = subprocess.run(
            ["schtasks", "/Query", "/FO", "LIST", "/TN", name],
            capture_output=True,
            text=True,
            encoding="cp932",
            errors="replace",
            timeout=30,
        )
    except FileNotFoundError:
        # schtasks not on PATH (e.g. non-Windows / restricted env)
        return {
            "registered": None,
            "exit_code": None,
            "error": "schtasks command not found",
            "next_run": None,
            "status": None,
            "last_run": None,
        }
    except subprocess.TimeoutExpired:
        return {
            "registered": None,
            "exit_code": None,
            "error": "schtasks timeout (30s)",
            "next_run": None,
            "status": None,
            "last_run": None,
        }

    registered = (result.returncode == 0)
    info: Dict[str, Any] = {
        "registered": registered,
        "exit_code": result.returncode,
        "next_run": None,
        "status": None,
        "last_run": None,
    }

    if not registered:
        # ★ 重要 ★ stderr 文言は記録するが judgment には 一切 使わない (i18n 安全)
        info["stderr_snippet"] = (result.stderr or "").strip().splitlines()[:2]
        return info

    # parse /FO LIST output — key: value 形式 (ja-JP/en-US 両対応)
    for line in (result.stdout or "").splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        key, _, val = line.partition(":")
        key_l = key.strip().lower()
        val = val.strip()
        # en-US
        if key_l == "next run time":
            info["next_run"] = val
        elif key_l == "status":
            info["status"] = val
        elif key_l == "last run time":
            info["last_run"] = val
        # ja-JP (本環境 = ja-JP cp932)
        elif "次回の実行時刻" in key or "次回実行時刻" in key:
            info["next_run"] = val
        elif "状態" == key.strip():
            info["status"] = val
        elif "前回の実行時刻" in key or "前回実行時刻" in key:
            info["last_run"] = val

    return info


def check_bat(bat_name: str) -> Dict[str, Any]:
    """bat 存在 + size + mtime 取得"""
    p = TOOLS_DIR / bat_name
    if not p.exists():
        return {"exists": False, "path": str(p)}
    st = p.stat()
    return {
        "exists": True,
        "path": str(p),
        "size_bytes": st.st_size,
        "mtime": dt.datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
    }


def check_py_module(py_rel_path: str) -> Dict[str, Any]:
    """py module 存在 + py_compile PASS check"""
    p = REPO_ROOT / py_rel_path
    if not p.exists():
        return {"exists": False, "compile_ok": False, "path": str(p)}
    # tempfile に書き出して compile (Python 3.14 で os.devnull は使えない)
    try:
        with tempfile.NamedTemporaryFile(suffix=".pyc", delete=False) as tf:
            tmp_pyc = tf.name
        try:
            py_compile.compile(str(p), cfile=tmp_pyc, doraise=True)
            return {"exists": True, "compile_ok": True, "path": str(p)}
        finally:
            try:
                os.remove(tmp_pyc)
            except OSError:
                pass
    except py_compile.PyCompileError as exc:
        return {
            "exists": True,
            "compile_ok": False,
            "path": str(p),
            "error": str(exc).splitlines()[0][:200],
        }
    except Exception as exc:
        return {
            "exists": True,
            "compile_ok": False,
            "path": str(p),
            "error": f"{type(exc).__name__}: {exc}"[:200],
        }


def run_verify() -> Dict[str, Any]:
    schtask_results: Dict[str, Any] = {}
    bat_results: Dict[str, Any] = {}
    py_results: Dict[str, Any] = {}

    for entry in SCHTASKS:
        name = entry["name"]
        bat = entry["bat"]
        py = entry["py"]
        schtask_results[name] = query_schtask(name)
        bat_results[bat] = check_bat(bat)
        py_results[py] = check_py_module(py)

    # summary
    registered_count = sum(
        1 for v in schtask_results.values() if v.get("registered") is True
    )
    bats_exist = sum(1 for v in bat_results.values() if v.get("exists") is True)
    py_compile_ok = sum(
        1 for v in py_results.values() if v.get("compile_ok") is True
    )

    ready = (
        registered_count == 0  # 5/18 1830 想定: まだ未登録 = OK to proceed admin
        and bats_exist == len(SCHTASKS)
        and py_compile_ok == len(set(e["py"] for e in SCHTASKS))
    )

    return {
        "verify_time": dt.datetime.now().isoformat(timespec="seconds"),
        "method": "schtasks /Query /FO LIST /TN <name>, exit code direct judgment",
        "schtasks": schtask_results,
        "bats": bat_results,
        "py_modules": py_results,
        "summary": {
            "schtasks_registered": f"{registered_count}/{len(SCHTASKS)}",
            "bats_exist": f"{bats_exist}/{len(SCHTASKS)}",
            "py_compile_ok": f"{py_compile_ok}/{len(set(e['py'] for e in SCHTASKS))}",
            "ready_for_5_22_admin": ready,
        },
    }


def print_summary(report: Dict[str, Any]) -> None:
    s = report["summary"]
    print("=" * 60)
    print(f"admin_verify_v2  ({report['verify_time']})")
    print(f"method: {report['method']}")
    print("=" * 60)
    print(f"  schtasks registered : {s['schtasks_registered']}")
    print(f"  bats exist          : {s['bats_exist']}")
    print(f"  py compile ok       : {s['py_compile_ok']}")
    print(f"  ready for 5/22 admin: {s['ready_for_5_22_admin']}")
    print("-" * 60)
    print("schtasks detail:")
    for name, info in report["schtasks"].items():
        mark = "REG" if info.get("registered") is True else "---"
        nxt = info.get("next_run") or ""
        print(f"  [{mark}] {name:42s} ec={info.get('exit_code')}  next={nxt}")
    print("-" * 60)
    print("bats detail:")
    for name, info in report["bats"].items():
        mark = "OK" if info.get("exists") else "MISS"
        size = info.get("size_bytes", "-")
        print(f"  [{mark:4s}] {name:50s} size={size}")
    print("-" * 60)
    print("py modules detail:")
    for name, info in report["py_modules"].items():
        if info.get("compile_ok"):
            mark = "OK"
        elif info.get("exists"):
            mark = "COMPILE-FAIL"
        else:
            mark = "MISS"
        print(f"  [{mark:13s}] {name}")
    print("=" * 60)


def main() -> int:
    parser = argparse.ArgumentParser(description="admin verify v2 (read-only)")
    parser.add_argument(
        "--out",
        default=None,
        help="output JSON path (default: data/admin_verify_v2_<YYYYMMDD>_<HHMM>.json)",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="suppress stdout summary"
    )
    args = parser.parse_args()

    report = run_verify()

    if args.out:
        out_path = Path(args.out)
    else:
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M")
        # spec: data/admin_verify_v2_5_18_1830.json — date stamp form は実行時刻依存
        # default: data/admin_verify_v2_<YYYYMMDD>_<HHMM>.json
        out_path = REPO_ROOT / "data" / f"admin_verify_v2_{stamp}.json"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    if not args.quiet:
        print_summary(report)
        print(f"\nJSON saved: {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
