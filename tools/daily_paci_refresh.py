#!/usr/bin/env python3
"""PACI 当日データ 自動取得 + 安全再生成 (2026-05-30 実装)。

背景: jrdb_paci.csv は parse_jrdb.py が PACIパック(Paci/KYI*.txt)から生成するが、
日次パイプライン(daily_jrdb_kyi.bat)にPACI取得+parse再生成が無く、 4/4頃から
更新停止 → 当日race_idがpaci.csvに入らず、 V15 gain 52.6%のPACI特徴が
stale/default化していた。

★ このスクリプトは「障害修正(取得経路の復旧)」。 V15 model / predict_core /
  daily_predict / app.py は一切不変。 投票・戦略ロジックにも触れない ★

安全機構: bare parse_jrdb.py は sed/tyb/kab/cyb csv も再生成し、 日次append維持の
  kab/cyb を破壊する (2026-05-30 確認: cyb 395→551,542行に暴走)。 そのため
  parse 前に sed/tyb/kab/cyb をバックアップし、 parse 後に復元して paci のみ更新する。

手順:
  1. 最新 PACI*.zip を download + extract (scrape_jrdb_paci.step_download、 --skip-parse相当)
  2. jrdb_{sed,tyb,kab,cyb}.csv をバックアップ
  3. parse_jrdb.py 実行 (jrdb_paci.csv を全件再生成)
  4. sed/tyb/kab/cyb を復元 (paci のみ新版を残す)
  5. 当日カバレッジを検証

usage:
  python tools/daily_paci_refresh.py                 # 当日含む最近を取得+再生成
  python tools/daily_paci_refresh.py --since 20260525
  python tools/daily_paci_refresh.py --skip-download  # 既存extractからparseのみ
"""
from __future__ import annotations

import argparse
import io
import os
import shutil
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

if sys.platform == "win32":
    # reconfigure はバッファを閉じない (TextIOWrapper 再ラップだと silent_runner の
    # ログリダイレクト下で元ラッパーGC→バッファclose→"I/O operation on closed file")。
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
sys.path.insert(0, str(BASE_DIR / "tools"))

# parse_jrdb.py が再生成するが、 日次append維持で保護すべき csv
PROTECT = ["jrdb_sed", "jrdb_tyb", "jrdb_kab", "jrdb_cyb"]
PACI_CSV = DATA_DIR / "jrdb_paci.csv"


def _rows(path: Path) -> int:
    if not path.exists():
        return -1
    with open(path, "r", encoding="utf-8-sig") as f:
        return sum(1 for _ in f)


def main():
    ap = argparse.ArgumentParser(description="PACI 当日取得 + 安全再生成")
    default_since = (datetime.now() - timedelta(days=7)).strftime("%Y%m%d")
    ap.add_argument("--since", default=default_since, help="この日以降のPACI取得")
    ap.add_argument("--skip-download", action="store_true")
    args = ap.parse_args()

    print("=" * 60)
    print(f" daily_paci_refresh  {datetime.now().isoformat(timespec='seconds')}")
    print(f"  since={args.since}  skip_download={args.skip_download}")
    print("=" * 60)

    # --- Step 1: download + extract PACI ---
    if not args.skip_download:
        try:
            import scrape_jrdb_paci as sp
            years = sorted({datetime.now().year, int(args.since[:4])})
            res = sp.step_download(years, args.since, dry_run=False)
            print(f"  download: {res}")
        except Exception as e:
            print(f"  [WARN] PACI download 失敗 (続行してparse試行): {e}")

    # --- Step 2: backup protected csvs ---
    paci_before = _rows(PACI_CSV)
    backups = {}
    for name in PROTECT:
        src = DATA_DIR / f"{name}.csv"
        if src.exists():
            bak = DATA_DIR / f"{name}.csv.paci_refresh_bak"
            shutil.copy2(src, bak)
            backups[name] = (bak, _rows(src))
    print(f"  backup: {list(backups.keys())}")

    # --- Step 3: parse_jrdb.py (regenerate jrdb_paci.csv) ---
    parse_script = BASE_DIR / "tools" / "parse_jrdb.py"
    print(f"  run: {parse_script.name}")
    rc = -1
    try:
        proc = subprocess.run([sys.executable, str(parse_script)], cwd=str(BASE_DIR),
                              capture_output=True, text=True, timeout=1800,
                              encoding="utf-8", errors="replace")
        rc = proc.returncode
        print(f"  parse_jrdb.py rc={rc}")
        if rc != 0:
            for ln in (proc.stderr or "").splitlines()[-10:]:
                print(f"    stderr: {ln}")
    except subprocess.TimeoutExpired:
        print("  [ERROR] parse_jrdb.py timeout")

    # --- Step 4: restore protected csvs (paci のみ新版を残す) ---
    restored = []
    for name, (bak, before_rows) in backups.items():
        cur = DATA_DIR / f"{name}.csv"
        after_rows = _rows(cur)
        # parse が変更した (=append維持を破壊した) 場合のみ復元
        if after_rows != before_rows:
            shutil.copy2(bak, cur)
            restored.append(f"{name}({before_rows}<-{after_rows})")
        try:
            os.remove(bak)
        except OSError:
            pass
    print(f"  restored (parse破壊から復元): {restored or 'なし(全て不変)'}")

    paci_after = _rows(PACI_CSV)
    print(f"  jrdb_paci.csv: {paci_before:,} → {paci_after:,} 行 ({paci_after-paci_before:+,})")

    # --- Step 3.5: ZE (ZED前走拡張) も再生成 ---
    # jrdb_ze.csv も paci と同じ「日次パイプライン外で静かに停止(5/1〜)」だった。
    # ZED は同じ PACIパックに含まれ既に extract 済。 parse_jrdb_extended.py --types zed は
    # jrdb_ze.csv のみ書く (他csv無干渉) ため backup/restore 不要で安全。 gain 12.7%。
    ze_csv = DATA_DIR / "jrdb_ze.csv"
    ze_before = _rows(ze_csv)
    try:
        ext_script = BASE_DIR / "tools" / "parse_jrdb_extended.py"
        proc_ze = subprocess.run([sys.executable, str(ext_script), "--types", "zed"],
                                 cwd=str(BASE_DIR), capture_output=True, text=True,
                                 timeout=900, encoding="utf-8", errors="replace")
        print(f"  parse_jrdb_extended --types zed rc={proc_ze.returncode}")
    except Exception as e:
        print(f"  [WARN] ZE再生成失敗: {e}")
    print(f"  jrdb_ze.csv: {ze_before:,} → {_rows(ze_csv):,} 行")

    # --- Step 5: 当日カバレッジ検証 (monitorに委譲) ---
    print("\n  当日カバレッジ検証:")
    try:
        mon = subprocess.run([sys.executable, str(BASE_DIR / "tools" / "data_freshness_monitor.py"),
                              "--no-notify"], cwd=str(BASE_DIR), capture_output=True,
                             text=True, timeout=120, encoding="utf-8", errors="replace")
        for ln in (mon.stdout or "").splitlines():
            if "jrdb_paci" in ln or "当日race_id" in ln or "閾値" in ln:
                print(f"    {ln.strip()}")
    except Exception as e:
        print(f"    [WARN] monitor 実行失敗: {e}")

    print("\n daily_paci_refresh 完了")
    return 0 if rc == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
