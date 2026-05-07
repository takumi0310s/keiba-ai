"""logs/ 配下 30日以上前 archive (Session #40 B4).

- logs/ 配下を walk
- mtime > 30 日前 のファイルを logs/archive/{YYYYMM}/ に移動
- gzip 圧縮で容量削減
- 移動 log を logs_cleanup_history.json に記録

usage:
  python tools/logs_cleanup.py --dry-run  # 試行のみ
  python tools/logs_cleanup.py            # 実行
  python tools/logs_cleanup.py --days 60  # 60 日以上前 を archive

V15 production 完全不変 (logs のみ操作)。
"""
from __future__ import annotations

import argparse
import datetime
import gzip
import json
import os
import shutil
import sys
import time
from pathlib import Path

BASE = Path(r"C:/Users/takum/keiba-ai")
LOGS_DIR = BASE / "logs"
ARCHIVE_DIR = LOGS_DIR / "archive"
HISTORY_FILE = BASE / "data" / "logs_cleanup_history.json"


def iter_log_files(threshold_sec: float):
    """LOGS_DIR 配下の通常ファイル (archive/ 以下は除外) で mtime < threshold のもの."""
    if not LOGS_DIR.exists():
        return
    for root, dirs, files in os.walk(LOGS_DIR):
        # archive/ 以下を skip
        rp = Path(root)
        try:
            rel = rp.relative_to(LOGS_DIR)
            if rel.parts and rel.parts[0] == "archive":
                continue
        except ValueError:
            continue
        for f in files:
            full = rp / f
            try:
                mtime = full.stat().st_mtime
            except Exception:
                continue
            if mtime < threshold_sec:
                yield full, mtime


def archive_path(src: Path, mtime: float) -> Path:
    dt = datetime.datetime.fromtimestamp(mtime)
    sub = ARCHIVE_DIR / f"{dt:%Y%m}"
    sub.mkdir(parents=True, exist_ok=True)
    return sub / (src.name + ".gz")


def gzip_move(src: Path, dst_gz: Path) -> int:
    """src を gzip 圧縮して dst_gz に移動。 元 file は削除。 size_saved (bytes) を返す."""
    src_size = src.stat().st_size
    with open(src, "rb") as fin, gzip.open(dst_gz, "wb") as fout:
        shutil.copyfileobj(fin, fout)
    src.unlink()
    dst_size = dst_gz.stat().st_size
    return src_size - dst_size


def main():
    p = argparse.ArgumentParser(description="logs/ 30日以上前 archive (Session #40 B4)")
    p.add_argument("--days", type=int, default=30)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    threshold = time.time() - args.days * 86400
    n_archived = 0
    n_skipped = 0
    bytes_saved = 0
    items = []

    for f, mtime in iter_log_files(threshold):
        dst = archive_path(f, mtime)
        if args.dry_run:
            print(f"  [DRY] {f.relative_to(BASE)} -> {dst.relative_to(BASE)}")
            n_archived += 1
            continue
        try:
            saved = gzip_move(f, dst)
            bytes_saved += saved
            n_archived += 1
            items.append({
                "src": str(f.relative_to(BASE)),
                "dst": str(dst.relative_to(BASE)),
                "saved_bytes": saved,
                "ts": datetime.datetime.now().isoformat(),
            })
        except Exception as e:
            print(f"  [ERR] {f}: {e}", file=sys.stderr)
            n_skipped += 1

    print(f"\nlogs_cleanup: archived={n_archived}, skipped={n_skipped}, "
          f"saved={bytes_saved/1024/1024:.1f} MB"
          + (" (dry run)" if args.dry_run else ""))

    if not args.dry_run and items:
        # history append
        prev = []
        if HISTORY_FILE.exists():
            try:
                prev = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
            except Exception:
                prev = []
        prev.extend(items)
        HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        HISTORY_FILE.write_text(
            json.dumps(prev, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8")
        print(f"  history: {HISTORY_FILE}")


if __name__ == "__main__":
    main()
