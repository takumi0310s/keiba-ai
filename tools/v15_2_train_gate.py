#!/usr/bin/env python
"""
v15.2 train gate
================

leak_audit_automation.py が PASS しない限り、 v15.2 学習を 機械的に禁止する gate。

★ design only ★ — 本 sub-task では gate logic の設計のみ実装。
実 v15.2 学習は親 (5/24+) で別途 着手。

usage:
    # v15.2 候補 features list を audit
    python tools/v15_2_train_gate.py --features data/v15_2/features_v152_candidates.txt

    # V15 cache 自体を audit (sanity check)
    python tools/v15_2_train_gate.py --v15-cache

exit code:
    0 = PASS (leak / suspect 0 件、 学習許可)
    1 = FAIL (leak / suspect 検出、 学習禁止)
    2 = ERROR (file not found 等)

★ V15 production 完全不変 ★ — read-only audit のみ。
"""
from __future__ import annotations

import argparse
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "tools"))

from leak_audit_automation import (
    audit_features,
    summarize,
    _load_v15_cache,
    _read_features_file,
)


def gate(features_path: str | None = None, use_v15_cache: bool = False,
         allow_suspect: bool = False, verbose: bool = True) -> int:
    """leak audit gate。

    Args:
        features_path: features list file (1 feature per line)
        use_v15_cache: True なら V15 cache (145 features) を audit
        allow_suspect: True なら HIGH_CORR_LEAK_SUSPECT を warning 扱い (FAIL → WARN)
        verbose: print details

    Returns:
        exit code (0=PASS, 1=FAIL, 2=ERROR)
    """
    cache_df = None
    try:
        cache_df, cache_features = _load_v15_cache()
    except FileNotFoundError:
        if use_v15_cache:
            print("[ERROR] V15 cache not found", file=sys.stderr)
            return 2

    if use_v15_cache:
        features = cache_features
        label = "V15 cache (145 features)"
    elif features_path:
        if not os.path.exists(features_path):
            print(f"[ERROR] features file not found: {features_path}", file=sys.stderr)
            return 2
        features = _read_features_file(features_path)
        label = features_path
    else:
        print("[ERROR] --features or --v15-cache required", file=sys.stderr)
        return 2

    if verbose:
        print(f"=== v15.2 train gate ===", flush=True)
        print(f"source: {label}", flush=True)
        print(f"features: {len(features)}", flush=True)

    results = audit_features(features, cache_df=cache_df)
    s = summarize(results)

    if verbose:
        print()
        print("--- summary ---")
        for v, n in sorted(s["by_verdict"].items()):
            print(f"  {v}: {n}")

    # 1. LEAK detect → 即 FAIL
    if s["leaks"]:
        print()
        print(f"[FAIL] {len(s['leaks'])} LEAK features detected (学習禁止):")
        for fname in s["leaks"]:
            r = next(r for r in results if r.feature == fname)
            print(f"  - {r.feature} | datatype={r.datatype} | timing={r.timing} | {r.note}")
        print()
        print("★ V15 production 維持、 v15.2 学習 NO-GO ★")
        return 1

    # 2. HIGH_CORR_LEAK_SUSPECT
    if s["suspects"]:
        print()
        print(f"[{'WARN' if allow_suspect else 'FAIL'}] "
              f"{len(s['suspects'])} HIGH_CORR_LEAK_SUSPECT detected:")
        for fname in s["suspects"]:
            r = next(r for r in results if r.feature == fname)
            corr = f"{r.corr_target:+.4f}" if r.corr_target is not None else "?"
            print(f"  - {r.feature} | corr={corr} | datatype={r.datatype} | {r.note}")
        print()
        print("★ 高 corr feature は monotonic + per-finish + train/test stability で")
        print("  individual audit 必要 (V15.1 SKB 教訓: corr 0.137 ですら NO-GO 判定)")
        if not allow_suspect:
            print("★ V15 production 維持、 v15.2 学習 NO-GO ★")
            return 1
        # allow_suspect=True なら warning 扱い (5/24+ 実 audit 用)

    # 3. UNKNOWN datatype
    if s["by_verdict"].get("UNKNOWN", 0) > 0:
        print()
        print(f"[WARN] {s['by_verdict']['UNKNOWN']} UNKNOWN datatype features:")
        unknowns = [r for r in results if r.verdict == "UNKNOWN"]
        for r in unknowns:
            print(f"  - {r.feature} | source={r.note}")
        print()
        print("★ unknown datatype は manual audit 必要 ★")
        return 1

    # 4. PREV_ONLY warning (LEAK ではないが要確認)
    if s["prev_only"]:
        print()
        print(f"[INFO] {len(s['prev_only'])} PREV_ONLY features (前走 ref として要確認):")
        for fname in s["prev_only"]:
            r = next(r for r in results if r.feature == fname)
            print(f"  - {r.feature} | datatype={r.datatype}")
        print()
        print("(prev_only は LEAK ではないが、 「current race 用に使われていないか」 manual check)")

    print()
    print(f"[PASS] {len(features)} features all clean (leak=0, suspect=0)")
    print("★ v15.2 学習 GO 条件: 本 gate PASS + WF AUC + LIVE retro 全 PASS が必要 ★")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="v15.2 train gate (leak audit)")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--features", help="features list file (1 feature per line)")
    src.add_argument("--v15-cache", action="store_true",
                     help="V15 cache (145 features) を audit (sanity)")
    parser.add_argument("--allow-suspect", action="store_true",
                        help="HIGH_CORR_LEAK_SUSPECT を warning 扱い (default FAIL)")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    return gate(features_path=args.features,
                use_v15_cache=args.v15_cache,
                allow_suspect=args.allow_suspect,
                verbose=not args.quiet)


if __name__ == "__main__":
    sys.exit(main())
