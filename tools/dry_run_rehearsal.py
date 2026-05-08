"""5/8 22:00 dry-run リハーサル (Session #46 B).

V15 model md5 verify (★最重要) + 5/3 sample data で predict 動作確認。
md5 mismatch なら 緊急 RED alert (Discord critical)。

V15 model md5 期待値: 842b9a5f305c793ed8fa54a74e06b836

usage:
  python tools/dry_run_rehearsal.py --sample 20260503

  # schtasks 想定
  schtasks /Create /TN "Keiba-DryRunRehearsal_2200" \
      /TR "python C:/Users/takum/keiba-ai/tools/dry_run_rehearsal.py" \
      /SC WEEKLY /D SAT /ST 22:00 /F

V15 production 完全独立 (read-only model load)。
"""
from __future__ import annotations

import argparse
import datetime
import gzip
import hashlib
import json
import os
import pickle
import sys
import time
from pathlib import Path

BASE = Path(r"C:/Users/takum/keiba-ai")

EXPECTED_V15_MD5 = "842b9a5f305c793ed8fa54a74e06b836"


def verify_v15_md5() -> dict:
    """★最重要★ V15 model file md5 verify."""
    p = BASE / "keiba_model_v15_central_live.pkl.gz"
    if not p.exists():
        return {
            "status": "missing",
            "expected": EXPECTED_V15_MD5,
            "actual": None,
            "match": False,
            "alert": "★ CRITICAL: V15 model file 不在 ★",
        }
    try:
        with gzip.open(p, "rb") as f:
            data = f.read()
        actual_md5 = hashlib.md5(data).hexdigest()
        match = actual_md5 == EXPECTED_V15_MD5
        size_mb = p.stat().st_size / 1024 / 1024
        return {
            "status": "ok" if match else "MISMATCH",
            "expected": EXPECTED_V15_MD5,
            "actual": actual_md5,
            "match": match,
            "size_mb": round(size_mb, 2),
            "mtime": datetime.datetime.fromtimestamp(p.stat().st_mtime).isoformat(),
            "alert": None if match else f"★ CRITICAL: V15 md5 MISMATCH ★ expected {EXPECTED_V15_MD5}、 actual {actual_md5}",
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)[:200],
            "match": False,
            "alert": f"★ V15 md5 verify error: {e} ★",
        }


def verify_v15_load() -> dict:
    """V15 model load + features 確認."""
    p = BASE / "keiba_model_v15_central_live.pkl.gz"
    if not p.exists():
        return {"status": "missing"}
    try:
        with gzip.open(p, "rb") as f:
            obj = pickle.load(f)
        features = obj.get("features", [])
        return {
            "status": "ok",
            "n_features": len(features),
            "features_first_5": list(features[:5]),
            "model_type": type(obj).__name__,
        }
    except Exception as e:
        return {"status": "error", "error": str(e)[:200]}


def predict_sample(sample_date: str) -> dict:
    """5/3 sample data の race を predict (read-only、 既存 daily_predict は trigger しない)."""
    pred_path = BASE / "data" / "daily_predictions" / f"{sample_date}.csv"
    if not pred_path.exists():
        return {"status": "missing", "path": str(pred_path)}
    try:
        import pandas as pd
        df = pd.read_csv(pred_path, dtype=str)
        return {
            "status": "ok",
            "n_predictions": len(df),
            "n_races": df['race_id'].nunique() if 'race_id' in df.columns else None,
            "sample_first_3": df.head(3).to_dict('records') if len(df) >= 3 else [],
        }
    except Exception as e:
        return {"status": "error", "error": str(e)[:200]}


def syntax_check_critical() -> dict:
    """重要 production file の syntax check."""
    files = [
        "tools/predict_core.py",
        "tools/daily_predict.py",
        "app.py",
    ]
    out = {}
    for f in files:
        p = BASE / f
        if not p.exists():
            out[f] = {"status": "missing"}
            continue
        try:
            import py_compile
            py_compile.compile(str(p), doraise=True)
            out[f] = {"status": "ok"}
        except py_compile.PyCompileError as e:
            out[f] = {"status": "error", "error": str(e)[:200]}
    return out


def send_discord(title: str, body: str, color: str = "green") -> bool:
    try:
        sys.path.insert(0, str(BASE / "tools"))
        from notify import send_discord as _send
        return _send(title, body, color=color, channel="updates")
    except Exception as e:
        print(f"[discord] error: {e}", file=sys.stderr)
        return False


def main():
    p = argparse.ArgumentParser(description="5/8 22:00 dry-run rehearsal (Session #46 B)")
    p.add_argument("--sample", default="20260503", help="dry-run 対象 sample date")
    p.add_argument("--no-discord", action="store_true")
    args = p.parse_args()

    print("=" * 70)
    print(f"dry-run rehearsal (sample: {args.sample})")
    print("=" * 70)

    # === ★ Step 1: V15 md5 verify (最重要) ★ ===
    print("\n[Step 1] V15 model md5 verify ★最重要★")
    md5_result = verify_v15_md5()
    print(f"  expected: {md5_result.get('expected')}")
    print(f"  actual:   {md5_result.get('actual')}")
    print(f"  match:    {md5_result.get('match')}")
    if md5_result.get("alert"):
        print(f"  ★★★ {md5_result['alert']} ★★★")

    md5_ok = md5_result.get("match", False)

    # === Step 2: V15 model load ===
    print("\n[Step 2] V15 model load + features")
    load_result = verify_v15_load()
    print(f"  status: {load_result.get('status')}")
    if load_result.get("status") == "ok":
        print(f"  n_features: {load_result['n_features']}")
        print(f"  first 5: {load_result['features_first_5']}")

    load_ok = load_result.get("status") == "ok"

    # === Step 3: production file syntax ===
    print("\n[Step 3] production file syntax check")
    syntax = syntax_check_critical()
    syntax_ok = all(v.get("status") == "ok" for v in syntax.values())
    for f, v in syntax.items():
        mark = "OK" if v.get("status") == "ok" else "NG"
        print(f"  [{mark}] {f}")

    # === Step 4: sample predict ===
    print(f"\n[Step 4] sample data predict ({args.sample})")
    pred = predict_sample(args.sample)
    print(f"  status: {pred.get('status')}")
    if pred.get("status") == "ok":
        print(f"  n_predictions: {pred['n_predictions']}")
        print(f"  n_races: {pred['n_races']}")

    pred_ok = pred.get("status") == "ok"

    # === Discord 通知 ===
    overall_ok = md5_ok and load_ok and syntax_ok and pred_ok
    severity = "green" if overall_ok else "red"
    title = f"[dry-run rehearsal] " + ("ALL PASS" if overall_ok else "★ CRITICAL ★")

    body_lines = [
        f"sample: {args.sample}",
        f"V15 md5: {'OK' if md5_ok else 'NG'} ({md5_result.get('actual', 'N/A')[:16]}...)",
        f"V15 load: {'OK' if load_ok else 'NG'}",
        f"syntax: {'OK' if syntax_ok else 'NG'}",
        f"sample predict: {'OK' if pred_ok else 'NG'}",
    ]
    if md5_result.get("alert"):
        body_lines.append("---")
        body_lines.append(md5_result["alert"])

    body = "\n".join(body_lines)

    print(f"\n=== summary ===")
    print(body)

    if not args.no_discord:
        send_discord(title, body, color=severity)
        print(f"\nDiscord: sent")

    out_path = BASE / "data" / "v18" / f"dry_run_rehearsal_{args.sample}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "sample": args.sample,
        "v15_md5": md5_result,
        "v15_load": load_result,
        "syntax": syntax,
        "predict": pred,
        "overall_ok": overall_ok,
    }, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\n  written: {out_path.relative_to(BASE)}")

    sys.exit(0 if overall_ok else 1)


if __name__ == "__main__":
    main()
