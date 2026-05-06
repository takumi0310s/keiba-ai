"""V15 + V18/V19 並列予測 + fall-back 機構 (試作、5/9 deploy せず)

Session #32 D。 GO 条件 #5 「fall-back 機構 (V18/V19 fail → V15 自動切替)」 のための試作。

絶対遵守 (5/9 投資保護):
  - 既存 V15 model file 触らない (read のみ)
  - predict_core.py に変更を加えない (新規 module で隔離)
  - schtasks に登録しない (手動実行のみ)
  - 5/9 当日 V15 単独投資の動作には完全無影響

Modes:
  v15_only            : V15 単独 (5/9 本番モード)
  v15_v18v19_parallel : V15 主、V18/V19 副 (5/16+ 試行モード、本セッションで試作)

Usage:
    python tools/v15_v18v19_orchestrator.py --race-id 202604010312 --mode v15_only
    python tools/v15_v18v19_orchestrator.py --race-id 202604010312 --mode v15_v18v19_parallel --dry-run

Discord 通知:
  - V15 success → 通常通知
  - V18/V19 success + V15 success → 並列通知 (主: V15、副: V18/V19)
  - V18/V19 fail → V15 単独通知 + Discord warn 「V18/V19 fall-back to V15」
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime
import traceback

BASE = Path(r"C:/Users/takum/keiba-ai")
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "tools"))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def predict_v15(race_id: str):
    """V15 単独予測。 既存 predict_one_race 経由。"""
    try:
        import predict_one_race as por
        ret = por.predict_one_race(race_id)
        if ret is None:
            return None, None, None, "predict_v15 returned None"
        result, race_name, rinfo = ret
        return result, race_name, rinfo, None
    except Exception as e:
        return None, None, None, f"predict_v15 exception: {type(e).__name__}: {e}"


def predict_v18_v19(race_id: str, race_name: str | None, rinfo: dict | None):
    """V18 単勝 + V19 複勝 予測 (試作)。

    本実装は 5/16+ で deploy 予定。 本セッションでは 設計のみ + skeleton。
    現状は NotImplementedError を返す (deploy 防止)。
    """
    # 設計:
    # 1. V18 lgb + xgb load (data/v18/models/v18_tansho_*)
    # 2. V19 lgb + xgb load (data/v18/models/v19_fukusho_*)
    # 3. predict_core で features 構築済 df を入力
    # 4. v18 ensemble 予測 → P(1着)
    # 5. v19 ensemble 予測 → P(top3)
    # 6. race-level normalize (softmax T=1.0、tools/race_normalize.py)
    # 7. EV 計算 (P × 単勝オッズ / 複勝オッズ)
    # 8. filter (単勝 p_norm>=0.5 ev>=1.2、複勝 p_norm>=0.7 ev>=1.1)
    # 9. bet 候補 (もしあれば) を返す

    return None, None, "v18_v19 not implemented (5/16+ deploy 予定、本セッション skeleton のみ)"


def notify_orchestrator_status(mode: str, race_id: str,
                                v15_result, v15_err: str | None,
                                v18_result, v18_err: str | None,
                                v19_result, v19_err: str | None,
                                fallback_triggered: bool) -> None:
    """Discord で orchestrator status を通知 (v15_v18v19_parallel mode のみ)"""
    import subprocess

    if mode == "v15_only":
        return  # 通知不要 (既存 race_auto_notify で対応)

    title = f"V15+V18/V19 並列予測 status ({race_id})"

    lines = [f"mode: {mode}"]
    lines.append(f"V15: {'OK' if v15_result is not None else 'FAIL'}")
    if v15_err:
        lines.append(f"  V15 err: {v15_err[:200]}")
    lines.append(f"V18: {'OK' if v18_result is not None else 'FAIL'}")
    if v18_err:
        lines.append(f"  V18 err: {v18_err[:200]}")
    lines.append(f"V19: {'OK' if v19_result is not None else 'FAIL'}")
    if v19_err:
        lines.append(f"  V19 err: {v19_err[:200]}")
    if fallback_triggered:
        lines.append("")
        lines.append("⚠️ fall-back triggered: V18/V19 fail → V15 単独で投資判断")

    body = "\n".join(lines)
    color = "yellow" if fallback_triggered else ("green" if v15_result is not None else "red")

    try:
        subprocess.run(
            [sys.executable, str(BASE / "tools/notify_done.py"),
             title, body[:1800], "--color", color],
            check=False, timeout=30,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )
    except Exception as e:
        print(f"[WARN] Discord 通知失敗: {e}", file=sys.stderr)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--race-id", type=str, required=True)
    p.add_argument("--mode", type=str, default="v15_only",
                   choices=["v15_only", "v15_v18v19_parallel"])
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    print(f"=== orchestrator mode={args.mode} race_id={args.race_id} ===")

    # 1. V15 (常時 主)
    v15_result, race_name, rinfo, v15_err = predict_v15(args.race_id)
    if v15_result is None:
        print(f"[NG] V15 fail: {v15_err}")
        return 1
    print(f"[OK] V15 success: {race_name}")

    if args.mode == "v15_only":
        # 5/9 本番モード: V15 単独で完了
        print(f"\nmode={args.mode} → V15 単独で完了")
        return 0

    # 2. V18/V19 並列 (5/16+ 試行モード、現状 skeleton のみ)
    print(f"\n--- V18/V19 並列予測 (試作) ---")
    v18_result, v18_err = None, "not impl"
    v19_result, v19_err = None, "not impl"
    try:
        v18_result, v19_err_pair, v18_err = predict_v18_v19(args.race_id, race_name, rinfo)
        # v19_err_pair は (v18_err, v19_err) の組または str だが現状は None
    except Exception as e:
        v18_err = f"orchestrator exception: {type(e).__name__}: {e}"
        traceback.print_exc()

    # fall-back 判定
    fallback = (v18_result is None) or (v19_result is None)
    if fallback:
        print(f"\n⚠️ V18/V19 fail → fall-back to V15 単独")

    # 3. 通知 (dry-run なら skip)
    if not args.dry_run:
        notify_orchestrator_status(
            args.mode, args.race_id,
            v15_result, v15_err,
            v18_result, v18_err,
            v19_result, v19_err,
            fallback_triggered=fallback,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
