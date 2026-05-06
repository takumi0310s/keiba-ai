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


# Session #36 B: 運用フィルタ (Niigata/京都 + 重〜不良 除外)
# Session #33 D 発見: Niigata 0%→28% / Kyoto top1_p3 -22.3pt の sample 構成シフト対応
EXCLUDE_COURSES_FOR_V18V19 = {'新潟', '京都'}
EXCLUDE_CONDITIONS_FOR_V18V19 = {'B', 'X'}  # 重〜不良 (top1_p3 不安定)


def is_v18v19_eligible(rinfo: dict) -> tuple[bool, str]:
    """運用フィルタ判定 (V18/V19 投票候補か)。 戻り値: (適格?, 理由)"""
    course = str(rinfo.get('course', '') if rinfo else '')
    if course in EXCLUDE_COURSES_FOR_V18V19:
        return False, f"運用フィルタ: {course} 除外 (sample 構成シフト)"
    cond = str(rinfo.get('condition_enc', rinfo.get('condition', '')) if rinfo else '')
    if cond in EXCLUDE_CONDITIONS_FOR_V18V19:
        return False, f"運用フィルタ: 馬場 {cond} 除外 (top1_p3 不安定)"
    return True, "適格 (V18/V19 投票候補)"


def predict_v18_v19(race_id: str, race_name: str | None, rinfo: dict | None,
                     v15_df=None):
    """V18 単勝 + V19 複勝 予測 (Session #36 B 本実装)。

    Args:
        race_id: race_id
        race_name: race_name
        rinfo: race_info dict (course, condition 等)
        v15_df: predict_core.predict_race の結果 DataFrame (features 含む)

    Returns:
        (v18_bets, v19_bets, error_msg) or (None, None, error_msg)
        v18_bets/v19_bets: [{'umaban': X, 'prob': Y, 'ev': Z, 'odds': W}] or []
    """
    # Step 1: 運用フィルタ
    if rinfo:
        eligible, reason = is_v18v19_eligible(rinfo)
        if not eligible:
            return [], [], f"フィルタで投票見送り: {reason}"

    # Step 2: V18/V19 model load
    try:
        import lightgbm as lgb
        v18_lgb = lgb.Booster(model_file=str(BASE / 'data/v18/models/v18_tansho_lgb.txt'))
        v19_lgb = lgb.Booster(model_file=str(BASE / 'data/v18/models/v19_fukusho_lgb.txt'))
    except Exception as e:
        return None, None, f"V18/V19 model load fail: {type(e).__name__}: {e}"

    # Step 3: features alignment
    if v15_df is None or len(v15_df) == 0:
        return None, None, "v15_df not provided or empty"

    v18_feats = v18_lgb.feature_name()
    # missing features は 0 fallback
    import numpy as np
    import pandas as pd
    df_v18 = pd.DataFrame(index=v15_df.index)
    for f in v18_feats:
        if f in v15_df.columns:
            df_v18[f] = pd.to_numeric(v15_df[f], errors='coerce').fillna(0).values
        else:
            df_v18[f] = 0.0  # default

    # Step 4: predict
    try:
        v18_p = v18_lgb.predict(df_v18.values)
        v19_p = v19_lgb.predict(df_v18.values)  # V19 is same features
    except Exception as e:
        return None, None, f"V18/V19 predict fail: {type(e).__name__}: {e}"

    # Step 5: race-level normalize (softmax T=1.0)
    def softmax(x, T=1.0):
        e = np.exp(np.array(x) / T - max(x) / T)
        return e / e.sum()
    v18_norm = softmax(v18_p, T=1.0)
    v19_norm = softmax(v19_p, T=1.0)

    # Step 6: bet 候補生成 (filter: 単勝 p>=0.5 ev>=1.2、複勝 p>=0.7 ev>=1.1)
    v18_bets, v19_bets = [], []
    if '馬番' in v15_df.columns and '単勝オッズ' in v15_df.columns:
        for i, row in v15_df.reset_index(drop=True).iterrows():
            uma = row.get('馬番', 0)
            odds = row.get('単勝オッズ', 0)
            if pd.notna(odds) and odds > 0:
                p18 = v18_norm[i]
                p19 = v19_norm[i]
                ev18 = p18 * odds
                ev19 = p19 * odds * 0.3  # 複勝オッズ ~ 単勝×0.3 仮定
                if p18 >= 0.5 and ev18 >= 1.2:
                    v18_bets.append({'umaban': int(uma), 'prob': float(p18), 'ev': float(ev18), 'odds': float(odds)})
                if p19 >= 0.7 and ev19 >= 1.1:
                    v19_bets.append({'umaban': int(uma), 'prob': float(p19), 'ev': float(ev19), 'odds': float(odds)})

    return v18_bets, v19_bets, None


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
