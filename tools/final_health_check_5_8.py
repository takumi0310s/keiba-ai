"""5/8 朝 final health check (Session #40 A4).

5/9 投資直前の総合 health check。 翌日 (5/9) に V15 daily_predict が
完全動作する前提を担保する。

Check 項目:
1. V15 model file が読み込めるか (keiba_model_v15_central_live.pkl.gz)
2. predict_core.py の syntax OK
3. daily_predict.py の syntax OK
4. netkeiba Cookie 有効 (refresh_cookie.py --check)
5. JRDB データ最新 (jrdb_kyi.csv の最新日付 ≥ 5/3)
6. jra_payouts.csv 最新化 (4/6 停止確認、 撤退対応情報のみ)
7. .env の Discord Webhook 設定 (BETS / UPDATES 両方)
8. data/cumulative_results.csv 累計収支確認 (≥ 0)
9. 当日タスクスケジューラ (DailyPredict / RaceAutoNotify) 登録確認
10. 撤退余裕 (累計 - 撤退ライン -50,000円)

Discord (channel: alerts) に結果通知。
critical 項目 1 つでも NG なら exit 1 → schtasks 連鎖停止。

usage:
  python tools/final_health_check_5_8.py
  python tools/final_health_check_5_8.py --no-discord  # 通知抑制
"""
from __future__ import annotations

import argparse
import datetime
import gzip
import os
import pickle
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

BASE = Path(r"C:/Users/takum/keiba-ai")


def check_v15_model() -> Tuple[bool, str]:
    p = BASE / "keiba_model_v15_central_live.pkl.gz"
    if not p.exists():
        return (False, f"V15 live model not found: {p.name}")
    try:
        with gzip.open(p, "rb") as f:
            obj = pickle.load(f)
        size_mb = p.stat().st_size / 1024 / 1024
        return (True, f"V15 model OK ({size_mb:.1f} MB)")
    except Exception as e:
        return (False, f"V15 model load failed: {e}")


def check_syntax(rel_path: str) -> Tuple[bool, str]:
    p = BASE / rel_path
    if not p.exists():
        return (False, f"file not found: {rel_path}")
    try:
        import py_compile
        py_compile.compile(str(p), doraise=True)
        return (True, f"{rel_path} syntax OK")
    except py_compile.PyCompileError as e:
        return (False, f"{rel_path} syntax error: {e}")


def check_cookie() -> Tuple[bool, str]:
    refresh = BASE / "tools" / "refresh_cookie.py"
    if not refresh.exists():
        return (False, "refresh_cookie.py not found")
    try:
        result = subprocess.run(
            [sys.executable, str(refresh), "--check"],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            return (True, "Cookie OK (--check)")
        return (False, f"Cookie check failed: {result.stdout[-200:]} {result.stderr[-200:]}")
    except subprocess.TimeoutExpired:
        return (False, "Cookie check timeout (>30s)")
    except Exception as e:
        return (False, f"Cookie check error: {e}")


def check_jrdb_freshness(min_date: str = "20260503") -> Tuple[bool, str]:
    """JRDB 鮮度: jrdb 系 csv の最新 mtime / または extracted/ Bac の最新ファイル日付."""
    p_extract = BASE / "data" / "jrdb" / "extracted" / "Bac"
    if p_extract.exists():
        files = list(p_extract.glob("BAC*.txt"))
        if files:
            # filename: BAC<YYMMDD>.txt → 26 04 25 → 2026-04-25
            dates = []
            for f in files:
                name = f.stem  # 'BAC260425'
                if len(name) == 9:
                    yy = name[3:5]
                    mm = name[5:7]
                    dd = name[7:9]
                    yyyy = "20" + yy
                    dates.append(yyyy + mm + dd)
            if dates:
                max_date = max(dates)
                if max_date >= min_date:
                    return (True, f"JRDB latest {max_date} >= {min_date}")
                return (False, f"JRDB stale: latest {max_date} < {min_date}")
    return (False, "JRDB extracted dir 未確認 or 空")


def check_payouts_freshness(min_date: str = "20260406") -> Tuple[bool, str]:
    """payouts は 4/6 で停止確認済 (CLAUDE.md 既知バグ)、 INFO のみ"""
    p = BASE / "data" / "jra_payouts.csv"
    if not p.exists():
        return (True, "[INFO] jra_payouts.csv missing (Phase 3 で JV-Link 切替予定)")
    try:
        import pandas as pd
        df = pd.read_csv(p, dtype={'race_date': str}, usecols=['race_date'])
        max_date = df['race_date'].max()
        is_fresh = max_date >= min_date
        return (True, f"[INFO] payouts latest {max_date} (4/6 停止 既知)")
    except Exception as e:
        return (True, f"[INFO] payouts check error (非critical): {e}")


def check_env_webhooks() -> Tuple[bool, str]:
    p = BASE / ".env"
    if not p.exists():
        return (False, ".env not found")
    try:
        env = {}
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip().strip('"').strip("'")
        bets = env.get("DISCORD_WEBHOOK_BETS", "")
        upd = env.get("DISCORD_WEBHOOK_UPDATES", "")
        fb = env.get("DISCORD_WEBHOOK_URL", "")
        ok_bets = bets.startswith("https://")
        ok_upd = upd.startswith("https://")
        ok_fb = fb.startswith("https://")
        if not (ok_bets or ok_fb):
            return (False, "DISCORD_WEBHOOK_BETS / URL 未設定")
        if not (ok_upd or ok_fb):
            return (False, "DISCORD_WEBHOOK_UPDATES / URL 未設定")
        return (True, f"Discord webhooks OK (bets={ok_bets}, updates={ok_upd}, fb={ok_fb})")
    except Exception as e:
        return (False, f"env check error: {e}")


def check_cumulative() -> Tuple[bool, str, int]:
    """累計収支の sanity check.

    CSV は BATCH 仮想 + USER 実投資 mixed のため、 撤退ライン -50,000 JPY のみ判定。
    CLAUDE.md の +13,530 円 (USER 実) と一致させる必要はない (生データ raw)。
    """
    p = BASE / "data" / "cumulative_results.csv"
    if not p.exists():
        return (False, "cumulative_results.csv not found", 0)
    try:
        import pandas as pd
        df = pd.read_csv(p, low_memory=False)
        df['profit_num'] = pd.to_numeric(df['profit'], errors='coerce').fillna(0)
        total = int(df['profit_num'].sum())
        margin = total - (-50_000)
        ok = total > -50_000  # 撤退ライン未達 で OK
        msg = f"raw cumulative={total:+,d} JPY, retire margin={margin:+,d} JPY (USER 実: +13,530 per CLAUDE.md)"
        return (ok, msg, total)
    except Exception as e:
        return (False, f"cumulative check error: {e}", 0)


def check_schtasks() -> Tuple[bool, str]:
    """Windows schtasks に DailyPredict / RaceAutoNotify が登録されているか確認"""
    try:
        result = subprocess.run(
            ["schtasks", "/Query", "/FO", "CSV"],
            capture_output=True, text=True, timeout=20,
        )
        if result.returncode != 0:
            return (False, "schtasks query failed")
        out = result.stdout
        critical_tasks = ["DailyPredict", "RaceAutoNotify"]
        missing = [t for t in critical_tasks if t not in out]
        if missing:
            return (False, f"schtasks 未登録: {missing}")
        return (True, f"schtasks OK ({len(critical_tasks)} 件登録確認)")
    except subprocess.TimeoutExpired:
        return (False, "schtasks query timeout")
    except Exception as e:
        return (False, f"schtasks error: {e}")


def send_discord(title: str, body: str, color: str = "green", channel: str = "updates") -> bool:
    try:
        sys.path.insert(0, str(BASE / "tools"))
        from notify import send_discord as _send
        return _send(title, body, color=color, channel=channel)
    except Exception:
        return False


def main():
    p = argparse.ArgumentParser(description="5/8 朝 final health check (Session #40 A4)")
    p.add_argument("--no-discord", action="store_true")
    args = p.parse_args()

    checks: List[Tuple[str, str, bool, str, bool]] = []  # (name, category, ok, msg, critical)

    ok, msg = check_v15_model()
    checks.append(("V15 model", "model", ok, msg, True))

    for f in ["tools/predict_core.py", "tools/daily_predict.py", "app.py"]:
        ok, msg = check_syntax(f)
        checks.append((f"syntax {f}", "syntax", ok, msg, True))

    ok, msg = check_cookie()
    checks.append(("netkeiba Cookie", "auth", ok, msg, True))

    ok, msg = check_jrdb_freshness("20260503")
    checks.append(("JRDB 鮮度", "data", ok, msg, True))

    ok, msg = check_payouts_freshness()
    checks.append(("jra_payouts.csv", "data", ok, msg, False))  # non-critical

    ok, msg = check_env_webhooks()
    checks.append((".env webhooks", "config", ok, msg, True))

    ok, msg, total = check_cumulative()
    checks.append(("累計収支", "finance", ok, msg, True))

    ok, msg = check_schtasks()
    checks.append(("schtasks", "ops", ok, msg, True))

    # ===== 集計 =====
    fails = [(n, cat, m) for n, cat, ok, m, crit in checks if not ok and crit]
    warnings = [(n, m) for n, _, ok, m, crit in checks if not ok and not crit]
    successes = [(n, m) for n, _, ok, m, _ in checks if ok]

    print("=" * 60)
    print(f"5/8 朝 final health check  ({datetime.datetime.now():%Y-%m-%d %H:%M:%S})")
    print("=" * 60)
    for n, _, ok, m, crit in checks:
        mark = "OK " if ok else ("NG " if crit else "WARN")
        print(f"  [{mark}] {n}: {m}")
    print()
    print(f"成功: {len(successes)}, 警告: {len(warnings)}, 失敗 (critical): {len(fails)}")

    severity = "green" if not fails else ("red" if fails else "yellow")
    title = f"[5/8 朝 health check] {'PASS' if not fails else 'FAIL'}"
    body_lines = [f"成功: {len(successes)}, 警告: {len(warnings)}, 失敗 (critical): {len(fails)}"]
    if fails:
        body_lines.append("---")
        body_lines.append("**critical 失敗**:")
        for n, _, m in fails:
            body_lines.append(f"  [NG] {n}: {m}")
    if warnings:
        body_lines.append("**警告**:")
        for n, m in warnings:
            body_lines.append(f"  [WARN] {n}: {m}")
    body = "\n".join(body_lines)

    if not args.no_discord:
        send_discord(title, body, color=severity, channel="updates")

    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
