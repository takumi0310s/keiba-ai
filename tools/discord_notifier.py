"""Discord 通知 薄いラッパー (重複防止 + 重要度管理).

既存 tools/notify_done.py を呼び出す前に dedup 判定を挟む。
- dedup_key 指定時: data/discord_dedup_state.json を見て 30 分以内の同一 key はスキップ
- severity=critical は dedup 無視で必ず送信

Usage (CLI):
    python tools/discord_notifier.py --title X --subtitle Y --body Z \
        --severity warning --color yellow [--dedup-key KEY] [--ttl 1800]

Usage (Python):
    from tools.discord_notifier import notify
    notify(title="X", subtitle="Y", body="Z", severity="warning", dedup_key="am3_fire_check_20260420")
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys
from pathlib import Path

BASE = Path(r"C:/Users/takum/keiba-ai")
STATE_PATH = BASE / "data" / "discord_dedup_state.json"
DEFAULT_TTL_SEC = 1800  # 30 分


def _load_state() -> dict:
    if not STATE_PATH.exists():
        return {}
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(
        json.dumps(state, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8")


def _should_skip(dedup_key: str, ttl_sec: int) -> bool:
    if not dedup_key:
        return False
    state = _load_state()
    last = state.get(dedup_key)
    if not last:
        return False
    try:
        last_dt = datetime.datetime.fromisoformat(last)
    except Exception:
        return False
    elapsed = (datetime.datetime.now() - last_dt).total_seconds()
    return elapsed < ttl_sec


def _record(dedup_key: str) -> None:
    state = _load_state()
    state[dedup_key] = datetime.datetime.now().isoformat()
    _save_state(state)


def notify(
    title: str,
    subtitle: str = "",
    body: str = "",
    severity: str = "info",
    color: str | None = None,
    dedup_key: str | None = None,
    ttl_sec: int = DEFAULT_TTL_SEC,
) -> bool:
    """Discord 通知。重複判定後 notify_done.py 経由で送信。

    Returns:
        True 送信した / False スキップ (dedup 中)
    """
    # critical は dedup 無視
    if severity != "critical" and _should_skip(dedup_key or "", ttl_sec):
        print(f"[discord_notifier] SKIP (dedup within {ttl_sec}s): {dedup_key}")
        return False

    if color is None:
        color = {"info": "blue", "warning": "yellow", "critical": "red", "ok": "green"}.get(severity, "blue")

    args = [sys.executable, str(BASE / "tools/notify_done.py"),
            title, subtitle, body, "--color", color]
    try:
        subprocess.run(args, check=False, timeout=30,
                       env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    except Exception as e:
        print(f"[discord_notifier] send err: {e}", file=sys.stderr)
        return False

    if dedup_key and severity != "critical":
        _record(dedup_key)
    return True


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--title", required=True)
    p.add_argument("--subtitle", default="")
    p.add_argument("--body", default="")
    p.add_argument("--severity", choices=["info", "ok", "warning", "critical"], default="info")
    p.add_argument("--color", default=None)
    p.add_argument("--dedup-key", default=None)
    p.add_argument("--ttl", type=int, default=DEFAULT_TTL_SEC)
    args = p.parse_args()

    sent = notify(
        title=args.title,
        subtitle=args.subtitle,
        body=args.body,
        severity=args.severity,
        color=args.color,
        dedup_key=args.dedup_key,
        ttl_sec=args.ttl,
    )
    return 0 if sent else 2


if __name__ == "__main__":
    sys.exit(main())
