"""Discord channel routing wrapper (Session #40 B1).

既存 tools/notify.py (bets / updates 2 channel) の上に
3 channel (alerts / investments / updates) を提供する薄い wrapper。

channel:
  - investments  (新): 5/9 当日 投資通知。 DISCORD_WEBHOOK_INVESTMENTS or BETS or fallback
  - alerts       (新): 失敗・障害。 DISCORD_WEBHOOK_ALERTS or fallback
  - updates      (既存): 通常進捗。 DISCORD_WEBHOOK_UPDATES or fallback

V15 production の既存通知経路 (notify.py / notify_done.py) は完全に不変、
本 wrapper を新規利用箇所のみ。

usage:
  from tools.discord_routing import notify
  notify("5/9 投資完了", "12R 1勝...", channel="investments", color="green")
  notify("Cookie 失効", "refresh_cookie 失敗", channel="alerts", color="red")
  notify("scrape 完了", "...", channel="updates", color="blue")

CLI:
  python tools/discord_routing.py --title X --body Y --channel alerts --color red
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

BASE = Path(r"C:/Users/takum/keiba-ai")
sys.path.insert(0, str(BASE / "tools"))


# ===== 拡張 webhook 解決 =====

def _load_env() -> dict:
    """既存 notify.py と同様 .env を直接 parse"""
    env = {}
    p = BASE / ".env"
    if not p.exists():
        return env
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def _resolve_webhook(channel: str) -> str | None:
    env = _load_env()
    keys_priority = []
    if channel == "investments":
        keys_priority = ["DISCORD_WEBHOOK_INVESTMENTS", "DISCORD_WEBHOOK_BETS"]
    elif channel == "alerts":
        keys_priority = ["DISCORD_WEBHOOK_ALERTS", "DISCORD_WEBHOOK_UPDATES"]
    elif channel == "updates":
        keys_priority = ["DISCORD_WEBHOOK_UPDATES"]
    elif channel == "bets":
        keys_priority = ["DISCORD_WEBHOOK_BETS"]
    keys_priority.append("DISCORD_WEBHOOK_URL")  # 共通 fallback
    for k in keys_priority:
        url = env.get(k, "")
        if url.startswith("https://"):
            return url
    return None


def notify(title: str, body: str, *, channel: str = "updates", color: str = "green") -> bool:
    """3 channel routing notify."""
    # 既存 notify.send_discord は (bets, updates) のみ対応のため、
    # investments / alerts は拡張 webhook を直接叩く実装に分岐
    if channel in ("bets", "updates"):
        try:
            from notify import send_discord
            return bool(send_discord(title, body, color=color, channel=channel))
        except Exception as e:
            print(f"[discord_routing] notify.send_discord error: {e}", file=sys.stderr)

    # investments / alerts は requests で直接 webhook 投げ
    url = _resolve_webhook(channel)
    if not url:
        print(f"[discord_routing] no webhook for channel={channel}", file=sys.stderr)
        return False
    color_map = {"green": 0x4ade80, "yellow": 0xf0c040, "red": 0xff4060, "blue": 0x60b0ff}
    cint = color_map.get(color, 0x4ade80)
    payload = {
        "embeds": [{
            "title": title,
            "description": body[:4000],
            "color": cint,
        }]
    }
    try:
        import requests
        r = requests.post(url, json=payload, timeout=10)
        return r.status_code in (200, 204)
    except Exception as e:
        print(f"[discord_routing] post error: {e}", file=sys.stderr)
        return False


def cli():
    p = argparse.ArgumentParser(description="Discord channel routing wrapper")
    p.add_argument("--title", required=True)
    p.add_argument("--body", default="")
    p.add_argument("--channel", default="updates",
                   choices=["investments", "alerts", "updates", "bets"])
    p.add_argument("--color", default="green",
                   choices=["green", "yellow", "red", "blue"])
    args = p.parse_args()
    ok = notify(args.title, args.body, channel=args.channel, color=args.color)
    print(f"{'OK' if ok else 'FAIL'}: {args.title} -> {args.channel}")


if __name__ == "__main__":
    cli()
