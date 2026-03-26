"""Discord通知モジュール（チャンネル振り分け対応）

Usage:
    from tools.notify import send_discord
    send_discord("予測完了", "...", color="green", channel="bets")
    send_discord("スクレイピング完了", "...", color="blue", channel="updates")

Channels:
    "bets"    → DISCORD_WEBHOOK_BETS (買い目通知)
    "updates" → DISCORD_WEBHOOK_UPDATES (システム通知)
    未指定     → DISCORD_WEBHOOK_URL (フォールバック)
"""
import os
import requests

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

COLORS = {"green": 0x4ade80, "yellow": 0xf0c040, "red": 0xff4060, "blue": 0x60b0ff}

_ENV_CACHE = None


def _load_env():
    global _ENV_CACHE
    if _ENV_CACHE is not None:
        return _ENV_CACHE
    _ENV_CACHE = {}
    env_path = os.path.join(BASE_DIR, '.env')
    if not os.path.exists(env_path):
        return _ENV_CACHE
    with open(env_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if '=' in line and not line.startswith('#'):
                key, val = line.split('=', 1)
                _ENV_CACHE[key.strip()] = val.strip('"').strip("'")
    return _ENV_CACHE


def _get_webhook_url(channel="updates"):
    env = _load_env()
    if channel == "bets":
        url = env.get('DISCORD_WEBHOOK_BETS', '')
        if url.startswith('https://'):
            return url
    if channel == "updates":
        url = env.get('DISCORD_WEBHOOK_UPDATES', '')
        if url.startswith('https://'):
            return url
    # Fallback
    url = env.get('DISCORD_WEBHOOK_URL', '')
    return url if url.startswith('https://') else None


def send_discord(title, message, color="green", fields=None, channel="updates"):
    """Discord Webhook通知を送信。URLが未設定ならスキップ。

    Args:
        channel: "bets" (買い目) or "updates" (システム通知)
    """
    url = _get_webhook_url(channel)
    if not url:
        return False

    embed = {
        "title": title[:256],
        "description": message[:2000],
        "color": COLORS.get(color, COLORS["blue"]),
    }
    if fields:
        embed["fields"] = [{"name": str(k)[:256], "value": str(v)[:200], "inline": True}
                           for k, v in list(fields.items())[:10]]

    try:
        resp = requests.post(url, json={"embeds": [embed]}, timeout=10)
        return resp.status_code in (200, 204)
    except Exception:
        return False
