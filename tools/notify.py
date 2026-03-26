"""Discord通知モジュール

Usage:
    from tools.notify import send_discord
    send_discord("予測完了", "本日24レース処理完了。的中: 8/24", color="green")
"""
import os
import json
import requests

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

COLORS = {"green": 0x4ade80, "yellow": 0xf0c040, "red": 0xff4060, "blue": 0x60b0ff}


def _get_webhook_url():
    env_path = os.path.join(BASE_DIR, '.env')
    if not os.path.exists(env_path):
        return None
    with open(env_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith('DISCORD_WEBHOOK_URL='):
                val = line.split('=', 1)[1].strip('"').strip("'")
                return val if val.startswith('https://') else None
    return None


def send_discord(title, message, color="green", fields=None):
    """Discord Webhook通知を送信。URLが未設定ならスキップ。"""
    url = _get_webhook_url()
    if not url:
        return False

    embed = {
        "title": title,
        "description": message[:2000],
        "color": COLORS.get(color, COLORS["blue"]),
    }
    if fields:
        embed["fields"] = [{"name": k, "value": str(v)[:200], "inline": True}
                           for k, v in list(fields.items())[:10]]

    try:
        resp = requests.post(url, json={"embeds": [embed]}, timeout=10)
        return resp.status_code in (200, 204)
    except Exception:
        return False
