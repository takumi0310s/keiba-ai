#!/usr/bin/env python
"""コマンドラインからDiscord通知を送信

Usage:
    python tools/notify_done.py                              # "タスク完了"
    python tools/notify_done.py "学習完了" "AUC 0.8110"      # タイトル + メッセージ
    python tools/notify_done.py "エラー" "詳細" --color red  # 色指定
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tools'))

from notify import send_discord

title = sys.argv[1] if len(sys.argv) > 1 else "タスク完了"
msg = sys.argv[2] if len(sys.argv) > 2 else ""
color = "green"
if "--color" in sys.argv:
    idx = sys.argv.index("--color")
    if idx + 1 < len(sys.argv):
        color = sys.argv[idx + 1]

ok = send_discord(title, msg, color=color, channel="updates")
print(f"{'OK' if ok else 'SKIP (URL未設定)'}: {title}")
