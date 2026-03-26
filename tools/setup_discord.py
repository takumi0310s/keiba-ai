#!/usr/bin/env python
"""Discord Webhook セットアップウィザード"""
import os
import sys
import io

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, 'tools'))
ENV_PATH = os.path.join(BASE_DIR, '.env')

print("=" * 50)
print("  Discord Webhook セットアップ")
print("=" * 50)
print("""
【手順】
1. Discordでサーバーを開く
2. テキストチャンネルの設定(歯車) → 連携サービス → Webhook
3. 「新しいWebhook」をクリック
4. 名前を「keiba-ai」に変更
5. 「WebhookURLをコピー」をクリック
6. 下にURLを貼り付け
""")

url = input("Webhook URL: ").strip()

if not url.startswith("https://discord.com/api/webhooks/"):
    print("ERROR: 無効なURL。https://discord.com/api/webhooks/ で始まる必要があります。")
    sys.exit(1)

# Save to .env
lines = []
replaced = False
if os.path.exists(ENV_PATH):
    with open(ENV_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip().startswith('DISCORD_WEBHOOK_URL='):
                lines.append(f'DISCORD_WEBHOOK_URL="{url}"\n')
                replaced = True
            else:
                lines.append(line)
if not replaced:
    lines.append(f'DISCORD_WEBHOOK_URL="{url}"\n')

with open(ENV_PATH, 'w', encoding='utf-8') as f:
    f.writelines(lines)

print(f"\n.envに保存しました。")

# Test
print("テスト通知を送信中...")
from notify import send_discord
ok = send_discord("keiba-ai 通知テスト", "Discord連携が正常に設定されました。", color="green",
                   fields={"ステータス": "OK", "モデル": "v10"})

if ok:
    print("成功! Discordを確認してください。")
else:
    print("失敗。URLを確認してください。")
