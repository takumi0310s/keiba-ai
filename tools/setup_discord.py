#!/usr/bin/env python
"""Discord Webhook セットアップウィザード（2チャンネル対応）"""
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
2. 2つのテキストチャンネルを作成:
   - #買い目 (レース予測通知用)
   - #アップデート (システム通知用)
3. 各チャンネルの設定(歯車) → 連携サービス → Webhook
4. 「新しいWebhook」→ 名前を「keiba-ai」に変更
5. 「WebhookURLをコピー」して下に貼り付け
""")

# Load existing
existing = {}
if os.path.exists(ENV_PATH):
    with open(ENV_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            for key in ['DISCORD_WEBHOOK_BETS', 'DISCORD_WEBHOOK_UPDATES', 'DISCORD_WEBHOOK_URL']:
                if line.strip().startswith(f'{key}='):
                    existing[key] = line.strip().split('=', 1)[1].strip('"').strip("'")

# Input URLs
def ask_url(label, key):
    cur = existing.get(key, '')
    if cur:
        print(f"  現在の{label}: {cur[:50]}...")
        change = input(f"  変更しますか? (y/N): ").strip().lower()
        if change != 'y':
            return cur
    url = input(f"  {label} Webhook URL: ").strip()
    if url and not url.startswith('https://discord.com/api/webhooks/'):
        print("  WARNING: 無効なURL（スキップ）")
        return cur
    return url or cur

bets_url = ask_url("買い目チャンネル", "DISCORD_WEBHOOK_BETS")
updates_url = ask_url("アップデートチャンネル", "DISCORD_WEBHOOK_UPDATES")

# Save to .env
lines = []
keys_written = set()
if os.path.exists(ENV_PATH):
    with open(ENV_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            key = line.strip().split('=', 0)[0] if '=' in line else ''
            if key == 'DISCORD_WEBHOOK_BETS' and bets_url:
                lines.append(f'DISCORD_WEBHOOK_BETS="{bets_url}"\n')
                keys_written.add('DISCORD_WEBHOOK_BETS')
            elif key == 'DISCORD_WEBHOOK_UPDATES' and updates_url:
                lines.append(f'DISCORD_WEBHOOK_UPDATES="{updates_url}"\n')
                keys_written.add('DISCORD_WEBHOOK_UPDATES')
            elif key == 'DISCORD_WEBHOOK_URL':
                # Keep as fallback
                lines.append(line if not line.endswith('\n') else line)
                keys_written.add('DISCORD_WEBHOOK_URL')
            else:
                lines.append(line if line.endswith('\n') else line + '\n')

if 'DISCORD_WEBHOOK_BETS' not in keys_written and bets_url:
    lines.append(f'DISCORD_WEBHOOK_BETS="{bets_url}"\n')
if 'DISCORD_WEBHOOK_UPDATES' not in keys_written and updates_url:
    lines.append(f'DISCORD_WEBHOOK_UPDATES="{updates_url}"\n')

with open(ENV_PATH, 'w', encoding='utf-8') as f:
    f.writelines(lines)

print(f"\n.envに保存しました。")

# Test
print("\nテスト通知を送信中...")
from notify import send_discord, _ENV_CACHE
_ENV_CACHE = None  # Force reload

ok1 = send_discord("🏇 テスト (買い目)", "買い目チャンネルのテスト通知です。", color="green", channel="bets")
print(f"  買い目チャンネル: {'OK' if ok1 else 'SKIPPED (URL未設定)'}")

ok2 = send_discord("テスト (アップデート)", "アップデートチャンネルのテスト通知です。", color="blue", channel="updates")
print(f"  アップデートチャンネル: {'OK' if ok2 else 'SKIPPED (URL未設定)'}")

if ok1 or ok2:
    print("\n成功! Discordを確認してください。")
else:
    print("\n両方とも未設定です。URLを確認してください。")
