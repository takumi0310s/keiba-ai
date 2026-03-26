# Discord通知セットアップ

## 1. Webhook作成

1. Discordでサーバーを開く
2. 通知を受けたいテキストチャンネルの **設定(歯車)** をクリック
3. **連携サービス** → **Webhook** → **新しいWebhook**
4. 名前を `keiba-ai` に変更（任意）
5. **WebhookURLをコピー**

## 2. セットアップ（自動）

```bash
python tools/setup_discord.py
```

URLを貼り付けるとテスト通知が送信されます。

## 3. セットアップ（手動）

`.env` に以下を追加:

```
DISCORD_WEBHOOK_URL="https://discord.com/api/webhooks/XXXX/YYYY"
```

## 4. 通知が届くタイミング

| スクリプト | 通知内容 |
|-----------|---------|
| daily_predict.py | 予測完了 |
| daily_results.py | 結果照合完了 |
| weekly_report.py | 週次レポート完了 |
| weekly_premium_update.py | Premium更新完了 |
| daily_premium_scrape.py | Daily Scrape完了 |
| scrape_speed_index.py | Speed Index取得完了 |
| scrape_premium_data.py | Premium Data取得完了 |
| scrape_super_premium.py | Super Premium取得完了 |
| train_v11_premium.py | 学習完了（採用/不採用） |

## 5. 通知をオフにする

`.env` から `DISCORD_WEBHOOK_URL` の行を削除するだけ。
