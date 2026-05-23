# 5/23 V21 Per-Race Paper 実機テスト

**起動時刻**: 2026-05-23 14:00 頃  
**目的**: 残り午後 R で V21 paper 通知の実機テスト

---

## 起動状態

| 項目 | 値 |
|------|-----|
| PID | **20476** |
| V21 model | models/v21_candidate.pkl.gz (2.0MB) ✅ |
| Discord | DISCORD_WEBHOOK_UPDATES (fallback、V21_PAPER 未設定) |
| Active timers | **11 R** |
| Log | logs/v21_paper_20260523.log |

---

## タイマー一覧 (残り 11 R)

| レース | 発走 | V21 fire 予定 |
|--------|------|--------------|
| 東京9R | 14:20 | **14:03** |
| 京都9R | 14:31 | 14:14 |
| 新潟10R | 14:45 | 14:28 |
| 東京10R | 14:55 | 14:38 |
| 京都10R | 15:05 | 14:48 |
| 新潟11R | 15:20 | 15:03 |
| 東京11R | 15:30 | 15:13 |
| 京都11R | 15:45 | 15:28 |
| 新潟12R | 16:01 | 15:44 |
| 東京12R | 16:10 | 15:53 |
| 京都12R | 16:30 | **16:13** |

---

## V15 並行稼働確認

| PID | プロセス | 状態 |
|-----|---------|------|
| 28500 | race_auto_notify (V15 本番) | ✅ 生存 |
| 20476 | v21_per_race_paper (V21 paper) | ✅ 生存 |

**干渉なし確認**: 完全独立プロセス、メモリ/ロック共有なし。

---

## Discord 設定状況

| webhook | 設定 |
|---------|------|
| DISCORD_WEBHOOK_V21_PAPER | **未設定** (要 Discord で webhook 作成) |
| DISCORD_WEBHOOK_UPDATES | ✅ 設定済 (fallback として使用中) |

**5/23 テスト**: #アップデート チャンネルに【V21 paper】通知が届く  
**5/24 正式稼働前**: Discord で V21 専用 webhook 作成 → `.env` に追加推奨

---

## V21 paper 通知 format

```
【V21 paper — 投票しないでください】
📝 東京9R (V21 paper)
V21候補モデル (TYB 全部入り) | V15+10特徴量 WF AUC 0.8696

三連複フォーメーション 7点 (paper)
1列目: X
2列目: X, X
3列目: X, X, X, X, X

TYB: [取得済/未取得]
V15との差異: 軸馬一致/相違 ← 注目

⚠ これはpaper予測です。実 cash 投票は V15 買い目のみ使用してください。
```

---

## 5/24 自動化手順

1. Discord で V21 paper 専用 Webhook URL 作成
2. `.env` に追加: `DISCORD_WEBHOOK_V21_PAPER=https://discord.com/api/webhooks/...`
3. schtask 登録:
   ```bat
   schtasks /Create /TN "Keiba-V21PaperNotify" /SC WEEKLY /D SAT,SUN /ST 08:50 /TR "C:\Users\takum\keiba-ai\tools\v21_paper_notify.bat" /RL HIGHEST
   ```
4. bat ファイル作成 (tools/v21_paper_notify.bat)

---

*起動: 2026-05-23 14:00 | V15 production 不変 | V21 paper のみ*
