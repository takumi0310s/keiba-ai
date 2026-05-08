# 5/9 緊急時 早見表 (Session #46 D)

**作成**: 2026-05-08 (Session #46 D)
**目的**: 緊急時 30 秒で対応、 印刷推奨 (A4 1 枚)

> **絶対遵守**: V15 model md5 = `842b9a5f305c793ed8fa54a74e06b836`
> 累計 -2,100 円超えたら **即停止**、 残 R 全 skip

---

## 5/9 タイムライン

```
05:00 起床
06:00 NightlySanity (5/8 23:00 起動分) Discord 確認
06:30 MorningChecklist 自動 → Discord で 8 項目 status 確認
07:00 MorningDigest
08:00 DailyPredict (V15 全レース、 ~10-15 min)
08:45 RaceAutoNotify → #bets / #investments
09:00 候補 list 確定 + PAT login
10:00- 投票 (1勝のみ、 700円 × max 3R = 2,100円)
14:50 multi_stage_predict race11_1450
15:45 multi_stage_predict race12_1545 ★主戦場
18:00 DailyResults 自動
20:30 振り返り
```

---

## 緊急時 8 シナリオ

| # | 症状 | 対処 command | 期待結果 |
|---|------|------------|--------|
| 1 | predict が動かない | `schtasks /Run /TN "Keiba-DailyPredict"` | Discord に 通知届く |
| 2 | Cookie 切れ | `python tools/refresh_cookie.py` | "OK" msg + cookies.json 更新 |
| 3 | JRDB 接続失敗 | `python tools/jrdb_health_alert.py` | "green" status |
| 4 | Discord 通知来ない | `.env` の DISCORD_WEBHOOK_* 確認 → `python tools/notify_done.py "test" "test"` | "OK: test" |
| 5 | 11R 重賞 (G1/G2/G3) | **絶対投票 NG** (案B改は 1勝のみ) | (記録のみ、 投票 skip) |
| 6 | 12R 1勝のみ | **700円 × 3R 上限** (案B改 維持) | (max 投資 2,100円) |
| 7 | PAT 投票失敗 | 諦めて記録のみ | `logs/pat_log.txt` に記録 |
| 8 | 累計 -2,100 円超 | **即停止**、 残 R 全 skip | 撤退余裕 +63,530 円 維持 |

---

## V15 不変 verify

```powershell
# 5/9 朝 即時 verify
python tools/dry_run_rehearsal.py --sample 20260503
# ↓ 期待
# [Step 1] V15 md5 verify ★最重要★
#   match: True
#   ALL PASS Discord 通知届く
```

**md5 mismatch なら ★ critical alert ★ → 5/9 投資 中止検討**

---

## 5/9 朝 自動 sequence

```
06:30 morning_checklist_generator → Discord 8 項目
07:00 morning digest
08:00 DailyPredict
08:45 RaceAutoNotify (戦略⑦ + 案B改)
09:00 ユーザー手動: 候補確認 + PAT login
10:00- 投票 (max 700円×3R = 2,100円)
```

---

## Cookie 切れ → 即対応

```powershell
# Step 1: 現状確認
python tools/refresh_cookie.py --check

# Step 2: NG なら refresh
python tools/refresh_cookie.py --auto

# Step 3: それでも NG なら interactive
python tools/refresh_cookie.py
# → netkeiba ID/PW 入力 (1-2 分)
```

---

## Discord webhook 死亡

```powershell
# .env 確認
type .env | findstr DISCORD

# test 送信
python tools/notify_done.py "test 5/9" "msg"
# → "OK: test 5/9" 返れば 復旧
```

---

## 投資判断 早見

| 状況 | 推奨 action |
|------|-----------|
| 11R 重賞 | **絶対 skip** |
| 12R 1勝 + 8頭+ + 良/稍重 | 案B改 候補 → 投票 (700円) |
| 12R 1勝 + 7頭以下 | **skip** (条件 E、 案B改 NG) |
| 12R 1勝 + 重/不良 | **skip** (条件 B、 案B改 NG) |
| 12R 1勝 + 京都 | **skip** (戦略⑦、 京都除外) |
| 12R 平場特別 | **skip** (06_特別、 -9,470円損失源) |
| 累計 < -2,100 円 | **完全停止** |

---

## ★ 印刷 推奨 ★

このページを A4 で印刷、 5/9 朝 デスクに置く。
緊急時 30 秒で読める。

---

**Session #46 D 完了 — 2026-05-08**
