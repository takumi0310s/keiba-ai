# 次セッション 起動チェックリスト

**用途**: 5/8 夜 or 5/9 朝、もしくはそれ以降のセッション 開始時

---

## 0. 即読み (1 分)

```
docs/HANDOFF_5_5_TO_5_9.md     ← 引き継ぎ書 v2 (これ 1 つで現状把握)
```

特に section 0 (v1 訂正)、section 1 (累計)、section 3 (5/9 戦略) を確認。

---

## 1. 環境 同期 (2 分)

```bash
cd C:\Users\takum\keiba-ai
git pull --rebase --autostash origin main
git log --oneline -5
```

最新 commit が **2b6dc4eb (Phase 2.5+: 5/9 本番最終調整)** か、それより新しいか確認。
本書 base = 2b6dc4eb (Session #14)。

Session #15 (本 振り返り) commit を含む場合は **次の number** が最新。

---

## 2. 重要 doc 確認 (5 分)

5/9 投資 セッション なら:

| 順 | doc | 用途 |
|----|-----|------|
| 1 | `data/results/20260509_pat_checklist.md` | **5/9 朝 まずこれ** (投票チェックリスト) |
| 2 | `data/results/20260509_operation_guide.md` | 時系列フロー |
| 3 | `data/results/20260509_pre_check.md` | 5/8 21:00 後 確認手順 |
| 4 | `data/v18/risk_management_5_9.md` | 撤退ライン |
| 5 | `data/results/20260509_final_plan_v2.md` | 採用方針 確定 |

5/9 後 振り返り セッション なら:

| 順 | doc | 用途 |
|----|-----|------|
| 1 | `data/results/20260509_summary.md` (or _auto.md) | 当日結果 |
| 2 | `data/v18/post_5_9_improvement_template.md` | 振り返りテンプレ (埋めて 5/16 改善材料化) |
| 3 | `data/v18/risk_management_5_9.md` | 5/10 判断基準 |

5/12+ NAR / Phase 2.5+ セッション なら:

| 順 | doc | 用途 |
|----|-----|------|
| 1 | `data/v18/jra_nar_integration_plan.md` | 5/12-5/24 並列運用 plan |
| 2 | `data/v18/nar_pipeline_design.md` | NAR pipeline 詳細 |
| 3 | `docs/HANDOFF_5_5_TO_5_9.md` section 4 | Phase 2.5 残タスク |

---

## 3. 状態 chk (3 分、5/9 朝なら必須)

### 3.1 Cookie 健全性

```bash
python tools/refresh_cookie.py --check
# expected: [OK] Premium 認証 OK
# 切れていたら: python tools/refresh_cookie.py --auto (期限切れ時のみ自動)
```

### 3.2 schtasks Ready 状態

```powershell
Get-ScheduledTask -TaskName 'Keiba-Morning_Sat','DailyPredict','DailyResults_Sat','Keiba-RaceDayReport_Sat','Keiba-NarDailyPredict' | ft TaskName, State, @{N='NextRun';E={(Get-ScheduledTaskInfo $_).NextRunTime}}
```

5/9 当日 必要なのは Morning_Sat (06:30) / DailyPredict (08:00) / DailyResults_Sat (18:00) / RaceDayReport_Sat (18:00)。
全 Ready で next が 5/9 当日であれば OK。

### 3.3 Discord 直近通知 確認

- Discord アプリ で #bets / #updates タブ 開く
- 最新通知が 24 時間以内 (Cookie/morning/results notify_done.py 等の trace)
- 5/8 21:00 後 / 5/9 06:30+ で受信あれば OK

### 3.4 daily_predictions/2026MMDD.csv 直近確認

```bash
ls -la data/daily_predictions/ | tail -5
```

5/9 09:00 以降なら 20260509.csv が 7-11 KB で存在 期待。

---

## 4. 静音化動作確認 (1 分、任意)

最近 ターミナルが見えてないこと:

```powershell
# 過去 24 時間で TYB monitor が 何回発火したか (毎時 X:30 = 24 回 想定)
Get-EventLog -LogName 'Microsoft-Windows-TaskScheduler/Operational' -After (Get-Date).AddDays(-1) | Where-Object {$_.Message -match 'TybPublishMonitor'} | Measure-Object
# OR
ls logs/tyb_publish*.log -lt | head -3
```

または `data/tyb_publish_log.csv` の rows が 5/4 以降で増えているか:

```bash
python -c "import pandas as pd; df = pd.read_csv('data/tyb_publish_log.csv'); print(len(df), 'rows'); print(df.tail(5))"
```

---

## 5. 並行 session 確認 (任意)

別 session が pre/post で動いている可能性:

```bash
git log --oneline --since="6 hours ago" 2>&1
# 自分の commit + 他 session commit が混在していたら rebase 必要 (autostash で OK)
```

---

## 6. tasks (TaskCreate) 確認

前回 session の TaskList が残っているか:

```
TaskList の出力を確認 (in_progress / pending あれば 引き継ぎ)
```

通常 session 越しに tasks は維持されるが、コンテキスト切り替え時 TaskGet 失敗するなら無視 OK。

---

## 7. ストイック作業時 (引き継ぎ密度 高)

長時間作業の繰り返しなら:

| step | doc |
|------|-----|
| 0 開始 | 本 chk list |
| 0 確認 | `docs/HANDOFF_5_5_TO_5_9.md` |
| 0 v1 訂正 | `docs/handoff_v1_v2_diff.md` |
| 履歴 詳細 | `docs/sessions_5_3_5_5_recap.md` |
| 教訓 | `docs/lessons_learned_5_5.md` |
| 5/9 当日 | `data/results/20260509_pat_checklist.md` |
| Phase 2.5 progress | `data/v18/phase_2_5_session10_final.md` |

---

## 8. 起動 完了

5 分で section 0-3 完了。10 分で section 0-7 完了。
迷ったら **`docs/HANDOFF_5_5_TO_5_9.md`** に戻る。

---

## 9. 緊急時のみ

| 症状 | 対応 |
|------|------|
| Cookie 切れ → premium 取得失敗 | `python tools/refresh_cookie.py --auto` |
| daily_predict 完了せず | watchdog ログ確認 → 手動 `python tools/daily_predict.py --date YYYYMMDD` |
| Discord 通知来ない | `.env` の DISCORD_WEBHOOK_URL 確認 + `python tools/notify_done.py "test" "test"` |
| schtasks 発火せず | Get-ScheduledTask で State=Ready 確認、`Start-ScheduledTask` で手動発火 |
| git pull 衝突 | `--autostash` で auto stash → rebase → unstash |
| python module 不在 | `pip install -r requirements.txt` |

---

## 10. NEVER list (禁止行為)

- ❌ 11R 投票 (重賞 + 距離不適合 全除外)
- ❌ 1R 700円 超え (案B改 上限)
- ❌ 1日 2,100円 超え
- ❌ TYB midday script 実行 (5/3 で 404 確認、Phase 2.5 観測待ち)
- ❌ v18/v19 5/9 投入 (5/16 以降)
- ❌ NAR 5/9 投入 (5/12 paper 開始)
- ❌ 累計 -50,000円 超え (絶対)
- ❌ 数字を v1 引き継ぎから transfusion (`docs/handoff_v1_v2_diff.md` 確認)

---

これだけで セッション 復帰 確実。
