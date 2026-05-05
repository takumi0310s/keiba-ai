# 次セッション 起動チェックリスト

**最終更新**: 2026-05-06 00:30 (Session #25 = Phase 2.5+ 最終総括後)
**ベース commit**: 86cd1da5 (緊急 3 件対応) + Session #25 commit

---

## 0. 即読み (1 分)

```
docs/PHASE_2_5_PLUS_FINAL_RECAP_5_5.md     ← 51.5h 全総括 (寝る前最終、これ 1 つで全体像)
docs/UPDATE_INVENTORY_20260505.md          ← 6 領域棚卸し (緊急 3 件は対応済)
docs/HANDOFF_5_5_TO_5_9.md                 ← 引き継ぎ書 v2 (5/9 投資詳細)
docs/system_self_diagnosis_5_5.md          ← システム自己診断 (改善余地 リスト)
```

---

## 1. 5/6 (火) 朝 - admin 1 コマンドのみ

### 必須 (1 分)

```powershell
# 管理者として PowerShell を起動して実行
PowerShell -ExecutionPolicy Bypass -File C:\Users\takum\keiba-ai\tools\register_process_watchdog_v2.ps1
```

これで ProcessWatchdog v2 切替完了 (Disabled v1 → Enable + 静音化済 v2)。

### 確認 (任意、3 分)

```powershell
Get-ScheduledTask -TaskName "ProcessWatchdog" | Select TaskName, State
# State=Ready で OK

# 5 分後にログ確認
Get-Content C:\Users\takum\keiba-ai\logs\watchdog_v2_20260506.log -Tail 20
```

---

## 2. 5/6 (火) 平日 隙間時間 推奨 (~3h で済む quick wins)

`docs/system_self_diagnosis_5_5.md` C 推奨アクション参照。

| # | タスク | 工数 |
|---|--------|------|
| 1 | 累計閾値 Discord alert 実装 (-10k/-30k/-50k) | 30min |
| 2 | memory/MEMORY.md 新設 (Claude session 圧縮対策) | 30min |
| 3 | data/v18/index.md + data/results/index.md | 45min |
| 4 | Cookie 失効時 Discord alert | 30min |

これらは 5/9 当日に必須ではないが、心理安全装置として効果絶大。

---

## 3. 5/7 (水) - 5/8 (金) 平日

### 高優先度 (5/9 投入準備)

- 🟠 SED260503 取得 + KKA/KAB 連結再実行 (15min、5/9 朝の前走成績結合率に直結)
- 🟠 speed_index 4-5 月 backfill (30min、条件 C/D 予測精度に直結)
- 🟠 戦略⑦ 5/2-5/3 retro 完全版 (2h、5/9 投入直前必須)
- 🟠 cumulative_results.csv 書き込みバグ修正 (4h、top1_num/score 95% 欠損)

### 5/8 (金) 21:00 後 (1 度だけ、必須)

```bash
cd C:\Users\takum\keiba-ai

# 12R race_name 確認
python -c "
import requests, re
for rid in ['202604010312','202605020512','202608030512']:
    r = requests.get(f'https://race.netkeiba.com/race/shutuba.html?race_id={rid}', headers={'User-Agent':'Mozilla/5.0'})
    m = re.search(r'<h1[^>]*>([^<]+)</h1>', r.text)
    print(rid, '→', (m.group(1).strip() if m else 'NOT_FOUND'))
"

# Cookie 健全性
python tools/refresh_cookie.py --check
```

---

## 4. 5/9 (土) 本番フロー

### 朝 (06:30 - 09:00)

1. `data/results/20260509_pat_checklist.md` を**最初に開く** (順番に従って投票完了まで進める)
2. 06:30 Keiba-Morning_Sat 自動 → Discord #bets 通知 確認
3. 08:00 DailyPredict (watchdog) 自動 → 35 races 完了
4. 09:00 12R race_name 確認 (1 勝クラスかどうか目視)

### 昼 (14:00 - 15:30)

- PAT 投票 (採用 R × 700 円、上限 2,100 円)
- 11R 全 3 場除外 (新潟駿風 S 距離不適合 / 東京エプソム C G3 / 京都京都新聞杯 G2)

### 夜 (18:00 - 20:30)

- 18:00 DailyResults_Sat + Keiba-RaceDayReport_Sat 自動 → Discord 結果通知
- 20:30 `data/v18/post_5_9_improvement_template.md` 振り返り埋め (5/16 改善材料化)

---

## 5. 環境同期 (毎セッション開始時、2 分)

```bash
cd C:\Users\takum\keiba-ai
git pull --rebase --autostash origin main
git log --oneline -5
```

最新 commit が **86cd1da5 以降** か確認。

---

## 6. 状態 chk (5/9 朝なら必須)

### 6.1 Cookie

```bash
python tools/refresh_cookie.py --check
# expected: [OK] Premium 認証 OK
# 切れていたら: python tools/refresh_cookie.py --auto
```

### 6.2 schtasks Ready 状態

```powershell
Get-ScheduledTask -TaskName 'Keiba-Morning_Sat','DailyPredict','DailyResults_Sat','Keiba-RaceDayReport_Sat','Keiba-NarDailyPredict','ProcessWatchdog' | ft TaskName, State, @{N='NextRun';E={(Get-ScheduledTaskInfo $_).NextRunTime}}
```

5/9 (土) 朝なら Morning_Sat / DailyPredict / DailyResults_Sat / RaceDayReport_Sat の State=Ready かつ NextRun が 5/9 当日。
ProcessWatchdog は 5/6 admin 実行後に State=Ready 確認。

### 6.3 daily_predictions/2026MMDD.csv 直近確認

```bash
ls -la data/daily_predictions/ | tail -5
```

### 6.4 Discord 直近通知 確認

- Discord アプリで #bets / #updates タブ開く
- 最新通知が 24 時間以内 (Cookie/morning/results notify_done.py 等の trace)

---

## 7. 緊急時のみ

| 症状 | 対応 |
|------|------|
| Cookie 切れ → premium 取得失敗 | `python tools/refresh_cookie.py --auto` |
| daily_predict 完了せず | watchdog ログ確認 → 手動 `python tools/daily_predict.py --date YYYYMMDD` |
| Discord 通知来ない | `.env` の DISCORD_WEBHOOK_URL 確認 + `python tools/notify_done.py "test" "test"` |
| schtasks 発火せず | Get-ScheduledTask で State=Ready 確認、`Start-ScheduledTask` で手動発火 |
| git pull 衝突 | `--autostash` で auto stash → rebase → unstash |
| python module 不在 | `pip install -r requirements.txt` |
| ProcessWatchdog v2 動かない | `tools/register_process_watchdog_v2.ps1 -Rollback` で v1 に戻す |
| pre_fire_check が cp932 で落ちる | 修正済 (Session #24)、最新 commit 確認 |

---

## 8. NEVER list (禁止行為)

- ❌ 11R 投票 (重賞 + 距離不適合 全除外)
- ❌ 1R 700 円超え (案B改 上限)
- ❌ 1 日 2,100 円超え
- ❌ TYB midday script 実行 (5/3 で 404、Phase 2.5 観測待ち)
- ❌ V18/V19 5/9 投入 (5/16 以降、条件達成後)
- ❌ NAR 5/9 投入 (5/12 paper 開始)
- ❌ 累計 -50,000 円超え (絶対撤退ライン)
- ❌ 数字を v1 引き継ぎから transfusion (`docs/handoff_v1_v2_diff.md` 確認)

---

## 9. ストイック作業時 (引き継ぎ密度 高)

| step | doc |
|------|-----|
| 0 開始 | 本 chk list |
| 0 全体像 | `docs/PHASE_2_5_PLUS_FINAL_RECAP_5_5.md` |
| 0 改善余地 | `docs/system_self_diagnosis_5_5.md` |
| 0 棚卸し | `docs/UPDATE_INVENTORY_20260505.md` |
| 0 引き継ぎ | `docs/HANDOFF_5_5_TO_5_9.md` |
| 0 v1 訂正 | `docs/handoff_v1_v2_diff.md` |
| 履歴 詳細 | `docs/sessions_5_3_5_5_recap.md` |
| 教訓 | `docs/lessons_learned_5_5.md` |
| 5/9 当日 | `data/results/20260509_pat_checklist.md` |
| Phase 2.5 progress | `data/v18/phase_2_5_session10_final.md` |

---

## 10. ユーザー方針 (絶対遵守)

- 取り返し禁止
- 累計 +14,140 円 死守 (5/5 朝時点)
- 撤退ライン -50,000 円 (絶対)
- 5/9 案B改 維持 (12R 1 勝のみ、上限 2,100 円)
- V18/V19 投入は 5/16 以降 (条件達成後)
- NAR 投入は 5/12 paper 開始、5/16 試行 500 円/日
- 静音化済 + admin elevation の手順書あり

---

## 11. 起動 完了

5 分で section 0-1 完了。10 分で section 0-6 完了。
迷ったら **`docs/PHASE_2_5_PLUS_FINAL_RECAP_5_5.md`** に戻る。
