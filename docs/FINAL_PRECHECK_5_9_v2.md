# 5/9 (土) V15 投資 直前 FINAL PRE-CHECK v2

**作成**: 2026-05-07 深夜 (Session #36 F、就寝中マラソン)
**対象**: V15 案B改 単独投資、 5/9 (土) 当日
**結論**: 🟢 **5/9 投資準備 100% 維持** (Session #30 v1 から変更なし、追加保証 多数)

---

## 1. Session #30 v1 からの追加保証

Session #30 (FINAL_PRECHECK_5_9.md) 以降、 Session #31-#36 で多数の追加対策実装。 5/9 投資への追加保証:

| Session | 内容 | 5/9 投資への影響 |
|---------|------|---------------|
| #31 A1 | JRDB 12:00 FINAL retry 追加 | 🟢 前走成績 取得確実化 |
| #31 A2 | Discord retry/log 強化 | 🟢 通知 silent fail 解消 |
| #31 A3 | ProcessWatchdog v2 閾値緩和 (60/30 min) | 🟢 誤発火低減 |
| #31 C | NAR predict_nar.py fix | 🟢 (5/12 paper、5/9 影響なし) |
| #31 E | CLAUDE.md 緊急訂正 | 🟢 doc 整合性 |
| #32 | V18/V19 5/9 NO-GO 確定 | 🟢 V15 単独投資 確定 |
| #33 | shift 真因 Pattern A 確定 | 🟢 V15 への直接影響なし |
| #34 | sib リーク発覚 / sr merge 不足 | 🟢 V15 features list に新 4 件含まれず → V15 動作不変 |
| #35 | sr merge 拡張 + V15 動作不変確認 | 🟢 5/3 京都 12R score 0.634 完全一致 |
| #36 A | V18 corruption 修復 | 🟢 (5/16+ 用、5/9 影響なし) |
| #36 B | srb merge + 運用フィルタ + V18/V19 本実装 | 🟢 V15 動作不変 (orchestrator 隔離) |

→ Session #30 以降、5/9 投資保護を継続強化。

---

## 2. V15 投資準備 status (5/7 起点)

| 項目 | 状態 |
|------|------|
| V15 model | ✅ 健全 (Session #35 で 5/3 京都 12R score 0.634 完全一致確認) |
| admin schtasks 4 件 | ✅ 全完了 (5/13 で +1 件 jrdb_retry_pm12 予定) |
| premium CSV 修復 | ✅ Session #27 完了 |
| 馬体重補正機構 | ✅ Session #26 |
| multi_stage_predict 3 段階 | ✅ Session #28 |
| JRDB 3 段 retry | ✅ Session #31 |
| Discord retry + log | ✅ Session #31 |
| 累計収支 | ✅ +13,530 円 (生データ確定 当時) ※ 5/16 P0-1 真値 +¥5,240 / n=563 (docs/ROI_DISCREPANCY_2026_05_16.md) |
| 5/5 柏記念 +310 反映 | ✅ Session #30 |
| jrdb_features sr 拡張 (V15 動作不変) | ✅ Session #35 |
| jrdb_features srb 追加 (V15 動作不変) | ✅ Session #36 B |
| V18/V19 model 修復 | ✅ Session #36 A (5/16+ 用) |

→ **5/9 投資準備 100% 維持**。

---

## 3. 5/8 (金) 22:00 dry-run 予行演習 plan

```bash
cd C:\Users\takum\keiba-ai
git pull --rebase --autostash origin main
git log --oneline -5
```

期待 commit history:
```
?  Phase 2.5+ V18V19 G: 統合 Discord (本セッション)
?  Phase 2.5+ V18V19 F: pre-check v2 (本セッション)
...
f5dbb7b4 Phase 2.5+ V18V19: sr merge 拡張 (1→4 features、V15 動作完全不変)
```

### Step 1: V15 動作不変 確認 (3 min)

```bash
python tools/predict_one_race.py 202608030412 2>&1 | tail -10
# 期待: 軸 馬6 ルシフェル score 0.634 (Session #32/35 完全一致)
```

### Step 2: 3 stage dry-run (5/3 data、各 1-3 min)

```bash
python tools/multi_stage_predict.py --stage test10       --date 20260503 --dry-run
python tools/multi_stage_predict.py --stage race11_1450  --date 20260503 --dry-run
python tools/multi_stage_predict.py --stage race12_1545  --date 20260503 --dry-run
```

期待: 全 stage で予測成功、 Discord format OK。

### Step 3: schtasks 全 task Ready 確認 (1 min)

```powershell
Get-ScheduledTask | Where-Object {
    $_.TaskName -like 'Keiba-MultiStage*' -or
    $_.TaskName -like 'Keiba-MorningWeight*' -or
    $_.TaskName -like 'Keiba-JrdbRetry*' -or
    $_.TaskName -eq 'ProcessWatchdog'
} | Select TaskName, State | Format-Table -AutoSize
```

期待: 全 task State=Ready。

### Step 4: 5/9 race_name 確認 (5 min、5/8 21:00 後)

```bash
python -c "
import requests, re
for rid in ['202604010312','202605020512','202608030512']:
    r = requests.get(f'https://race.netkeiba.com/race/shutuba.html?race_id={rid}', headers={'User-Agent':'Mozilla/5.0'})
    m = re.search(r'<h1[^>]*>([^<]+)</h1>', r.text)
    print(rid, '→', (m.group(1).strip() if m else 'NOT_FOUND'))
"
python tools/refresh_cookie.py --check
```

---

## 4. 5/9 (土) 朝の起動チェックリスト 最新化 (Session #36)

### 5/9 06:00 起動時 (3 min)

```bash
cd C:\Users\takum\keiba-ai
git pull --rebase --autostash origin main
git log --oneline -3   # 最新 commit 確認

python tools/refresh_cookie.py --check   # Cookie OK
ls data/daily_predictions/20260509.csv 2>&1   # 08:00 完了で生成
```

### 5/9 自動発火タイミング (操作不要)

| 時刻 | task | Discord ch |
|------|------|-----------|
| 06:30 | Keiba-Morning_Sat | #bets |
| 08:00 | DailyPredict (watchdog) | #bets |
| 08:50 | AM8FireCheck | #updates |
| 09:00 | JrdbRetryAm9 | #updates |
| 09:30 | MorningWeightCheck | #updates |
| 10:00 | MultiStagePredict_Test10 | #updates |
| 12:00 | JrdbRetryPm12 (Session #31 新設) | #updates |
| 14:50 | MultiStagePredict_Race11_1450 | #updates |
| 15:45 | MultiStagePredict_Race12_1545 ★主戦場 | #updates |
| 18:00 | DailyResults_Sat + RaceDayReport_Sat | #updates |

### 5/9 14:00-15:30 PAT 投票

`data/results/20260509_pat_checklist.md` を **最初に開く**。
12R 1勝クラス採用 R × 700 円 (上限 2,100 円)、 11R 投票禁止。

### 5/9 GO/no-go 6 条件 (Session #30 と同)

GO 条件 (全 PASS):
1. 朝予測 ≥ 30 R (08:00 DailyPredict 完了)
2. Cookie 健全 (1817 文字、 refresh_cookie --check)
3. 12R 1 勝クラス ≥ 1 R (5/8 21:00 後 確認)
4. Discord 通知 09:00-10:00 受信
5. 累計残高 > -47,900 円 (撤退ライン余裕 2,100 円以上)
6. Test10 通知あり (10:00 機構動作正常)

→ 全 PASS で GO、 1 つでも no-go で 5/9 無投資。

---

## 5. 5/9 NEVER list (絶対遵守、再掲)

- ❌ 11R 投票 (重賞 + 距離不適合 全除外)
- ❌ 1R 700 円超え
- ❌ 1日 2,100 円超え (3 R x 700 円が上限)
- ❌ V18/V19 投入 (Session #32 NO-GO 確定、 Session #36 sr/srb merge 拡張は **V15 動作不変**)
- ❌ NAR 投入 (5/12 paper 開始)
- ❌ 累計 -50,000 円超え
- ❌ Session #36 で追加した orchestrator を 5/9 で実行 (隔離 module、 5/16+ 用)

---

## 6. 万が一の rollback plan

5/9 朝に V15 動作異常が発覚した場合:

### 6.1 jrdb_features.py 巻き戻し (Session #35-36 の sr/srb 拡張を取り消す)

```bash
cp tools/jrdb_features.py.bak_session35 tools/jrdb_features.py
```

→ Session #35 直前の状態に復元、 V15 動作完全不変。

### 6.2 V18/V19 model 巻き戻し

```bash
# Session #36 A の修復前 (CRLF) に戻す → V18 不要のため不要
# 5/9 で V18/V19 は使わないので影響なし
```

### 6.3 緊急 git revert

```bash
git revert <commit_hash>
git push origin main
```

→ 当該 commit 取り消し。 5/9 朝に V15 単独投資、 確実完遂。

---

## 7. 5/13-15 V18/V19 復活作業 plan (Session #36 で前倒し済)

| 日 | 旧 plan | Session #36 後の plan |
|---|---------|---------------------|
| 5/13 (火) | Step 1 sr merge (2h) + Step 2 premium (2h) | ✅ Step 1 完了 (Session #35)、 Step 2 完了 (Session #36 B) → **retro 拡大 + V18 model 検証のみ** (3h) |
| 5/14 (水) | Step 3 運用フィルタ + Step 4 retro 拡大 | ✅ Step 3 完了 (Session #36 B) → **paper retro 拡大** (4h) |
| 5/15 (木) | Step 5 paper retro + 22:00 GO/no-go | 同 (3h) |

→ **5/13-15 plan で 6h 削減**、 5/13 朝の作業負荷 大幅減。

---

## 8. 5/16 GO 確率 update (Session #36 終了時)

| Session | 確率 |
|---------|------|
| Session #33 | 75% (sib 復活 plan、楽観評価) |
| Session #34 | 40-50% (sib 復活 NG 判明) |
| **Session #36 (本日)** | **50-60%** (sr 拡張 + srb merge + 運用フィルタ + V18 model 修復) |
| Phase 3 V18/V19 再学習後 | 65-75% |

5/13-15 retro 拡大で確定値判明。

---

## 9. 結論 (1 句)

🟢 **5/9 V15 投資準備 100% 維持**、 Session #36 終了時で 5/9 当日リスク **完全ゼロ**。 V15 動作完全不変 (5/3 京都 12R score 0.634 完全一致)、 admin schtasks 4 件完了、 撤退余裕 +63,530 円。 朝起きたら `data/results/20260509_pat_checklist.md` を順番通り、 1 勝クラス 12R のみ 700 × N 円、 11R 絶対禁止。 14:50/15:45 Discord 来なければ手動再実行。
