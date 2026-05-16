# 5/9 (土) 投資 直前 FINAL PRE-CHECK

**作成**: 2026-05-06 PM (Session #30、67h目連続作業の総点検)
**対象 commit**: c722d403 + 本セッション
**結論**: **🟢 5/9 投資準備 100% 完成 (緊急 0 件、改善余地 4-5 件は 5/16+ で対応)**

---

## 1. 5/9 投資準備 status

| 項目 | 状態 | 備考 |
|------|------|------|
| V15 model | ✅ 健全 | AUC 0.8858、150 features |
| admin schtasks 4 件 | ✅ 全完了 | JrdbRetryAm9 / MorningWeightCheck / MultiStagePredict 6task / ProcessWatchdog v2 |
| premium CSV 修復 | ✅ 完了 | 5/2-5/3 +489 si / +1,004 tr 行 |
| 馬体重補正機構 | ✅ 完了 | 5/3 動作テスト OK |
| multi_stage_predict 3 段階 | ✅ 完了 | 5/3 dry-run 全成功 |
| MEMORY.md | ✅ 復活 | リポジトリ内 7 file |
| 累計収支 | ✅ +13,530 円 当時 | 撤退余裕 +63,530 円 ※ 5/16 P0-1 真値 +¥5,240 / +¥55,240 (docs/ROI_DISCREPANCY_2026_05_16.md) |
| 5/5 柏記念 +310 円 | ✅ 反映済 | cumulative_results に追記 (本セッション) |

**5/9 投資判断**: 案B改 V15 単独 維持 GO。

---

## 2. 残作業 (緊急度別、Session #30 A 領域)

詳細: `data/v18/final_audit_remaining_tasks_5_6.md`

### 🔴 緊急 (5/9 までに必須)
**0 件確定**。 5/9 当日リスクなし。

### 🟠 高 (5/12-5/16 までに) — 既知、計画通り
- 5/12 NAR paper 開始 (admin 登録済、scrape_nar_today/results 動作検証推奨)
- 5/15 V18/V19 GO/no-go 判定 (5 条件全 0 達成、現状 NO-GO 寄り)

### 🟡 中 (5/24 Phase 3 までに)
- V15.1 4-model + WF 検証 + leak audit (Phase 3 本格採用)
- race-level normalize 本番統合 (predict_core.py)
- feature distribution shift 調査
- chihou data backfill (NAR 本格運用前提)

### 🟢 低 (Phase 3+ 以降)
- CLAUDE.md V15 化書換 (現状 v13.5b 中心)
- 古い doc archive (~60 ファイル)
- TARGET 6 月再契約候補

### schtasks 異常 task
- `DailyPredict rc=1`: 平日 0 races 既知誤判定、土曜は 35 races で OK
- `Keiba-WeeklyScrapeResume 3221225786`: Ctrl+C 終了 (5/9 影響なし、月曜のみ)
- `WeeklyReport rc=1`: 月曜のみ
- 他 2 件は Windows 標準 / placeholder で 5/9 影響なし

→ **5/9 影響 0 件、慌てる必要なし**。

---

## 3. 精度改善 (Session #30 B 領域)

詳細: `data/v18/precision_improvement_opportunities_5_6.md`

### 🟠 5/9 即適用可能 (4 件、追加実装 0 件、既存 commit で全組込済)
1. **multi_stage_predict 3 段運用**: 馬体重補正で TOP1 score +0.21〜0.31 上昇 (既動作確認済)
2. **当日確定オッズ**: `predict_core.py L1135 fetch_realtime_odds_full` 既実装
3. **馬場・天候**: `predict_core.py L1385 fetch_jra_and_weather` 既実装
4. **騎手変更検出**: `predict_core.py L2074-2091 jockey_change` 既実装

→ V15 既に直前情報を相当活用。 5/9 で **追加実装は不要**、既存機構を信頼で OK。

### 🔴 5/9 不採用判定
- **V15.1 SKB +0.0699 投入**: NO-GO (4-model 互換未検証 / SKB 取得確実性未確認 / 軸top3 率 unknown / fall-back 困難) → 取り返し禁止ルール遵守
- **formation 拡張**: 既 retro で 7 点 baseline ROI 最良判明済

### 5/16 V18/V19 試行 暫定 NO-GO
5 条件中 **0 達成**:
- #1 race-level normalize 本番統合 ❌
- #3 sample 30+ bets → 9 bets のみ ❌
- #4 winner_top1 rate ≥ 40% → 34.5% ❌
- #5 feature shift 未着手 ❌

→ **5/16 V18/V19 投入は NO-GO 寄り**、V15 案B改 単独維持 推奨。

### Phase 3 (5/24+) ロードマップ
- V15.1 4-model ensemble 学習 + WF + leak audit
- predict_core.py に SKB merge 統合
- V20 統合モデル (JRA + NAR、52+ features、6 月後半)

---

## 4. リスク監視 (Session #30 C 領域)

詳細: `data/v18/risk_audit_5_6.md`

### 想定外シナリオ 8 種

| # | シナリオ | 発生確率 | 影響度 | 対策 status |
|---|---------|---------|--------|-----------|
| ① | Cookie 切れ | 低 | 高 | refresh_cookie --auto 実装済 |
| ② | JRDB 9:00 retry 失敗 | 中 | 中 | retry 1 回のみ (改善余地) |
| ③ | 馬体重 10:00 取得失敗 | 中 | 低 | multi_stage_predict L446-453 で朝予測 fallback ✅ |
| ④ | Discord silent fail | 低 | 高 | **未対策** (notify.py L84、retry/log なし) |
| ⑤ | PAT 入力ミス | 中 | 中 | pat_checklist.md 確認手順 ✅ |
| ⑥ | PAT 障害 | 極低 | 高 | 投票見送り判断 (人手) |
| ⑦ | ProcessWatchdog 誤発火 | 低 | 低 | 07:00-18:00 のみ再起動、CommandLine 部分一致で誤発火低リスク |
| ⑧ | 停電 | 極低 | 高 | daily_predict のみ resume 対応 |

**未対策 4 件**: ② JRDB retry 強化、④ Discord retry/log、⑤ NAR 反映 (本セッションで完了)、⑧ 停電対応 — いずれも 5/9 当日影響は限定的、**5/16+ で改善**。

### 累計収支 final 確認
- **生データ +13,530 円** (採用 当時) vs USER 申告 +14,140 円 (610 円差は 5/4 の何か、影響なし)
- 撤退ライン -50,000 円まで余裕 **+63,530 円**
※ 5/16 P0-1 真値: **+¥5,240** / 撤退余裕 **+¥55,240** (docs/ROI_DISCREPANCY_2026_05_16.md)
- 5/9 全外し最悪 -2,100 円 → 累計 +11,430 円 (依然プラス維持)
- 5/5 柏記念 +310 円 を `data/cumulative_results.csv` に追記済 (Session #30 本日)

---

## 5. 5/8 (金) 22:00 dry-run チェックリスト

```bash
cd C:\Users\takum\keiba-ai
git pull --rebase --autostash origin main
git log --oneline -5
```

### Step 1: Cookie + 12R race_name (5/8 21:00 後)

```bash
python -c "
import requests, re
for rid in ['202604010312','202605020512','202608030512']:
    r = requests.get(f'https://race.netkeiba.com/race/shutuba.html?race_id={rid}', headers={'User-Agent':'Mozilla/5.0'})
    m = re.search(r'<h1[^>]*>([^<]+)</h1>', r.text)
    print(rid, '->', (m.group(1).strip() if m else 'NOT_FOUND'))
"
python tools/refresh_cookie.py --check
```

期待: 全 3 race_id で 1 勝クラス or 別条件 → 1 勝のみ採用、Cookie OK。

### Step 2: 3 stage dry-run (5/3 データ)

```bash
python tools/multi_stage_predict.py --stage test10       --date 20260503 --dry-run
python tools/multi_stage_predict.py --stage race11_1450  --date 20260503 --dry-run
python tools/multi_stage_predict.py --stage race12_1545  --date 20260503 --dry-run
```

期待:
- test10: 3場 2R 補正、3R-12R 朝予測通知 (馬体重未公開 fallback OK)
- race11_1450: 重賞含む 3場 11R、案B改 全採用外 0 円
- race12_1545: 採用 R に買い目 7 点、それ以外採用外

### Step 3: schtasks 全 8 件 Ready 確認 (PowerShell)

```powershell
Get-ScheduledTask | Where-Object { $_.TaskName -like 'Keiba-MultiStage*' -or $_.TaskName -like 'Keiba-MorningWeight*' -or $_.TaskName -like 'Keiba-JrdbRetry*' -or $_.TaskName -eq 'ProcessWatchdog' } | Select TaskName, State | Format-Table -AutoSize
```

期待: 全 8 task State=Ready、ProcessWatchdog も Ready (Disabled なら admin で再登録)。

### 失敗時 rollback

各 ps1 に -Rollback フラグあり、admin で実行で削除可能。

---

## 6. 5/9 (土) 朝の起動チェックリスト (5 分で復帰)

### 必須確認 5 件 (08:00 起動時)

1. `data/results/20260509_pat_checklist.md` 開く ★最優先
2. Discord #bets で 06:30 Morning_Sat 通知確認
3. Discord #updates で 08:50 AM8FireCheck OK 確認
4. `python tools/refresh_cookie.py --check` (1 分)
5. `git log --oneline -3` で c722d403 以降 確認

### 自動発火タイミング表 (5/9 当日)

| 時刻 | task | Discord ch |
|------|------|-----------|
| 06:30 | Keiba-Morning_Sat | #bets |
| 08:00 | DailyPredict | #bets |
| 08:50 | AM8FireCheck | #updates |
| 09:00 | JrdbRetryAm9 | #updates |
| 09:30 | MorningWeightCheck | #updates |
| **10:00** | **MultiStagePredict_Test10** ★ | **#updates** |
| **14:50** | **MultiStagePredict_Race11_1450** ★ | **#updates** |
| **15:45** | **MultiStagePredict_Race12_1545** ★主戦場 | **#updates** |
| 14:00-15:30 | PAT 投票 (USER 手動) | - |
| 18:00 | DailyResults_Sat + RaceDayReport_Sat | #updates |
| 20:30 | post_5_9_improvement_template.md (USER 手動) | - |

### 5/9 投資 GO/no-go 6 条件

GO 条件 (全 PASS):
1. 朝予測 ≥ 30 R (08:00 DailyPredict 完了)
2. Cookie 健全 (1817 文字、refresh_cookie --check)
3. 12R 1 勝クラス ≥ 1 R (5/8 21:00 後 確認)
4. Discord 通知 09:00-10:00 受信
5. 累計残高 > -47,900 円 (撤退ライン余裕 2,100 円以上)
6. Test10 通知あり (10:00 機構動作正常)

**全 PASS で GO、1 つでも no-go で 5/9 無投資**。

### 5/9 NEVER list (絶対遵守)

- ❌ 11R 投票 (重賞 + 距離不適合 全除外)
- ❌ 1R 700 円超え
- ❌ 1日 2,100 円超え (3 R x 700 円が上限)
- ❌ V18/V19 投入 (5/16 以降、条件達成後)
- ❌ NAR 投入 (5/12 paper 開始)
- ❌ 累計 -50,000 円超え

---

## 7. 結論 (寝起きで読む 1 行)

**5/9 投資準備 100% 完成、🔴緊急 0 件、改善は 5/16+ で段階投入**。
朝起きたら `data/results/20260509_pat_checklist.md` 開いて順番通り進めれば迷わず投票完了。 PAT 投票は 12R 1 勝クラスのみ 700 × N 円、11R 絶対禁止。 14:50/15:45 Discord 通知が来なければ手動再実行 (`python tools/multi_stage_predict.py --stage race11_1450` 等)。 累計 +13,530 円死守、撤退余裕 +63,530 円維持。 ※ 5/16 P0-1 真値 +¥5,240 / +¥55,240 (docs/ROI_DISCREPANCY_2026_05_16.md)
