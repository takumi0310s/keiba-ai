# RUNBOOK: 5/10 朝 fire 失敗時の対応手順

作成: Session #73 D (2026-05-09 18:30+)
対象: 2026-05-10 (Sun) 朝の自動運用

## fire 順 (期待)

| 時刻 | task | 役割 |
|------|------|------|
| 06:30 | Morning_Sun | dashboard |
| 07:00 | MorningDigest | 朝 digest |
| 07:30 | JrdbHealthCheck_Sun | JRDB 健全性 |
| **08:00** | **DailyPredict** | **★ 当日全 R 朝予測 ★** |
| 08:45 | RaceAutoNotify_Sun | top3 通知 |
| 09:30 | MorningWeightCheck_Sun | 馬体重 alert |
| **09:30** | **SaveAllHorseScores_0930** | **★ 全頭 V15 score 保存 ★** |
| 10:00+ | MultiStagePredict_* | multi stage |

## 失敗 case + 対応

### Case 1: DailyPredict 8:00 失敗

**症状**: 8:30 過ぎても data/daily_predictions/20260510.csv 生成されない

**原因候補**:
1. netkeiba block (HTTP 4XX)
2. SCRAPER-GUARD 誤停止 (土日朝 OPERATIONAL_CALLERS 漏れ)
3. python crash / Cookie 期限切れ
4. JV-Link 32-bit 問題 (Premium scrape の連動失敗)

**対応 step**:
```
# 1. log 確認
type C:\Users\takum\keiba-ai\logs\daily_predict.log | tail -50

# 2. Cookie 確認
python tools/refresh_cookie.py --check

# 3. 期限切れなら自動更新
python tools/refresh_cookie.py --auto

# 4. manual 再実行
python tools/daily_predict.py --date 20260510

# 5. 完了後 RaceAutoNotify 自動 chained 動作 (8:45 fire の logic に依存)
```

**Discord 期待動作**: 8:00-8:30 に 「daily_predict 完了」 通知。 無ければ要対応。

---

### Case 2: SaveAllHorseScores_0930 失敗 (Session #71 機能)

**症状**: 9:30+ に data/daily_predictions_full/20260510.csv 生成されない

**原因候補**:
1. daily_predict.py 8:00 未完了 → race_id 一覧見つからず → graceful exit (csv 未生成、 想定動作)
2. V15 model load fail
3. parse_shutuba 全 R 失敗 (netkeiba block)
4. kill-switch 誤発動 (data/v18/save_all_horse_scores.kill 存在)

**対応 step**:
```
# 1. log 確認
type C:\Users\takum\keiba-ai\logs\save_all_horse_scores.log | tail -100

# 2. kill-switch 確認 (誤って残ってないか)
dir C:\Users\takum\keiba-ai\data\v18\save_all_horse_scores.kill

# 3. daily_predictions/20260510.csv 確認
dir C:\Users\takum\keiba-ai\data\daily_predictions\20260510.csv

# 4. manual 再実行 (dry-run で確認)
python tools/save_all_horse_scores.py --date 20260510 --dry-run

# 5. 問題なければ本実行
python tools/save_all_horse_scores.py --date 20260510

# 6. 1 R だけ test
python tools/save_all_horse_scores.py --date 20260510 --race-id 202608030601 --dry-run
```

**注**: 仮に 9:30 失敗しても V15 production / 投票 logic に影響なし。
SaveAllHorseScores は「全頭分の score 並行保存」 の補助機能。

---

### Case 3: PreRacePredict_Watchdog_5_9 (★ 5/10 動作不能 確定 ★)

**症状**: 5/10 朝 中 fire するが Discord 通知 0 通

**原因 (Session #73 B 確定)**:
1. `pre_race_predict_runner.bat` は dev/two-stage commit、 main 不在 → 物理欠如で silent exit
2. stage2_predict.py 内部 `DATE = "20260509"` hardcode → 5/10 race_id matching 全敗
3. `RACE_START_TIMES` も 5/9 race_id のみ hardcode

**対応 step (5/10 の最低限 救命策)**:

#### Option A: task disable (簡単、 silent fail を明示停止)
```
schtasks /Change /TN "\Keiba-PreRacePredict_Watchdog_5_9" /DISABLE
```
影響: Session #72 機能 完全停止。 RaceAutoNotify (top3) は健在。

#### Option B: dev/two-stage checkout (bat 復活、 但し DATE は 5/9 のまま)
```
git checkout dev/two-stage
```
影響:
- pre_race_predict_runner.bat 在中化 → fire 起動可
- でも stage2_predict.py 内 DATE = 20260509 → 5/10 race_id 全 skip
- → 結局 silent skip と同じ

#### Option C: stage2_predict.py 動的化 (本筋、 30-60 分作業)
別 Session で 以下 修正:
1. `DATE = datetime.now().strftime("%Y%m%d")`
2. `CACHE_PATH` 動的化
3. `RACE_START_TIMES` を data から動的取得 (data/jra_calendar_{date}.csv 経由)
4. `out_path` 動的化
5. main merge or 個別 cherry-pick
6. dev/two-stage に main merge して bat も含めて 反映

→ **5/10 朝に向けては Option A (disable) を推奨**。
   Option C は 5/16 開催前までに別 Session で実施。

---

### Case 4: Discord 通知届かない (全般)

**症状**: 期待 task が動いてるのに Discord 通知 来ない

**対応 step**:
```
# 1. Discord webhook 設定確認
type C:\Users\takum\keiba-ai\.env | findstr DISCORD

# 2. dedup state 確認 (誤って block してないか)
type C:\Users\takum\keiba-ai\data\discord_dedup_state.json

# 3. 5min hash dedup の reset (Session #59 logic)
del C:\Users\takum\keiba-ai\data\discord_send_cache.json

# 4. test 通知
python tools/notify_done.py "manual test" "5/10 morning recovery"
```

---

### Case 5: V15 model 読み込み失敗

**症状**: 「[ERROR] model load fail」 が log に出る

**対応 step**:
```
# 1. model file 存在確認
dir C:\Users\takum\keiba-ai\keiba_model_v15_central_live.pkl.gz

# 2. file size 確認 (壊れてないか、 通常 数十 MB)

# 3. version check
python -c "from tools.predict_core import load_models; m=load_models(); print(m.get('is_live'))"

# 4. fallback (v13.5b に retreat、 但し ROI 落ちる)
# 緊急時のみ

# kill-switch 経由で停止
echo. > C:\Users\takum\keiba-ai\data\v18\save_all_horse_scores.kill
```

---

### Case 6: process_watchdog kill-switch 誤発動

**症状**: 全タスク fire しない (system 全停止)

**対応 step**:
```
# 1. process_watchdog log 確認
type C:\Users\takum\keiba-ai\logs\process_watchdog.log | tail -50

# 2. 誤検知の場合、 watchdog 一時停止
schtasks /Change /TN "\ProcessWatchdog" /DISABLE

# 3. 個別 task 手動 fire
schtasks /Run /TN "\keiba-ai\DailyPredict"

# 4. 復旧後 watchdog 再 enable
schtasks /Change /TN "\ProcessWatchdog" /ENABLE
```

---

## 緊急 escalation

5/10 朝 段階で 投票 直前に重大 fail が起きた場合:

1. **手動予測 (1 R)**:
   ```
   python tools/predict_one_race.py 202608030612
   ```

2. **手動投票判定**: V15 投票方針 (絶対遵守):
   - 5/10 案 (要確認、 5/9 同様の戦略想定): 1勝クラス 12R のみ ¥700
   - 累計 +13,530 円 死守、 撤退ライン -50,000 円

3. **完全停止判断**: 累計 -10,000 円 / -50,000 円 で投票停止。

## 5/10 朝 fire test 完了 chronology

- 06:30 Morning_Sun → Discord notification 期待
- 08:00 DailyPredict → daily_predictions/20260510.csv 生成 期待
- 09:30 SaveAllHorseScores_0930 → daily_predictions_full/20260510.csv 生成 期待
- 09:30+ MorningWeightCheck → 体重 alert (該当馬居れば)

各段階で fail が無ければ 5/10 投票運用 OK。

## 投資保護 (再掲)

- V15 model 不変 (Session #73 で 一切 触らず)
- main HEAD 5f5c3d43 (Session #71) 不変
- daily_predict.py / predict_core.py / app.py / race_auto_notify.py 不変
- 累計 +13,530 円 維持
