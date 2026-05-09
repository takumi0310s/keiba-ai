# Session #73 B: stage2_predict.py 動作試験

実行日時: 2026-05-09 18:30+
対象: tools/stage2_predict.py (Session #72 dev/two-stage 在中)

## 結論

**stage2_predict.py は 5/9 完全 hardcode、 5/10 動作不能 (重要)。**

加えて呼び出し bat (pre_race_predict_runner.bat) は dev/two-stage に在中、
main 不在。 現在 main checkout 中 → bat 物理欠如。

## 詳細 hardcode 箇所

| 箇所 | 内容 | 修正必要 |
|------|------|---------|
| `DATE = "20260509"` | 全処理の対象日 | "20260510" or 動的化 |
| `CACHE_PATH = "pre_race_predict_cache_5_9.json"` | dedup cache | "_5_10" or 動的化 |
| `DAILY_PRED = "20260509.csv"` | 朝予測 read 元 (DATE 経由) | DATE 修正で連動 |
| `RACE_START_TIMES` | 5/9 race_id 33 件 hardcode | 5/10 race_id 追加必須 |
| `out_path = "pre_race_predict_5_9_R*.json"` | 結果 file 名 | "_5_10" or 動的化 |

## 5/10 RACE_START_TIMES 必要 race_id (推定)

5/9 と同じ 3 場 6 日目想定:
- 京都 06: 202608030601 〜 202608030612 (12 R)
- 東京 06: 202605020601 〜 202605020612 (12 R)
- 新潟 04: 202604010401 〜 202604010412 (12 R、 5/9 と同じく 1R/4R 不在の可能性)

## bat file 欠如

- `pre_race_predict_runner.bat`: dev/two-stage に commit、 main 不在
- 現在 (main) では schtasks 起動しても 即 失敗 (silent_runner.vbs から見て bat 不在 → no-op)
- save_all_horse_scores_runner.bat は main 在中 (Session #71) → 動作可

## fallback 動作

stage2_predict.py 内部 logic:
- `load_full_predictions(race_id, DATE)` → daily_predictions_full/{DATE}.csv read
- DATE = 20260509 のため、 5/10 fire しても 5/9 csv を read
- 5/10 race_id は 5/9 csv に無い → None → top3 fallback (build_message)
- 但し morning_row も 5/9 csv から (DAILY_PRED) → 5/10 race_id は None → "not_in_morning" skip

→ 結果: 5/10 朝 fire → 全 R skip "not_in_morning"、 Discord 通知 0 通

## 5/10 の影響と必要 action

| 影響 | 詳細 |
|------|------|
| 1h 前 stage 2 通知 | **動作しない** (Session #72 機能 完全停止) |
| 朝予測 (Stage 1) | DailyPredict 8:00 で正常動作 |
| 全馬 score 保存 | SaveAllHorseScores 9:30 で正常動作 (Session #71) |
| RaceAutoNotify | 8:45 で正常動作 (top3) |

5/10 朝の通知不足は深刻ではない (Stage 1 + RaceAutoNotify 健在)。
但し Session #72 の「全馬 V15 score 順 table」 1h 前通知 機能は失われる。

## 必要 action (5/16 開催前まで)

### Option 1: dev/two-stage に main checkout (簡易)
```
git checkout dev/two-stage
```
- 利点: pre_race_predict_runner.bat 復活
- 欠点: stage2_predict.py 5/9 hardcode は残る (5/10 動かない)
- 結論: 不十分

### Option 2: stage2_predict.py 動的化 + main merge (本筋)
- DATE = datetime.now().strftime("%Y%m%d") に変更
- CACHE_PATH を date suffix 動的化
- RACE_START_TIMES を data/jra_calendar_{date}.csv または schedule API から動的取得
- main merge or cherry-pick
- 推定 工数: 30-60 分 (本 Session 73 の B/C/D/E 完了後 検討)

### Option 3: PreRacePredict_Watchdog_5_9 を disable (一時退避)
```
schtasks /Change /TN "\Keiba-PreRacePredict_Watchdog_5_9" /DISABLE
```
- 5/10 朝の silent failure を防ぐ
- Session #72 機能 一時停止
- 5/16 開催前に Option 2 + 再 enable

### 推奨 (Session #73 範囲外)

- **5/9 残り時間**: stage2_predict 動的化 commit を別 Session で実施
- **本 Session #73**: documentation + 5/10 fire 状態 報告に limit
- **5/10 朝 9:30 SaveAllHorseScores**: Session #71 機能は完全動作

## 確認 result

- ✗ stage2_predict.py 5/9 hardcode (DATE/CACHE/RACE_TIMES/出力名)
- ✗ pre_race_predict_runner.bat main 不在
- ✓ stage2_predict.py 内 全馬 table 構築 logic は schema 互換 (Session #71 csv 読める)
- ✓ Session #72 commit `812eaf54` の change は意図通り

## 次 action

→ Session #73 C (schtasks) + D (runbook) で 5/10 朝の状態を doc 化、
   5/9 内に Option 2 / Option 3 を 別 Session で対応 推奨。
