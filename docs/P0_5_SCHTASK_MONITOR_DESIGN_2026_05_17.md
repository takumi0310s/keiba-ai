# P0-5-C: -15 min 再計算 schtask + monitor 設計

> Sub-task P0-5-C (Sole author: design only, 2026-05-17)
> 前提: P0-5-A (data fetch) / P0-5-B (recalc logic) は別 sub-task で別途設計
> 関連 commit: T6 anomaly auto detector = 2993f0b5
> **★ 実 schtasks /create 禁止。 admin 実行は 5/18+ user 判断後 ★**

---

## 0. 結論

| 項目 | 値 |
|------|----|
| 新規 schtask | **1 個** (`Keiba-LiveOrchestrator-15min`) |
| dynamic 動的生成 | **不要** (1 task が internal loop で全 race を polling) |
| T6 拡張 trigger | **2 件追加** (5 → 7 trigger) |
| 既存 schtasks への変更 | **なし** (★ V15 production 完全不変 ★) |
| 実装着手 | 5/17 21:00+ |
| paper shadow 期間 | 5/19-6/16 (約 30R 蓄積) |
| 採用判定 | 6/17+ (統計検定 + ROI delta) |

---

## 1. 既存 schtasks list (read-only query 結果、 5/17)

`schtasks /query /fo LIST` 抽出 (`\keiba-ai\` 配下 + ルート `Keiba-*`)。

### 1-1. ルート配下 (G1 day 用 + 既存 daily)

| TaskName | 推定時刻 | 用途 |
|----------|----------|------|
| `\Keiba-AM3FireCheck` | 03:00? | DailyPremiumScrape 起動 check |
| `\Keiba-AM6FireCheck` | 06:00? | DailyJrdbKyi 起動 check |
| `\Keiba-AM8FireCheck` | 08:00? | DailyPredict 起動 check |
| `\Keiba-PreFireCheck` | 朝 | pre-fire 健全性 |
| `\Keiba-MorningDigest` | 朝 | digest 通知 |
| `\Keiba-Morning_Sat` / `_Sun` | 朝 | 週末 morning task |
| `\Keiba-MorningWeightCheck_Sat` / `_Sun` | 09:30 | 馬体重補正 (CLAUDE.md 既存) |
| `\Keiba-SaveAllHorseScores_0930` | 09:30 | 馬体重 + score snapshot |
| `\Keiba-MultiStagePredict_Race11_1450_*` | 14:50 | G1 day 用 multi-stage |
| `\Keiba-MultiStagePredict_Race12_1545_*` | 15:45 | 同上 |
| `\Keiba-MultiStagePredict_Test10_*` | test | 動作試験 |
| `\Keiba-VoteCandidates_1400_5_9` | 14:00 (5/9) | 投票候補 snapshot |
| `\Keiba-Cumulative_1700_5_9` | 17:00 (5/9) | 累計 audit |
| `\Keiba-Summary_2030_5_9` | 20:30 (5/9) | summary |
| `\Keiba-Verdict_R11_*` / `R12_*` | 各 R 後 | 結果 verdict |
| `\Keiba-RaceDayReport_Sat` / `_Sun` | 週末 | report |
| `\Keiba-PreRacePredict_Watchdog_5_9` | 5/9 | watchdog |
| `\Keiba-JrdbRetryAm9_Sat` / `_Sun` | 09:00 | JRDB retry |
| `\Keiba-FridayWeekendScrape` | 金 | 週末 pre-scrape |
| `\Keiba-TybPublishMonitor` | — | TYB 公開監視 |
| `\Keiba-NightlySanity` | 23:00 | nightly check (CLAUDE.md 既存) |
| `\Keiba-NarDailyPredict` / `Results` / `Scrape` 等 | — | NAR 系 |
| `\Keiba-NarLiveOddsRefresh` | — | NAR live odds |
| `\Keiba-NarMidDayCalendar` | — | NAR calendar |
| `\KeibaAI_DriftDetector` | — | drift 検出 |
| `\ProcessWatchdog` | 5min | 既存 watchdog (CLAUDE.md) |

### 1-2. `\keiba-ai\` 配下 (production core)

| TaskName | 時刻 (CLAUDE.md) | 用途 |
|----------|------------------|------|
| `DailyJrdbKyi` | 06:00 | JRDB 全種別 DL |
| `DailyPremiumScrape` | 03:00 (週末 + 月) | netkeiba premium |
| `DailyPredict` | 08:00 | 当日全 R 予測 |
| `DailyResultsEvening` | 20:00 | 結果照合 |
| `DailyResults_Sat` / `_Sun` | 18:00 | 週末結果照合 |
| `JrdbHealthCheck_Sat` / `_Sun` | 07:30 | JRDB 健全性 |
| `RaceAutoNotify_Sat` / `_Sun` | 08:45 | 5 min 前 通知 + 投票 (production 核) |
| `WeeklyReport` | 月 08:00 | 週次 |
| `Keiba-ScrapeProgress` | — | scrape progress |
| `Keiba-WeeklyScrapeResume` | — | scrape resume |

★ 既存 production 核 = `RaceAutoNotify_Sat/_Sun` (08:45 起動、 各 R の 5min 前で予測 + 通知 + 投票判断) ★

→ 新規 `LiveOrchestrator` は **08:45 RaceAutoNotify と並走** (independent、 影響 0)。

---

## 2. 新規 schtask 仕様 (単一 task、 internal loop)

### 2-1. dynamic schtask が **不要** な理由

候補 A (dynamic、 不採用):
- 1 day 35 R × 3 phase (-20/-15/-10 min) = **105 schtasks/day** 動的生成
- 翌日に削除 → 翌々日に再生成 ... admin 権限要求 + 失敗 recovery 複雑
- schtasks XML 編集 + UAC 衝突リスク

候補 B (★ 採用 ★、 単一 task + internal polling loop):
- `Keiba-LiveOrchestrator-15min` 1 個
- 朝 08:30 起動 (DailyPredict 完了直後)
- 21:00 まで running (last race 終了後 + buffer)
- internal で 各 race の `発走 - 20 / 15 / 10 min` を `time.sleep()` で polling
- admin 権限 不要 (一度 register すれば 以後 schtasks /create 不要)

### 2-2. schtask 案

```bat
:: ★ 5/18+ user GO 後、 admin 1 回だけ実行 ★
schtasks /Create ^
  /TN "\keiba-ai\Keiba-LiveOrchestrator-15min" ^
  /TR "C:\Users\takum\keiba-ai\tools\live_orchestrator.bat" ^
  /SC WEEKLY /D SAT,SUN ^
  /ST 08:30 ^
  /RL HIGHEST ^
  /F
```

備考:
- **WEEKLY SAT,SUN** に限定 (NAR 用 monkey-patch は別 task、 ここでは触れない)
- `/RL HIGHEST` ... port 8501 etc. 影響 0 (read-only fetch + Discord notify のみ)
- 月-金 は 起動しない (中央競馬 開催なし、 fail-safe)
- 重複起動防止: `live_orchestrator.bat` 先頭で lock file check (P0-5-B 設計に従う)

### 2-3. `tools/live_orchestrator.bat` (★ 5/18+ 実装、 design only ★)

```bat
@echo off
:: Keiba-LiveOrchestrator-15min — 朝 08:30 起動、 21:00 自然終了
cd /d C:\Users\takum\keiba-ai
call .venv\Scripts\activate.bat
python tools\live_orchestrator.py > logs\live_orchestrator_%date:~0,4%%date:~5,2%%date:~8,2%.log 2>&1
exit /b %errorlevel%
```

### 2-4. RaceAutoNotify との 重複排除

- 既存 `RaceAutoNotify_Sat/_Sun` は **5 min 前 で 予測 + 通知 + 投票判断**
- 新規 `LiveOrchestrator` は **-20/-15/-10 min で fetch + recalc + (shadow) 通知**
- **paper shadow 期間中**:
  - 通知は `#updates` channel のみ (production `#bets` には送らない)
  - 投票判断 = 既存 RaceAutoNotify のまま (★ V15 production 完全不変 ★)
- 採用後 (6/17+):
  - `#bets` 通知に切替 + RaceAutoNotify との timing 調整 (別 sub-task で設計)

---

## 3. `tools/live_orchestrator.py` 設計

### 3-1. 全体構造

```python
"""
live_orchestrator.py (★ 5/18+ 実装、 design only ★)

朝 08:30 起動、 daily_predictions/{YYYYMMDD}.csv + netkeiba fetch から
各 race の発走時刻を取得し、 -20/-15/-10 min で順次実行。

paper shadow 期間 (5/19-6/16) は #updates のみ通知、
production 投票判断は 既存 RaceAutoNotify_Sat/_Sun のまま。

★ V15 production 完全不変 ★、 predict_core.py / daily_predict.py /
race_auto_notify.py / app.py には触れない。
"""
import time
from datetime import datetime, timedelta
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SHADOW_MODE = True  # ★ 6/17+ 採用判定で False に切替 ★


def main():
    date_str = datetime.now().strftime('%Y%m%d')

    # 1. 当日 race list 取得 (★ race_auto_notify.fetch_race_list_with_times 再利用 ★)
    from tools.race_auto_notify import fetch_race_list_with_times
    races = fetch_race_list_with_times(date_str)
    # races = [{'race_id', 'race_name', 'course', 'race_num', 'start_time'}, ...]

    # 2. 戦略⑦案 C 除外 (CLAUDE.md 4/27 仕様)
    races = [r for r in races if not is_strategy7c_excluded(r)]

    # 3. 各 race を 順次 process (single-threaded、 chronological order)
    for race in races:
        process_one_race(race, date_str)

    # 4. summary log
    write_daily_summary(date_str)


def process_one_race(race: dict, date_str: str):
    race_id = race['race_id']
    start_dt = parse_start_dt(race['start_time'])

    # phase 1: -20 min pre-fetch
    if not wait_until(start_dt - timedelta(minutes=20), max_wait_min=600):
        log_warn(f'{race_id}: -20min already passed, skip')
        return
    try:
        pre = fetch_pre_features(race_id)  # ★ P0-5-A 別設計 ★
        save_csv(pre, REPO / f'data/live_pre_features/{race_id}.csv')
    except Exception as e:
        log_err(f'{race_id} pre-fetch fail: {e}')
        return  # fallback: 朝 base prediction で運用継続 (★ production 影響 0 ★)

    # phase 2: -15 min recalc
    if not wait_until(start_dt - timedelta(minutes=15), max_wait_min=10):
        return
    try:
        recalc = run_recalc(race_id, pre)  # ★ P0-5-B 別設計 ★
        save_csv(recalc, REPO / f'data/recalc_predictions/{race_id}.csv')
    except Exception as e:
        log_err(f'{race_id} recalc fail: {e}')
        return

    # phase 3: -10 min 通知 (順位変動あり + shadow channel)
    if not wait_until(start_dt - timedelta(minutes=10), max_wait_min=10):
        return
    if has_rank_change(race_id, recalc):
        channel = '#updates' if SHADOW_MODE else '#bets'
        send_discord_rank_change(race_id, recalc, channel=channel)
```

### 3-2. helper 必要 (P0-5-A / B から渡される想定)

| function | 出力 | 提供元 |
|----------|------|--------|
| `fetch_pre_features(race_id)` | DataFrame | P0-5-A |
| `run_recalc(race_id, pre)` | DataFrame (top3 + score) | P0-5-B |
| `has_rank_change(race_id, recalc)` | bool | P0-5-B (朝 prediction との top3 diff) |
| `is_strategy7c_excluded(race)` | bool | 既存 race_auto_notify L186-200 同等 |
| `fetch_race_list_with_times(date_str)` | list[dict] | 既存 race_auto_notify.py L38-96 再利用 |

★ 既存 module は import のみ、 modify せず ★

### 3-3. 出力 artifact

| path | 内容 | 1 day size 想定 |
|------|------|----------------|
| `data/live_pre_features/{race_id}.csv` | -20 min 取得 features | 35 R × ~20 KB = 700 KB |
| `data/recalc_predictions/{race_id}.csv` | -15 min 再計算 top3 + score | 35 R × ~5 KB = 175 KB |
| `data/recalc_summary/{date_str}.csv` | daily summary (朝 vs 再計算 diff) | ~50 KB |
| `logs/live_orchestrator_{date_str}.log` | 全 phase log | ~500 KB |

---

## 4. T6 連携 (2 trigger 追加)

### 4-1. 既存 5 trigger (commit 2993f0b5、 不変)

`tools/anomaly_auto_detector.py`:
1. `check_predictions` — daily_predictions/{date}.csv 不在/0 行
2. `check_vote_candidates` — race_auto_notify log で 投票候補 0R
3. `check_streamlit` — :8501 HTTP GET 失敗
4. `check_discord_recent` — log mtime 古い / 通知 0 messages
5. `check_strategy7c` — 京都 R に対し Skip log なし

### 4-2. 新規 2 trigger 追加 (★ severity = warning ★、 production 影響 0)

#### trigger 6: `check_live_prefetch`

```python
def check_live_prefetch(date_str: str, repo: Path = REPO) -> dict:
    """data/live_pre_features/ 直下に当日 race_id ファイルが何件あるか"""
    pred_path = repo / 'data' / 'daily_predictions' / f'{date_str}.csv'
    pre_dir = repo / 'data' / 'live_pre_features'
    if not pred_path.exists():
        return _result('warning', 'check skipped (predictions 不在)')
    import pandas as pd
    try:
        n_total = len(pd.read_csv(pred_path))
    except Exception as e:
        return _result('warning', f'predictions 読込失敗: {e}')
    if not pre_dir.exists():
        return _result('warning',
                       f'live_pre_features/ 不在 (LiveOrchestrator 未起動 or 全 fail)')
    # 当日 race_id prefix = YYYYMMDD の 6桁 prefix (race_id = YYYYCCKKDDRR 12桁 から 推測)
    # 簡易: 当日生成 ファイル数 を mtime で count
    today = datetime.now().date()
    n_pre = sum(1 for p in pre_dir.glob('*.csv')
                if datetime.fromtimestamp(p.stat().st_mtime).date() == today)
    if n_pre == 0:
        return _result('warning',
                       f'-20min fetch 全 fail (predictions {n_total} R に対し pre 0 件)')
    if n_pre < n_total * 0.5:
        return _result('warning',
                       f'-20min fetch fail rate > 50% ({n_pre}/{n_total})')
    return _result('ok', f'-20min fetch {n_pre}/{n_total} OK')
```

#### trigger 7: `check_live_recalc`

```python
def check_live_recalc(date_str: str, repo: Path = REPO) -> dict:
    """data/recalc_predictions/ の当日 race_id ファイル数"""
    pred_path = repo / 'data' / 'daily_predictions' / f'{date_str}.csv'
    recalc_dir = repo / 'data' / 'recalc_predictions'
    if not pred_path.exists():
        return _result('warning', 'check skipped (predictions 不在)')
    import pandas as pd
    try:
        n_total = len(pd.read_csv(pred_path))
    except Exception as e:
        return _result('warning', f'predictions 読込失敗: {e}')
    if not recalc_dir.exists():
        return _result('warning', 'recalc_predictions/ 不在 (orchestrator 未起動?)')
    today = datetime.now().date()
    n_recalc = sum(1 for p in recalc_dir.glob('*.csv')
                   if datetime.fromtimestamp(p.stat().st_mtime).date() == today)
    if n_recalc == 0:
        return _result('warning', f'-15min recalc 全 fail (0/{n_total})')
    if n_recalc < n_total * 0.5:
        return _result('warning',
                       f'-15min recalc fail rate > 50% ({n_recalc}/{n_total})')
    return _result('ok', f'-15min recalc {n_recalc}/{n_total} OK')
```

### 4-3. `run_all` 拡張 (★ 1 行追加 ★)

```python
def run_all(date_str: str) -> list[tuple[str, dict]]:
    return [
        ('predictions',      check_predictions(date_str)),
        ('vote_candidates',  check_vote_candidates(date_str)),
        ('streamlit',        check_streamlit()),
        ('discord_recent',   check_discord_recent(date_str)),
        ('strategy7c',       check_strategy7c(date_str)),
        ('live_prefetch',    check_live_prefetch(date_str)),    # +T6.6
        ('live_recalc',      check_live_recalc(date_str)),      # +T6.7
    ]
```

`main` の `5 - critical - warning` ハードコードを `len(triggers) - critical - warning` に修正必要 (★ 既存 line 263、 修正 1 行 ★)。

### 4-4. severity policy

| trigger | severity | rollback 推奨 |
|---------|----------|---------------|
| 6 (prefetch fail) | warning | なし (production 影響 0) |
| 7 (recalc fail) | warning | なし (production 影響 0) |

→ critical にしない理由: shadow 期間中 = production 投票判断は 既存 RaceAutoNotify (5 min 前) のまま、 LiveOrchestrator 全失敗でも production 不変。

---

## 5. error handling + fallback chain

### 5-1. -20min fetch fail
1. log: `logs/live_orchestrator_{date_str}.log` (ERROR 行)
2. Discord `#updates` warning (★ shadow mode ★)
3. fallback: 朝 08:00 prediction が production で 使われる (既存 通り)

### 5-2. -15min recalc fail
1. 同様
2. 朝 base で 5 min 前 RaceAutoNotify が 通知 + 投票判断 (既存 通り)

### 5-3. -10min Discord notify fail
1. log のみ
2. production 影響 0

### 5-4. orchestrator 全死 (process crash)
1. ProcessWatchdog (既存 5min interval) が 検出? — ★ ただし `live_orchestrator.py` は ProcessWatchdog の 対象 list に未登録 ★
2. 5/18+ admin 登録時、 ProcessWatchdog config に 追加 する option あり (別 sub-task)
3. fallback: RaceAutoNotify は 独立 task、 LiveOrchestrator 死亡しても production 不変

### 5-5. race_auto_notify 重複起動 risk
- LiveOrchestrator は race_auto_notify と並走、 ただし **netkeiba fetch / Discord 通知 のみ** で 投票候補生成 (`cumulative_results.csv` 書込) は **しない**
- → race_auto_notify と書込競合なし

---

## 6. 5/18+ 実装 step

### Step 1: 設計 review (5/17 21:00、 30 min、 ★ 本 doc 含む 3 doc 統合 ★)
- P0-5-A (data fetch 設計) ← 別 sub-task で生成済 想定
- P0-5-B (recalc logic 設計) ← 別 sub-task で生成済 想定
- P0-5-C (本 doc)
- 親が 3 doc 統合 review

### Step 2: `live_orchestrator.py` 実装 (5/17 21:30-25:00、 約 4h)
- `tools/live_orchestrator.py` 新規
- `tools/live_data_fetcher.py` 新規 (P0-5-A 設計に従う)
- `tools/recalc_15min.py` 新規 (P0-5-B 設計に従う)
- 単体 test: `tests/test_live_orchestrator.py`

### Step 3: T6 拡張 (5/18 AM、 1-2h)
- `tools/anomaly_auto_detector.py` に `check_live_prefetch` + `check_live_recalc` 追加
- line 263 ハードコード `5` を `len(triggers)` に修正
- `tests/T6_anomaly_detection_test.py` に 2 case 追加

### Step 4: schtask 登録 (5/18 user GO 後、 admin 5 min)
- `tools/register_live_orchestrator_schtask.bat` 新規 (本 doc § 2-2 の schtasks /Create コマンド)
- ★ admin で実 schtasks /Create ★ (★ design 段階では 絶対実行しない ★)

### Step 5: paper shadow eval (5/19-6/16、 約 30R 蓄積)
- SHADOW_MODE=True で 4 週末運用
- `data/recalc_summary/` で 朝 vs 再計算 diff 蓄積

### Step 6: 採用判定 (6/17+、 別 sub-task)
- 統計検定 + ROI delta で GO/no-go
- GO なら SHADOW_MODE=False + #bets channel 切替

---

## 7. paper shadow eval plan (5/19-6/16)

### 7-1. 蓄積 metrics

| metric | source | 集計 |
|--------|--------|------|
| top3 diff rate | 朝 prediction vs 再計算 top3 一致率 | race 別 + daily 集計 |
| top1 shift count | 朝 top1 と 再計算 top1 不一致数 | daily |
| 投票候補 diff | 戦略⑦案 C 適用後の 推奨 race 数差 | daily |
| recalc fail rate | trigger 7 の warning 率 | daily |
| (paper) ROI delta | 再計算 top1 軸 vs 朝 top1 軸 | 全期間集計 |

### 7-2. 想定週末数

5/19 から 6/16 まで:
- 5/23-24, 5/30-31, 6/6-7, 6/13-14 = **4 週末**
- 1 週末 ~70R × 4 = ~280R 想定 (NAR 含まず)
- 戦略⑦案 C 後の 推奨 race ~30-40R 想定 (★ paper shadow eval 対象 ★)

### 7-3. 蓄積 file

```
data/recalc_summary/
  20260523.csv  # 各 race 別 (race_id, 朝 top1, 再計算 top1, diff, rank_change)
  20260524.csv
  ...
  paper_roi_delta.csv  # 期間集計
```

---

## 8. 採用判定基準 (6/17+、 別 sub-task で詳細設計)

### 8-1. quantitative

| 基準 | 閾値 | 想定 |
|------|------|------|
| top3 diff rate | < 15% | 想定 5-10% (馬体重 / オッズ shift で) |
| recalc fail rate | < 10% (= warning trigger 7 で warning率) | 想定 < 5% |
| paper ROI delta | +5pt 以上 (95% CI 下限) | ★ 想定 +0〜+8pt、 honest だと達成不透明 ★ |
| 統計検定 | bootstrap p < 0.05 | N=30 では 検出力 不足の可能性 |

### 8-2. qualitative

- production 投票判断不変
- V15 base AUC / ROI に影響なし
- T6 trigger 6/7 が 安定 ok (warning 率 < 10%)

### 8-3. NO-GO 時

- SHADOW_MODE=True のまま 7/末まで延長 (60R 蓄積)
- それでも NO-GO なら schtask 停止 (`schtasks /End /TN "..."` + `/Delete /F`)

---

## 9. V15 production 不変保証 ✅

| 既存 component | 5/18+ 変更 | 5/17 確認 |
|----------------|------------|-----------|
| `train/features_v15_new.py` | 触れない | ✅ |
| `tools/predict_core.py` | 触れない | ✅ |
| `tools/daily_predict.py` | 触れない | ✅ |
| `tools/race_auto_notify.py` | import 再利用のみ、 modify 0 | ✅ |
| `app.py` | 触れない | ✅ |
| `keiba_model_v15_central*.pkl.gz` | 触れない | ✅ |
| 既存 schtasks (DailyPredict / RaceAutoNotify 等) | 触れない | ✅ |
| 戦略⑦案 C 適用 (race_auto_notify L186-200) | 触れない | ✅ |

新規 artifact (★ V15 production と 独立 ★):
- `tools/live_orchestrator.py` (新規)
- `tools/live_orchestrator.bat` (新規)
- `tools/live_data_fetcher.py` (新規、 P0-5-A 設計に従う)
- `tools/recalc_15min.py` (新規、 P0-5-B 設計に従う)
- `data/live_pre_features/` (新規 dir)
- `data/recalc_predictions/` (新規 dir)
- `data/recalc_summary/` (新規 dir)
- `tools/anomaly_auto_detector.py` (★ 2 function 追加 のみ、 既存 5 function 不変 ★)

---

## 10. honest 注記

- 本 doc は **設計のみ**。 実 schtasks /Create / コード生成は **5/18+ 着手**
- `top3 diff rate` / `paper ROI delta` の 想定値 は **未検証**。 5/19-6/16 paper eval で 実測
- T6 trigger 6/7 の false positive 率 は 実 test で 計測 (現時点 不明)
- `fetch_race_list_with_times` の再利用は 「import 想定」 のみ、 実 import test 未実施
- 30R で の 統計検定力 は 不足の可能性あり (CI 幅広く GO 判定困難リスク)
- 月-金 開催 (祝日 + 平日特例) には 本 design 未対応 (★ 当面 SAT,SUN 限定、 平日対応は 採用後 ★)

---

## 11. 関連 doc

- T6 anomaly detector: `tools/anomaly_auto_detector.py` (commit 2993f0b5)
- 戦略⑦案 C 仕様: CLAUDE.md § "戦略⑦ 自動化 (4/27 適用済)"
- 既存 schtasks list: CLAUDE.md § "定期タスク (Windows タスクスケジューラ)"
- V15 production 仕様: CLAUDE.md § "現行モデルのベースライン"
- P0-5-A (data fetch 設計): 別 sub-task で生成 (★ 本 doc では import 想定 ★)
- P0-5-B (recalc logic 設計): 別 sub-task で生成 (★ 本 doc では 関数 signature 想定 ★)
