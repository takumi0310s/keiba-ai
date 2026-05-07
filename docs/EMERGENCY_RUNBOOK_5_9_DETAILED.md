# Emergency Runbook 5/9 詳細版 (Session #40 B2)

**作成**: 2026-05-07 (Session #40 B2)
**対象**: 5/9 V15 案B改 投資日 + 以後 V15 単独運用期間
**目的**: 障害シナリオ 15 件 + 検出 → 対応 → 復旧 手順 完備

---

## 0. 連絡 / 通知 channel

| channel | webhook env | 用途 |
|---------|-------------|------|
| `#investments` | `DISCORD_WEBHOOK_INVESTMENTS` | 5/9 当日 投資進捗 / 結果 |
| `#alerts` | `DISCORD_WEBHOOK_ALERTS` | 障害・失敗 |
| `#bets` | `DISCORD_WEBHOOK_BETS` | 平常 race_auto_notify 買い目 |
| `#updates` | `DISCORD_WEBHOOK_UPDATES` | 通常進捗 |

ヘルパー: `python tools/discord_routing.py --title X --body Y --channel alerts --color red`

---

## 1. シナリオ 1〜15

### S01: Cookie 切れ + refresh 自動失敗

**検出**:
- daily_predict.py / scrape_premium_data.py が 403 で停止
- nightly_sanity の `refresh_cookie.py --check` が NG
- Discord #alerts に "Cookie failed" 通知

**対応**:
1. PowerShell で `python tools/refresh_cookie.py` (対話式)
2. netkeiba ID/PW を入力
3. Cookie 更新後 `--check` で confirm
4. daily_predict 再 trigger

**復旧時間**: 5-10 分

**escalation**: Cookie 自動 fail 3 連続 → 5/9 case では **race_auto_notify 投票見送り、 PAT 手動投票切替**

---

### S02: JRDB 全 retry 失敗

**検出**:
- DailyJrdbKyi (06:00) で extracted/Bac/BAC{今日}.txt 不在
- jrdb_health_check が NG
- Discord #alerts に "JRDB 取得失敗"

**対応**:
1. `python tools/check_jrdb_status.py` で詳細確認
2. JRDB Advance サイトに手動 login → ZIP 取得
3. `data/jrdb/raw/Bac/BAC{date}.zip` に手動配置
4. `python tools/parse_jrdb.py --date {YYYYMMDD}` で extract
5. daily_predict 再 trigger

**復旧時間**: 15-30 分

**5/9 case**:
- JRDB なしでも V15 は動作可 (jrdb 系 features は 0 fallback)
- ただし AUC 落ちる → race_auto_notify 投票は信頼度低下、 PAT 手動投票推奨

---

### S03: 馬体重取得失敗 (一部開催)

**検出**:
- predict_core で `horse_weight = 0` の馬が発生
- daily_predict log で warning

**対応**:
1. netkeiba 開催ページを直接アクセスして馬体重を確認
2. data/race_card_{race_id}.json を手動編集
3. predict_one_race.py で再実行

**復旧時間**: 10-15 分 / race

**5/9 case**:
- 案B改 12R 1勝クラス のみ → 該当 race の馬体重を絶対確認
- 不明なら 該当 race を投資除外

---

### S04: Discord webhook 死亡 (404 / rate limit)

**検出**:
- notify_done.py / discord_routing.py が FAIL
- log に 404 / 429

**対応**:
1. Discord サーバーで webhook を再生成
2. `.env` の `DISCORD_WEBHOOK_*` を新 URL で更新
3. test: `python tools/discord_routing.py --title test --body test --channel updates`

**復旧時間**: 5 分

**5/9 case**:
- 通知が届かなくても投票自体は可能
- final_health_check 結果は手動確認 (PowerShell stdout 直視)

---

### S05: PAT サーバー障害

**検出**:
- 5/9 当日 PAT login 不能
- A-PAT (即PAT) 経由でも buy 不能

**対応**:
1. JRA 公式 (`https://www.jra.go.jp/`) で障害情報確認
2. 緊急 hotline 03-3592-2000 に連絡
3. 復旧後 即 buy
4. 復旧時刻が レース 5 分前以内なら 該当 race **投票見送り** (損失回避)

**復旧時間**: JRA 次第 (通常 30 分以内)

**5/9 case**:
- 案B改 投資額が小さい (上限 2,100円) → 投票見送り = 損失 0
- 1 R missed = -700 円 (機会損失) で済む
- 焦らず **投票見送り 推奨**

---

### S06: ProcessWatchdog 誤発火 (生きている process を kill)

**検出**:
- daily_predict.py が突然 kill された
- log に "watchdog killed PID xxxx"

**対応**:
1. `python tools/daily_predict.py --resume` で再開
2. resume が無理なら `python tools/daily_predict.py` 全実行
3. ProcessWatchdog の log を確認、 false positive ルール調整

**復旧時間**: 10-15 分

**5/9 case**:
- 朝 08:00 daily_predict が kill されたら 09:00 まで再 trigger
- 09:00 前に終了しないなら 当日 全 race 投票見送り

---

### S07: 落雷・停電 (V15 production 全停止)

**検出**:
- PC 強制終了
- UPS 警告音

**対応**:
1. PC 復旧後 起動
2. `git status` で uncommitted 確認
3. `python tools/final_health_check_5_8.py` で状態 confirm
4. 投資判断: 5/9 当日 復旧前なら **投票見送り**

**復旧時間**: 数分 〜 数時間 (停電次第)

**5/9 case**:
- 雷雨予報なら前日 PC バックアップ電源確保
- 停電中は 完全見送り

---

### S08: NW障害

**検出**:
- daily_predict が timeout
- ping 不通

**対応**:
1. router 再起動 (10s 電源 OFF)
2. ISP 障害情報確認
3. モバイル tethering で fallback
4. PAT は スマホ A-PAT で代替投票可能

**復旧時間**: 数分 〜 数時間

**5/9 case**:
- スマホ A-PAT 事前 setup 推奨
- PC 復旧不能なら スマホで案B改 投票実行

---

### S09: PC ハング (Windows 11)

**検出**:
- マウス/キーボード反応なし
- TaskManager (Ctrl+Shift+Esc) でも不能

**対応**:
1. 電源ボタン 長押し → 強制再起動
2. 起動後 `git status` で状態 confirm
3. ProcessWatchdog 再起動: `python tools/process_watchdog.py`

**復旧時間**: 5-10 分

**5/9 case**:
- 朝 06:00 final_health_check が動かない → 手動実行
- 投票時刻 (10:00 +) に PC 復旧不能なら スマホ A-PAT 切替

---

### S10: PowerShell 起動不能

**検出**:
- schtasks の bat が動かない
- powershell.exe で error 0x80072EE2 等

**対応**:
1. CMD で代替実行: `cd C:\Users\takum\keiba-ai && python tools/daily_predict.py`
2. PowerShell 修復: `sfc /scannow` (管理者)
3. Python 直接実行で迂回

**復旧時間**: 10-30 分

**5/9 case**:
- python 単体で動くので CMD で十分
- final_health_check は CMD で `python tools\final_health_check_5_8.py`

---

### S11: python crash (predict_core 異常終了)

**検出**:
- daily_predict log に Traceback
- exit code != 0

**対応**:
1. log の error 種別を確認
   - ImportError → `pip install ...` で再 install
   - MemoryError → 不要 process kill (ProcessWatchdog 含む)
   - KeyError (列不在) → data file 確認、 必要なら scrape 再実行
2. `python tools/predict_one_race.py {race_id}` で 1 race 単位 retry
3. それでも NG なら 該当 race 投票見送り

**復旧時間**: 10-30 分

**5/9 case**:
- predict_core の crash は V15 model load 失敗 が一番怖い
- final_health_check で前日 confirm 済 → 当日新規 crash は env 起因が多い

---

### S12: git 衝突

**検出**:
- git pull / push で error
- "Your branch and 'origin/main' have diverged"

**対応**:
1. `git status` で uncommitted 確認
2. 必要なら stash: `git stash push -u -m "emergency"`
3. `git pull --rebase origin main`
4. 衝突したら手動 merge → `git rebase --continue`
5. push: `git push origin main`

**復旧時間**: 5-15 分

**5/9 case**:
- 投資直前は git operation 不要
- 5/9 朝 health check 後 commit ない方針

---

### S13: 学習 data 破損 (jra_races_full.csv 等)

**検出**:
- pandas read_csv で error
- file size 0 byte

**対応**:
1. `git log --oneline -- data/jra_races_full.csv` で最新 commit 確認
2. `git checkout HEAD -- data/jra_races_full.csv` で復元 (gitignored だが backup あれば)
3. backup なし → scrape_jra.py 等で再生成

**復旧時間**: 30 分 〜 数時間

**5/9 case**:
- daily_predict は learn_pickle (V15 model file) のみで動作
- jra_races_full は learning 用、 5/9 production には不要
- 慌てず V15 production を継続

---

### S14: predict_core 異常終了 (5/9 当日 朝)

**検出**:
- daily_predict log に "predict_core failed for race xxx"

**対応**:
1. 該当 race を `python tools/predict_one_race.py {race_id}` で個別実行
2. それでも NG → そのレースは 投票見送り
3. 他の race は通常 process

**復旧時間**: 10 分 / race

**5/9 case**:
- 案B改 12R 1勝クラス → 該当 race の数は 1-3 R
- 個別失敗時は 該当 R 見送り、 他 R 継続

---

### S15: 全 system fallback (manual override)

**検出**:
- 上記 S01-S14 が複数発生 + 復旧不能
- 5/9 朝 06:00-09:00 で system 不動

**manual override 手順**:
1. PAT に 直接 login (PC or スマホ A-PAT)
2. netkeiba で 5/9 出馬表確認
3. **過去の 案B改 BT で「上位 3 番人気から trio 7 点」** を仮買い
4. 投票額: 700円/R × 最大 3R = 2,100 円
5. 結果 翌日 `python tools/daily_results.py` で照合

**復旧時間**: 30-60 分

**重要**:
- system 復旧優先で 最悪 当日 投票見送り も選択肢
- 撤退余裕 +63,530円 の中で 機会損失 -2,100円 (1 日分) は 3.3% のみ
- **無理せず 翌週 (5/16+) に judgment 持ち越し**

---

## 2. 5/9 当日 朝 全体フロー (時系列)

```
05:00  PC ON、 sleep 解除
06:00  Keiba-FinalHealthCheck (schtasks 自動)、 NG なら #alerts 通知
06:30  health check 結果 手動 confirm (Discord 確認)
07:00  Keiba-MorningDigest (dashboard) 自動
08:00  DailyPredict 自動実行 (V15 全レース 推論)
08:45  RaceAutoNotify 自動 (戦略⑦ + 案B改 → #bets / #investments)
09:00  予測結果 手動 確認 + 投票候補 list 確定
09:30  PAT login + 入金確認
10:00- レース開始時刻に応じて投票 (1勝 のみ)
18:00  DailyResults 自動 結果照合
20:30  振り返り (data/v18/post_5_9_improvement_template.md)
```

各時点で 上記 S01-S15 シナリオ発生時は対応 → manual override に escalation。

---

## 3. 撤退判定 (緊急)

| 累計 (1 日終了時) | 状態 | 翌日 |
|----------------|------|------|
| ≥ -2,100 円 (案B改 損失内) | 想定内 | 通常運用継続 |
| -2,100 〜 -10,000 円 | 注意 | 翌週投資停止、 原因調査 |
| -10,000 〜 -50,000 円 | 警告 | 全停止、 5/16+ 再判定 |
| < -50,000 円 | **撤退** | 完全停止、 全 model 廃止判断 |

5/9 単日 想定 max loss = -2,100円 (案B改 全外し)。
従って 5/9 中の撤退は **発生しない** (撤退ライン 1/24 にしか達しない)。

---

## 4. 結論

✅ 15 シナリオ 全 検出 / 対応 / 復旧 手順 整備
✅ Discord 3 channel routing (investments / alerts / updates) で priority 分離
✅ manual override 経路 確保 (S15)
✅ 撤退判定 累計収支 ベース で明文化
✅ 5/9 当日 想定 max loss = -2,100円 (撤退ライン 1/24)

→ **5/9 投資 緊急対応 完備**

---

**Session #40 B2 完了**
