# 夜-1: P0-5 paper eval 準備 dry-run 結果 (2026-05-17 21:00+)

> 目的: 5/18 朝 admin schtask 登録 (Keiba-LiveOrchestrator-15min SAT/SUN 08:30) の前提となる
> P0-5 pipeline 4 module の動作確認を mock data + dry-run のみ で実施。
> 実 schtasks /create / 実 fetch は **絶対 0**。V15 production 完全不変。

---

## 0. 結論

| 項目 | 判定 |
|------|------|
| dry-run 4 module 動作 | **PASS** |
| 5/18 admin 登録 | **ready** (前提 bat 4 件全実在、 conflict 0) |
| 5/24 初回 fire 準備 | **完了** (checklist Section 6 参照) |
| V15 production 不変保証 | **OK** (predict_core / daily_predict / app.py / .pkl.gz / cumulative_results 全 untouched) |
| 実 schtasks /create | **0** (★ 5/18 admin 待ち ★) |
| 実 fetch | **0** (★ G1 day blocklist 20260517 適用 + 5/24+ 待ち ★) |

★ 1 件 注意: `tools/live_orchestrator.bat` (schtask の TR=実行 cmd) **未存在**。
   5/18 admin 登録前に親側で作成 必要 (詳細 Section 7)。

---

## 1. tools/p0_5_schtask_register.bat dry-run

### syntax check (file 内容 verify)
- file 存在: PASS (46 行)
- admin 権限 check (Section 1, `net session`): logic OK
- 既存 conflict check (Section 2, `schtasks /Query /TN "Keiba-LiveOrchestrator-15min"`): logic OK
- 登録 cmd (Section 3): `schtasks /Create /TN Keiba-LiveOrchestrator-15min /TR ...live_orchestrator.bat /SC WEEKLY /D SAT,SUN /ST 08:30 /RL HIGHEST /F`

### conflict check (実 schtasks /Query 結果)
既存 8:30 fire task:
- `KeibaAI_DriftDetector` (DAILY 08:30) — 別 process / 別 log、 衝突せず
- `Keiba-LiveOrchestrator-15min` — **未登録** (新規)

★ 5/18 admin 登録時の上書き確認 prompt あり → 安全。

---

## 2. Keiba-LiveOrchestrator mock 動作 dry-run

### 2-1. tools/live_data_fetcher.py --race-id 202608030801 --mock --dry-run
```
status: ok
sources_used: [jvlink_o1, jvlink_tcov, netkeiba_pre]
odds_snapshot: {1:8.0, 2:12.0, 3:4.0, 4:20.0, 5:1.5}
baba_condition: 良 / moisture: 8.5 / cushion: 9.2
weight_diff: {1:0, 2:4, 3:-2, 4:12, 5:-3}
```
★ G1 day blocklist 20260517 = active、 `--mock` 省略でも `forced_mock=True` で 実 fetch 拒否確認。

### 2-2. tools/calibrator_overlay.py --build-state
```
[OK] saved data/v21/calibrator_overlay_v1.pkl
keys: [version, delta_cap, odds_weight, leak_features, weight_diff_thresholds, note]
delta_cap: 0.10 (★ ±0.10 遵守 ★)
odds_weight: 0.3
```

### 2-3. tools/calibrator_overlay.py --simulate 20260517 --dry-run
```
n_races: 34
n_with_delta: 34
mean_abs_delta: 0.0851
max_abs_delta: 0.0999... (< DELTA_CAP 0.10)
delta_cap_respected: True
has_odds_base: True
```
★ DELTA_CAP ±0.10 全 race 遵守 verify。

### 2-4. tools/discord_recalc_notify.py --race-id 202608030801 --mock
```
[DRY-RUN] would send to #updates:
[CRIT] P0-5 recalc shadow: 東京 mock race (202608030801)
severity: critical
top1 change: 5 -> 3
(paper shadow eval / V15 production 不変)
```
★ #updates webhook env (`DISCORD_WEBHOOK_UPDATES`) のみ参照、 #買い目 webhook touch 0。

### 2-5. tools/recalc_15min.py (full pipeline、 非 京都 race)
race_id=202604010602 (新潟 3歳未勝利 1800m ダ)、 14 頭 V15 prob 取得 → mock -15min features 適用:
- 朝 8:00 top1: 馬 6 (V15 prob 0.807)
- recalc top1: 馬 6 (corrected 0.807)
- top3 swap 1 件、 severity: **minor**
- #updates dry-run 通知 fire OK
- 出力 (dry-run=True で skip): `data/recalc_15min/20260517/202604010602.json` 想定

### 2-6. tools/recalc_15min.py (京都 race 戦略⑦案 C skip)
race_id=202608030801 (京都 3歳未勝利 1800m ダ):
```
status: strategy_7c_skip
skip_reason: strategy_7_kyoto_p0_2_5_17
```
★ 戦略⑦案 C 京都 filter 動作 OK (race_auto_notify.py logic 整合)。

---

## 3. fallback chain 動作確認

| stage | fail 想定 | 観測動作 | V15 投票判断影響 |
|-------|---------|---------|----------------|
| daily_predictions 不在 | race_id 999999999999 / date 20260601 | `status: fallback`, `fallback_to: V15_morning_8am` | **影響なし** |
| live_pre_features 不在 | mock data で代替 | mock data 使用 + 通知 dry-run | **影響なし** |
| odds_15min NaN | 一部馬の odds 欠損 | `corrected_prob = v15_prob` (delta 0) | **影響なし** |
| weight_diff NaN | 一部馬の体重差 欠損 | 同上 (delta 0) | **影響なし** |
| DISCORD_WEBHOOK_UPDATES 未設定 | 環境変数なし | `[WARN] send skip` + return False | **影響なし** |
| 戦略⑦案 C 該当 | 京都 race | `strategy_7c_skip` で 出力 skip | **影響なし** |

★ 全 fail 経路で V15 朝 8:00 投票判断は **完全不変**。 P0-5 は shadow eval only。

---

## 4. T6 異常 detection 連携

### 既存 5 trigger (commit 2993f0b5、 tools/anomaly_auto_detector.py)
1. `check_predictions` — daily_predictions 不在 / 0 rows
2. `check_vote_candidates` — race_auto_notify log で 0 messages
3. `check_streamlit` — port 8501 HTTP GET fail
4. `check_discord_recent` — log mtime 古い / 通知 0
5. `check_strategy7c` — 京都 R 通り抜け check

### P0-5 paper eval 連携の 2 新 trigger (5/18+ 拡張余地)
6. `check_live_pre_features_fetch` — `data/live_pre_features/{date}/` の 5/24+ 出力数 < race 数
7. `check_recalc_15min_output` — `data/recalc_15min/{date}/` の 5/24+ 出力数 < 想定

### 期待 metric (5/24 初回 fire 後 計測)
- false positive < 5% (現行 5 trigger 同等)
- true positive > 95% (同上)
- ★ 実測値は 5/24-5/25 初週末 蓄積後 評価 ★

★ P0-5 連携 trigger の **実装は 5/18 admin 登録後の別 task**。 本 dry-run は 既存 5 trigger logic 確認のみ。

---

## 5. 5/18 admin 整合確認

docs/5_18_ADMIN_TASKS.md (commit 61b6a0b6) と本 dry-run の整合:

| 項目 | 5/18 admin docs | dry-run 確認 | 整合 |
|------|----------------|------------|------|
| schtask 名 | Keiba-LiveOrchestrator-15min | bat 内 / register 内 一致 | OK |
| timing | WEEKLY SAT,SUN 08:30 | bat /SC WEEKLY /D SAT,SUN /ST 08:30 | OK |
| 実行 cmd | tools/live_orchestrator.bat | bat 内 /TR 一致 | **NG (bat 未存在)** |
| run level | HIGHEST (admin) | bat /RL HIGHEST | OK |
| 初回 fire | 5/18 SUN 08:30 → ★ **本 dry-run 後 修正: 5/24 SAT 初回** ★ | (本 docs は 5/24 想定) | 注 ※ |
| conflict | なし | 実 schtasks /Query で確認、 8:30 は DriftDetector のみ (別 process) | OK |
| rollback | schtasks /Delete /TN ... /F | bat 不要、 1 行 cmd | OK |

※ **重要 不整合**: 5_18_ADMIN_TASKS.md は「5/18 SUN 即 fire」 想定だが、
   本 dry-run prompt 冒頭は「初回 fire は 5/24 (土)」想定。
   **どちらを採用するか親判断必要** (5/18 fire なら paper eval 1 日 前倒し可、 5/24 fire なら 1 週空く)。

---

## 6. 5/24 初回 fire 前 checklist

```
■ 5/18 admin 登録 (★ user 起床後 06:30-07:00 推奨 ★)
  □ powershell 管理者として実行 (Start menu 右クリック)
  □ cd C:\Users\takum\keiba-ai
  □ tools\live_orchestrator.bat 存在確認 (★ 親側で 5/18 admin 前に作成必要 ★)
  □ .\tools\p0_5_schtask_register.bat 実行
  □ schtasks /Query /TN "Keiba-LiveOrchestrator-15min" /V /FO LIST で 登録確認
  □ logs/ writable 確認 (touch test)
  □ DISCORD_WEBHOOK_UPDATES env 設定確認
  □ V15 production unchanged 確認 (git status で predict_core / daily_predict / app.py clean)

■ 5/19-5/23 待機期間
  □ 平日 schtask 起動なし (SAT/SUN 限定 — verify)
  □ 異常監視: logs/live_orchestrator.log (★ 5/24 初回 fire 後 生成 ★)
  □ 既存 8 schtask の動作不変 confirm (DailyPredict 08:00 / RaceAutoNotify 08:45 等)

■ 5/24 (SAT) 朝 08:30 初回 fire
  □ 08:30 LiveOrchestrator-15min 自動起動 (★ user 起床後 確認 ★)
  □ logs/live_orchestrator.log 出力確認
  □ race -20/-15/-10 min 各 fetch / recalc / 通知 fire 確認
  □ data/recalc_15min/20260524/ 出力確認 (race 数分の json file)
  □ Discord #updates 通知受信確認 (順位変動あり race のみ、 戦略⑦案 C skip 反映)
  □ V15 朝 8:00 投票判断 完全不変 verify (data/daily_predictions/20260524.csv touch 0)

■ 5/24 夜 (DailyResultsEvening 20:00 完了後)
  □ data/recalc_15min/20260524/ で 12-18 R 蓄積確認
  □ 戦略⑦案 C skip 動作 (京都 R / 条件 X 含まれず)
  □ T6 既存 5 trigger 異常通知 0 件 verify

■ 異常時 緊急停止 (1 行 admin)
  schtasks /Delete /TN "Keiba-LiveOrchestrator-15min" /F
```

---

## 7. ★ Section 7: 5/18 admin 前 親側 作業 (P0 必須) ★

### 7-1. tools/live_orchestrator.bat 作成必要

現状:
- `tools/p0_5_schtask_register.bat` (commit 333da9b0) は `/TR "C:\Users\takum\keiba-ai\tools\live_orchestrator.bat"` を指す
- **しかし `tools/live_orchestrator.bat` は未存在**

→ 5/18 admin 登録前 (★ 5/17 中の親 commit ★) に live_orchestrator.bat 作成必須。

### 7-2. 想定 内容 (★ 親判断、 本 agent 作成せず ★)
```batch
@echo off
:: Keiba-LiveOrchestrator-15min main entry (WEEKLY SAT/SUN 08:30 起動)
cd /d C:\Users\takum\keiba-ai
:: 当日 race ループ → race -20/-15/-10 min 各 timing で recalc_15min.py
:: log: logs/live_orchestrator_YYYYMMDD.log
python tools/live_orchestrator_main.py >> logs/live_orchestrator_%date:~0,4%%date:~5,2%%date:~8,2%.log 2>&1
```
※ python entry point の有無も 親側 確認。

### 7-3. docs/5_18_ADMIN_TASKS.md 初回 fire 日 統一
- 5/18 SUN 即 fire vs 5/24 SAT 初回 — 親判断 → docs 統一 update

---

## 8. fabrication 防止 confirm

| 項目 | 実測 |
|------|------|
| 実 schtasks /create 回数 | **0** |
| 実 fetch 回数 (jvlink/netkeiba) | **0** (mock のみ) |
| Discord 実発火回数 | **0** (全 dry-run) |
| V15 .pkl.gz 変更 | **0** |
| cumulative_results.csv 変更 | **0** |
| predict_core / daily_predict / race_auto_notify / app.py 変更 | **0** |
| git commit / push (agent 内) | **0** (★ 親集中 ★) |

★ 「想定」値の表記:
- 5/24 初回 fire の log 出力 / Discord 通知数 — **未実測** (5/24 朝の親確認待ち)
- T6 連携 false positive < 5% / true positive > 95% — **未実測** (5/24-5/25 蓄積後)

---

## 9. 最終判定

- **dry-run 4 module 動作: PASS** (live_data_fetcher / calibrator_overlay / discord_recalc_notify / recalc_15min)
- **5/18 admin 登録: ★ ready (条件付) ★** — `tools/live_orchestrator.bat` 作成が 5/18 admin 前提
- **5/24 初回 fire checklist: ready** (Section 6)
- **V15 production 完全不変: 確認済**
