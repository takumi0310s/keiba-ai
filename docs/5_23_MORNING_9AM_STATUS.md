# 5/23 朝 9:00 真の運用初日 統合確認

**確認時刻**: 2026-05-23 09:28 (watchdog_v2 最終 fire 時刻より)  
**対象**: DailyPredict / LiveOrchestrator / 買い目通知 / v2 paper tracking / TYB観測 / AnomalyCheck

---

## 総合 verdict: **一部要対応** ⚠

| コンポーネント | 状態 | 詳細 |
|--------------|------|------|
| DailyPredict (08:00) | ✅ **正常** | 34/34 R 予測成功 |
| race_auto_notify (08:45) | ✅ **正常** | 36R スケジュール済、8 買い目通知 |
| race_notify_log v2 phase1 | ✅ **正常 (c-fix確認)** | 34 JSON 全非 None |
| LiveOrchestrator (08:30) | ❌ **未 fire** | Python stub 問題 (下記) |
| AnomalyCheck 0630/0830 | ❌ **失敗** | 同上 Python stub 問題 |
| TYB 観測 mode | ⏳ **待機中** | 09:25~ (1R 前) 初 fetch 予定 |
| KAB 影響 | ✅ **なし** | 予測に KAB 依存なし、34/34 正常 |

---

## 1. DailyPredict ✅

| 項目 | 値 |
|------|-----|
| 完了時刻 | 2026-05-23 08:39:11 |
| 予測対象 | 34 R (全 36 - 障害 2) |
| 成功 | 34/34 (失敗 0) |
| 会場 | 新潟 + 東京 + 京都 (各 12R 前後) |
| 総投資額 | ¥23,800 |
| JRDB | SED 前走特徴量 取得確認 (12/12 馬等) |
| CSV | data/daily_predictions/20260523.csv (34行) ✅ |

**KAB 影響**: なし。予測 log に KAB error 記載なし、34/34 完走。

---

## 2. LiveOrchestrator ❌ 未 fire

**verdict: 本日 5/23 に LiveOrchestrator は実際に fire していない**

| ファイル | 内容 | 変更日時 |
|---------|------|---------|
| data/live_orchestrator_log/20260523.log | 5/18 テスト実行分 (171 bytes) | **2026-05-18** 18:19 |
| 同ファイル内容 | `{"event":"orchestrator_start","mock":true,"dry_run":true,"timestamp":"2026-05-18T..."}` + `no_races` | — |

**原因**: `tools/live_orchestrator.bat` が以下 Python パスを使用:
```
C:\Users\takum\AppData\Local\Microsoft\WindowsApps\python.exe
```
これは Microsoft Store Python スタブ (alias) であり、**Task Scheduler から起動すると「指定されたプログラムは実行できません」** エラーで失敗する。

→ AnomalyCheck (0630/0830) も同じパスで同じ理由で失敗している。

**リスク評価**: V15 production には無影響。race_auto_notify.py は別の schtask で正常動作済み。LiveOrchestrator はまだ mock/dry-run モードのため、買い目への影響なし。

**要対応**: Python パスを `%PYTHON_EXE%` から `python` または絶対パス (非 Store stub) に変更が必要。→ 5/24 以前に修正推奨。

---

## 3. 買い目通知 ✅

| 項目 | 値 |
|------|-----|
| 起動時刻 | 2026-05-23 08:45 |
| 検出レース数 | 36 R (新潟 + 東京 + 京都 各 12R) |
| 通知スケジュール | 新潟1R 09:40 〜 京都12R 16:25 (36 件) |
| 整形済み買い目通知 | **8 messages** 送信済 |

発走スケジュール確認:
- 新潟 1R: 09:45 / 東京 1R: 09:55 / 京都 1R: 10:05
- 新潟 12R: 16:01 / 東京 12R: 16:10 / 京都 12R: 16:30

**戦略⑦案 C / C4 適用**: log に `[STRATEGY_C4]` / `[STRATEGY7]` の SKIP エントリが 2 件 ("障害レース除外" は別扱い)。買い目通知は 36R 中 8R = フィルタ後の通知数。

---

## 4. race_notify_log v2 phase1 ✅ (c-fix 確認)

| 項目 | 値 |
|------|-----|
| phase1 JSON 数 | **34 ファイル** (data/race_notify_log_v2/20260523/phase1/) |
| None 値 | **0件** ← c-fix 前は全 None だった |
| distance | 記録済 (例: 1800, 1200, 1400) ✅ |
| surface | 記録済 (例: ダ, 芝) ✅ |
| condition | 記録済 (例: C, D, A) ✅ |
| ranking_top5 | 馬名 / umaban / score 全記録 ✅ |
| formation_planned | trio 7点 formation 文字列 ✅ |

サンプル (202604010702 = 新潟2R):
```json
{
  "race_meta": {"distance": 1800, "surface": "ダ", "condition": "C"},
  "ranking_top5": [{"rank":1,"umaban":15,"name":"ウェルム","score":0.242},...],
  "formation_planned": "2-3-15; 2-8-15; 3-5-15; ..."
}
```

**paper_shadows = {}**: phase2 は発走 -5 min に記録 → 09:40〜 蓄積開始予定。現時点 (09:28) は空で正常。

★ **c-fix 確認完了**: 5/23 が 8 strategy paper tracking の初の真の蓄積日 ★

---

## 5. TYB 観測 mode ⏳ 待機中

| 項目 | 状態 |
|------|------|
| TYB_SHADOW_OBSERVE_MODE | True (設定済) |
| TYB_OBSERVE_LAUNCH_DATE | "20260523" |
| 5/23 fetch 実績 | **なし** (09:28 時点、第 1R 発走 09:45 のため未到達) |
| 初回 fetch 予定 | 09:25〜 (新潟1R 発走 09:45 の -20min) |
| 5/22 log エントリ | ERROR entries のみ (単体テストの mock 失敗、本番影響なし) |

**Discord 非表示確認**: TYB_SHADOW_ENABLED=False のまま。fetch_tyb_observe() は log のみ。

---

## 6. KAB 影響 ✅

Daily predict log に KAB 関連エラーなし。JRDB SED / PACI は正常取得 (SED 12/12、PACI 馬名フォールバック含)。予測 34/34 完走。

---

## 7. AnomalyCheck 0630/0830 ❌

| ファイル | 内容 |
|---------|------|
| logs/keiba_anomaly_check_0630_20260523.log | 40 bytes — "指定されたプログラムは実行できません" |
| logs/keiba_anomaly_check_0830_20260523.log | 40 bytes — 同上 |

**原因**: bat ファイル内 `PYTHON_EXE=C:\Users\takum\AppData\Local\Microsoft\WindowsApps\python.exe` が Task Scheduler から実行不可。

**要対応**: LiveOrchestrator と同じ Python パス問題。V15 / 買い目には影響なし。

---

## 要対応事項 (優先度順)

| 優先度 | 問題 | 影響 | 修正方法 |
|--------|------|------|---------|
| ★ | Python Store stub エラー | LiveOrchestrator 未 fire / AnomalyCheck 全失敗 | bat ファイルの `PYTHON_EXE` を有効な Python パスに変更 |
| 中 | LiveOrchestrator 5/24 mock 解除 | 5/24 以降に live fetch が必要 | Python パス修正後、mock/dry-run 解除 |
| 低 | TYB observe 午前 R 確認 | 09:25〜 初 fetch — 本日夕方に結果確認 | summarize_observe_log('20260523') |

---

## Python パス問題 — 暫定修正案

```bat
:: 修正前 (Store stub — Task Scheduler では動かない)
SET PYTHON_EXE=C:\Users\takum\AppData\Local\Microsoft\WindowsApps\python.exe

:: 修正後案 (要ユーザー確認：実際の python.exe パスに変更)
:: 候補 1: venv を使用する場合
SET PYTHON_EXE=C:\Users\takum\AppData\Local\Programs\Python\Python3xx\python.exe
:: 候補 2: keiba-ai の venv
SET PYTHON_EXE=C:\Users\takum\keiba-ai\venv\Scripts\python.exe
:: 候補 3: conda/miniforge の場合
SET PYTHON_EXE=C:\Users\takum\miniforge3\python.exe
```

**確認コマンド (Powershell)**:
```powershell
(Get-Command python).Source
```

---

*確認日時: 2026-05-23 09:28 | read-only audit | V15 production 不変*
