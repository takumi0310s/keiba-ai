# 2026-05-23 AnomalyCheck schtask 登録確認レポート

## 1. AnomalyCheck 5 bat files: Python path 確認

| ファイル | PYTHON_EXE | 状態 |
|---------|-----------|------|
| tools/keiba_anomaly_check_0630.bat | C:\Users\takum\AppData\Local\Python\pythoncore-3.14-64\python.exe | OK |
| tools/keiba_anomaly_check_0830.bat | C:\Users\takum\AppData\Local\Python\pythoncore-3.14-64\python.exe | OK |
| tools/keiba_anomaly_check_0940.bat | C:\Users\takum\AppData\Local\Python\pythoncore-3.14-64\python.exe | OK |
| tools/keiba_anomaly_check_1410.bat | C:\Users\takum\AppData\Local\Python\pythoncore-3.14-64\python.exe | OK |
| tools/keiba_anomaly_check_1700.bat | C:\Users\takum\AppData\Local\Python\pythoncore-3.14-64\python.exe | OK |
| tools/anomaly_auto_detector.bat (TR target) | C:\Users\takum\AppData\Local\Python\pythoncore-3.14-64\python.exe | OK |

全 6 ファイル: pythoncore-3.14-64 真パス使用、WindowsApps stub なし。

## 2. schtask クエリ結果: AnomalyCheck

**AnomalyCheck 5 タスク = 未登録**

`schtasks /query /fo LIST /v` 全出力を "AnomalyCheck" で検索 → ヒット 0 件。

| 期待タスク名 | 状態 |
|------------|------|
| Keiba-AnomalyCheck-0630 | 未登録 |
| Keiba-AnomalyCheck-0830 | 未登録 |
| Keiba-AnomalyCheck-0940 | 未登録 |
| Keiba-AnomalyCheck-1410 | 未登録 |
| Keiba-AnomalyCheck-1700 | 未登録 |

=> `tools/register_anomaly_detector_schtask.bat` を管理者権限で実行して登録が必要。

## 3. register_anomaly_detector_schtask.bat: 文字化け確認

`tools/register_anomaly_detector_schtask.bat` 全行確認:
- 全コメント行: `::` プレフィックス付き (正常)
- 問題行（bare comment text = コマンドとして実行される行）: なし
- SET 行: `set TR="C:\Users\takum\keiba-ai\tools\anomaly_auto_detector.bat"` (正常)
- schtasks /Create 行: 5 行すべて正常

**文字化け / エンコーディング問題: なし。修正不要。**

## 4. FridayWeekendScrape: PACI line 確認

`friday_weekend_scrape.bat`:
- PACI scrape 行: **存在** (line 18-19: `python tools\scrape_jrdb_paci.py`)
- schtask: **登録済み** (`\Keiba-FridayWeekendScrape`)
  - Status: Ready
  - Next Run: 2026/05/29 10:00:00 (次の金曜)
  - Last Run: 2026/05/22 10:00:00 (先週金曜、正常稼働)

**注意**: `friday_weekend_scrape.bat` は bare `python` コマンドを使用 (明示的パスなし)。
現状は `python` が真パス (`pythoncore-3.14-64`) に解決されているなら問題なし。
確認推奨: `where python` の出力が WindowsApps 以外であること。

## 5. 全 schtask 健全性チェック

WindowsApps (stub) python を使用するタスク: **0 件**

主要 Keiba タスク (抜粋):

| タスク名 | Status | Next Run |
|---------|--------|----------|
| \Keiba-FridayWeekendScrape | Ready | 2026/05/29 10:00 |
| \KeibaAI_DriftDetector | Ready | 2026/05/25 8:30 |
| \keiba-ai\DailyPredict | Ready | 2026/05/24 8:00 |
| \keiba-ai\DailyPremiumScrape | Ready | 2026/05/24 3:00 |
| \keiba-ai\RaceAutoNotify_Sat | Ready | 2026/05/30 8:45 |
| \keiba-ai\WeeklyReport | Ready | 2026/05/25 8:00 |

全タスク Status = Ready。N/A (1回限り実行済み) タスクも正常終了済み。

## 6. アクション要否

| 項目 | 要否 | 内容 |
|-----|------|------|
| AnomalyCheck schtask 登録 | **要** | 管理者権限で `tools/register_anomaly_detector_schtask.bat` を実行 |
| register bat 文字化け修正 | 不要 | 問題なし |
| friday_weekend_scrape.bat PACI 追加 | 不要 | 既に存在 |
| WindowsApps stub 修正 | 不要 | 全タスク clean |
| friday_weekend_scrape.bat python パス明示化 | 推奨 | bare `python` → `%PYTHON_EXE%` (明示パス) に変更推奨 |
