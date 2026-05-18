# 完成-4: admin_verify_v2 (正しい existence check 規律確立)

**日付**: 2026-05-18
**status**: 完了
**scope**: ★ read-only ★ (schtasks /Create / /Delete 一切実行せず)

---

## 1. 背景: 5/18 17:00 false positive

### 事故概要

5/18 17:00 status_verify 実行時、 旧 bash one-liner grep が `schtasks /Query` 結果と
pattern 不一致を起こし、 **★ 9/9 schtask 全てを 「✅ 登録済」 と誤判定 ★**。

B-5 dry-run verify (5/22 admin 操作前 想定) で ★ 真値 0/9 登録 ★ が判明、
旧 method の信頼性ゼロが確定。

### root cause

旧 method:
```bash
schtasks /Query /TN <name> 2>&1 | grep -E "(ERROR|エラー|not exist)" || echo "登録済"
```

問題点:
1. **stdout / stderr 混在**: `2>&1` で stderr を stdout に merge してから grep するが、
   ja-JP locale (cp932) の error 文言 「エラー: 指定されたファイルが見つかりません。」 を
   そのまま grep にかけても **byte-level encoding mismatch** で no-match になる場合あり。
2. **case-sensitive な en-US pattern**: `"ERROR"` (大文字) は 「エラー」 (ja-JP) と byte 不一致。
3. **exit code 無視**: `schtasks /Query` の returncode (0 = 存在 / 1 = 不在) は捨てられ、
   stdout/stderr 文字列のみに頼っていた。
4. **pattern 漏れ**: 「指定された TN ... が存在しません」 「タスクが見つかりません」 等
   ja-JP の error 文言 variant は pattern に含まれず、 fallback `echo "登録済"` に着地。

実 stderr 出力 (5/18 19:11 admin_verify_v2 で確認):
```
エラー: 指定されたファイルが見つかりません。
```

→ 旧 pattern `["ERROR", "エラー", "not exist"]` のうち 「エラー」 は含まれているのに
grep が拾えなかった = encoding (cp932 / utf-8 / bash WSL の locale 不整合) の問題。

---

## 2. 正しい method (exit code 直接判定)

### 採用 method

```python
result = subprocess.run(
    ["schtasks", "/Query", "/FO", "LIST", "/TN", name],
    capture_output=True, text=True,
    encoding="cp932",  # Windows ja-JP の native encoding
    errors="replace",
    timeout=30,
)
registered = (result.returncode == 0)
```

**判定根拠**: `schtasks /Query` の returncode 仕様 (Microsoft 公式):
- `0` = 指定タスクが存在し Query 成功
- `1` = タスク不在 / Query 失敗

**文字列 pattern に依存しない** ため:
- locale (ja-JP / en-US / 他言語) 非依存
- encoding (cp932 / utf-8 / shift_jis) 非依存
- error 文言 variant に対しても robust

---

## 3. 実装

### file

- `tools/admin_verify_v2.py` (★ read-only ★)
- `tests/test_admin_verify_v2.py` (8 tests、 5 spec + 3 bonus)

### scope

9 schtask + 9 bat + 5 py module を全数 verify:

| # | schtask name                            | bat                                       | py module                                |
|---|-----------------------------------------|-------------------------------------------|------------------------------------------|
| 1 | Keiba-LiveOrchestrator-15min            | live_orchestrator.bat                     | tools/live_orchestrator_main.py          |
| 2 | Keiba-FeaturesIntegrity-Daily           | keiba_features_integrity.bat              | tools/features_integrity_monitor.py      |
| 3 | Keiba-AnomalyCheck-0630                 | keiba_anomaly_check_0630.bat              | tools/anomaly_auto_detector.py           |
| 4 | Keiba-AnomalyCheck-0830                 | keiba_anomaly_check_0830.bat              | tools/anomaly_auto_detector.py           |
| 5 | Keiba-AnomalyCheck-0940                 | keiba_anomaly_check_0940.bat              | tools/anomaly_auto_detector.py           |
| 6 | Keiba-AnomalyCheck-1410                 | keiba_anomaly_check_1410.bat              | tools/anomaly_auto_detector.py           |
| 7 | Keiba-AnomalyCheck-1700                 | keiba_anomaly_check_1700.bat              | tools/anomaly_auto_detector.py           |
| 8 | Keiba-CumulativeAudit-Daily             | daily_cumulative_audit.bat                | tools/daily_cumulative_audit.py          |
| 9 | Keiba-RaceNotifyLogV2Aggregator         | keiba_race_notify_log_v2_aggregator.bat   | tools/race_notify_log_v2_aggregator.py   |

### test 一覧 (8/8 PASS)

| # | test                                          | 内容                                                |
|---|-----------------------------------------------|-----------------------------------------------------|
| 1 | test_parse_schtasks_query_success             | mock /Query 成功時の next_run / status 抽出 確認    |
| 2 | test_parse_schtasks_query_not_exist           | exit code 1 -> registered=False 判定 確認           |
| 3 | test_bat_existence_check_real                 | 実 bat 存在 + size 取得 OK                          |
| 4 | test_bat_existence_check_missing              | 不在 bat -> exists=False                            |
| 5 | test_py_module_compile_ok                     | 実 py module compile PASS                           |
| 6 | test_py_module_missing                        | 不在 py -> compile_ok=False                         |
| 7 | test_5_18_17_00_false_positive_detected       | ★ 旧 grep 法 false positive vs 新 exit code 法 diff ★ |
| 8 | test_run_verify_full_shape                    | run_verify が 9/9 全 entry 処理 + summary OK        |

---

## 4. 5/18 19:11 実行結果 (真値)

```
============================================================
admin_verify_v2  (2026-05-18T19:11:08)
method: schtasks /Query /FO LIST /TN <name>, exit code direct judgment
============================================================
  schtasks registered : 0/9
  bats exist          : 9/9
  py compile ok       : 5/5
  ready for 5/22 admin: True
```

### diff vs 5/18 17:00 旧 status_verify

| 項目                    | 旧 status_verify (17:00) | admin_verify_v2 (19:11) |
|-------------------------|--------------------------|--------------------------|
| 判定 method             | bash grep pattern        | exit code 直接           |
| schtask registered      | **9/9 (false positive)** | **0/9 (真値)**           |
| bat existence           | (不明 / 未確認)          | 9/9                      |
| py module compile       | (不明 / 未確認)          | 5/5                      |
| 5/22 admin 操作 ready   | (信頼不能)               | True                     |

★ ★ ★ false positive 完全 検出。 真値は **9/9 全て未登録**。 ★ ★ ★

### 全 schtask 詳細

全 9 件 exit_code=1 / stderr_snippet=「エラー: 指定されたファイルが見つかりません。」

→ 全件 schtask 不在 = B-5 dry-run 結果 と完全一致。

### 全 bat 詳細

| bat                                              | size |
|--------------------------------------------------|------|
| live_orchestrator.bat                            | 618  |
| keiba_features_integrity.bat                     | 716  |
| keiba_anomaly_check_0630.bat                     | 720  |
| keiba_anomaly_check_0830.bat                     | 656  |
| keiba_anomaly_check_0940.bat                     | 672  |
| keiba_anomaly_check_1410.bat                     | 664  |
| keiba_anomaly_check_1700.bat                     | 667  |
| daily_cumulative_audit.bat                       | 269  |
| keiba_race_notify_log_v2_aggregator.bat          | 899  |

9/9 存在、 size >0、 全て完成-1 sub-task で作成済 (一部は既存)。

### 全 py module 詳細

5 unique modules、 全 5/5 py_compile PASS:
- tools/live_orchestrator_main.py
- tools/features_integrity_monitor.py
- tools/anomaly_auto_detector.py
- tools/daily_cumulative_audit.py
- tools/race_notify_log_v2_aggregator.py

---

## 5. 5/22 admin 操作前後の verify 手順

### 手順 (★ 必ず守ること ★)

#### before admin 操作 (5/22 朝、 想定)

```bash
cd C:\Users\takum\keiba-ai
python tools/admin_verify_v2.py --out data/admin_verify_v2_5_22_before.json
```

期待 output:
- `schtasks_registered`: **0/9**
- `bats_exist`: **9/9**
- `py_compile_ok`: **5/5**
- `ready_for_5_22_admin`: **True**

★ 1 つでも条件不一致 → admin 操作中止、 完成-1 / 完成-2 sub-task の再確認 ★

#### admin 操作 (5/22、 管理者権限 PowerShell で手動実行)

```
schtasks /Create /TN "Keiba-LiveOrchestrator-15min" /TR "C:\Users\takum\keiba-ai\tools\live_orchestrator.bat" /SC DAILY /ST 08:30 /RL HIGHEST /F
schtasks /Create /TN "Keiba-FeaturesIntegrity-Daily" /TR "C:\Users\takum\keiba-ai\tools\keiba_features_integrity.bat" /SC DAILY /ST 22:00 /RL HIGHEST /F
schtasks /Create /TN "Keiba-AnomalyCheck-0630" /TR "C:\Users\takum\keiba-ai\tools\keiba_anomaly_check_0630.bat" /SC DAILY /ST 06:30 /RL HIGHEST /F
schtasks /Create /TN "Keiba-AnomalyCheck-0830" /TR "C:\Users\takum\keiba-ai\tools\keiba_anomaly_check_0830.bat" /SC DAILY /ST 08:30 /RL HIGHEST /F
schtasks /Create /TN "Keiba-AnomalyCheck-0940" /TR "C:\Users\takum\keiba-ai\tools\keiba_anomaly_check_0940.bat" /SC DAILY /ST 09:40 /RL HIGHEST /F
schtasks /Create /TN "Keiba-AnomalyCheck-1410" /TR "C:\Users\takum\keiba-ai\tools\keiba_anomaly_check_1410.bat" /SC DAILY /ST 14:10 /RL HIGHEST /F
schtasks /Create /TN "Keiba-AnomalyCheck-1700" /TR "C:\Users\takum\keiba-ai\tools\keiba_anomaly_check_1700.bat" /SC DAILY /ST 17:00 /RL HIGHEST /F
schtasks /Create /TN "Keiba-CumulativeAudit-Daily" /TR "C:\Users\takum\keiba-ai\tools\daily_cumulative_audit.bat" /SC DAILY /ST 21:00 /RL HIGHEST /F
schtasks /Create /TN "Keiba-RaceNotifyLogV2Aggregator" /TR "C:\Users\takum\keiba-ai\tools\keiba_race_notify_log_v2_aggregator.bat" /SC DAILY /ST 20:30 /RL HIGHEST /F
```

★ 実 admin 操作は user 手動、 本 sub-task では一切自動化しない ★

#### after admin 操作 (5/22 朝、 admin 完了直後)

```bash
python tools/admin_verify_v2.py --out data/admin_verify_v2_5_22_after.json
```

期待 output:
- `schtasks_registered`: **9/9** (★ 全件 registered=True / exit_code=0 / next_run あり ★)
- `bats_exist`: 9/9
- `py_compile_ok`: 5/5

★ 9/9 未満 → 即 user に報告、 不足 schtask を再 /Create ★

#### 差分検証 (before vs after)

```bash
diff data/admin_verify_v2_5_22_before.json data/admin_verify_v2_5_22_after.json
```

期待 diff: 全 9 schtask が `"registered": false` -> `"registered": true` に flip。
他の field (bats / py_modules) は不変。

---

## 6. 規律: false positive 二度禁止

### 旧 method 全面廃止

以下の bash one-liner 系 schtask check は ★ 二度と使わない ★:

```bash
# ★ 禁止 ★
schtasks /Query /TN <name> 2>&1 | grep -E "(ERROR|エラー|not exist)" || echo "登録済"

# ★ 禁止 ★
schtasks /Query | findstr <name>  # 大量出力 + 部分一致 false positive 多発
```

### 新 method 一本化

schtask existence check は ★ 必ず ★ `tools/admin_verify_v2.py` 経由:

```bash
python tools/admin_verify_v2.py --quiet --out /tmp/verify.json
python -c "import json; r=json.load(open('/tmp/verify.json')); print(r['summary'])"
```

### 教訓

- ★ exit code 直接判定 > 文字列 pattern match ★ (i18n / locale / encoding に robust)
- ★ test で false positive を再現 ★ (test 5 = 旧 vs 新 diff 検出)
- ★ read-only tool は ★ 必ず read-only ★ ★ (subprocess /Create / /Delete 入れない)

---

## 7. 完成-4 完了

- [x] tools/admin_verify_v2.py 新規作成 (★ read-only ★)
- [x] tests/test_admin_verify_v2.py 新規作成 (8 tests、 8/8 PASS)
- [x] data/admin_verify_v2_5_18_1830.json 実行結果保存
- [x] docs/完成-4_ADMIN_VERIFY_V2_2026_05_18.md (本 file) 作成
- [x] 5/18 17:00 false positive root cause 詳細記録
- [x] 正しい method (exit code 直接判定) 確立
- [x] 5/22 admin 操作前後の verify 手順 文書化

**verdict**: ★ 完成-4 完了、 admin_verify_v2 ready、 schtask 0/9、 bat 9/9、 test 8/8 PASS ★
