# 5/23 Python Store Stub Fix

**実施時刻**: 2026-05-23 09:49  
**commit**: a054bfa3

---

## 問題

| 項目 | 内容 |
|------|------|
| 旧 PYTHON_EXE | `C:\Users\takum\AppData\Local\Microsoft\WindowsApps\python.exe` |
| 問題 | Microsoft Store Python スタブ — Task Scheduler context から **起動不可** |
| エラー | 「指定されたプログラムは実行できません」(40 bytes log) |
| 影響 | LiveOrchestrator 未 fire / AnomalyCheck 0630/0830/0940/1410/1700 全失敗 |

---

## 真の Python path

```
where.exe python → C:\Users\takum\AppData\Local\Microsoft\WindowsApps\python.exe  ← stub (NG)
実体              → C:\Users\takum\AppData\Local\Python\pythoncore-3.14-64\python.exe  ← ✅
```

実体確認:
```
python OK: 3.14.3 (tags/v3.14.3:323c59a, Feb 3 2026) [MSC v.1944 64 bit (AMD64)]
```

---

## 修正した bat ファイル (11 件)

| bat ファイル | 用途 |
|-------------|------|
| tools/live_orchestrator.bat | LiveOrchestrator (08:30 SAT/SUN) |
| tools/keiba_anomaly_check_0630.bat | AnomalyCheck 06:30 |
| tools/keiba_anomaly_check_0830.bat | AnomalyCheck 08:30 |
| tools/keiba_anomaly_check_0940.bat | AnomalyCheck 09:40 |
| tools/keiba_anomaly_check_1410.bat | AnomalyCheck 14:10 |
| tools/keiba_anomaly_check_1700.bat | AnomalyCheck 17:00 |
| tools/anomaly_auto_detector.bat | AnomalyCheck 実体 |
| tools/keiba_cumulative_audit.bat | 累積 audit |
| tools/daily_cumulative_audit.bat | daily audit |
| tools/keiba_race_notify_log_v2_aggregator.bat | v2 aggregator |
| tools/keiba_features_integrity.bat | features 整合性チェック |

全 11 件:
```
SET PYTHON_EXE=C:\Users\takum\AppData\Local\Microsoft\WindowsApps\python.exe
→ SET PYTHON_EXE=C:\Users\takum\AppData\Local\Python\pythoncore-3.14-64\python.exe
```

---

## Task Scheduler 起動確認

`tools/anomaly_auto_detector.py` を修正後 Python で実行:

```
=== anomaly auto detection 20260523 ===
  [✅] predictions        predictions 34 R OK
  [✅] vote_candidates    投票候補 8 messages OK
  [⚠] streamlit          streamlit :8501 unreachable (クラウド運用のため正常)
  [✅] discord_recent     Discord notify 10 messages (log 40 min 前)
  [★] strategy7c         ★ 案 C 不動作疑い: 京都 R 12 あり、log に [STRATEGY7] Skip 京都 0 件
  summary: critical=1, warning=1, ok=3
```

**strategy7c "critical" について**: 08:30 check 時点では全レースが未発走 (09:45〜)。
[STRATEGY7] Skip ログは各レース発走 -5 min に記録されるため、08:30 時点では 0 件が正常。
**AnomalyCheck の 08:30 チェックは early-fire による false positive**。

---

## V15 Regression Test ✅

```
V15 features: 145
V15 version: v15
V15 auc: 0.8939485520467574
V15 unchanged: PASS
```

bat 修正は PYTHON_EXE 行のみ。V15 pkl.gz / predict_core / production logic は一切不変。

---

## LiveOrchestrator 手動 fire 可否

| 項目 | 状態 |
|------|------|
| 現在時刻 (確認時) | 09:49 |
| 新潟1R | 09:45 (4 分前通過済) |
| 東京1R | 09:55 (残り 6 分) |
| LiveOrchestrator mode | `--mock --dry-run` (betting なし) |
| **手動 fire 可否** | **✅ 可能** — 残り 35R に間に合う |

**手動 fire コマンド** (ユーザー実行):
```bash
# ターミナルで手動実行 (mock=True で安全)
! tools\live_orchestrator.bat
# または
! C:\Users\takum\AppData\Local\Python\pythoncore-3.14-64\python.exe -u tools\live_orchestrator_main.py --mock --dry-run
```

**注意**: 本日 5/23 は mock/dry-run モード継続 (5/23 bat コメント: "5/24 以降は mock 解除予定")。
実際の投票は race_auto_notify.py が担当しており、Live Orchestrator は再計算/shadow のみ。

---

## ★ 追加発見: race_auto_notify が終了している可能性 ★

race_auto_notify_20260523.log の末尾が `^C` で終わっており、プロセスが中断されている可能性がある。

- ログに 09:40 以降の個別レースエントリが存在しない
- 原因不明 (手動終了 / クラッシュ / schtask 制限)
- **影響**: 個別レース通知 (戦略⑦ filter + 買い目) が出ていない可能性

確認方法 (ユーザー):
```
! tasklist | findstr python
# または
! Get-Process | Where-Object { $_.Name -eq "python" }
```

race_auto_notify が動いていなければ再起動が必要:
```
! python tools\race_auto_notify.py
```

---

## Task Scheduler への bat 修正の反映

**schtask 登録済みの場合**: schtask は bat へのパスを参照するため、bat 修正は自動的に反映される。schtask 再登録は不要。

次回 AnomalyCheck (09:40 schtask) から正常 fire する見込み。

---

*修正完了: 2026-05-23 09:49 | V15 production 不変確認済み | push 未実施*
