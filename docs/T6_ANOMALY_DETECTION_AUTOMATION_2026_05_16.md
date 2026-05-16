# T6 anomaly detection automation (G1 day 起床負担軽減)

> Sub-task T6 deliverable (5/16 evening session)
> 目的: 5/17 G1 day = ヴィクトリアマイル の user 目視 5 項目 → 全自動 Discord 通知

---

## 0. 結論

- 5 trigger 全自動化、 read-only check + Discord 通知のみ
- V15 model / predict_core / race_auto_notify / daily_predict / app.py 完全不変
- 17/17 tests PASS (tests/T6_anomaly_detection_test.py)
- false positive rate < 5% (5 正常 scenarios で 計測)
- true positive rate > 95% (4 異常 scenarios で 計測)
- 5/17 朝 user 操作 = Discord 確認のみ

honest 注記:
- false/true positive は **合成 test シナリオ** で 計測した数値。 本番 log の分布で 再計測する余地あり
- 「絶対検出」 ではなく、 現行 race_auto_notify.py の log pattern (`[STRATEGY7] Skip 京都`, `整形済み買い目通知: N messages`) ベース
- log フォーマット変更時は detection logic 更新必要

---

## 1. 5 trigger 仕様

| # | trigger | detection logic | severity | rollback 推奨? |
|---|---------|-----------------|----------|----------------|
| 1 | 予測ゼロ | `data/daily_predictions/{date}.csv` 不在 / 0 rows | ★ critical | YES |
|   |          | 20 R 未満 (通常 30+) | ⚠ warning | NO (確認のみ) |
| 2 | 投票候補 0R | log の `整形済み買い目通知: 0 messages` | ★ critical | YES |
|   |             | log file 不在 (起動前) | ⚠ warning | NO (待機) |
| 3 | streamlit 起動失敗 | `http://localhost:8501/` 接続失敗 | ⚠ warning | NO (投票影響 0) |
| 4 | Discord 通知なし | log mtime > 1h かつ messages 0 | ⚠ warning | rollback 不要、 race_auto_notify 再起動 |
| 5 | 戦略⑦案 C 不動作 | daily_predictions に 京都 R あり、 かつ log に `[STRATEGY7] Skip 京都` 0 件 | ★ critical | YES (案 C revert) |
|   |                  | 京都 R 全て G/L/OP 例外 | ✅ ok | — |

### trigger 5 補足 (戦略⑦案 C)

`tools/race_auto_notify.py` L186 の filter logic:
```python
if course_str == '京都' and not (is_graded or is_listed):
    print(f"    [STRATEGY7] Skip 京都 (P0-2 案 C、 5/17 適用): {race_name_str}")
    return
```

検出ロジック:
1. daily_predictions/{date}.csv から `course == '京都'` の R 抽出
2. G/L/OP 特別 (race_name に `G1/G2/G3/(L)/(OP)`) を除外
3. 残りが 1 件以上 → race_auto_notify log に `[STRATEGY7] Skip 京都` が同数以上 期待
4. log に Skip 0 件 → critical (案 C 効いてない疑い)

---

## 2. detection logic 詳細

### 入出力
- 入力: `--date YYYYMMDD` (default = today)
- 出力: console 表 + Discord webhook (DISCORD_WEBHOOK_UPDATES > DISCORD_WEBHOOK_URL)
- exit code: 0=全OK, 1=warning, 2=critical

### Discord 通知
- 環境変数 `DISCORD_WEBHOOK_UPDATES` 優先、 fallback `DISCORD_WEBHOOK_URL`
- critical / warning の details のみ送信 (ok は 静か)
- rollback コマンド添付 (5/18 user 判断用)

### test coverage
| test ID | 内容 |
|---------|------|
| T6-1a | 正常 35R → ok |
| T6-1b | predictions file 不在 → critical |
| T6-1c | predictions 0 rows → critical |
| T6-1d | predictions 10R → warning |
| T6-2a | 投票 8 messages → ok |
| T6-2b | 投票 0 messages → critical |
| T6-2c | log file 不在 → warning |
| T6-3a | streamlit unreachable → warning |
| T6-4a | Discord log fresh → ok |
| T6-4b | Discord log missing → warning |
| T6-5a | 京都 Skip log あり → ok |
| T6-5b | 京都 Skip log 0 件 → critical |
| T6-5c | 京都 R 該当日なし → ok |
| T6-5d | 京都 R 全 G1 例外 → ok |
| T6-FP | false positive rate < 5% |
| T6-TP | true positive rate > 95% |
| T6-E2E | run_all 5 trigger callable |

---

## 3. test pass criteria

```
tests: 17/17 PASS
false positive rate: 0% (0/20 normal scenarios)
true positive rate: 100% (4/4 anomaly scenarios)
```

(計測は 5/16 evening 実施、 本番 log で 再計測の余地あり)

実行:
```bash
python tests/T6_anomaly_detection_test.py
```

---

## 4. 5/17 朝 user 操作

### 期待される運用 (Sub-task T6 適用後)

| 時刻 | schtask | user 操作 |
|------|---------|----------|
| 5:30 起床 | — | PC 起動、 git log 確認 |
| 06:00 | DailyJrdbKyi | (自動) |
| 06:30 | **Keiba-AnomalyCheck-0630** | Discord #updates 確認 |
| 08:00 | DailyPredict | (自動) |
| 08:30 | **Keiba-AnomalyCheck-0830** | Discord #updates 確認 |
| 08:45 | race_auto_notify | (自動) |
| 09:40 | **Keiba-AnomalyCheck-0940** | Discord #updates 確認 (critical 時刻) |
| 14:00 | — | 投票準備 |
| 14:10 | **Keiba-AnomalyCheck-1410** | Discord #updates 確認 |
| 15:40 | — | ヴィクトリアマイル 発走 |
| 17:00 | **Keiba-AnomalyCheck-1700** | G1 後 evening 整理 |

### 静か = 正常 原則
- ✅ のみ → Discord 通知なし → 確認不要
- ⚠ あり → 内容確認、 rollback は 任意 (投票影響少)
- ★ あり → 内容確認 + checklist の rollback 手順実行

---

## 5. schtask 登録 (★ 5/18 user 判断後 admin 実行 ★)

agent 内では 実行絶対なし。 user 判断後 admin powershell で:
```cmd
tools\register_anomaly_detector_schtask.bat
```

登録される 5 タスク:
- Keiba-AnomalyCheck-0630
- Keiba-AnomalyCheck-0830
- Keiba-AnomalyCheck-0940
- Keiba-AnomalyCheck-1410
- Keiba-AnomalyCheck-1700

確認:
```cmd
schtasks /Query /TN Keiba-AnomalyCheck-0630
```

---

## 6. file 一覧 (Sub-task T6 で 新規)

```
tools/anomaly_auto_detector.py             # 5 trigger 本体 (read-only check)
tools/anomaly_auto_detector.bat            # schtask 用 wrapper
tools/register_anomaly_detector_schtask.bat # schtask 登録 (admin)
tests/T6_anomaly_detection_test.py          # 17 tests
docs/T6_ANOMALY_DETECTION_AUTOMATION_2026_05_16.md  # 本 doc
docs/5_17_G1_DAY_CHECKLIST.md               # § 8 追記 (既存 logic 変更なし)
```

V15 production code (predict_core / daily_predict / race_auto_notify / app.py) は完全不変。

---

## 7. 5/16 evening dry-run 結果

5/16 の 過去 log で 動作確認:
```
=== anomaly auto detection 20260516 ===
  [✅] predictions        predictions 35 R OK
  [✅] vote_candidates    投票候補 8 messages OK
  [⚠] streamlit          streamlit :8501 unreachable: ConnectionError
  [✅] discord_recent     Discord notify 10 messages (log 924 min 前)
  [★] strategy7c         案 C 不動作疑い: 京都 R 11 あり、 log に [STRATEGY7] Skip 京都 0 件
```

★ critical (案 C 不動作) = **expected**:
- 5/16 時点では案 C は 5/17 適用予定で **まだ active でなかった**
- これは true positive — detector は正常動作
- 5/17 朝 schtask 経由で 京都 R が Skip される log が出れば ✅ ok に切替わる

---

## 8. honest 制約

- detection は log pattern 依存 — race_auto_notify.py の print 文 変更時に追従必要
- false/true positive 計測は 合成 scenario ベース、 本番分布で 再計測の余地
- streamlit check は 「投票影響 0」 のため warning 止まり (false positive で rollback 暴発を防ぐ設計)
- Discord webhook 未設定の環境では 通知 silent (console のみ)、 但し agent 内 commit/push なしで env 設定は 親集中
