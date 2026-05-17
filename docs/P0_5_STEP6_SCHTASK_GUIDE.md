# P0-5 schtask 登録 guide (★ 5/18+ user 判断後 admin 実行 ★)

> 作成日: 2026-05-17 (Session #87)
> 関連 design: [P0_5_SCHTASK_MONITOR_DESIGN_2026_05_17.md](P0_5_SCHTASK_MONITOR_DESIGN_2026_05_17.md) (commit 2646bf9b)
> ★ 本 doc は登録 guide のみ。 実 schtasks /create は 5/18+ user 判断後 admin 実行 ★

---

## 0. 前提

- 5/17 21:00+ 順 1-6 全て完了 (新規 file 作成、 schtask 登録なし)
- 5/18 朝までに `tools/live_orchestrator.bat` 動作確認 (今は順 1 で register/unregister script のみ完成、 live_orchestrator.bat 自体は順 2+ で作成)
- 5/18 朝 8:00 DailyPredict 完了後 `daily_predictions/20260518.csv` 確認

---

## 1. 登録 timing 案

### 案 A: 5/18 朝 (土曜、 weekend before run) ★ 推奨 ★
- 5/18 朝 06:00 起床、 admin で実行
- 8:30 自動 fire、 当日 race に対して live fetch + recalc + #updates 通知開始
- 24R 蓄積開始 (5/18 sat、 5/24 sat、 5/25 sun、 5/31 sat、 6/1 sun、 6/7 sat、 ...)
- 6/17 採用判定

### 案 B: 5/18 夜 (weekend after run)
- 5/18 G1 day 結果回収後 (20:00 以降)
- 5/24 朝 8:30 から fire 開始
- 6/17 採用判定

★ 推奨: 案 A ★ (5/18 が weekend なので、 即 live data 取得開始)

---

## 2. 登録手順

```powershell
# 管理者として powershell 起動
cd C:\Users\takum\keiba-ai
.\tools\p0_5_schtask_register.bat
```

期待出力:
```
[OK] Keiba-LiveOrchestrator-15min 登録成功
TaskName: Keiba-LiveOrchestrator-15min
Schedule: WEEKLY (SAT,SUN)
Start: 08:30
```

---

## 3. 動作確認手順

### 5/18 朝 08:30 (admin 登録後初回 fire)
- `logs/live_orchestrator.log` で 起動確認
- race -20 / -15 / -10 min の fetch / recalc / 通知 log 確認
- `data/live_data/20260518/` / `data/recalc_15min/20260518/` で 出力 file 確認
- Discord #updates channel で 順位変動通知あれば確認

### 補助 schtasks query (read-only)
```powershell
schtasks /Query /TN "Keiba-LiveOrchestrator-15min" /V /FO LIST
```

---

## 4. 異常時 rollback コマンド

```powershell
# 管理者として
.\tools\p0_5_schtask_unregister.bat
```

★ 1 行で 完全 rollback、 既存 V15 + 戦略⑦案 C はそのまま動作 ★

---

## 5. 既存 schtasks との conflict check

5/17 21:00 query で確認した既存 Keiba schtasks 一覧 (主要なもののみ):

| schtask | 時刻 | 内容 |
|---------|------|------|
| `\keiba-ai\DailyPredict` | 08:00 | 当日全レース予測 (V15) |
| `\keiba-ai\DailyPremiumScrape` | 03:00 | netkeiba Premium 事前取得 |
| `\keiba-ai\RaceAutoNotify_Sat/Sun` | (race time -5min) | 戦略⑦案 C 自動通知 |
| `\keiba-ai\DailyResultsEvening` | 20:00 | 結果照合・ROI 計算 |
| `\keiba-ai\DailyJrdbKyi` | 06:00 | JRDB 全種別 DL |
| `\keiba-ai\JrdbHealthCheck_Sat/Sun` | 07:30 | JRDB 取得健全性 check |
| `\Keiba-NightlySanity` | 23:00 | 翌日 task 事前 check |
| `\Keiba-MorningWeightCheck_Sat/Sun` | 09:30 | 馬体重補正機構 |
| `\Keiba-MultiStagePredict_Race11/12_*` | (固定時刻) | multi-stage 予測 |

★ 新規 `Keiba-LiveOrchestrator-15min` 08:30 fire は ★
- 8:00 DailyPredict 完了後 30 min (★ 完全独立、 conflict なし ★)
- 9:30 MorningWeightCheck の前 (★ conflict なし ★)
- internal polling loop で 1 schtask のみ、 同時 fire 想定なし
- V15 production schtasks 全 unchanged

---

## 6. 採用判定 (6/17 Wed)

- 5/18-6/16 で SAT/SUN ~30R 蓄積
- paper shadow ROI vs V15 production ROI 比較
- 統計的有意性 (Welch's t-test、 p<0.05)
- 採用判定 5 項目:
  1. V15 AUC 維持 (production 完全不変)
  2. paper shadow ROI vs V15 production ROI 改善
  3. LEAK PASS (calibrator_overlay は live data のみ、 post-race feature なし)
  4. LIVE 安定 (schtask 連続 fire 失敗 < 5%)
  5. 統計有意 (Welch's t-test p<0.05)

→ GO → 6/18+ production 投入候補 (calibrator_overlay を strategy_layer に統合検討)
→ NO-GO → 5/18+ paper shadow eval 継続、 24R 追加蓄積後 再判定 (7/15 まで)

---

## 7. V15 production 不変保証 ✅

- predict_core / daily_predict / race_auto_notify / app.py / .pkl.gz 全 unchanged
- 既存 8+ schtasks 不変
- cumulative_results.csv read のみ
- Discord #買い目 channel 完全不変
- live_orchestrator は #updates channel のみ、 production 投票判断不変

---

## 8. 関連 file

| file | 用途 |
|------|------|
| `tools/p0_5_schtask_register.bat` | admin 登録 script (★ 5/18+ 実行 ★) |
| `tools/p0_5_schtask_unregister.bat` | admin rollback script |
| `tools/live_orchestrator.bat` | schtask 実行 entry (★ 順 2+ で作成 ★) |
| `tools/live_orchestrator.py` | internal polling loop (★ 順 2+ で作成 ★) |
| `docs/P0_5_SCHTASK_MONITOR_DESIGN_2026_05_17.md` | 設計 doc (commit 2646bf9b) |
| `docs/P0_5_RECALC_LOGIC_DESIGN_2026_05_17.md` | recalc 設計 doc |
| `docs/P0_5_DATA_SOURCE_AUDIT_2026_05_17.md` | data source audit |
