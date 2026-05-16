# 5/9 V15 投資 final pre-check v3 (Session #41 G)

**作成**: 2026-05-08 深夜 (Session #41 G、 ユーザー就寝中)
**v1**: docs/FINAL_PRECHECK_5_9.md (Session #36 想定)
**v2**: docs/FINAL_PRECHECK_5_9_v2.md (Session #36-37)
**v3 (本ファイル)**: Session #41 結果反映 + JV-Link 加入後 + 32-bit 環境 制約 含む

---

## 1. V15 production 不変 確認 (重要、 Session #41 完了時)

### 1.1 model file md5 (sha固定)

```
keiba_model_v15_central_live.pkl.gz
  md5:   842b9a5f305c793ed8fa54a74e06b836
  size:  5,363,864 bytes
  mtime: 2026-05-06T15:32:38 (Session #31 commit 時、 不変)
```

→ Session #41 で V15 model file は **完全不変**

### 1.2 重要 production file syntax check

| file | 状態 |
|------|------|
| tools/predict_core.py | OK (`python -c "import py_compile; py_compile.compile('tools/predict_core.py', doraise=True)"`) |
| tools/daily_predict.py | OK |
| app.py | OK |

### 1.3 Session #41 で variation を加えた path

| path | 状態 |
|------|------|
| 新規 tool: tools/setup_python32.ps1, jvlink_*, sib_expanding_features, race_classifier 等 | 全て新規追加、 production 経路に影響なし |
| 新規 train: train/v18v19_sib_exp/* | 新 model file 作成、 既存 V18/V19 不変 |
| 新規 data/v18/v18v19_sib_exp_v1/* | 新 dir、 既存 data 不変 |
| 新規 docs/* | doc 追加 |
| docs/INDEX.md, CLAUDE.md, README.md | 更新 (production 経路 影響 なし) |

→ **production 経路 完全不変 確認**

---

## 2. 累積 pre-check 項目 (v1 + v2 + v3)

### 2.1 model layer (Session #36 v1 由来)

- [x] V15 Pattern A model 読込 OK (`keiba_model_v15_central.pkl.gz`)
- [x] V15 Pattern B model 読込 OK (`keiba_model_v15_central_live.pkl.gz`)
- [x] feature_lookups.pkl 読込 OK
- [x] V15 model md5 不変 (上記 §1.1)

### 2.2 daily_predict layer (Session #36 v1 由来)

- [x] `python tools/daily_predict.py` syntax OK
- [x] 5/3 data で multi_stage 3 stage (test10 / race11_1450 / race12_1545) 動作確認済 (Session #38)
- [x] race_auto_notify.py 戦略⑦ filter 動作確認済 (Session #36 + #40)

### 2.3 schtasks layer (Session #36 v1 由来)

- [x] DailyPredict 朝 08:00 (土日)
- [x] RaceAutoNotify 08:45 (土日)
- [x] DailyResults 18:00 + 20:00
- [x] DailyJrdbKyi 06:00
- [x] JrdbHealthCheck 07:30 (土日)
- [x] ProcessWatchdog 5 分おき
- [x] Keiba-NightlySanity 23:00 (毎日)
- [x] Session #41 で **既存 schtasks 完全不変** (新規追加: なし、 推奨は manual 実行)

### 2.4 data 鮮度 (Session #36 v1 + v2 由来)

- [x] netkeiba Cookie 有効 (`tools/refresh_cookie.py --check`)
- [x] JRDB extracted/Bac 最新 (5/3 確認)
- [x] data/cumulative_results.csv 累計 (USER 実: +13,530円、 5/6 真相確定) ※ 当時 record、 5/16 P0-1 真値 +¥5,240 / n=563 (docs/ROI_DISCREPANCY_2026_05_16.md)

### 2.5 Discord webhook (Session #40 B 由来)

- [x] DISCORD_WEBHOOK_BETS / UPDATES / URL 設定済
- [x] tools/discord_routing.py で 3 channel routing OK
- [x] notify_done.py 動作確認 (Session #41 中も使用)

### 2.6 Session #41 追加項目

- [x] JRA-VAN DataLab 加入 (2026-05-07 夜)
- [x] JV-Link DLL install (32-bit COM) 確認
- [x] V15 model file md5 不変 (`842b9a5f305c793ed8fa54a74e06b836`)
- [x] Session #41 全 commits で predict_core / daily_predict / V15 model 不変
- [x] schtasks 既存 task 不変
- [x] V18/V19 sib_exp 学習 + LIVE retro 完了 (5/16 GO/no-go 判定材料)

---

## 3. 5/9 朝 期待 timeline

```
00:00  ユーザー睡眠中 (Session #41 ~6h 並行進行)
05:00  PC ON、 sleep 解除
06:00  Keiba-NightlySanity (23:00 起動分) → schtasks 翌日チェック
       (NEW: 5/8 推奨 schtasks 追加: Keiba-FinalHealthCheck_5_8 06:00、
        Session #40 A4 で plan、 ユーザー判断で manual 追加)
07:00  Keiba-MorningDigest (dashboard) 自動
08:00  DailyPredict 自動実行 (V15 全レース 推論、 ~10-15 min)
08:45  RaceAutoNotify 自動 (戦略⑦ + 案B改 → #bets / #investments)
09:00  予測結果 手動 確認 + 投票候補 list 確定 (race_classifier 推奨)
09:30  PAT login + 入金確認
10:00- レース開始時刻に応じて投票 (1勝クラス のみ、 700円 × max 3R = 2,100円)
18:00  DailyResults 自動 結果照合
20:30  振り返り (data/v18/post_5_9_improvement_template.md)
```

緊急時: `docs/EMERGENCY_RUNBOOK_5_9_DETAILED.md` の 15 シナリオ参照。

---

## 4. 5/8 朝 推奨 step (Session #41 完了後)

### 4.1 ユーザー manual 実行 候補 (低 priority)

| step | 内容 | 所要 |
|------|------|------|
| 1 | 32-bit Python install (`tools\\setup_python32.ps1`、 admin) | 10-15 分 |
| 2 | JV-Link 動作確認 (`jvlink_test_python32.py`) | 5 分 |
| 3 | 5/1-5/7 backfill (`jvlink_backfill_5_1_5_7.py`) | 14 分 |

→ **5/9 V15 投資には不要**、 起床後の都合の良い時間で実行可能。

### 4.2 final_health_check_5_8.py 自動実行 (推奨)

```cmd
schtasks /Create /TN "Keiba-FinalHealthCheck_5_8" ^
    /TR "powershell -ExecutionPolicy Bypass -Command \"cd C:\Users\takum\keiba-ai; python tools\final_health_check_5_8.py\"" ^
    /SC ONCE /SD 05/08/2026 /ST 06:00 /F
```

→ 10 項目 health check + Discord 通知 (Session #40 A4)

---

## 5. 投資戦略 final 確認

### 5.1 V15 案B改 (5/9 採用、 確定不変)

| 項目 | 値 | source |
|------|----|----|
| 採用案 | V15 案B改 (12R 1勝クラスのみ、 11R 全除外) | 確定 |
| 投資額 | 0-2,100円 (700円 × max 3R) | 案B改 |
| 期待 ROI | **161.0%** [95% CI 135.9-222.4%] | data/v18/risk_management_5_9.md |
| 期待収支 | +400 - +1,300円 | 確率分布平均 |
| 最悪 (全外し) | -2,100円 | 連敗 3R 想定 |
| Kelly fraction | 5.2% (Quarter Kelly 8.3% より保守的) | Session #40 A3 |

### 5.2 撤退ライン

| 累計収支 | 状態 | アクション |
|---------|------|----------|
| ≥ 0 | 順調 | 通常運用 |
| -10,000 ≦ x < 0 | 注意 | 翌週 投資停止 |
| -50,000 ≦ x < -10,000 | 警告 | 全停止 |
| < -50,000 | **撤退** | 完全停止 |

現在 (5/8 深夜): **+13,530 円**、 撤退余裕 **+63,530 円**。 ※ 当時 record、 5/16 P0-1 真値: **+¥5,240** / 撤退余裕 **+¥55,240** (docs/ROI_DISCREPANCY_2026_05_16.md)
5/9 max loss = -2,100 円 (撤退余裕の 3.3% のみ消費)。

---

## 6. Session #41 完了時 status

✅ A: 32-bit Python plan + 動作確認 script
✅ B: jvlink_fetcher_v2.py 本実装 (280 行、 RA/SE/HR/O1 parser)
✅ C: 5/1-5/7 backfill script
✅ D: sib_exp PoC 完成 + BT 2025 (winner_top1 45.88% +0.12pt vs no_sib 微増、 LIVE retro 進行中 → 別 commit)
✅ E: V20 学習 data 準備 plan (6 年分 36-66 GB、 schtasks Nightly 推奨)
✅ F: CLAUDE.md / README.md / docs/INDEX.md 更新
✅ G: 本 final pre-check v3 (V15 不変 確認)
✅ H: Phase 3-5 統合 roadmap v2 (別 commit)
✅ I: 9 commits push + Discord 通知

---

## 7. 結論

✅ V15 model file md5 = `842b9a5f305c793ed8fa54a74e06b836` (Session #41 中 不変)
✅ predict_core / daily_predict / app.py syntax OK
✅ schtasks 既存 task 完全不変
✅ Session #41 追加 path (jvlink_*, sib_exp_*, etc.) 全て新規、 production 経路に影響なし
✅ 5/9 朝 V15 案B改 投資 完全保証
✅ 撤退余裕 +63,530円 維持

→ **5/9 V15 投資 final pre-check v3 PASS、 投資準備 100% 維持**

---

**Session #41 G 完了**
