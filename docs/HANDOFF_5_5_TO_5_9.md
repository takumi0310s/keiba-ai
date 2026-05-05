# 引き継ぎ書 v2: 5/5 PM → 5/9 (土) 本番 + Phase 2.5+

**作成**: 2026-05-05 17:35 (Session #15)
**ベース commit**: 2b6dc4eb (5/9 本番最終調整)
**ユーザー方針** (絶対):
- 取り返し禁止
- 累計損失拡大NG (現在 +14,140円)
- 撤退ライン -50,000円

---

## 0. v1 と違う点 (絶対 再生産しない)

詳細: `docs/handoff_v1_v2_diff.md`。要点:

| v1 (誤) | v2 (正) | source |
|--------|--------|--------|
| training_times 2025 = 2,551 件 | **192,296** | `data/training_times.csv` |
| 5/2 USER 損失 -23,800円 | **-8,820円** | USER 報告 (実投資 ベース) |
| v15 batch ROI 31.3% (USER ROI と誤解) | **全 R 仮想 ROI**、USER 案B改 ROI は 161% (BT) | `data/cumulative_results.csv` + healthy 4日分析 |
| TYB 17:00 確実公開 | **不明** (5/4 12:25 / 5/9 12:25 共に 404、観測継続中) | `data/tyb_publish_log.csv` |
| NAR モデル AUC 0.789 | **0.8145** (v4 復活、OOS 0.8519) | `data/nar/models/keiba_model_nar_v4.pkl` |
| 累計 約 -25,000円 | **+14,140円** | USER 報告 (5/5 朝) |

**重要原則**: 数字は **必ず生データで再検証**。引き継ぎ書 v1 の数字を session 越しに transfusion しない。

---

## 1. 累計収支 + 撤退ライン

| 項目 | 値 | source |
|------|----|--------|
| **5/5 朝 累計** | **+14,140円** | USER 報告 (`data/results/20260505_kashiwa_kinen.md` 内) |
| 撤退ライン (絶対) | -50,000円 | 全 session 一貫 |
| **撤退まで余裕** | **+64,140円** | 50,000 + 14,140 |
| 5/9 最悪 | -2,100円 → 累計 +12,040円 | 案B改 上限 |
| 5/9-5/10 最悪 | -4,200円 → 累計 +9,940円 | 両日上限 |

**撤退判定基準** (詳細 `data/v18/risk_management_5_9.md`):
- 5/9 単日 ROI < 50% → 5/10 投資停止
- 5/9-5/10 累計 -10,000円 → 翌週投資停止
- 累計 -50,000円 → 完全撤退

---

## 2. 現行モデル構成

### 2.1 V15 (本番、JRA 案B改 5/9 以降)

| 項目 | 値 |
|------|----|
| ファイル | `keiba_model_v15_central_live.pkl.gz` (2,050 KB) |
| version | v15_live |
| AUC | **0.8939** (本番)、0.8858 (4-model ensemble) |
| features | **150** (Pattern B、150 列) |
| 訓練データ | 527,280行 (2015/01-2025/12) |
| load 時間 | 1.1 秒 |
| 役割 | **JRA 案B改 主モデル**、5/9-5/10 12R 1勝クラス予測 |

### 2.2 V17 (実質 morning 同等)

- v17 morning: 11R/12R 用 (`predict_v17_morning_pipeline.py`)
- v17 ULTRA-CLEAN: midday 想定だったが TYB 公開時刻不明で 404 → **5/9 では使わない**
- TYB 観測完了後 (5/11 月) 再判断

### 2.3 v18 (単勝) / v19 (複勝)

| 項目 | 値 |
|------|----|
| ファイル | `data/v18/models/v18_tansho_lgb.txt` + `_xgb.json` (CRLF 復旧済 S5) |
| AUC | v18 0.85+ (BT)、retro distribution shift で 全 bet=0 |
| 状態 | **5/16 試行候補** (条件付き、`data/v18/v18_v19_integration_plan_5_4_pm.md`) |
| 必須前提 | race-level normalize (softmax T=1.0) を本番統合 |
| Platt scaling | 試作済 (S8)、max prob 0.154→0.213 (不十分) |
| race-level normalize | 試作済 (S10)、softmax T=1.0 で sum=1 強制 → bet>0 化 |

### 2.4 NAR v4 (復活 → 体系化)

| 項目 | 値 |
|------|----|
| ファイル | `data/nar/models/keiba_model_nar_v4.pkl` (167 KB) |
| AUC | **0.8145** (reported), **0.8519** (OOS 2025) |
| features | 22 (Pattern B、odds_log + pop_rank dominant) |
| 学習データ | 4,821 races / 49,213 rows (NAR 2020-2024) |
| 騎手 stats | 315 人 |
| 使い方 | `tools/predict_nar.py --shutuba-csv ...` (柏記念で 0.777 完全再現確認) |
| 役割 | **5/12 paper → 5/16 試行 500円/日** |
| 既知の限界 | chihou_races_2020_2025.csv 不在で strict OOS 評価 不能 |

### 2.5 旧モデル (動作確認用、削除候補)

- `keiba_model_v9_nar.pkl` (479 KB)
- `archive/nar/keiba_model_*.pkl` (multiple)
- `keiba_model_v12_central*.pkl.gz` (旧 base)
- `keiba_model_v134_*.pkl.gz` (旧)

---

## 3. 5/9 投資戦略 (確定、変更不可)

詳細: `data/results/20260509_final_plan_v2.md`

| 項目 | 値 |
|------|----|
| 採用案 | **V15 案B改** (12R 1勝クラスのみ、11R 全除外) |
| 採用予定 R | 0-3 R (12R × 3場、要 5/8 21:00 後 race_name 確認) |
| 投資額 | **0-2,100円** (700円 × 採用R数) |
| 期待 ROI | **161.0%** [95%CI 135.9-222.4%] (healthy 4日 base) |
| 期待収支 | +400-1,300円 |
| 最悪 | -2,100円 (全外し) |
| 11R | **全 3場 除外** (新潟駿風S 距離不適合, 東京エプソムC G3, 京都京都新聞杯 G2) |

### 5/9 朝のフロー (5/9 朝 起きたらまず開く)

1. **`data/results/20260509_pat_checklist.md`** 開く
2. 06:30 Keiba-Morning_Sat 自動 → Discord #bets 通知 確認
3. 08:00 DailyPredict (watchdog) 自動 → 35 races 完了
4. 09:00 12R race_name 確認 (1勝クラスかどうか)
5. 14:00-15:30 PAT 投票 (採用 R × 700円)
6. 18:00 DailyResults_Sat + Keiba-RaceDayReport_Sat 自動 → Discord 結果通知
7. 20:30 振り返りテンプレ埋め (`data/v18/post_5_9_improvement_template.md`)

詳細: `data/results/20260509_operation_guide.md`

### 5/8 (金) 21:00 後 確認 (1度)

```bash
# 5/9 全 race_id + 12R race_name 確認
python -c "
import requests, re
for rid in ['202604010312','202605020512','202608030512']:
    r = requests.get(f'https://race.netkeiba.com/race/shutuba.html?race_id={rid}',
                     headers={'User-Agent':'Mozilla/5.0'})
    m = re.search(r'<h1[^>]*>([^<]+)</h1>', r.text)
    print(rid, '→', (m.group(1).strip() if m else 'NOT_FOUND'))
"

# Cookie 確認
python tools/refresh_cookie.py --check
```

詳細: `data/results/20260509_pre_check.md`

---

## 4. Phase 2.5 残タスク (優先順位)

### 🔴 緊急: なし

すべて完了 (Cookie / morning task / watchdog / odds_base / TYB monitor / Platt / normalize / 静音化 / NAR / 5/9 準備)。

### 🟠 高 (5/12 までに)

| # | task | 工数 | 出力先 |
|---|------|------|--------|
| H1 | tools/scrape_nar_today.py 実装 (NAR 出馬表当日取得) | 60min | data/nar_today_shutuba_DATE.csv |
| H2 | tools/scrape_nar_results.py 実装 (NAR 結果照合) | 60min | data/nar_daily_results/ |
| H3 | 5/8 21:00 後 12R race_name 手動確認 | 5min | (人手) |

NAR 5/12 paper trading 開始 を possible にするため H1/H2 必須。

### 🟡 中 (Phase 2.5 完了 5/24 までに)

| # | task | 工数 | 出力先 |
|---|------|------|--------|
| M1 | TYB publish 観測 完了判定 | 5min | data/tyb_publish_log.csv 解析 |
| M2 | feature distribution shift 調査 (BT vs production) | 90min | data/v18/feature_shift_5_*.md |
| M3 | race-level normalize 本番 pipeline 統合 (predict_core.py) | 30min | tools/predict_core.py |
| M4 | chihou_races_2020_2025.csv 生成 (NAR strict OOS 評価) | 60min | data/chihou_races_2020_2025.csv |
| M5 | 条件別 NAR ROI 計算 (jra_payouts相当 NAR データ) | 60min | data/v18/nar_roi_by_condition.md |
| M6 | v18/v19 5/16 試行 (条件達成後) | 1日 | 5/16 paper / 1,000円試行 |
| M7 | NAR 5/12-5/15 paper → 5/16 試行 (500円) | 5日 | 5/16 NAR 試行 |

### 🟢 低 (Phase 3 構想、6 月以降)

| # | task | 内容 |
|---|------|------|
| L1 | v15.1 features 拡張 (157f想定、ra_score/sc_score 完全復活) | KKA/SKB/SR 等 V15 逆輸入 |
| L2 | v20 統合モデル設計 (JRA + NAR 共通 features 52+) | Phase 3 |
| L3 | 古いモデル削除 (v9/v12/v134) | git LFS or .gitignore |
| L4 | predict_v20.py で 統合 inference 化 | Phase 3 |

---

## 5. タスクスケジューラ 全 28 件 (Keiba 系のみ、5/5 PM 確認)

すべて Ready (静音化済 vbs ラッパー経由)。

### JRA 既存 16 件 (Session #9 で静音化済)

| # | TaskName | スケジュール | 役割 |
|---:|----------|--------------|------|
| 1 | Keiba-AM3FireCheck | 03:15 | (確認) |
| 2 | Keiba-AM6FireCheck | 06:15 | (確認) |
| 3 | Keiba-AM8FireCheck | 08:50 | (確認) |
| 4 | Keiba-FridayWeekendScrape | 金 10:00 | 週末出馬表事前 (元想定 21:00) |
| 5 | Keiba-MorningDigest | 07:00 | dashboard |
| 6 | Keiba-Morning_Sat | 土 06:30 | morning_top_races (V17 11R/12R) |
| 7 | Keiba-Morning_Sun | 日 06:30 | 同上 |
| 8 | Keiba-NightlySanity | 23:00 | 翌日 task pre-check |
| 9 | Keiba-PreFireCheck | 02:55 | (確認) |
| 10 | Keiba-TybPublishMonitor | 毎時 X:30 | TYB 公開時刻 観測 (5/4-5/10 蓄積中) |
| 11 | KeibaAI_DriftDetector | 週次 月 08:30 | drift 検出 |
| 12 | DailyJrdbKyi | 06:00 | JRDB KYI/SED/TYB 等 全種 DL |
| 13 | DailyPredict | 08:00 | V15 当日全レース推論 (watchdog 化済) |
| 14 | DailyPremiumScrape | 03:00 | premium data |
| 15 | DailyResultsEvening | 20:00 | 結果照合 (二重) |
| 16 | DailyResults_Sat | 土 18:00 | 結果照合 |
| (16+) | DailyResults_Sun | 日 18:00 | 同上 |
| (16+) | JrdbHealthCheck_Sat | 土 07:30 | JRDB 鮮度 chk |
| (16+) | JrdbHealthCheck_Sun | 日 07:30 | 同上 |
| (16+) | Keiba-ScrapeProgress | 07:00 | scrape progress monitor |
| (16+) | Keiba-WeeklyScrapeResume | 月 06:30 | scrape 再開 |

(計 21 task — 16 件と数えていたのは old count、現在 21+ 件)

### NAR 5 件 (Session #13 で登録 ps1 提供 → admin 実行済 確認)

| # | TaskName | スケジュール | 役割 |
|---:|----------|--------------|------|
| 22 | Keiba-NarMidDayCalendar | 13:00 | NAR カレンダー (placeholder) |
| 23 | Keiba-NarDailyScrape | 16:30 | NAR 出馬表 + 前夜オッズ (placeholder) |
| 24 | Keiba-NarDailyPredict | 17:00 | NAR 推論 + 候補抽出 |
| 25 | Keiba-NarLiveOddsRefresh | 19:00 | live odds (placeholder) |
| 26 | Keiba-NarDailyResults | 21:30 | 結果照合 (placeholder) |

注: placeholder は同じ `nar_daily_pipeline.bat` を呼ぶ (no-op 相当)。 5/12 paper trading 開始までに実 script 追加予定。

### RaceDayReport 2 件 (Session #14 で登録 ps1 提供 → admin 実行済 確認)

| # | TaskName | スケジュール | 役割 |
|---:|----------|--------------|------|
| 27 | Keiba-RaceDayReport_Sat | 土 18:00 | race_day_report.py + Discord |
| 28 | Keiba-RaceDayReport_Sun | 日 18:00 | 同上 |

→ **5/9 18:00 に自動レポート発火** (USER 手動操作 不要)。

### Disabled (1 件)

- ProcessWatchdog (Disabled): 過去のメモリ監視、現在 不使用

---

## 6. 重要ファイル 一覧

### モデル

| path | 役割 |
|------|------|
| `keiba_model_v15_central_live.pkl.gz` | **V15 本番 (Pattern B、150 features)** |
| `keiba_model_v15_central.pkl.gz` | V15 Pattern A (リークフリー、評価用) |
| `data/v18/models/v18_tansho_lgb.txt` | v18 単勝 LGB (5/16 試行候補) |
| `data/v18/models/v18_tansho_xgb.json` | v18 単勝 XGB |
| `data/v18/models/v19_fukusho_lgb.txt` | v19 複勝 LGB |
| `data/v18/models/v19_fukusho_xgb.json` | v19 複勝 XGB |
| `data/nar/models/keiba_model_nar_v4.pkl` | **NAR v4 (5/12 paper 開始)** |

### 主要スクリプト (本番運用)

| path | 役割 |
|------|------|
| `tools/daily_predict.py` | V15 当日全 R 推論 (08:00 自動) |
| `tools/daily_predict_watchdog.py` | watchdog 監視 + 自動再起動 (S4) |
| `tools/morning_top_races.bat` + `.sh` | 06:30 morning V17 |
| `tools/refresh_cookie.py` | netkeiba premium Cookie 自動 refresh (--auto) |
| `tools/silent_runner.vbs` | wscript hidden 起動 (S9 静音化) |
| `tools/race_day_report.py` | 18:00 自動 結果サマリー + Discord (S14) |
| `tools/predict_nar.py` | NAR v4 汎用 predict (S13) |
| `tools/race_normalize.py` | race-level normalization (S10) |
| `tools/notify_done.py` | Discord 通知 utility |
| `tools/predict_core.py` | V15 core inference (将来 race_normalize 統合候補) |

### 投票準備 doc (5/9 で読む順)

1. `data/results/20260509_pre_check.md` (事前) ← 5/8 21:00 後
2. `data/results/20260509_operation_guide.md` (時系列フロー)
3. `data/results/20260509_pat_checklist.md` (投票時 ← **5/9 朝 まずこれ**)
4. `data/results/20260509_dry_run_5_5.md` (動作 検証 結果)
5. `data/v18/risk_management_5_9.md` (撤退ライン)
6. `data/results/20260509_final_plan_v2.md` (採用方針 確定)
7. `data/v18/post_5_9_improvement_template.md` (5/9 終了後 振り返り)

### NAR 関連 doc (5/12 paper 開始までに読む)

| path | 役割 |
|------|------|
| `data/v18/nar_v4_current_state.md` | model + データ + V15比較 |
| `data/v18/nar_pipeline_design.md` | 自動化 + JRA並列 + 撤退基準 |
| `data/v18/nar_v4_backtest_5_5.md` | AUC 再現 + 限界 |
| `data/v18/nar_schtasks_user_guide.md` | (既に admin 実行済) |
| `data/v18/jra_nar_integration_plan.md` | 5/16-5/24 並列運用 + Phase 3 v20 構想 |

### Phase 2.5 進捗

| path | 役割 |
|------|------|
| `data/v18/phase_2_5_progress_5_4.md` | 4/27 〜 5/4 の進捗 (S7) |
| `data/v18/phase_2_5_progress_5_4_pm.md` | 5/4 PM 続き (S10) |
| `data/v18/phase_2_5_session10_final.md` | Session #10 終了 サマリー |

### 引き継ぎ書 (Session #15、本書まわり)

| path | 役割 |
|------|------|
| `docs/HANDOFF_5_5_TO_5_9.md` | **本書 (v2)** |
| `docs/handoff_v1_v2_diff.md` | v1 誤情報 訂正 |
| `docs/sessions_5_3_5_5_recap.md` | 14 セッション 収穫マップ |
| `docs/lessons_learned_5_5.md` | 5/3-5/5 教訓 |
| `docs/next_session_checklist.md` | 次回 起動 手順 |

---

## 7. 出力フォーマット (PAT 投票)

### 三連複 7 点 (V15 trio_bets 列、案B改 12R 1勝クラス)

```
軸 = top1
2列目: top2, top3
3列目: top2, top3, top4, top5, top6  (5 通り)
→ 1×2×5 から重複除く 7 点
```

例 (柏記念 NAR v4 検証時):
```
軸: 8 ミッキーファイト (p_ens=0.777)
2列目: 2, 10
3列目: 2, 10, 13, 3, 1
→ 7 点: 2-8-10, 1-2-8, 2-8-13, 2-3-8, 1-8-10, 8-10-13, 3-8-10
```

5/9 案B改 では `data/daily_predictions/20260509.csv` の **trio_bets 列をそのまま使う** (再計算しない)。

### 馬連 2 点 (条件 E のみ、5/9 該当なし想定)

5/9 12R は 1勝クラスで頭数 8+ 想定 → 条件 E (7 頭以下) は通常該当しない。万一条件 E なら馬連:
```
TOP1-TOP2, TOP1-TOP3
各 350 円
```

---

## 8. 次セッション (5/8 夜 or 5/9 朝) で何から始めるか

詳細: `docs/next_session_checklist.md`

要点:
1. `git pull --rebase`
2. `git log --oneline -5` で最新 commit 確認 (本書 base = 2b6dc4eb)
3. **`data/results/20260509_pat_checklist.md`** を開く (5/9 朝)
4. もしくは `data/results/20260509_pre_check.md` の 5/8 21:00 後 確認手順 (5/8 夜)
5. Discord 直近通知確認 (#updates / #bets)

---

## 9. 既知の問題 (継続 monitor)

| issue | 状態 | 対応 |
|-------|------|------|
| TYB 公開時刻 不明 | 観測中 (5/4-5/10) | 5/11 月 結果判定 |
| jra_races_full 2026 過去分 一部欠損 | 4-5月分 復旧済 (S10)、6+ は 5/12 scrape 待ち | continuous |
| ot/ov/ow/oz 4 種 33日 stale | V15 学習に未使用、影響軽微 | Phase 3 v15.1 で再評価 |
| jra_payouts.csv 4/26 まで | 5/2-5/3 は別途 verify_real_roi で得た | 自動取得復旧待ち |
| FridayWeekendScrape 10:00 | 元想定 21:00 → 補完で 5/8 21:00 後 手動 scrape 推奨 | 別 task |
| chihou_races_2020_2025.csv 不在 | NAR strict OOS 評価 不能 | 別 60min task |
| daily_predict_watchdog 火-金で fatal alert (誤報) | 0 races (JRA 開催なし) で正常終了→ alert | 仕様、土日のみ alert を信じる |

---

## 10. ユーザー方針 (絶対遵守)

- **取り返し禁止** (損失出ても無理に回収しない)
- **累計損失拡大NG** (現在 +14,140円 死守)
- **撤退ライン -50,000円** (絶対)
- 5/9 案B改 維持 (12R 1勝のみ、上限 2,100円)
- v18/v19 投入は 5/16 以降 (条件達成後)
- NAR 投入は 5/12 paper 開始、5/16 試行 500円/日
- 静音化済、admin elevation 必要部分は手順書あり

---

## 11. 連絡 + 通知

- Discord #bets: morning_top_races, race_auto_notify, predict_nar, race_day_report
- Discord #updates: notify_done.py 経由 (今 session 進捗 等)
- フォールバック: DISCORD_WEBHOOK_URL

`.env` に DISCORD_WEBHOOK_URL / DISCORD_WEBHOOK_BETS / NETKEIBA_PASSWORD / NETKEIBA_COOKIE 設定済 確認 (S14 で確認)。

---

## 12. 結論 (一句)

**5/9 朝、`data/results/20260509_pat_checklist.md` を開いて 順序通り進めれば 迷わず投票完了**。
あとは 18:00 の自動レポート と 20:30 の振り返り。
累計 +14,140円 を死守、5/16 から段階 ramp。
