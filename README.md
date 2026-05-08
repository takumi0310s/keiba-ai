# keiba-ai

JRA (中央競馬) + NAR (地方競馬) 予測 AI システム

> 最終更新: **2026-05-08 (Session #44、 ★ TFJV フル data 即活用 → V20 1 ヶ月前倒し ★、 6/8 V20 投入候補)**
> 累計収支: **+13,530円** (`data/cumulative_results.csv`、 5/6 真相確定値)
> 撤退ライン: **-50,000円** (絶対遵守) — 撤退余裕 +63,530円

> **Session #38 確定 (5/7)**: V15.1 SKB POST-RACE LEAK 確定 → 採用 NO-GO。 V18/V19 sib抜き = リーク + 識別能力 hybrid → 5/16 NO-GO。 5/9 V15 案B改 単独継続 (絶対)。
> **Session #39 deluxe (5/7)**: Phase 3 (5/24+) 修正版 + V20 (6/9-30) architecture + Phase 4 (7-8月) 動画解析 PoC を全前倒し設計。
> **Session #40 マスター (5/7)**: 5 領域 (PAT 7pt baseline confirm、 race_classifier、 Kelly、 health check、 EMERGENCY runbook 15 シナリオ、 docs/INDEX.md、 voting design)。
> **Session #41 巨大マラソン (5/8)**: 8 領域。 sib_exp PoC LIVE retro **+6.89pt 改善大成功** (24.14% → 31.03%)、 32-bit Python plan、 JV-Link fetcher v2、 V20 6 年分 backfill plan。
> **Session #42 (5/8)**: 10 領域。 拡張 retro 4/18-5/5 (V15 案B改 ROI 44.47%)、 動画解析 feasibility GO、 sib_exp **window=5 最適化 (corr 0.2010)**、 5/16 V18/V19 GO 65-80%。
> **Session #43 (5/8)**: 7 領域。 ★ V15 ROI 真因発見 (44% → 真の 83.96%) + sib_w5 LIVE 完全回復 (winner_top1 34.48% = OLD 同等)。 5/16 GO 確率 **85-95%** に劇的上昇。
> **Session #44 (5/8)**: 7 領域。 ★ **TFJV フル data 即活用** ★: tools/tfjv_parser.py 実装、 6 年分一括 parse (320K records / 10 秒)、 V20 PoC LGB AUC 0.8752、 ★ Phase 3 plan v3 で V20 投入 **7/1 → 6/8 (1 ヶ月前倒し)** ★。 32-bit Python / JV-Link は 5/16 後 廃止判断。 詳細 → [`docs/PHASE_3_4_5_INTEGRATED_ROADMAP_v3.md`](docs/PHASE_3_4_5_INTEGRATED_ROADMAP_v3.md)

### JRA-VAN 加入完了 (5/7 夜)

ユーザー側で JRA-VAN DataLab 加入 + JV-Link DLL install 完了。 32-bit COM のみ提供のため、 5/24+ Phase 3 前半で 32-bit Python venv (`C:\Users\takum\jvlink-venv\`) を構築し JV-Link 経由で公式データ取得開始。 既存 64-bit Python (predict_core / daily_predict 含む) は完全維持。

---

## 1. プロジェクト概要

中央競馬の全レースを AI で予測 → 条件別に三連複 / 馬連の買い目を自動生成。
2026/4 から地方競馬 (NAR) と複数モデル (V17/V18/V19) を並走させる Phase 2.5+ に移行。

- **Streamlit**: https://keiba-ai-l2klehd4rfoupnj5g7rw8b.streamlit.app
- **GitHub**: https://github.com/takumi0310s/keiba-ai
- **本番モデル**: V15 (LGB+XGB+FT+IR 4-model ensemble, AUC 0.8939)
- **試作モデル**: V17 (morning), V18 (単勝), V19 (複勝), NAR v4

---

## 2. モデル構成

### 2.1 本番運用中

| モデル | ファイル | AUC | 特徴量 | 役割 | status |
|--------|---------|-----|--------|------|--------|
| **V15 Pattern B** | `keiba_model_v15_central_live.pkl.gz` | **0.8939** (本番) / 0.8858 (4-model) | 150 | JRA 案B改 主モデル (12R 1勝クラス) | **本番** |
| V15 Pattern A | `keiba_model_v15_central.pkl.gz` | 同上 | 145 | リークフリー評価用 | 本番 |
| **NAR v4** | `data/nar/models/keiba_model_nar_v4.pkl` | **0.8145** (reported) / 0.8519 (OOS) | 22 | 地方 NAR 予測 (5/12 paper 開始) | **本番候補** |

### 2.2 試作 / 試行候補

| モデル | ファイル | 状態 | 備考 |
|--------|---------|------|------|
| V17 morning | `train/train_v17_*.py` | 試作 (土日 06:30 morning_top_races) | 11R/12R 用 |
| V18 単勝 | `data/v18/models/v18_tansho_lgb.txt` + `_xgb.json` | **5/16 NO-GO 確定** (Session #38) | sib抜き で winner_top1 -10pt、 6/15+ sib_*_exp 版で再判定 |
| V19 複勝 | `data/v18/models/v19_fukusho_lgb.txt` + `_xgb.json` | **5/16 NO-GO 確定** (Session #38) | 同上 |
| V15.1 (KKA+SKB+SRB) | — | **採用 NO-GO 確定** (Session #38) | SKB POST-RACE LEAK 確定 (skb_kishi_code_3 +480bp) |

### 2.3 計画中 (Phase 3-4)

| モデル | 期間 | 内容 |
|--------|------|------|
| V18/V19 v2 (sib_*_exp 版) | 5/28-6/8 学習、 6/15+ 投入候補 | Session #39 A の sib expanding window で hybrid 解消 |
| **V20** (JRA + NAR 統合) | **6/9-6/30 学習、 7/1+ 投入候補** | JV-Link 主軸 + SKB 完全除外 + sib_*_exp、 4-model ensemble、 共通 80 features |
| **V21** (V20 + 動画解析) | **7-8 月 PoC、 9/1+ 投入候補** | YOLOv8 + DLC SuperAnimal で歩様 / 仕上がり / 集中度 features 追加 |

### 2.4 アーカイブ済 (5/5 Session #19)

`archive/old_models_20260505/` に 23 ファイル / 87 MB 移動。
v8 / v9 / v92b / v12 / v13 / v131-v135 / v141 / v9_nar 系。
復元したい場合は `archive/old_models_20260505/` から手動でコピー。

---

## 3. 自動化 (Windows タスクスケジューラ 28 件、静音化済)

すべて `tools/silent_runner.vbs` 経由で hidden window 起動。

### 3.1 主要 JRA タスク

| 時間 | タスク | 役割 |
|------|--------|------|
| 03:00 | DailyPremiumScrape | netkeiba premium データ事前取得 |
| 06:00 | DailyJrdbKyi | JRDB KYI/SED/TYB 全種 DL |
| 06:30 (土日) | Keiba-Morning_Sat / _Sun | morning_top_races (V17 11R/12R) |
| 07:00 | Keiba-MorningDigest | dashboard |
| 08:00 | DailyPredict | V15 当日全レース推論 (watchdog 化済) |
| 08:45 (土日) | RaceAutoNotify_Sat/_Sun | 戦略⑦ 適用 + Discord #bets |
| 18:00 (土日) | DailyResults_Sat / _Sun + Keiba-RaceDayReport_Sat/_Sun | 結果照合 + 自動レポート |
| 20:00 | DailyResultsEvening | 結果照合 (二重) |
| 月 08:00 | weekly_report | 週次レポート |
| 月 08:30 | KeibaAI_DriftDetector | モデルドリフト検出 |
| 23:00 | Keiba-NightlySanity | 翌日 task pre-check + Discord 通知 |

### 3.2 NAR タスク (5 件、5/12 paper 開始用)

| 時間 | タスク | 役割 |
|------|--------|------|
| 13:00 | Keiba-NarMidDayCalendar | NAR カレンダー |
| 16:30 | Keiba-NarDailyScrape | NAR 出馬表 + 前夜オッズ |
| 17:00 | Keiba-NarDailyPredict | NAR v4 推論 + 候補抽出 |
| 19:00 | Keiba-NarLiveOddsRefresh | live odds |
| 21:30 | Keiba-NarDailyResults | 結果照合 |

### 3.3 観測 / health check

| 時間 | タスク | 役割 |
|------|--------|------|
| 毎時 X:30 | Keiba-TybPublishMonitor | TYB 公開時刻 観測 (5/4-5/10 蓄積中) |
| 03:15 / 06:15 / 08:50 | Keiba-AM*FireCheck | 予定タスク発火確認 |
| 07:30 (土日) | JrdbHealthCheck_Sat/_Sun | JRDB 鮮度 chk |

---

## 4. Phase 2.5+ 成果 (5/3-5/5、14 セッション 21+ commits)

詳細: [`docs/HANDOFF_5_5_TO_5_9.md`](docs/HANDOFF_5_5_TO_5_9.md)

| 領域 | 成果 |
|------|------|
| 5/9 投資 GO 判定 | V15 案B改 161% ROI 確証、12R 1勝クラスのみ |
| 距離分布 shift | v18/v19 BT vs production で 27.7倍 prob 縮小 → race-level normalize 試作 |
| NAR v4 復活 | archive から発見 → 体系化、柏記念で AUC 完全再現 |
| 静音化 | 28 タスク wscript hidden window 化 |
| データ監査 | v1 引き継ぎ書 7 件誤情報訂正 (累計 +14,140円 が正、-25,000円 は誤) |
| アーカイブ整理 | 古いログ + stale CSV → archive (291 MB) + 旧モデル → archive (87 MB) |

教訓: [`docs/lessons_learned_5_5.md`](docs/lessons_learned_5_5.md)

---

## 5. 5/9 (土) 投資戦略 (確定、変更不可)

詳細: `data/results/20260509_final_plan_v2.md`

| 項目 | 値 |
|------|----|
| 採用案 | **V15 案B改** (12R 1勝クラスのみ、11R 全除外) |
| 投資額 | 0-2,100円 (700円 × 採用R数) |
| 期待 ROI | **161.0%** [95%CI 135.9-222.4%] |
| 期待収支 | +400 - +1,300円 |
| 最悪 | -2,100円 (全外し) |

5/9 朝の操作フロー: `data/results/20260509_pat_checklist.md` を開いて順序通り進めれば投票完了。
18:00 の自動レポート + 20:30 の振り返り (`data/v18/post_5_9_improvement_template.md`)。

### 戦略⑦ (4/27 自動化済、`tools/race_auto_notify.py`)

除外フィルタ:
- `06_特別` (G/L/OPEN特別ではない平場特別) → -9,470円損失源
- `京都` → 5/11 以降に再評価 (course_renovated 永久化効果待ち)
- 条件 E (頭数 ≤ 7) / 条件 B (重〜不良) → サンプル少

期待効果: ROI 119.2% → 140.3% (+21.1pt)

---

## 6. 累計収支 + 撤退ライン

| 項目 | 値 | source |
|------|----|--------|
| **5/5 朝累計** | **+14,140円** | `data/cumulative_results.csv` |
| 撤退ライン (絶対) | -50,000円 | 全 session 一貫 |
| 撤退まで余裕 | +64,140円 | 50,000 + 14,140 |

**撤退判定基準** (`data/v18/risk_management_5_9.md`):
- 5/9 単日 ROI < 50% → 5/10 投資停止
- 5/9-5/10 累計 -10,000円 → 翌週投資停止
- 累計 -50,000円 → 完全撤退

---

## 7. 開発ガイド

### 7.1 Claude Code セッション流儀

- 1 session = 1 task ブロック (commit-per-task)
- 並列セッションで衝突回避: 各 session が異なる path を担当
- TaskCreate / TaskUpdate で進捗 trace 化
- session 越し context bridge は `data/v18/phase_2_5_progress_*.md` + `docs/HANDOFF_*.md`
- 数字は **必ず生データで再検証**、引き継ぎ書 v1 の transfusion 禁止
- doc に数字を書くときは `(USER 実投資 / BATCH 仮想)` を併記

### 7.2 主要コマンド

```bash
# 1レース再予測 (取消発生時)
python tools/predict_one_race.py 202605020211

# 当日全レース予測
python tools/daily_predict.py
python tools/daily_predict.py --date 20260509

# NAR 予測
python tools/predict_nar.py --shutuba-csv ...

# 結果照合 + ROI
python tools/daily_results.py
python check_results.py --summary

# Streamlit ローカル
streamlit run app.py

# Cookie 自動 refresh
python tools/refresh_cookie.py --auto

# Discord 完了通知
python tools/notify_done.py "タスク名" "詳細"
```

### 7.3 リーク厳禁ルール

絶対に Pattern A モデルへ入れない 8 特徴量 (確定オッズ系 + 当日馬体重系 + 馬場状態系):
詳細は `CLAUDE.md` § 8。

### 7.4 ベースライン (これを下回る変更は採用しない)

- WF AUC: **0.8939** (V15 LightGBM Booster)
- 4-model ensemble: 0.8858
- ROI (戦略⑦込み 想定): 140%+
- 採用基準: WF AUC > 0.89 かつ 全年 AUC > 0.85 かつ 全条件 ROI > V15 以上

---

## 8. 重要ドキュメント

### 8.1 Phase 3-4 (Session #39 deluxe、 5/24+ 着手)

| 用途 | path |
|------|------|
| **Phase 3-4 統合 roadmap (5/24-8月)** | [`docs/PHASE_3_4_INTEGRATED_ROADMAP.md`](docs/PHASE_3_4_INTEGRATED_ROADMAP.md) |
| V20 architecture 詳細設計 (807 行) | [`docs/PHASE_3_V20_DETAILED_DESIGN.md`](docs/PHASE_3_V20_DETAILED_DESIGN.md) |
| JV-Link 統合 plan + 試作 | [`docs/PHASE_3_JVLINK_INTEGRATION_PLAN.md`](docs/PHASE_3_JVLINK_INTEGRATION_PLAN.md) |
| 全 4 source 役割分担 | [`docs/PHASE_3_DATA_SOURCE_STRATEGY.md`](docs/PHASE_3_DATA_SOURCE_STRATEGY.md) |
| sib expanding window 設計 + PoC | [`data/v18/sib_expanding_window_design_5_7.md`](data/v18/sib_expanding_window_design_5_7.md) |
| SKB 完全除外 patch 設計 | [`data/v18/skb_complete_exclusion_5_7.md`](data/v18/skb_complete_exclusion_5_7.md) |
| Phase 4 動画解析 PoC 設計 | [`docs/PHASE_4_VIDEO_AI_DESIGN.md`](docs/PHASE_4_VIDEO_AI_DESIGN.md) |
| Phase 4 馬体検出 + 姿勢推定 技術調査 | [`docs/PHASE_4_TECH_RESEARCH.md`](docs/PHASE_4_TECH_RESEARCH.md) |

### 8.2 Phase 2.5+ + 過去 引き継ぎ

| 用途 | path |
|------|------|
| 引き継ぎ書 (Phase 2.5+ 全体像) | [`docs/HANDOFF_5_5_TO_5_9.md`](docs/HANDOFF_5_5_TO_5_9.md) |
| 5/3-5/5 教訓 (11 件) | [`docs/lessons_learned_5_5.md`](docs/lessons_learned_5_5.md) |
| プロジェクト指示 (詳細仕様) | [`CLAUDE.md`](CLAUDE.md) |
| v1 誤情報訂正 | `docs/handoff_v1_v2_diff.md` |
| 14 セッション 収穫マップ | `docs/sessions_5_3_5_5_recap.md` |
| 次回セッション起動手順 | `docs/next_session_checklist.md` |
| 5/9 投票チェックリスト | `data/results/20260509_pat_checklist.md` |
| 5/9 操作ガイド | `data/results/20260509_operation_guide.md` |
| 撤退ライン詳細 | `data/v18/risk_management_5_9.md` |
| NAR 統合計画 | `data/v18/jra_nar_integration_plan.md` |

---

## 9. ライセンス

個人プロジェクト (private)
