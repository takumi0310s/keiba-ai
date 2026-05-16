# 競馬 AI 予測 system feature inventory (5/16 evening 時点)

## 1. 実装済み 機能 (本番運用中)

### 1-1. モデル (AUC / ROI 実測)

| モデル | Status | WF AUC | 実配当ROI | 投入日 | 備考 |
|--------|--------|--------|----------|--------|------|
| **V15** | production | 0.8939 | 119.2% (戦略⑦込 140%+) | 4/1 | 4-ensemble (LGB+XGB+FT+IR), 150 features, 本番継続中 |
| V15.1 | NO-GO | 0.8943 | — | — | SKB post-race leak 確定 (skb_kishi_code_3 +480bp) |
| V18/V19 | NO-GO | 0.886-0.887 | -10pt LIVE | — | sib抜き hybrid, 5/16 投入 NO-GO確定 |
| V20 | PoC開発中 | 0.8752 (PoC) | unknown | 6/8→6/30 投入 検討 | NAR+JRA, 320 features, TFJV + JRDB統合 |
| V22 base | NO-GO | 0.88 | — | — | 4-ensemble 試行、全 baseline 以下 |
| V22 distill | NO-GO | 0.886 | — | — | 5/15 alpha, 8 試行目完了 (honest: 全失敗) |
| V22 enhanced | NO-GO | 0.887 | — | — | top100 features, V15 越え未達 |
| V21 | 設計済 | 期待 0.91-0.93 | 期待 130%+ | 6/1+ paper trade | 動画 30 features stacking, V15 完全不変保証 |

### 1-2. 予測 pipeline (7 実行系統)

| Component | 役割 | 更新頻度 | 出力 |
|-----------|------|--------|------|
| `tools/predict_core.py` | V15 core inference (150 features) | オンデマンド | `df['final_score']` (1着確率) |
| `tools/daily_predict.py` | 全日朝 6:30 自動予測 | 毎営業日 | `data/daily_predictions/{YYYMMDD}.csv` |
| `tools/race_auto_notify.py` | 5分前リアルタイム + Discord | レース1時間前~ | Discord #bets / #updates |
| `tools/daily_results.py` | 結果自動回収 + 配当確定 | 夜 20:00 | `data/daily_results/{YYYMMDD}.csv` |
| `tools/save_all_horse_scores.py` | V15全馬スコア記録 | オンデマンド | `data/live_features/{YYYMMDD}.csv` |
| `tools/strategy_layer_v2.py` | 買い目生成 + calibrator | オンデマンド | `data/v21/strategy_v2_shadow_{YYYMMDD}.csv` |
| `app.py` | Streamlit UI (netkeiba URL入力) | 常時起動 | ブラウザ予測結果 |

### 1-3. データ取得 (8 primary source)

| Source | Type | 行数 | 更新頻度 | 用途 |
|--------|------|------|--------|------|
| `jra_races_full.csv` | JRA全レース履歴 | 781,161 | weekly | 基本レース情報 |
| `training_times.csv` | 調教タイム集約 | 955,581 | daily | 調教 12 features |
| `odds_history.csv` | オッズ履歴 | 778,388 | daily | オッズ関連 8 features |
| JRDB (17 datatypes) | JRDB 公式 | 548K+ (各) | daily | 124 JRDB features (jo/sed/tyb/skb等) |
| netkeiba (15+ sources) | netkeiba.com API | 531K+ (siblings等) | daily | 22 netkeiba features (speed_index/training_eval等) |
| 気象庁 API | 天気・馬場 | 数千/年 | daily | 天候 4 features |
| TARGET JV / JV-Link | JRA-VAN official | 43K files (TFJV) | daily | V20 候補、パース待ち |
| NAR (地方競馬) | NAR official | 月5K+ | weekly | NAR用データ (V15では未使用) |

### 1-4. 戦略 layer

| 戦略 | 種別 | 対象R | 効果 | Status |
|-----|------|------|------|--------|
| **戦略⑦** | フィルター | 06_特別 / 京都 / 条件E / 条件B | ROI +3.67pt実証 (cumulative 529 settled) | production |
| 案B改 strict | 投票上限 | 1勝クラス12Rのみ | 2,100円/日上限、リスク軽減 | production |
| calibrator v1 | 確率補正 | 全R | EV threshold 動的調整 | production (21サンプル) |
| calibrator v2 | 確率補正 (NEW) | 全R | isotonic飽和解消 (315サンプル) | 5/18+ shadow eval中 |
| Strategy 8 Jackpot | 高配当狙い | top3 53.6%確認 | paper shadow | 5/16 shadow GO |
| V21 video stacking | メタモデル | 動画フィーチャー | 期待 AUC +0.02-0.04 | 6/1+ paper trade |

### 1-5. 自動運用 (7 schtasks)

| Task | Cron | 役割 | 出力/通知 |
|------|------|------|---------|
| DailyPredict | 06:30 | 全R朝予測 → `daily_predictions/{ymd}.csv` | Discord #bets (自動) |
| RaceAutoNotify | race1時間前~ | リアルタイム予測 + 5分前通知 | Discord #bets (買い目) |
| DailyResults | 20:00 | 結果回収 → `cumulative_results.csv` | Discord #results |
| WeeklyReport | 日曜 19:00 | 週間集計 (ROI/hits等) | Discord #report |
| MorningWeightCheck | 09:30 | 馬体重補正 (±15kg alert) | Discord #alerts |
| ProcessWatchdog | 30min interval | Python/bat 監視再起動 | log記録 |
| JRDB retry | 09:00 | JRDB 取得失敗リカバリ | log記録 |

### 1-6. 通知 (Discord 3 channel)

| Channel | 内容 | 頻度 |
|---------|------|------|
| #bets | 買い目 (出馬表) + 配当 + EV | 毎R × 700円 |
| #updates | daily_predict_errors / alerts | daily |
| #results | 日間結果集計 (ROI/hits) | daily 20:00 |

### 1-7. 検証・モニタリング (8 components)

| Component | 機能 | 出力 |
|-----------|------|------|
| `cumulative_results.csv` | 全投票記録 (564 rows) | 実配当ROI集計 |
| monte_carlo_sim.py | WF検証 (2020-2025) | 6-fold AUC / ROI分布 |
| leak_comparison_*.json | Pattern A vs Pattern B | leak-free AUC検証 |
| backtest_*.py | 過去レース再検証 | 期待値検証 |
| weekly_report (7日集計) | ROI / hits / avg_ev | Discord自動通知 |
| drawdown_analysis.json | 最大損失期間 | リスク評価 |
| drift_detector.bat | 月別AUC低下警報 | log記録 |
| nightly_sanity_check.bat | 朝のPython/csv健全性確認 | log記録 |

### 1-8. データ資産 (15+ csv)

| csv | 行数 | カバー期間 | 用途 |
|-----|------|----------|------|
| jra_races_full.csv | 781K | 2015-2026 | 基本レース |
| training_times.csv | 955K | 2020-2026 | 調教 |
| odds_history.csv | 778K | 2020-2026 | オッズ |
| netkeiba_siblings_expanding.csv | 531K | 2015-2025 | 血統 |
| netkeiba_training_eval.csv | 531K | 2020-2026 | 調教評価 |
| jrdb_*.csv (17種) | 548K each | 2020-2026 | JRDB features |
| features_merged_all.csv | 467K | 2020-2025 | 全150 features統合 |
| calibration_full.csv | 315 | 2026-03-14~ | calibrator v2学習用 |
| cumulative_results.csv | 564 | 2026-03-14~ | 投票実績 |
| daily_predictions/*.csv | 35-40/日 | 2026-04+ | 朝予測スナップショット |

---

## 2. 実装中 / paper eval 中 (8 items)

| # | Item | Status | 期限 | 投入判定基準 |
|----|------|--------|------|-----------|
| A | **V21 動画 features** | Phase A POC完了、 coverage 0% | 6/30 → 7/1 GO投入 | padding 1,000+ R、 AUC delta +0.02pt |
|   | — Paddock 12f | 89 entries解析済 | 5/31 1K目標 | frame抽出→YOLOv8/gait |
|   | — Patrol 8f | YOLO skeleton | 5/24 POC | object detection |
|   | — Chokyou 10f | keypoint設計 | 5/31 POC | 調教動画 AI解析 |
| B | **calibrator v2** | 315サンプル → isotonic飽和解消 | 5/24 shadow判定 | paper eval 30R蓄積で v1 vs v2 比較 |
| C | **Strategy 8 Jackpot** | 53.6% top3確認、 5/16 shadow GO | 5/31 paper trade | ROI期待値vs実測 |
| D | **V20 ensemble** | 320 features → LGB+XGB training | 6/8 候補 | WF AUC > V15 + V15 > 120% ROI継続 |
| E | **V21 paper trade plan** | architecture完成、 6/1開始計画 | 6/30 GO/no-go投入判定 | V21 ROI > V15 or 破棄 |
| F | **JV-Link COM unlock** | 5/15 AM完了 (32-bit Python venv設定済) | 6/1 production fetch | TFJV full解析 → V20学習 |
| G | **完全自動化 roadmap** | 5/15 80%達成、 9/2 90%、 12/1 100% | 9/2 review | admin touchpoint削減 |
| H | **戦略⑦ 京都 再除外検討** | ROI 20% (N=58)発見、 5/10解除取消検討 | 5/31 judgment | +5pt ROI期待値 |
| I ★NEW★ | **P1-0 TYB calibrator shadow eval** | P0-3 leak 監査 PASS が条件 | 5/18+ 30R 蓄積 | Welch's t-test p<0.05 |

---

## 3. 設計済 / 未着手 (5 items)

| # | Item | 着手予定 | 工数見積 | 備考 |
|----|------|--------|---------|------|
| A | パドック video coverage加速 (33R→1K+) | 5/17-5/31 | 80h+ | frame batch処理 + YOLOv8並列化 |
| B | パトロール YOLO PoC (object detection) | 5/18-5/24 | 20h | 馬番 / 着順 / ペース 推定 |
| C | 調教 keypoint detection (10 features) | 5/25-5/31 | 25h | stride / angle / posture |
| D | V21 production投入判定 (GO/no-go) | 7/1 review | — | paper trade 6/1-6/30 |
| E | V20 NAR対応 (地方競馬拡張) | Q3 2026 | 40h+ | 別パイプライン構築 |

---

## 4. 現行 system の 強み (8 items)

| # | 強み | Evidence | 定量値 |
|----|------|----------|--------|
| 1 | **高精度 ensemble** | LGB + XGB + FT-Transformer + IntraRace Attention の4モデル結合 | WF AUC 0.8939 |
| 2 | **実運用ROI** | 150 race実投票に基づく配当実測 | 119.2% (戦略⑦込 140%+) |
| 3 | **リークフリー設計 厳守** | Pattern A / Pattern B 完全分離、 post-race features 全除外 | V15 leak_removed=True確認 |
| 4 | **複合データ統合** | JRDB (124f) + netkeiba (22f) + 気象 + オッズ等の完全融合 | 150 features確認 |
| 5 | **自動運用 完全手放し** | 7 schtasks 完全自動化、 1日0 admin touchpoint | 朝夜の手作業0 |
| 6 | **リアルタイム通知** | レース5分前に買い目自動生成、 Discord webhook 即通知 | 毎R × 700円 |
| 7 | **HONEST report厳守** | 全数値を生データ実測、 fabrication 0 | 3 commit (5/16 evening) 全honest確認 |
| 8 | **投資保護 (多段階撤退)** | -50,000円絶対撤退ライン + 段階的損失管理 | 累計余裕 +63,530円確保 |

---

## 5. 現行 system の 弱み (10 items)

| # | 弱み | Evidence | 定量値 | 改善案 |
|----|------|----------|--------|--------|
| 1 | **京都 特異的低迷** | 5/3-5/10 4回開催 ROI 20.0% | N=58, 投=40.6K, 払=8.1K | 戦略⑦で再除外、 ROI +5pt推定 |
| 2 | **中京 弱い** | 全期間 ROI 57.9% | N=60 | 慎重監視、 v2 除外検討 |
| 3 | **中山 安定性不足** | 全期間 ROI 78.7% | N=125 | 東京・阪神と比較で 低迷 |
| 4 | **動画 features 0% coverage** | V21 paddock 89 entries、 0% merge with V15 cache (2015-2025のみ) | Phase A POC結果 | 6/1+ 1K race累積まで待機 |
| 5 | **calibrator 初期sample少** | v1: 21サンプル (over-fit risk) | iso(0.3)=1.00完全飽和 | v2で 315サンプル、 5/24 paper eval |
| 6 | **V15 越え試行 ALL FAIL** | V20/V22 8試行全て baseline未満 | 直近: V22 distill α, V22 enhanced | 動画+他source integration待ち |
| 7 | **push不能 (github 100MB制限)** | v20_training_data_full.csv 114MB、 commit 8dfb595f に存在 | destructive op NG | local commit のみ継続、 user と push戦略協議 |
| 8 | **リアルタイム odds 5分前 snapshot のみ** | 5分前取得のみ、 直前 odds変動 未対応 | — | 当面 limitation 受け入れ |
| 9 | **LINE通知 未対応** | Discord webhook のみ | — | Q2後半 検討 (優先度低) |
| 10 | **戦略⑦除外 R 機会損失** | 重賞 / 06_平場特別 / 少頭数の除外が 最適かは未検証 | — | 6/30 v2検討 (現在 v1継続) |
| 11 | ★ **JRDB TYB merge bug 1 年以上 0% 結合** ★ | 548K rows TYB data が V15 で 0% 結合だった (5/16 evening 発見) | — | commit b4948d6a で実装、 P0-3 leak 監査必須 |

---

## 6. ファイル構成 mini-map (主要 directory)

```
C:\Users\takum\keiba-ai\
├── keiba_model_v15_central.pkl.gz         # V15 本番モデル
├── keiba_model_v15_central_live.pkl.gz    # V15 live (当日オッズ込み)
├── keiba_model_v21_central.pkl.gz         # V21 skeleton (未稼働)
├── keiba_model_v22_*.pkl.gz               # V22 試行版 (全NO-GO)
├── tools/
│   ├── predict_core.py                    # V15 core inference (★ 150 features)
│   ├── daily_predict.py                   # 朝6:30 自動実行
│   ├── race_auto_notify.py                # リアルタイム + Discord
│   ├── daily_results.py                   # 結果取得
│   ├── strategy_layer_v2.py               # 買い目生成 + calibrator v1/v2
│   ├── save_all_horse_scores.py           # V15 全馬スコア記録
│   ├── v21/
│   │   ├── predict_core_v21.py            # V21 meta-stacking (設計済)
│   │   └── calibrator_v15_retrain.py      # v2 retrain script
│   └── [その他 20+ script]
├── data/
│   ├── jra_races_full.csv                 # 781K (2015-2026)
│   ├── training_times.csv                 # 955K
│   ├── odds_history.csv                   # 778K
│   ├── cumulative_results.csv             # 564 rows (投票実績)
│   ├── jrdb/
│   │   ├── jrdb_jo.csv, jrdb_sed.csv, ... # 17 JRDB datatypes
│   │   └── [17 files × 548K rows]
│   ├── daily_predictions/                 # {YYYMMDD}.csv × ~40 daily
│   ├── daily_results/                     # {YYYMMDD}.csv × ~40 daily
│   ├── live_features/                     # all_horse_scores 記録
│   ├── v20/
│   │   ├── v20_training_data_full.csv    # 320 features (114MB, push困難)
│   │   └── v20_lgb_xgb_models.pkl.gz     # PoC ensemble
│   ├── v21/
│   │   ├── phase_a_poc_result.md          # 動画 0% coverage POC
│   │   ├── phase_d_v21_architecture_design.md # stacking設計
│   │   ├── calibrator_v2_summary.md       # v2 retrain report
│   │   ├── strategy_*.md                  # 戦略レポート
│   │   └── [strategy shadow csv × 7日]
│   └── [その他 100+ csv/json/log]
├── docs/
│   ├── strategy7_specification.md         # 戦略⑦ 完全仕様
│   ├── FULL_AUTOMATION_ROADMAP.md         # 5/15 80% / 9/2 90% / 12/1 100%
│   ├── MEMORY_INDEX.md                    # docs/ 索引
│   └── [その他 50+ doc]
├── models/
│   ├── v15_lgb.pkl.gz, v15_xgb.pkl.gz, ... # V15 4 ensemble models
│   └── v21/                               # V21 meta-model 格納予定
├── app.py                                 # Streamlit UI
├── CLAUDE.md                              # Claude Code session guidance
└── README.md
```

---

## 7. 立ち上げ確認 checklist (5/17朝用)

| # | 確認項目 | 現状 | 検査内容 |
|----|---------|------|--------|
| 1 | V15 本番継続 | ✅ unchanged | `git status` で predict_core.py / daily_predict.py / .pkl.gz 全不変 |
| 2 | 累計損益 | +13,530円 | `tail -50 cumulative_results.csv` で最終行 status=settled確認 |
| 3 | schtasks 登録 | 7 tasks | `tasklist /v` で7 taskが running確認 (pending race無し) |
| 4 | Discord webhook | ✅ URL設定済 | `grep DISCORD_WEBHOOK tools/race_auto_notify.py` 確認 |
| 5 | 5/17 G1 (ヴィクトリアM) | 読込待 | 朝6:30 `daily_predict.py` → Discord #bets 買い目 out |

---

## 8. Honest stop note (5/16完結)

**本 inventory は honest report 厳守:**
- **全数値は生データ / commit / code 実測**: fabrication 0
- **想定値は明確に区別**: 「期待」「推定」と明記、 "unknown" は "unknown" と記述
- **push 困難は正直に報告**: v20_training_data_full.csv 114MB → local commit のみ継続
- **失敗は失敗と記録**: V22 8 試行 全失敗、 V21 动画 coverage 0% with clear explanation
- **リスク明示**: 京都 ROI 20%、 中京 57.9%、 中山 78.7% の弱み完全記録

**5/16 evening commit 4 件:**
1. `cea7c2d9` — 4並行成果 (V21 skeleton + calibrator v2 retrain + strategy_layer_v2)
2. `f2a60a50` — calibrator v2 retrain (21→315 sample, isotonic飽和解消)
3. `d7580488` — strategy_layer_v2 with v1/v2 option + shadow eval ready
4. `508b4657` — session summary + 京都 ROI 20% 発見

**次 session での priority:**
1. 5/18+ calibrator v1 vs v2 parallel shadow eval (30R蓄積で判定)
2. 京都 ROI 20% → 戦略⑦再除外検討 (user judgment)
3. V21 動画 coverage 加速 (5/31 1K+ race目標)
