# tools/ 全 270+ files 索引 (Phase 21D-25、 categorized)

## 📂 Phase 21D-25 で新規追加した tools (5/11 marathon)

### A. 動画 capture (paddock / race / oikiri / YouTube)
| tool | 機能 | 状態 |
|------|------|------|
| paddock_video_capture.py | netkeiba paddock iframe screenshot | ✅ 動作 |
| netkeiba_movie_capture.py | paddock/oikiri/race 統一 capture | ✅ 動作 |
| youtube_jra_live_record.py | YouTube JRA 公式 LIVE 録画 (yt-dlp) | ✅ 動作 |
| paddock_pipeline.py | 全 race × top-N 自動 capture | ✅ 動作 |
| paddock_weekend_archive_build.py | 過去 開催 一括 build | ✅ 動作 |
| video_pipeline_unified.py | capture → YOLOv8 → gait → body chain | ✅ 動作 |
| jra_racing_viewer_capture.py | **JRA RV web Playwright wrapper** | 🟡 skeleton |
| jvlink_movie_wrapper.py | JV-Link MovieType API | 🟡 skeleton (32-bit Python 必須) |

### B. 動画 AI 解析
| tool | 機能 | 状態 |
|------|------|------|
| video_ai_yolov8.py | YOLOv8 馬 bbox 検出 | ✅ 動作 |
| video_ai_gait_features.py | gait / motion 20 features | ✅ 動作 |
| video_ai_body_condition.py | 馬体 condition 18 features | ✅ 動作 |
| video_ai_multi_track.py | race multi-horse tracking | ✅ 動作 |

### C. features 抽出
| tool | 機能 | rows |
|------|------|------|
| build_event_effect_features.py | 騎手/厩舎/升降級 events | 532K |
| build_pace_features.py | race-level pace 統計 | 94K |
| build_pace_features_expanding.py | expanding window 版 (LEAK-free) | 94K |
| build_hot_streak_features.py | jockey/trainer/horse recent | 188K |
| build_layoff_features.py | 休養日数 / 1年+ -12.7pt | 188K |
| build_distance_surface_change_features.py | 馬場 変更 -9.3pt | 188K |
| build_sire_class_down_features.py | 父馬 × class_down boost | 188K |
| build_remarks_features.py | 短評 8 categorical | 277K |
| v20_training_data_full_builder.py | 全 features merge | 190K × 101 cols |
| rebuild_all_features.py | 1 コマンド 全 features rebuild | - |

### D. 30 年 backtest + JV-Link parser
| tool | 機能 |
|------|------|
| backtest_30year_collect.py | TFJV 1995-2024 collector |
| jvlink_parser.py | 8 dataspec parser (32-bit Python) |
| jvlink_movie_wrapper.py | MovieType wrapper |

### E. Phase 23 運用最適化
| tool | 機能 |
|------|------|
| calibrate_confidence.py | isotonic + Platt scaling |
| kelly_bet_sizer.py | Kelly fractional 0.25x |
| exotic_optimizer.py | 三連複 EV 最大化 |
| drawdown_circuit_breaker.py | 累計/連敗 自動 monitor |
| v21_multimodal_poc.py | V15 + video stacking PoC |
| build_calibration_from_daily.py | 314 sample calibrator 学習 |
| extract_calibration_data.py | cumulative → (pred,label) |

### F. signal verify / pattern miner
| tool | 機能 |
|------|------|
| verify_new_features_signal.py | LGB AUC + importance |
| signal_scanner.py | 全 features +5pt scan |
| signal_interaction_analysis.py | 3-way interaction matrix |
| sire_signal_analysis.py | 父馬 × class_down |
| per_condition_signal_analysis.py | 6 条件別 signal |
| golden_pattern_miner.py | 4-way mining (top3 64.8%) |
| pattern_miner_5way.py | 5-way mining (top3 73.9%) |
| golden_pattern_roi_sim.py | 単勝 ROI 推定 |
| jackpot_pattern_roi.py | 4-way Jackpot ROI 184% |
| jackpot_multi_ticket_roi.py | 券種別 (Wide 178%) |
| pattern_5way_roi_full.py | 5-way 詳細 |
| yearly_jackpot_stability.py | 4 年 stability |
| course_jackpot_stability.py | 8 コース stability |
| seasonal_jackpot_stability.py | 月/季節 stability |
| per_condition_jackpot_roi.py | 6 条件別 Jackpot |
| trio_jackpot_combined.py | trio + Jackpot 軸 (skeleton) |

### G. Strategy 8 / production
| tool | 機能 |
|------|------|
| strategy8_shadow_runner.py | V15+Jackpot combine 通知 |
| strategy8_vs_v15_4year_backtest.py | 4 年 backtest |
| monthly_strategy8_projection.py | 月別 ROI projection |
| live_jackpot_detector.py | 当日 race で Jackpot 検出 |
| race_recommendation_api.py | race_id → 推奨 action |
| race_id_mapper.py | TFJV ↔ netkeiba 変換 |

### H. 5/17 開催 準備
| tool | 機能 |
|------|------|
| register_all_phase24_schtasks.py | 6 schtask 一括登録 |
| register_youtube_schtask.py | YouTube schtask |
| register_paddock_pipeline_schtask.py | paddock schtask |
| morning_briefing_5_17.py | 朝 06:30 統合 briefing |
| rehearsal_5_17.py | 全 chain リハーサル |
| check_video_sources.py | Phase 22-24 source 健全性 |
| phase23_smoke_test.py | 10 項目 verify |
| daily_phase23_impact_report.py | 当日 V15 vs Shadow 比較 |
| shadow_log_aggregator.py | 過去 log 集計 |

### I. その他 scraper / data
| tool | 機能 |
|------|------|
| bulk_scrape_stable_comments_v2.py | 厩舎コメント拡張 |
| bulk_scrape_expert_marks.py | 専門家 印 |
| scrape_jra_payouts_v2.py | 払戻 4/6 停止 復旧 |
| scrape_amedas_1min.py | アメダス 1 分粒度 |
| scrape_jra_finish_photos.py | 入線写真 skeleton |
| odds_5min_capture.py | 5 分 interval オッズ |
| closing_odds_drift.py | drift detection PoC |
| v20_full_features_auc_verify.py | LGB 単独 AUC 0.817 |
| v20_training_data_full_builder.py | 全 features merge → 学習 ready |
| train_v20_skeleton.py | V20 学習 spec |

## 📂 既存 tools (Phase 21D 以前、 V15 production 関連)

主要 production tools (V15 投資保護 で 不変):
- daily_predict.py / daily_results.py / race_auto_notify.py / morning_go_check.py
- refresh_cookie.py (拡張: NETKEIBA_EMAIL fallback + cookies.json export)
- weekly_report.py / project_status.py / etc.

## 📊 Phase 21D-25 統計

- 新規 tools: 75+
- 新規 docs: 11
- 新規 commits: 60+
- 既存 V15 production: 完全不変
- 検証 features: 40+
- 発見 patterns: 4-way Jackpot (top3 64.8%) + 5-way (73.9%) + 父馬 boost

## 🚀 5/12+ user action 優先 順

1. **5/12 月** admin: register_all_phase24_schtasks.py
2. **5/12 月** 32-bit Python: jvlink_parser.py --test-com
3. **5/12 月** 32-bit Python: jvlink_movie_wrapper.py --probe
4. **5/13 火** scrape: bulk_scrape_stable_comments_v2.py + expert_marks
5. **5/14 水** paddock archive build: paddock_weekend_archive_build.py 20260504 20260510
6. **5/15 木** JRA RV web: jra_racing_viewer_capture.py --probe (login flow 確認)
7. **5/16 金** rehearsal: rehearsal_5_17.py
8. **5/17 土** 本番: V15 案 B 改 + 戦略⑦ 単独継続

## V15 投資保護 (12h marathon 全期間 絶対遵守、 確認済)

predict_core / daily_predict / app.py / V15 model `.pkl.gz` ALL **不変**。
全 75+ Phase 21D-25 tools は new file / post-process / 検証 / 分析、 production 影響 0。
