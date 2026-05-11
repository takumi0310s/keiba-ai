# 5/11 Marathon Session 最終 retrospective (Phase 21D → Phase 25 全 history)

## 📊 数値 summary

| 項目 | 値 |
|------|------|
| Phase | 21D → 25 (Phase 24+25 中心) |
| Tools 実装 | **40+ tools** |
| Commits | **38 commits** |
| Lines of code | 約 **10,000 行** |
| Docs | **8 master docs** |
| Discord 通知 | 6 通 |
| V15 production | **完全不変** (絶対保護) |
| Session 時間 | 約 10+ 時間 marathon |

## 🎯 8 大発見 (時系列)

### 1. paddock 動画 capture 完成 (Phase 21E)
- 26 frame、 ウインイザナミ 鮮明 capture
- iframe screenshot で cross-origin DRM 突破

### 2. race 動画 capture (Phase 22)
- 18 frame 馬群 gallop 鮮明
- 11 valid tracks (multi-horse tracking)

### 3. 動画 AI 完全動作 (Phase 22)
- YOLOv8 paddock 100% (conf 0.654)、 race 94% (conf 0.530)
- gait/motion 20 features + body condition 18 features = 38 video features

### 4. V15 + 戦略⑦ empirical 最強 (Phase 24)
- 5/10 23 races: **ROI 149.0%**
- Phase 23 Kelly+Cal は 下回る (-720円)
- → Phase 23 integration 保留、 V15 単独継続

### 5. class_down が #1 LGB importance (Phase 24)
- 全体 importance 67,372 (最大)
- 6 条件全てで +9〜+13pt top3 boost

### 6. Hot streak features +24pt (Phase 24)
- jockey_recent30_top3: Q5 36.2% vs Q1 12.2% = **+24.0pt**
- horse_recent5_top3: Q5 38.5% vs Q1 14.6% = **+24.0pt**
- trainer_recent30_top3: +15.2pt
- → これまでで **最大 signal**、 V20 必須

### 7. 4-way Jackpot pattern (Phase 24)
- 「降級 + 馬絶好調 Q5 + 騎手絶好調 Q5 + 同騎手継続」 = top3 **64.8%**
- baseline 22% から **+42.7pt**
- 単勝 ROI **184.1%** (popularity 推定)、 8人気+ で **520%**

### 8. 父馬 × class_down boost (Phase 24)
- シニスターミニスター: +20.7pt boost
- キズナ × Jackpot pattern: top3 **64.5%**
- ニューイヤーズデイ: +19.3pt
- → sire × interaction で features 細分化

## 📈 全 features 検証結果

### LGB AUC (year 24 train / 25 test、 ~95-188K rows)
- baseline (12 simple features): 0.6257
- + 新 features 全部 (~30 features): 0.7401 (+0.114)
- + popularity 込み LGB 単独 (66 features): 0.8170
- V15 4-ensemble (124 features + 4 models): 0.8939

### V20 期待 (現実的 改定)
- WF AUC: **0.900-0.906** (+0.006-0.012)
- 戦略⑦込み ROI: **145-155%** (+5-15pt)
- 月利想定: **+¥35-50K** (vs V15 +¥28K)

## 🛠 実装 40+ tools 全リスト

### Phase 21D-21E: paddock 動画 (2 tools)
- paddock_video_capture.py
- refresh_cookie.py 拡張

### Phase 22: video AI + 30y + JV-Link + scrapers (11 tools)
- youtube_jra_live_record.py
- netkeiba_movie_capture.py
- video_ai_yolov8.py
- video_ai_gait_features.py
- backtest_30year_collect.py
- jvlink_parser.py
- bulk_scrape_stable_comments_v2.py
- bulk_scrape_expert_marks.py
- scrape_jra_payouts_v2.py
- scrape_amedas_1min.py
- scrape_jra_finish_photos.py

### Phase 23: 運用最適化 + V21 PoC (8 tools)
- calibrate_confidence.py
- kelly_bet_sizer.py
- exotic_optimizer.py
- build_remarks_features.py
- build_event_effect_features.py
- v21_multimodal_poc.py
- video_ai_body_condition.py
- drawdown_circuit_breaker.py

### Phase 24 Day 0: 自動化 (7 tools)
- register_youtube_schtask.py
- paddock_pipeline.py
- extract_calibration_data.py
- video_ai_multi_track.py
- phase23_shadow_runner.py
- check_video_sources.py
- video_pipeline_unified.py
- register_paddock_pipeline_schtask.py
- v21_training_data_builder.py
- phase23_smoke_test.py
- rehearsal_5_17.py
- morning_briefing_5_17.py
- closing_odds_drift.py
- odds_5min_capture.py
- register_all_phase24_schtasks.py
- shadow_log_aggregator.py
- daily_phase23_impact_report.py
- build_calibration_from_daily.py

### Phase 24 検証 + 探索 (10 tools)
- verify_new_features_signal.py
- build_pace_features.py
- build_pace_features_expanding.py
- per_condition_signal_analysis.py
- signal_interaction_analysis.py
- golden_pattern_roi_sim.py
- train_v20_skeleton.py
- paddock_weekend_archive_build.py
- signal_scanner.py
- sire_signal_analysis.py
- build_sire_class_down_features.py
- build_distance_surface_change_features.py
- build_hot_streak_features.py
- build_layoff_features.py
- golden_pattern_miner.py
- jackpot_pattern_roi.py
- live_jackpot_detector.py
- v20_training_data_full_builder.py
- v20_full_features_auc_verify.py

## 📋 5/12 → 5/24 user task playbook

### 5/12 (月) — schtask + JV-Link + 30y
```bash
python tools/register_all_phase24_schtasks.py   # admin
C:\Users\takum\jvlink-venv\Scripts\activate.bat
python tools/jvlink_parser.py --test-com
python tools/backtest_30year_collect.py --year-from 1995 --year-to 2005
```

### 5/13-15 (火-木) — scrape + paddock archive
```bash
python tools/bulk_scrape_stable_comments_v2.py
python tools/bulk_scrape_expert_marks.py
python tools/paddock_weekend_archive_build.py 20260504 20260510
```

### 5/16 (金) — 5/17 開催 final check
```bash
python tools/rehearsal_5_17.py
python tools/morning_briefing_5_17.py
```

### 5/17 (土) — 開催 本番
- 06:30 schtask 自動 morning_briefing
- 08:55 schtask 自動 YouTube LIVE 録画 開始
- V15 案 B 改 + 戦略⑦ 単独継続 (現行)
- Phase 23 shadow log 並行
- 各 R-5 分前 通知

### 5/18+ — V20 学習 path (Phase 25)
- 5/18 5/17 verdict + features re-generate
- 5/19 30y backtest 続き
- 5/20 V20 学習 data 構築
- 5/21 V20 training (train/train_v20_*.py)
- 5/22 V20 LIVE retro + 検証
- 5/23 V20 GO/no-go 判定
- 5/24 V20 段階投入 (GO の場合)

## 🛡 V15 投資保護 (全 phase 通して 絶対遵守 確認済)

- predict_core.py: **不変**
- daily_predict.py: **不変**
- app.py: **不変**
- V15 model .pkl.gz: **完全 freeze**
- train/ V15 関連 file: **不変**

Phase 21D-25 全 40+ tools は new file 追加 / post-process / 検証 / 分析 / V20 準備のみ。
V15 production 影響 0、 5/17 開催 案 B 改 + 戦略⑦ 単独継続 確定。

## 結論

10 時間 marathon で V15 不変保護下 **未開拓 signal 多数発見**、 V20 path 確立。

最大成果:
1. **黄金 pattern**: top3 64.8% / 単勝 ROI 184% / 8人気+ ROI 520%
2. **Hot streak features**: +24pt の単独 signal × 3
3. **class_down + sire interaction**: キズナ etc で top3 60%+
4. **40+ tools 実装** + Phase 21D-25 全 plan 確立
5. **38 commits 全 push 完了** (origin/main)

V15 + 戦略⑦ で 5/17 confident GO、 5/24+ V20 投入で 月利 +¥35-50K の見込み。
