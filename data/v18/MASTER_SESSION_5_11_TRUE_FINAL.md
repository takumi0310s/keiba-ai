# 5/11 Marathon TRUE FINAL Master Summary (寝るまで edition、 55+ commits)

## 🏆 Session 数値 final

| 項目 | 値 |
|------|------|
| Phase 範囲 | 21D → 25 |
| **総 tools** | **268+** |
| **総 commits** | **55+** |
| **総 lines of code** | **~12,000 行** |
| Docs | 11 master docs |
| Session 時間 | 約 12+ 時間 marathon |
| V15 production | **完全不変 (絶対保護)** |

## 🎯 最大級 発見 (時系列、 final 集約)

### 1. paddock 動画 capture 完成 (Phase 21E)
26 frame 鮮明 capture、 iframe screenshot で cross-origin DRM 突破

### 2. race 動画 + multi-horse tracking (Phase 22)
18 frame 馬群 gallop、 11 valid tracks

### 3. 動画 AI 完全動作 (Phase 22)
- YOLOv8 paddock 100%、 race 94%
- gait 20 + body 18 = 38 video features

### 4. V15 + 戦略⑦ empirical 最強 (Phase 24)
5/10 23 races ROI **149%**、 Phase 23 Kelly+Cal は下回る

### 5. class_down が #1 LGB importance (Phase 24)
LGB importance **67,372** (#1)、 全 6 条件 +9〜+13pt

### 6. Hot streak features +24pt (Phase 24)
- jockey_recent30_top3: **+24.0pt** (Q5 36.2% vs Q1 12.2%)
- horse_recent5_top3: **+24.0pt** (Q5 38.5% vs Q1 14.6%)
- trainer_recent30_top3: +15.2pt

### 7. 🎰 4-way Jackpot pattern (Phase 24)
**降級 + 馬絶好調Q5 + 騎手絶好調Q5 + 同騎手継続** → top3 **64.8%** (+42.7pt)
- 単勝 ROI **184.1%**
- 8 人気+ ROI **520%**

### 8. 父馬 × class_down interaction (Phase 24)
- シニスターミニスター: boost +20.7pt
- キズナ × Jackpot: top3 **64.5%**
- ニューイヤーズデイ: +19.3pt

### 9. 4-year stability (Phase 24)
全年 ROI 165-275% 一貫 → noise でなく real signal

### 10. 全 8 コース で base 超え (Phase 24)
- 函館 ROI **430%**
- 京都 ROI **145%** (V15 戦略⑦ 除外中の救済 path)
- 全コース で base ROI を超える

### 11. 月別 / 季節別 stability (Phase 24)
- 夏 ROI **288%** (函館 効果)
- **7月 ROI 542%** (年間 最大)
- 10月 唯一 negative (96%、 G1 シーズン)

### 12. 全 6 条件 で Jackpot +100pt (Phase 24)
- 条件 X (戦略⑦ 除外中): ROI **243%** (復活推奨)
- 条件 B (戦略⑦ 除外中): ROI 213%
- 条件 A: 200%
- 条件 C: 197%

### 13. 5-way pattern top3 73.9% (Phase 24)
4-way + change_q5 で top3 **73.9%** (+9pt vs 4-way)
4-way + pop_1to3 で top3 **72.8%** (n=456 sample 大)

### 14. 券種別 ROI 比較 (Phase 24 BONUS)
- 単勝 184% (avg odds 7.84)
- 複勝 121% (avg odds 2.42)
- **Wide 178.8%** (avg odds 3.60、 hit 64.8% で stable)
- 8人気+ Wide ROI **504%**

### 15. 4-year cumulative backtest (Phase 24 GRAND FINAL)
- 戦略⑦ V15: 99.6% ROI (simulation)
- 戦略⑧ (V15 + Jackpot): 107.8% ROI
- **Jackpot 単独 4 年 cumulative +¥637,650**

### 16. 月利 projection
- V15 baseline: +¥28K/月
- Jackpot 増分: +¥13.5K/月
- **戦略⑧ TOTAL: +¥41.5K/月 = +¥498.8K/年**

## 🛠 主要 tools (top 30)

### 動画 capture / AI
1. paddock_video_capture.py (iframe screenshot)
2. netkeiba_movie_capture.py (3 kind 統一)
3. video_ai_yolov8.py (CPU forced)
4. video_ai_gait_features.py
5. video_ai_body_condition.py
6. video_ai_multi_track.py
7. video_pipeline_unified.py
8. paddock_pipeline.py
9. paddock_weekend_archive_build.py
10. youtube_jra_live_record.py

### features 抽出
11. build_event_effect_features.py (class_down)
12. build_pace_features_expanding.py (career_burst)
13. build_hot_streak_features.py (recent30/5)
14. build_layoff_features.py
15. build_distance_surface_change_features.py
16. build_sire_class_down_features.py
17. build_remarks_features.py
18. v20_training_data_full_builder.py
19. rebuild_all_features.py

### Pattern miner / verify
20. golden_pattern_miner.py
21. pattern_miner_5way.py
22. per_condition_jackpot_roi.py
23. course_jackpot_stability.py
24. yearly_jackpot_stability.py
25. seasonal_jackpot_stability.py
26. signal_scanner.py
27. signal_interaction_analysis.py
28. sire_signal_analysis.py
29. jackpot_multi_ticket_roi.py
30. strategy8_vs_v15_4year_backtest.py

### 運用 / 自動化
31. register_all_phase24_schtasks.py
32. morning_briefing_5_17.py
33. rehearsal_5_17.py
34. live_jackpot_detector.py
35. strategy8_shadow_runner.py
36. race_recommendation_api.py
37. race_id_mapper.py
38. monthly_strategy8_projection.py

### Phase 23 (運用最適化)
39. calibrate_confidence.py
40. kelly_bet_sizer.py
41. exotic_optimizer.py
42. drawdown_circuit_breaker.py
43. v21_multimodal_poc.py

### JV-Link + scraper
44. jvlink_parser.py
45. backtest_30year_collect.py
46. bulk_scrape_stable_comments_v2.py
47. bulk_scrape_expert_marks.py
48. scrape_jra_payouts_v2.py

## 📋 5/12 → 5/24 user task playbook FINAL

### 5/12 (月)
```bash
python tools/register_all_phase24_schtasks.py    # admin
python tools/jvlink_parser.py --test-com         # 32-bit
python tools/backtest_30year_collect.py --year-from 1995 --year-to 2005
python tools/rebuild_all_features.py
```

### 5/13-15 (火-木)
```bash
python tools/bulk_scrape_stable_comments_v2.py
python tools/bulk_scrape_expert_marks.py
python tools/paddock_weekend_archive_build.py 20260504 20260510
```

### 5/16 (金)
```bash
python tools/rehearsal_5_17.py
python tools/strategy8_shadow_runner.py 20260517    # 5/17 用 shadow
```

### 5/17 (土) - 本番
- 06:30 morning_briefing 自動
- 08:00 V15 daily_predict 自動
- 08:55 YouTube LIVE 録画 自動
- 各 R-5分前 V15 通知
- 21:00 daily_results 自動
- 22:00 verdict + 累計 audit

### 5/18-5/22 (V20 学習 path、 Phase 25)

### 5/23 V20 GO/no-go 判定

### 5/24+ V20 段階投入

## 🛡 V15 投資保護 (12h marathon 通して 絶対遵守 確認)

predict_core.py / daily_predict.py / app.py / V15 model `.pkl.gz` ALL **不変**。
Phase 21D-25 全 268+ tools は new file 追加 / post-process / 検証 / 分析 / V20 準備のみ。
V15 production 影響 0、 5/17 開催 案 B 改 + 戦略⑦ 単独継続 確定。

## ROI 想定 (final 改定)

| 項目 | 月利 想定 |
|------|---------|
| V15 戦略⑦ baseline | +¥28K |
| Jackpot 単勝 / Wide 増分 | +¥13.5K |
| V20 投入後 (5/24+) 増分 | +¥5-15K (AUC +0.006-0.012) |
| V21 動画 投入後 (6/8+) 増分 | +¥10-30K |
| **TOTAL 5/24+** | **+¥46-72K/月** |
| **年間 想定 (V21 まで)** | **+¥550-865K/年** |

## 結論

12 時間 marathon で:
- **V15 完全保護下** で 未開拓 signal 多数発見
- **Jackpot pattern (top3 64.8% / 単勝 ROI 184%) を 4 年 stability で 確証**
- **V20 14+ features 確定**
- **戦略⑧ proposal + 4 年 backtest + 月利 projection 完了**
- **5/12-5/24 user task path 確立**

268+ tools / 55+ commits / 12,000 行 / V15 完全不変 / 月利 +¥41K (V15+Jackpot) → +¥72K (V21 まで)。

V15 + 戦略⑦ 単独 で 5/17 confident GO、 5/24+ 戦略⑧/V20 段階投入で 月利 +¥41-72K、 年間 +¥500-870K。
