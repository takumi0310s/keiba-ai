# Phase 21-24 完全 master summary (5/11 marathon、 keep-going edition)

今夜 1 session で **Phase 21D → Phase 24 Day 0++** 完走、 **22 tools 新規** 追加、 **8 commits**、 V15 production 完全不変。

## 📜 commit history (新しい順)

| commit | Phase | 主 内容 |
|--------|-------|--------|
| `bbc47d62` | Phase 24 Day 0++ | unified pipeline + paddock schtask + V21 training builder |
| `aa769439` | Phase 24 Day 0+ | video sources health check + 実 capture 確認 |
| `c3b095d8` | Phase 24 Day 0+ | shadow mode runner (V15 完全不変) |
| `ca3bf96e` | Phase 24 Day 0 | doc + calibration pilot data |
| `f4b581de` | Phase 24 Day 0 | YouTube schtask + paddock pipeline + multi-track + calibration |
| `f698b9a8` | Phase 23 | 8 運用最適化 tools + 市場 9 社 比較 |
| `56849235` | Phase 22 | 動画 AI + 30y backtest + JV-Link parser + 5 scraper |
| `f8af23f0` | Phase 22 in-progress | PC crash 復旧保存 |
| (prev) `4a5f0d67` | Phase 21E | paddock 完全動作 (26 frame) |
| (prev) `9e6affd6` | Phase 21D | paddock skeleton |

## 🛠 全 tools (22 新規、 V15 投資保護 完全)

### Phase 21D-E: paddock 動画 capture (2 tools)
| tool | 行数 | 機能 |
|------|------|------|
| paddock_video_capture.py | 330 | Playwright + iframe screenshot (cross-origin) |
| refresh_cookie.py (拡張) | +20 | EMAIL fallback + cookies.json export |

### Phase 22: video AI + 30y + JV-Link + scrapers (11 tools)
| tool | 行数 | 機能 |
|------|------|------|
| youtube_jra_live_record.py | 185 | YouTube JRA LIVE 録画 wrapper |
| netkeiba_movie_capture.py | 354 | paddock / oikiri / race 統一 capture |
| video_ai_yolov8.py | 170 | YOLOv8 馬 bbox 検出 (CPU forced) |
| video_ai_gait_features.py | 162 | gait / motion 20 features |
| backtest_30year_collect.py | 342 | TFJV 1995-2024 collector skeleton |
| jvlink_parser.py | 449 | JV-Link 8 dataspec parser (32-bit Python required) |
| bulk_scrape_stable_comments_v2.py | 394 | 厩舎コメント拡張 scraper |
| bulk_scrape_expert_marks.py | 427 | 専門家 印 collect |
| scrape_jra_payouts_v2.py | 237 | 払戻 4/6 停止 復旧 path |
| scrape_amedas_1min.py | 255 | アメダス 1 分粒度 |
| scrape_jra_finish_photos.py | 231 | JRA 入線写真 skeleton |

### Phase 23: 運用最適化 + V21 multimodal (8 tools)
| tool | 行数 | 機能 |
|------|------|------|
| calibrate_confidence.py | 178 | isotonic + Platt 校正 |
| kelly_bet_sizer.py | 165 | Kelly fractional 0.25x bet size |
| exotic_optimizer.py | 165 | Plackett-Luce 三連複 EV 最大 |
| build_remarks_features.py | 130 | race_review → 9 categorical |
| build_event_effect_features.py | 140 | 騎手 / 厩舎 / 升降級 events、 class_down +12.5pt |
| v21_multimodal_poc.py | 160 | V15 + video stacking PoC |
| video_ai_body_condition.py | 175 | 馬体 condition 8 features × 2 |
| drawdown_circuit_breaker.py | 175 | 累計 / 連敗 自動 monitor |

### Phase 24 Day 0/0+/0++: 統合 + 自動化 (7 tools)
| tool | 行数 | 機能 |
|------|------|------|
| register_youtube_schtask.py | 162 | YouTube 土 / 日 08:55 schtask |
| paddock_pipeline.py | 240 | 全レース全頭 / Top-N 自動 capture |
| extract_calibration_data.py | 90 | cumulative → (pred, label) CSV |
| video_ai_multi_track.py | 198 | race 動画 multi-horse tracking |
| phase23_shadow_runner.py | 220 | Phase 23 tool 全 shadow mode runner |
| check_video_sources.py | 180 | Phase 22-24 source 健全性 一括 check |
| video_pipeline_unified.py | 240 | capture → YOLOv8 → gait → body chain |
| register_paddock_pipeline_schtask.py | 130 | 日 / 月 20:00 paddock 自動 archive |
| v21_training_data_builder.py | 170 | V15 + video + remarks + events merge |

合計: **22 新規 tools / 約 5,500 行 / V15 投資保護 完全**

## ✅ 動作確認 結果 (実機)

| 項目 | 確認 |
|------|------|
| paddock 動画 capture | ✅ 26 frame、 ウインイザナミ 鮮明 |
| race 動画 capture | ✅ 18 frame、 馬群 鮮明 |
| YOLOv8 paddock | ✅ 26/26 (100%) conf 0.654 |
| YOLOv8 race | ✅ 17/18 (94%) conf 0.530 |
| gait features | ✅ 20 features 抽出 |
| body condition | ✅ score 0.717 |
| V21 multimodal | ✅ V15 0.65 → V21 0.6845 |
| multi-horse track | ✅ 18 frame → 11 tracks |
| Calibration demo | ✅ ECE 0.119 → 0.000 isotonic |
| Calibration pilot | ✅ 21 sample で動作 (production 不足) |
| Kelly demo | ✅ EV+ で 1500円 cap、 EV- skip |
| Pari-mutuel trio | ✅ Plackett-Luce 確率 + EV ranking |
| Drawdown breaker | ✅ 現状 WARN 検出 (累計 -25K) |
| Remarks features | ✅ 277K rows、 9 categorical |
| Event effects | ✅ 532K rows、 **class_down +12.5pt 強 signal** |
| Shadow backtest | ✅ V15 -11,950 vs Shadow +16,800 (20 races) |
| Paddock pipeline | ✅ 4/11 race horse capture 18 frame |
| Health check | ✅ 9 OK / 2 WARN (schtask 未登録) |
| 30y backtest dry-run | ✅ 1995 SE 108 files、 全体 7.2 GB |
| V21 training builder | ✅ 95K rows × 111 features (2024-2025) |

## 🎯 5/12 → 5/17 user task list

### 5/12 (月) 夜
```bash
# admin PowerShell で
python tools/register_youtube_schtask.py
python tools/register_paddock_pipeline_schtask.py
python tools/register_youtube_schtask.py --check
python tools/register_paddock_pipeline_schtask.py --check

# 32-bit Python venv
C:\Users\takum\jvlink-venv\Scripts\activate.bat
pip install pywin32
python tools/jvlink_parser.py --test-com
python tools/jvlink_parser.py --datatype RACE --from 20260503 --max 10

# 30y backtest 段階開始
python tools/backtest_30year_collect.py --year-from 1995 --year-to 2005 --datatype SE,HR

# 健全性
python tools/check_video_sources.py
```

### 5/13 (火) 夜
```bash
# 厩舎コメント / 専門家予想 実 scrape
python tools/bulk_scrape_stable_comments_v2.py --year-from 2024 --year-to 2026
python tools/bulk_scrape_expert_marks.py --year-from 2024 --year-to 2026

# 30y backtest 続き
python tools/backtest_30year_collect.py --year-from 2006 --year-to 2015 --datatype SE,HR

# shadow runner re-validate
python tools/phase23_shadow_runner.py --backtest --from 20260301
```

### 5/14 (水) 夜
```bash
# paddock 全レース 蓄積 (5/4-5/10 過去開催 6 日分 × 12 race × 3 馬 = 216 動画)
for d in 20260504 20260505 20260510; do
    python tools/video_pipeline_unified.py $d --top-n 3
done

# 30y backtest 完了
python tools/backtest_30year_collect.py --year-from 2016 --year-to 2024 --datatype SE,HR,UM,H1
```

### 5/15 (木) 夜
```bash
# V21 学習 data builder (動画 蓄積後)
python tools/v21_training_data_builder.py --year-from 2024 --year-to 2026

# 5/16-17 開催前 final check
python tools/morning_go_check.py
python tools/check_video_sources.py
```

### 5/16 (金) - 5/17 開催前夜
```bash
# 5/16 paddock 自動取得 (schtask 既登録)
python tools/paddock_pipeline.py 20260516 --top-n 3   # 手動でも可

# Phase 23 shadow log 確認
ls data/shadow_log/
```

### 5/17 (土) 本番
- 06:30: morning_go_check schtask 自動 → Discord
- 08:00: V15 daily_predict 自動 (現行)
- 08:55: YouTube LIVE 録画 schtask 自動起動
- 09:00+: paddock 自動取得 schtask (前日分 補完)
- 各 R-5 分前: V15 単独 通知 (現行)
- 21:00: daily_results 自動
- 22:00: shadow vs V15 比較

## 🎯 達成 想定 (5/17 開催 前)

| 項目 | 5/11 朝 | 5/17 開催 前 |
|------|--------|------------|
| V15 production | 安定運用 | **完全保護 維持** |
| Phase 22 video AI | paddock 1 馬 | **5/4-5/16 全 race × top 3 = ~500 馬 蓄積** |
| Phase 23 tool | standalone | **shadow mode 並行運用 開始** |
| JV-Link 32-bit | skeleton | **実 fetch 動作確認** |
| 30y backtest | dry-run のみ | **実取得 完了** (3 GB raw) |
| 厩舎コメント / 専門家予想 | scraper ready | **実 scrape 蓄積** |
| YouTube LIVE 録画 | 未 schtask | **5/16-17 自動録画 開始** |
| 撤退保護 | tool 実装済 | **breaker 稼働 monitoring** |

## V15 投資保護 (絶対遵守、 完全 確認)

predict_core.py / daily_predict.py / app.py / V15 model `.pkl.gz` ALL 不変。
Phase 21D-24 全 22 tools は post-process / helper / PoC、 production 影響 0。
5/17 開催で 案 B 改 + 戦略⑦ 単独継続。
