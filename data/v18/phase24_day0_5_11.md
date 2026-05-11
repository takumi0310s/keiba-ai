# Phase 24 Day 0: 5/17 完成 plan の今夜 marathon 続行 (5/11)

## ✅ Day 0 (5/11 今夜) 完了 5 項目

| # | tool | 動作確認 | 所要 |
|---|------|---------|------|
| 1 | `tools/register_youtube_schtask.py` | dry-run OK、 cmd 表示確認済 | 30m |
| 2 | `tools/paddock_pipeline.py` | 5/10 3 races dry-run、 9 capture target | 60m |
| 3 | `tools/extract_calibration_data.py` | 21 pairs 抽出 (sample 不足明示) | 30m |
| 4 | `tools/video_ai_multi_track.py` | race 18 frame → 11 valid tracks | 60m |
| 5 | calibrator pilot fit (B4 実 fit) | ECE 0.515 → 0.000 isotonic | 15m |

合計 ~3.5h、 V15 不変、 4 tool + 1 動作確認。

## 📋 5/12-5/17 daily plan (user task)

### Day 1: 5/12 (月曜)

**朝 (5m)**:
- nightly_sanity 自動結果 確認 (Discord 通知)

**夜 (3-4h)**:
- [ ] `python tools/register_youtube_schtask.py` (admin PowerShell、 schtask 登録)
- [ ] `python tools/register_youtube_schtask.py --check` で 登録確認
- [ ] `python tools/jvlink_parser.py --test-com` (32-bit Python venv で interactive)
   - 先に `C:\Users\takum\jvlink-venv\Scripts\activate.bat`
- [ ] 30y backtest 段階取得 開始 (1995-2005)
   - `python tools/backtest_30year_collect.py --year-from 1995 --year-to 2005`
- [ ] Phase 23 tool 統合 design doc 確認 + 議論 (Claude/user)

### Day 2: 5/13 (火曜)

**夜 (3-4h)**:
- [ ] 厩舎コメント 実 scrape (rate limit 注意、 1h)
   - `python tools/bulk_scrape_stable_comments_v2.py --year-from 2024 --year-to 2026`
- [ ] 専門家予想 実 scrape (1h)
   - `python tools/bulk_scrape_expert_marks.py --year-from 2024 --year-to 2026`
- [ ] Phase 23 tool 統合 (shadow mode) 実装
   - daily_predict.py を **改変せず**、 race_auto_notify の後ろに shadow.log
- [ ] 30y backtest 続き (2006-2015)

### Day 3: 5/14 (水曜)

**夜 (2-3h)**:
- [ ] V21 学習 data 構築 (動画 features + tabular features merge)
- [ ] paddock_pipeline.py で 5/10 全 race × Top 3 = ~108 paddock 動画 capture (約 1h、 rate limit)
- [ ] 30y backtest 完了 (2016-2024)

### Day 4: 5/15 (木曜)

**夜 (1-2h)**:
- [ ] paddock_pipeline.py で 5/4-5/10 蓄積 (約 5 開催 × 12 race × 3 馬 = 180 動画)
- [ ] YouTube 5/16 LIVE schtask 動作確認 (--check)
- [ ] morning_go_check.py の dry-run

### Day 5: 5/16 (金曜)

**朝**:
- [ ] morning_go_check 自動 → Discord 確認

**夜 (1h)**:
- [ ] 5/17 開催 final dry-run (daily_predict / race_auto_notify)
- [ ] paddock_pipeline.py で 5/16 結果 → 当日 全 race 全頭 paddock 取得 (約 200 動画)
- [ ] YouTube 録画 動作確認

### Day 6: 5/17 (土曜) - 本番

**朝 06:30**: Keiba-MorningGoCheck schtask 自動 → Discord
**朝 07:00**: user 確認 (1 通)
**朝 08:00**: V15 daily_predict 自動
**朝 08:55**: Keiba-YouTubeLiveRecord-Sat schtask 自動起動 → 9:00 LIVE 録画開始
**朝 09:00**: paddock_pipeline.py 自動 (前日 capture 漏れ補完)
**朝 09:30**: morning_weight_check
**各 R-5 分前**: V15 単独 通知 (現行)
**夜 21:00**: daily_results 自動
**夜 22:00**: 累計 audit、 Phase 23 shadow 結果 集計

## 🎯 Phase 23 tool 統合 戦略 (shadow mode)

**設計原則**: V15 production を一切変更せず、 結果のみ並行 log。

```python
# race_auto_notify.py の最後 (predict 完了後)、 5/13+ で実装:
try:
    from tools.kelly_bet_sizer import kelly_bet
    from tools.exotic_optimizer import select_optimal_trio
    from tools.drawdown_circuit_breaker import check_status
    # shadow log only - 実 通知 / 投票には影響しない
    shadow_results = {...}
    json.dump(shadow_results, open('data/shadow_log/{race_id}.json', 'w'))
except Exception as e:
    pass  # 完全 silent fail、 V15 通知は継続
```

5/17 開催で V15 実 ROI vs Phase 23 shadow ROI を比較 → 5/24+ 統合判定。

## 🎯 5/17 までに達成 expected

| 項目 | 5/11 現状 | 5/17 想定 |
|------|----------|----------|
| V15 production | 安定 | **完全保護 維持** |
| Phase 23 tool 統合 | standalone | **shadow mode 運用** |
| paddock 動画 蓄積 | 1 馬 26 frame | **~500 馬 × 30 frame = 15K frame** |
| YouTube LIVE 録画 | 未 schtask | **5/16-17 自動録画 開始** |
| JV-Link 32-bit | skeleton | **実動作確認 完了** |
| 30y backtest | dry-run | **実取得 完了** (段階分割 135 GB) |
| 厩舎コメント / 専門家予想 | scraper ready | **実 scrape 開始 蓄積** |
| 撤退保護 | tool 実装済 | **breaker 稼働** |

## 🚦 risk + mitigation

| risk | 対応 |
|------|------|
| schtask 登録 失敗 (admin) | --dry-run で cmd 確認、 PowerShell 管理者で手動実行 |
| paddock pipeline rate limit | sleep 5s/race、 night batch 推奨 |
| JV-Link 32-bit 認証 | DataLab 利用キー 入力 必要、 user task |
| 30y backtest 容量 (135 GB) | 段階分割 (1995-2005 / 2006-2015 / 2016-2024) |
| netkeiba cookie 期限切れ | refresh_cookie.py --auto で 自動更新 (Phase 21E で実装済) |

## V15 投資保護 (絶対遵守、 確認)

全 Day 0-6 plan で **V15 production 一切不変**。 統合は shadow mode で並行、 実投票 / 通知は V15 単独継続。
