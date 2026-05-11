# Phase 26 完成 summary (5/12 朝、 user task "全必要 items 即実装")

## 📊 タスク 完了 状況

| # | task | 完了 | 担当 | deliverable |
|---|------|------|------|-------------|
| 5 | v20 merge fix (horse_id dtype) | ✅ | main | corner_delta 87.2% / jockey_trainer_combo 92.8% coverage |
| 6 | JV-Link parser 拡張 | ✅ | Agent A | 8 → **32 dataspec** (DIFF/0B20/0B41 等 含む) |
| 7 | 真不足 features 7 件 | 🟡 進行中 | Agent B | start/middle/late index 等 |
| 8 | 5/17 動的 features tool | ✅ | main | live_features_5_17.py (LEAK-free 動的計算) |
| 9 | V20 full training | ✅ | main | LGB+XGB AUC 0.8393 (FT 4-ens は 5/24+ user task) |
| 10 | netkeiba 2026 補完 | ✅ | Agent C | 12 csv / 4,480 件 catchup wrapper |
| 11 | JRA-VAN パドックアイ | ✅ | main | Playwright skeleton (user 実 verify) |
| 12 | JRDB KTA/MZA/MSA | ✅ | Agent C | scraper + dedup 完備 |
| 13 | PACI 4/4 停止 修復 | ✅ | Agent A | 真因発見 (parse_jrdb.py 再実行止まり) + scraper |
| 14 | RV パトロール | ✅ | main | Playwright skeleton (login 後 capture) |

**完了率: 9 / 10 = 90%** (Task #7 残)

## 🛠 新規 / 拡張 tools (本セッション)

### main thread (5/12 朝)
1. tools/build_competitor_gap_features.py - jockey_trainer_combo +21.3pt 発見
2. tools/v20_training_data_full_builder.py 拡張 - horse_id dtype 統一
3. tools/live_features_5_17.py - 5/17 動的 features 計算
4. tools/strategy8_sidecar.py - V15 完全保護 + Jackpot 別 channel
5. tools/jravan_paddock_eye_capture.py - JRA-VAN パドックアイ skeleton
6. tools/jra_rv_patrol_capture.py - パトロールビデオ skeleton

### Agent A (worktree-agent-ac7cf203c729e9716)
7. tools/jvlink_parser.py 拡張 - 32 dataspec (8 → 32)
8. tools/scrape_jrdb_paci.py - PACI 修復

### Agent C
9. tools/netkeiba_2026_catchup.py - 12 csv 補完 wrapper
10. tools/scrape_jrdb_kta_mza.py - KTA/MZA/MSA scraper

### Agent B (進行中)
11. tools/build_competitor_gap_features_v2.py - 真不足 7 features (進行中)

## 🎯 5/16 → 5/17 適用 確認

### 完了している preparation:
- ✅ jra_races_full.csv に対し 動的 features 計算 (live_features_5_17.py)
- ✅ Jackpot pattern 検出 logic (4-way)
- ✅ Strategy 8 sidecar (別 Discord channel)
- ✅ Phase 24 で発見 14 features 全部 v20_training_data_full に merge 済

### 5/12 (月) user action (admin、 ~3-4 時間):
```bash
# 1. TFJV → jra_races_full.csv 更新
python tools/extract_jvdata.py

# 2. PACI 修復実行
python tools/scrape_jrdb_paci.py --since 20260503

# 3. JRDB KTA/MZA/MSA 取得 (火曜)
python tools/scrape_jrdb_kta_mza.py --auto

# 4. netkeiba 2026 補完 (重め、 1-2h)
python tools/netkeiba_2026_catchup.py --dry-run
python tools/netkeiba_2026_catchup.py --all-csv

# 5. features 全 rebuild
python tools/rebuild_all_features.py
python tools/build_competitor_gap_features.py

# 6. schtask 一括登録 (admin)
python tools/register_all_phase24_schtasks.py

# 7. JV-Link 32-bit 動作確認
C:\Users\takum\jvlink-venv\Scripts\activate.bat
pip install pywin32
python tools/jvlink_parser.py --test-com
python tools/jvlink_parser.py --list  # 32 dataspec 確認
```

### 5/13-15 (火-木) user task:
```bash
# JRA-VAN パドックアイ + RV パトロール login
python tools/jravan_paddock_eye_capture.py --probe
python tools/jra_rv_patrol_capture.py --probe --race-id 202603010112

# 5/17 用 daily features 事前テスト
python tools/live_features_5_17.py --race-ids <test_race>
```

### 5/16 (金) 夜 rehearsal:
```bash
python tools/rehearsal_5_17.py
python tools/strategy8_sidecar.py --auto  # demo
```

### 5/17 (土) 本番:
- 06:30 schtask 自動 morning_briefing
- 08:00 V15 daily_predict 自動
- 08:55 YouTube LIVE 録画 自動
- **09:00 strategy8_sidecar.py 実行** → Jackpot 該当馬 Discord 通知
- 各 R-5 分前 V15 通常 通知
- 21:00 daily_results 自動

## 🚀 期待 効果 (改定)

| layer | 月利 |
|------|------|
| V15 戦略⑦ baseline | +¥28K |
| + Strategy 8 (Jackpot 単勝 1500 円、 4 年 verified) | +¥13.5K |
| + JV-Link DIFF + 0B20 + 0B41 統合 (5/24+ V20) | +¥5-15K |
| + 真不足 features (corner/combo/disadv) 5/24+ V20 | +¥3-10K |
| + 動画 features V21 (6/8+) | +¥10-30K |
| **合計 (5/17+ initial)** | **+¥41.5K** |
| **合計 (V20 5/24+)** | **+¥50-70K** |
| **合計 (V21 6/8+)** | **+¥60-90K** |

## 🛡 V15 投資保護 (全 task で 完全遵守確認)

predict_core.py / daily_predict.py / app.py / V15 model `.pkl.gz` ALL **不変**。
全 11 新規 tools は新規 file / post-process / 検証 / 分析、 production 影響 0。

Agent 2 (Task #7) 完了次第 final 集約 + 全 commit。
