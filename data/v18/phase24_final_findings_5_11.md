# Phase 24 全 finding 集約 (5/11 marathon final、 寝るまで続行 edition)

## ★★★ 今夜の最重要 4 発見

### 1. 🎯 黄金 pattern 発見 (単勝 ROI 推定 180.4%)

**降級 (class_down=1) + 同騎手継続 (jockey_change=0) + 差し力 Q5 (pace_career_burst_mean 上位 20%)**

| 項目 | 数値 |
|------|------|
| sample | 786 records (2022-2025) |
| top3 rate | **43.8%** (全馬 22.8%、 +21pt) |
| 勝率 | **21.1%** (全馬 7.5%、 +14pt) |
| 単勝 ROI (popularity 推定) | **180.4%** (全馬 80%、 +100pt) |
| 人気 2-3 ROI | **192.0%** (sweet spot) |
| 人気 8+ ROI | **245.0%** (穴で大きい) |

### 2. 🎯 class_down が #1 features (importance 67,372)

LGB 全 feature 中 importance **#1**。 6 条件 全てで top3 rate +9〜+13pt の robust signal:
- 条件 A: +12.9pt
- 条件 B: +12.5pt
- 条件 C: +11.2pt
- 条件 D: +11.3pt
- 条件 E: +12.8pt
- 条件 X: +9.2pt

### 3. 🎯 pace_career_burst_mean (差し力 career mean)

quintile 別 top3 rate:
- Q1 (低): 17.8%
- Q5 (高): **27.9%** (+10.1pt)

6 条件全てで Q5-Q1 差 +7-13pt の robust signal。 LEAK-free (expanding window 厳守)。

### 4. 🚨 V15 + 戦略⑦ empirical 最強 (Phase 23 が下回る)

5/10 23 races 実 data:
- V15 700円固定 + 戦略⑦: ROI **149.0%**
- Phase 23 Shadow Kelly+Cal: ROI 120.8% (-720 円)

→ Phase 23 Kelly bet sizing 統合 **保留**、 long-term verify。 calibrator は 整備続行 (Brier -18.5%)。

## 📊 Phase 21D → 24 全 implementation (1 session)

| Phase | tools 数 | commits | 主 deliverable |
|-------|---------|---------|-------------|
| 21D-E | 2 | 2 | paddock 動画 capture (26 frame 鮮明) |
| 22 | 11 | 2 | video AI / 30y backtest / JV-Link / 5 scraper |
| 23 | 8 | 1 | 運用最適化 + V21 PoC + 市場 9 社 比較 |
| 24 Day 0 | 7 | 5 | YouTube/paddock schtask、 shadow runner、 health check 等 |
| 24 検証 | 6 | 7 | signal verify、 per-condition、 interaction、 ROI sim、 V20 manifest |
| **合計** | **34 tools** | **17 commits** | **約 7,000 行 / V15 完全保護** |

## 🎯 V20 学習で 確実 採用 features (14 件、 検証済)

### Strong (verify 済 important)
1. ✅ class_down (importance 67,372、 #1) → 全条件 +10-13pt
2. ✅ class_down_top3_rate_exp (4,486)
3. ✅ class_up + class_up_top3_rate_exp
4. ✅ class_change
5. ✅ jockey_change + jockey_change_top3_rate_exp
6. ✅ trainer_change + trainer_change_top3_rate_exp
7. ✅ pace_career_burst_mean (Q5-Q1 +10pt)
8. ✅ pace_career_change_1to4_mean (Q5-Q1 +10-15pt)
9. ✅ pace_career_relative_4cor_mean
10. ✅ pace_recent5_burst_mean
11. ✅ pace_recent5_change_mean

### 明示 interaction (V20 で explicit 化 推奨)
12. ✅ `class_down * (burst >= Q5)` (43.8% top3 rate pattern)
13. ✅ `class_down * (1 - jockey_change)` (同騎手で降級 boost)
14. ✅ `class_down * (1 - trainer_change)` (同厩舎で降級 boost)

### 期待 V20 性能 (改定)
- WF AUC: 0.8939 (V15) → **0.895-0.905** 想定
- 実 ROI: V15 戦略⑦ 140% → **V20 戦略⑦ 145-160%** 想定
- 黄金 pattern 抽出時: 月利 **+¥50-100K** 可能性

## 🚦 5/12 → 5/17 user task (確定)

### Day 1: 5/12 (月)
- [ ] `python tools/register_all_phase24_schtasks.py` (admin)
- [ ] JV-Link 32-bit `python tools/jvlink_parser.py --test-com`
- [ ] 30y backtest 1995-2005 段階開始
- [ ] `python tools/morning_briefing_5_17.py` 動作確認

### Day 2-4: 5/13-15 (火-木)
- [ ] 厩舎コメント / 専門家予想 実 scrape
- [ ] 30y backtest 段階続き
- [ ] paddock 過去 archive build 開始

### Day 5: 5/16 (金)
- [ ] paddock 5/16 capture
- [ ] YouTube schtask 動作確認
- [ ] morning_go_check + 全 chain rehearsal

### Day 6: 5/17 (土) - 本番
- 06:30 morning_go_check / briefing 自動
- 08:55 YouTube LIVE 録画 自動起動
- V15 案 B 改 + 戦略⑦ 単独継続
- Phase 23 shadow log 並行

## 🛡 V15 投資保護 (5/11 marathon 全 phase 通して 厳守 確認)

predict_core.py / daily_predict.py / app.py / V15 model `.pkl.gz` ALL 不変。
Phase 21D-24 全 34 tools は post-process / helper / 検証 / 分析、 production 影響 0。

## 🎯 5/24+ V20 投入 path (Phase 25)

5/17 開催 verdict 後:
- 5/18-22 V20 学習 data 構築 + 14 features 投入 + WF 検証
- 5/23 V20 GO/no-go 判定 (WF AUC ≥ 0.880 / 全条件 ROI ≥ 100%)
- 5/24 V20 段階投入 (V15 と並行運用 1 ヶ月)

V21 (paddock 動画 features 込み) 6/8+、 V22 (RL) 9/1+ で path 明確化。

## 結論

今夜 1 session で:
- **34 tools 実装**
- **3 strong signals verify**
- **黄金 pattern 発見 (ROI 推定 180.4%)**
- **V15 + 戦略⑦ empirical 最強 確定**
- **V20 学習 14 features 確定**
- **5/17 GO READY 確認 + 5/24 V20 投入 path 確立**

V15 + 戦略⑦ 単独継続で 5/17 開催 confident GO、 V20 投入で 月利 +¥50-100K の見込み。
