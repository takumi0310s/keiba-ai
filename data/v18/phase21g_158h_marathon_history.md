# Phase 21G - 158h+ マラソン (5/3-5/10) 全 history 集約

> 作成: 2026-05-11 (Phase 21G)
> 期間: 2026-05-03 → 2026-05-10 (約 158-170h、 commit 160 件)
> 目的: 1 年後 (2027/5/10) 振り返り資料 + 5/12+ 集中作業の base

## 1. マラソン総括

| 指標 | 値 |
|------|----|
| 期間 | 2026-05-03 〜 2026-05-10 |
| 総 commit 数 | 約 160 件 |
| Session 数 | #37 → #100+ (約 65 session) |
| 5/10 単日 commit | 26 件 (Phase 1-21 + 並行) |
| V15 production | 完全不変 (predict_core.py 触らず) |
| 累計収支 | +¥14,140 円 死守 (撤退余裕 +¥64,140) |

## 2. 主要 milestone (時系列)

### 5/3-5/4 (Phase 2.5 launch)
- Phase 2.5: V13.5b → V15 案 B 改 移行確定
- 戦略⑦ (06_特別/京都/条件E/条件B 除外) 実装 → ROI 119.2% → 140%+ 想定
- 累計 +¥13,530 突破

### 5/5-5/6 (Phase 2.5+ V18/V19)
- V18/V19 sib抜き 学習試行
- shift 真因 (30.4x → 8.3x、 winner_top1 -10pt) 解析
- 5/9 GO/NO-GO: NO-GO 確定 (Phase 2.5+ V18V19 E)

### 5/7 (Session #37-#40 — SKB LEAK 大発見)
- **★ Session #38 A: V15.1 SKB POST-RACE LEAK 確定 ★**
  - skb_kishi_code_3 +480bp / corr_target 0.137 / monotonic 1着→364, 10着→176
  - 採用 NO-GO 確定、 V20 で SKB 全 10 features 完全除外
- Session #38 B: V18/V19 sib抜き hybrid (LIVE shift 改善 + winner_top1 悪化)
- Session #39 A: sib_*_exp 修正版 PoC + Phase 3-4 統合 roadmap (V20 7/1+, V21 9/1+)
- Session #40 A-F: 6 領域並行 (PAT/Kelly/health check + alert + 新features + voting)

### 5/8 (Session #41-#48 — TFJV + V20 PoC)
- **★ Session #43 A: V15 ROI 44% 真因発見 → 真の ROI 83.96% ★** (NaN 集計 bug)
- **★ Session #43 C: sib_exp w5 LIVE retro 完全回復 (+6.89pt) ★**
- **★ Session #44 A-G: TFJV フル data 即活用 ★**
  - 構造把握 (43,000 files / 6 GB / 14 datatypes)
  - tools/tfjv_parser.py 本実装 (Shift-JIS OK)
  - V20 6 年分 一括 parse (320K records / 10 秒)
  - V20 PoC AUC 0.8752 (LGB single fold)
  - **★ Phase 3 V20 投入 7/1 → 6/8 に 1 ヶ月前倒し ★**
- AUDIT-1: 3 source 全要素 audit、 未活用 features Top 30

### 5/9 (Session #59-#89 — 完全自動化 plan)
- Session #71: 全馬 score 完全保存機能 (5/10+ 並行運用)
- Session #74-#77: 5/16 V18 trial final plan v5 + silent_runner emergency fix
- Session #78: PreRacePredict 緊急対応 (schtask disable + hardcode 撤廃)
- Session #79-#86: V20 構築 + JRA-VAN RV trial + Phase 4 plan v2 + JRA-VAN NEXT 自動分配 + hybrid 戦略 + V22 RL 設計 + 30 年 backtest 設計 + 完全自動化 ロードマップ (5/15 80% / 9/2 90% / 12/1 100%)
- Session #87-#89: 5/10 朝確認 + 全機能 audit prompt 永続化 + 5/10 投票候補 事前抽出 plan

### 5/10 (Phase 1-21 + 並行 — 史上最多 22 phase 1 day)
詳細は [phase21g_5_10_day_22_phases.md](phase21g_5_10_day_22_phases.md)。

### 5/11 (Phase 21A-D + 21G — 締め)
- Phase 21A: 5/17 GO worksheet 強化 + 06:30 schtask
- Phase 21B: V18 paper trade 動作確認 + V15 retrain 同等性 honest report
- Phase 21C: 5/10 score 帯別 深掘り + 案 B 改 strict 再評価 (重大訂正)
  - G section 統計的有意性検証: N=3 CI [0.04, 0.82] = 結論不能
- Phase 21D: 5/12 / 5/13-5/14 詳細 plan
- Phase 21G: 本 doc 集約

## 3. Model 進化系譜

| 世代 | AUC (WF) | 特徴量 | 採用日 | 備考 |
|------|---------|-------|-------|------|
| V8 | 0.78 台 | 基本 | 〜2025 末 | Pre-V9.x 系列 |
| V9.1-V9.3 | 0.80-0.81 | 67 | 2026-Q1 | リークフリー基盤 |
| V12 | 0.8037 | 74 | 2026-Q1 末 | LGB 単体 |
| V13.4 | 0.8656 | 124 | 2026-Q2 前半 | JRDB +50、 LGB+XGB |
| V13.5 | 0.8722 (試算) | 124 | 2026-04 | + FT-Transformer |
| V13.5b | 0.8788 | 124 | 2026-04-03 | + IntraRace、 Grid Ensemble |
| **V14.1** | 0.886+ | 138 | 2026-04 末 | 統合改善 |
| **V14.2** | 0.890+ | 145 | 2026-04 末 | 微調整 |
| **V15 (現行)** | **0.8939** | **150** | 2026-05-03 〜 | 戦略⑦込み 140%+ 想定 |
| V18 候補 | TBD | 165 | 5/16 trial 候補 | sib_*_exp + JRDB 真値 6 |
| V19 候補 | NO-GO | — | — | sib抜き、 5/9 NO-GO |
| V20 候補 | 0.8752 (PoC) | 80 共通 | 6/8-7/1 投入候補 | TFJV 6 年、 SKB 完全除外 |
| V21 候補 | TBD | 237 | 9/1 投入候補 | + RV 動画 (馬体 + 姿勢) |
| V22 候補 | RL | — | 12/1 投入候補 | PPO Gymnasium env |

## 4. データ source 進化

| Source | 5/3 状態 | 5/10 状態 |
|--------|---------|----------|
| netkeiba | shutuba/odds/結果 ベース | + master DOM probe + 過去 backfill 安全基盤 |
| JRDB | KYI 統合 75.9%、 SKB 別 | KYI 6 features 真値化 (40%)、 SKB 完全除外確定 |
| TFJV (TARGET JV) | SE/CK/HY/BR/KT 抽出済 | フル audit (43K files / 6GB / 14 datatypes)、 V20 base |
| JV-Link | 加入確定のみ | 1 ヶ月 backfill PoC + 17 features skeleton |
| RV 動画 | 未着手 | YOLOv8 138ms PoC + V21 candidate skeleton |

## 5. Bug 修復 (主要 12 件)

| # | bug | 修復 commit | 日時 |
|---|-----|-----------|------|
| 1 | V15 ROI 44% (actual_payout NaN 集計) | d3e8827c | 5/8 Session #43 A |
| 2 | SKB POST-RACE LEAK | 7c2f9ce1 / 84d52a1d | 5/7 Session #37/38 |
| 3 | sib_top3_rate hybrid leak | a95f77db | 5/7 Session #39 A |
| 4 | RACE_START_TIMES 31/35 R ズレ | 0c32d0cd | 5/10 Phase 6 |
| 5 | Stage 2 通知 score 表示 (CSV 列名不一致) | 1e95d73b | 5/10 Phase 5 |
| 6 | 買い目 2 重送信 root cause | 82a9c022 | 5/10 Phase 2 |
| 7 | Discord 二重送信 (5min hash dedup) | e803a826 | 5/9 Session #59 |
| 8 | 毎時間全 R 通知バグ | 8fc4e13b | 5/9 Session #64 |
| 9 | silent_runner.vbs Line 24 ERROR_FILE_NOT_FOUND | 384a0187 | 5/9 Session #77 |
| 10 | model corruption (CRLF→LF) | a3b57f9f | 5/7 Phase 2.5+ |
| 11 | predict_nar.py blocker (5/12 paper) | 05f4c39c | 5/6 Phase 2.5+ C |
| 12 | jrdb_paci.csv 4/4 から更新停止 | JV-Link O1 で代替経路 (Session #39 B) | 5/7 |

## 6. V15 投資保護 (絶対遵守 — 5/3-5/10 完全遵守)

- 🔴 **predict_core.py / V15 model 一切変更なし**
- 🔴 **destructive git op ゼロ件**
- 戦略⑦ (06_特別/京都/条件E/条件B 除外) 維持
- 5/9 案 B 改 12R 1勝クラスのみ上限 ¥2,100 戦略
- 撤退ライン -¥50,000 (現在 +¥14,140 = 余裕 +¥64,140)

## 7. 関連 doc 索引

- 5/10 day 22 phases 詳細: [phase21g_5_10_day_22_phases.md](phase21g_5_10_day_22_phases.md)
- 5/10 真の集大成 list: [phase21g_achievements.md](phase21g_achievements.md)
- 5/11 → 2027/5/10 timeline: [phase21g_timeline.md](phase21g_timeline.md)
- 用途別 doc 索引: [../../docs/MEMORY_INDEX.md](../../docs/MEMORY_INDEX.md)
- 完全自動化 ロードマップ: [../../docs/FULL_AUTOMATION_ROADMAP.md](../../docs/FULL_AUTOMATION_ROADMAP.md)

## 8. 1 年後 (2027/5/10) 振り返りチェック項目

- [ ] V15 → V20 → V21 → V22 段階投入 全成功か
- [ ] 累計収支 +50 万 / +100 万 / +200 万 のいずれを達成したか
- [ ] 完全自動化 100% (12/1 目標) は維持されているか
- [ ] 撤退ライン -¥50,000 を一度も触れなかったか
- [ ] V15 廃止 (V20 並行運用 1 ヶ月後 = 8/1 候補) は完了したか
- [ ] RV 動画解析 (V21、 9/1 投入候補) の実 ROI 寄与は何 pt だったか
- [ ] V22 RL (PPO) は production に投入できたか
