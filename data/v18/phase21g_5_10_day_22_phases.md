# Phase 21G - 5/10 day 22 phase 詳細 (★史上最多★)

> 作成: 2026-05-11 (Phase 21G)
> 5/10 単日 commit: 26 件 / Phase 1-21 + 並行 (Phase 5.5/5.6/5.7/5.8 含む)
> V15 production: 完全不変

## Block A: 朝〜昼 緊急 audit (Phase 1-10)

| Phase | 内容 | commit |
|-------|------|--------|
| Sleep audit (前夜) | GO 判定 + branch 復元 (#71/#77 file 物理 restore) | afb1c832 |
| Phase 2 | 緊急 audit: 買い目 2 重送信 root cause + 9:30 SaveAllHorseScores 初稼働 + 戦略⑦ filter | 82a9c022 |
| Phase 3 | 通知 + 特徴量 + 体重 audit (修復: なし、 5/15/16 plan 化) | f9e05312 |
| Phase 4 | 緊急: 5/10 当日 3 機能即実装 (京都通知 + Stage2 30分前予測 + 朝一全頭9:00) | 1ba675d0 |
| Phase 5 | 緊急: Stage 2 通知 score 表示修正 (CSV 列名不一致) | 1e95d73b |
| Phase 5.5 | 5/10 12:30 fire 検証 + 32R V15 score 順 list 生成 | 5d6a02a8 |
| Phase 5.6 | 5/10 全 35 R 全頭 V15 score 一覧出力 (read-only doc) | c7683af2 |
| Phase 5.7 | 5/10 14:00 PAT 投票 入力 list (上位 3R 軸/流し/三連複 7 点) | fd4095c3 |
| Phase 5.8 | 5/10 全 35 R V15 top1 score 順 ranking (band 分類 + 戦略⑦ filter) | 1439709b |
| Phase 6 | 緊急: 5/10 RACE_START_TIMES 真値 patch (31/35 R ズレ修正) | 0c32d0cd |
| Phase 7 | 緊急: 5/10 残 R + 5/11+ 全予想 audit + dynamic 取得化 実装 | c102b2fc |
| Phase 8 | 京都R12 体重統合再予想 (top1 不変) + V15 150 features 全 list audit | 9c9969e5 |
| Phase 9 | 強化版: 5/10 全 35R 完全照合 + 精度向上分析 + V15 → V18/V20 改善 plan | 8cee3543 |
| Phase 10 | 全 4 source 完全 audit + V20 統合 plan (期待 AUC 0.91-0.93) | d24adc95 |

## Block B: V18 candidate (Phase 11-14)

| Phase | 内容 | commit |
|-------|------|--------|
| Phase 11 | JRDB 未統合 features 実装 (V18 candidate predict_core_v18.py、 165 features) | a2a2279b |
| Phase 11 真値化 | JRDB KYI 6 features 真値 lookup 実装 (15 中 6/40%) | 376f494f |
| Phase 12 | JRA-VAN DataLab 未統合 17 features skeleton 実装 | b1751da5 |
| Phase 12 PoC | JV-Link 1 ヶ月 backfill PoC + 17 features 真値化試行 (honest report) | c7d668c1 |
| Phase 13 | netkeiba マスター scraping PoC + 統合 plan (25 features 候補) | f4d813bf |
| Phase 14 | paper_trade_engine.py 新規 + V15 vs V18 vs V20 比較 docs | 14d667a1 |

## Block C: V20-V22 構築 (Phase 15-18)

| Phase | 内容 | commit |
|-------|------|--------|
| Phase 15 | V20 4-model ensemble 学習 infrastructure (RTX 4070 Ti SUPER) | eba1347d |
| Phase 16 | RV 動画解析 PoC skeleton (V21 candidate 237 features) | d11a0bd8 |
| Phase 17 | 30 年 backtest 環境 + V22 RL (PPO) 初期学習 (現実 scope 調整版) | 93a3e45e |
| Phase 18 | netkeiba マスター DOM probe + 過去 backfill 安全基盤 + V18 再学習 plan | ce5a52b9 |

## Block D: 学習 + paper trade (Phase 19-20)

| Phase | 内容 | commit |
|-------|------|--------|
| Phase 19 | V18 真値版 学習 script + WF 評価 framework + 5/16 user CLI ready | 51e22ebe |
| Phase 20 | paper trade engine 強化 + 5/17 本番運用 ready (5-model 並行) | 79fee6a7 |

## 5/10 phase 数 集計

| 区分 | 数 |
|------|---|
| 主要 Phase 番号 | 1-20 (20 個) |
| 細分 (Phase 5.5/5.6/5.7/5.8) | 4 個 |
| 前夜 Sleep audit | 1 件 |
| **合計** | **約 22 phase** |

## 並行実行 pattern

5/10 は以下を並行 (Terminal A/B/C/D):
- A: 朝の audit (Phase 1-10)
- B: V18 candidate (Phase 11-14)
- C: V20-V22 構築 (Phase 15-18)
- D: 学習 + paper trade (Phase 19-20)

→ V15 production を一度も触らずに 4 並行進行を成立。

## 5/10 朝〜夜 timeline

| 時刻 | event |
|------|------|
| 〜06:00 | Sleep audit + branch restore |
| 06:00-09:00 | Phase 2-5 (買い目 audit + 9:30 SaveAllHorseScores 初稼働) |
| 09:00-12:00 | Phase 5.5-5.8 (12:30 fire 検証 + 全 35R score list) |
| 12:00-14:00 | Phase 6-7 (RACE_START_TIMES patch + dynamic 取得化) |
| 14:00-17:00 | Phase 8-9 (京都R12 統合再予想 + 5/10 完全照合) |
| 17:00-20:00 | Phase 10-14 (全 4 source audit + V18 candidate + paper trade) |
| 20:00-23:59 | Phase 15-20 (V20 ensemble + RV PoC + RL + V18 真値学習 + paper trade 強化) |

## 5/10 結果 (V15 production)

- 35R 中、 戦略⑦ filter 後の対象: 約 N=3 (Phase 21C で確認)
- 累計 +¥14,140 死守
- 投資保護 完全遵守

## 関連

- 全 history: [phase21g_158h_marathon_history.md](phase21g_158h_marathon_history.md)
- 真の集大成 list: [phase21g_achievements.md](phase21g_achievements.md)
- timeline: [phase21g_timeline.md](phase21g_timeline.md)
