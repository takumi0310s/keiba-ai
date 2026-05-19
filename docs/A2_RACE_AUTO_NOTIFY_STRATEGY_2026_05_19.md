# race_auto_notify.py 戦略実装 summary (2026-05-19)

Session A-2/A-3 + B-1/B-2/C-1/C-2/C-3 の 7 戦略を sequential commit で実装。

## 実装戦略一覧

| # | 戦略 | Commit | 区分 | 内容 |
|---|------|--------|------|------|
| 1 | C4 | c36614b1 | **production active** | Cond-A 1600-1800m drag 除外 (+8.62pt 確定) |
| 2 | C3 (pos2) | 5c8e9fd2 | **production active** | trio T1-T2-T4 (bet2) 除外 → 7→6点 |
| 3 | B-1 | 5f0a43a6 | paper only | V15 top1 = 市場1番人気 → skip log (N=41 N不足) |
| 4 | B-2 | a1cbf0bc | paper only | V15-市場 divergence: pop_rank < 3 → skip log |
| 5 | C-1 | 58ad7089 | paper only | EV>1 trio bet-level 計算 → log |
| 6 | C-2 | 9c9a33e9 | paper only | odds 帯除外 (過剰人気<1.5/極値>20/東京5-10x) |
| 7 | C-3 | bd13a82b | paper only | 場別 grid 高 ROI pocket 検出 log |

## production active (実際に bet をスキップ)

### C4: Cond-A 1600-1800m 除外
- `STRATEGY_C4_ENABLED = True`
- 条件: `cond_key == 'A' and 1600 <= distance <= 1800`
- 効果: +8.62pt 重-2 backtest confirmed
- 位置: 条件判定後 skip ブロック (L282 付近)

### C3: pos2 (T1-T2-T4) bet 除外
- `STRATEGY_C3_ENABLED = True`
- 条件: trio bet の T1-T2-T4 コンボを結果リストから除外
- 効果: 7点 → 6点 (bet2 除去)
- 位置: generate_trio_bets 直後 (L317 付近)

## paper only (ログのみ、実ベット不変)

### B-1: 1番人気除外
- `STRATEGY_B1_PAPER_ONLY = True`
- `_b1_top1_pop = df.iloc[0].get('pop_rank')`
- skip 条件: `_b1_top1_pop == 1`
- 採用判定: 6/17 (N=41 では不足)

### B-2: divergence フィルタ
- `STRATEGY_B2_PAPER_ONLY = True`
- `STRATEGY_B2_MIN_POP_RANK = 3`
- skip 条件: `pop_rank < 3` (V15 が市場と一致)
- 採用判定: N 蓄積後

### C-1: EV>1 trio
- `STRATEGY_C1_PAPER_ONLY = True`
- EV = score[i]*score[j]*score[k] / total_score^3 * 5000 / 100
- 閾値 >= 1.0 の bet 数を log

### C-2: odds 帯
- `STRATEGY_C2_PAPER_ONLY = True`
- skip 条件: top1_odds < 1.5 / > 20.0 / 東京5-10x

### C-3_VENUE: 場別 grid
- `STRATEGY_C3_VENUE_PAPER_ONLY = True`
- 高 ROI pocket: 中山A / 阪神C / 東京C / 中京A
- 結果 log のみ

## rollback 方法

各戦略は独立 commit。rollback は `git revert <hash>` で個別に可能。

| 戦略 | rollback hash |
|------|--------------|
| C4   | c36614b1 |
| C3   | 5c8e9fd2 |
| B-1  | 5f0a43a6 |
| B-2  | a1cbf0bc |
| C-1  | 58ad7089 |
| C-2  | 9c9a33e9 |
| C-3  | bd13a82b |

## 不変保証

- V15 .pkl.gz: 変更なし
- predict_core.py: 変更なし
- daily_predict.py: 変更なし
- app.py: 変更なし
- betting core logic (predict_race 呼び出し): 変更なし
