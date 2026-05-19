# race_notify_log v2 — 8 strategy 並行追跡拡張

Session #91 B-3 (2026-05-19) 実施。

## 概要

`tools/race_notify_log_v2.py` と `tools/race_notify_log_v2_aggregator.py` を拡張し、
1 race ごとに 8 strategy の "would have bet" formation を記録・集計できるようにした。

- V15 `.pkl.gz` / `predict_core.py` / `daily_predict.py` / `app.py` は **完全不変**
- log fail は stderr のみ、既存 logic への影響ゼロ

---

## 8 strategy 一覧

| key | 説明 | skip 条件 |
|-----|------|-----------|
| `actual` | 実際の bet (V15 + 戦略⑦ベース) | B/E/X 条件 または 京都 開催 |
| `c3` | + C3 (pos4 bet 除外、6 点化) | 同上 |
| `c4` | + C4 (条件 A 1600-1800m skip) | 同上 + cond=A かつ 1600≤dist≤1800 |
| `c3c4` | + C3+C4 複合 | 同上 |
| `no_1pop` | B-1: top1 予測馬が 1 番人気ならば skip | 同上 + pop_rank==1 |
| `divergence` | B-2: top1 予測馬が 3 番人気以内ならば skip | 同上 + pop_rank<3 |
| `ev_filter` | C-1: 現在は actual と同一 (EV 計算は将来実装予定) | 同上 |
| `odds_filter` | C-2: top1 単勝 1.5-20 倍 帯フィルタ | 同上 + odds 範囲外 |

---

## phase2 拡張: `strategy_formations`

`log_phase2()` に `predictions` 引数を追加。

```python
log_phase2(
    race_id,
    ...,
    predictions=[
        {'horse_num': 5, 'pop_rank': 3, 'odds': 8.0},
        {'horse_num': 2, 'pop_rank': 1, 'odds': 2.5},
        ...  # top 6 以上推奨
    ],
)
```

`predictions` が渡された場合、`build_strategy_formations()` が 8 strategy の formation を計算して
phase2 JSON の `strategy_formations` フィールドに保存する。

phase2 JSON 例:
```json
{
  "phase": 2,
  ...,
  "strategy_formations": {
    "actual":      [[2,5,7],[2,3,5],...],
    "c3":          [[2,5,7],[2,5,9],...],
    "c4":          null,
    "c3c4":        null,
    "no_1pop":     [[2,5,7],...],
    "divergence":  null,
    "ev_filter":   [[2,5,7],...],
    "odds_filter": [[2,5,7],...]
  }
}
```

---

## phase3 拡張: `strategy_results`

`log_phase3()` は phase2 JSON を自動読み込みして 8 strategy の hit/pnl を計算する。
呼び出し側は変更不要。

phase3 JSON 例:
```json
{
  "phase": 3,
  ...,
  "strategy_results": {
    "actual":     {"n_bets":7, "skipped":false, "hit":true, "payout":3500, "investment":700, "pnl":2800},
    "c3":         {"n_bets":6, "skipped":false, "hit":true, "payout":3500, "investment":700, "pnl":2800},
    "c4":         {"n_bets":0, "skipped":true,  "hit":false,"payout":0,    "investment":0,   "pnl":0},
    ...
  }
}
```

---

## aggregator 拡張: `strategy_stats`

`aggregate_day()` の返り値に `strategy_stats` フィールドを追加。
各 strategy について N / hits / hit_rate / inv / pay / ROI / PnL を集計。

```
python tools/race_notify_log_v2_aggregator.py --date 20260519
```

出力例:
```
=== 20260519 ===
  ...
  8 strategy paper ROI
  strategy             N  hits    hit%      inv      pay     ROI%      PnL
  ---------------- ----- ----- ------- -------- -------- -------- --------
  actual               2     1   50.0%     1400     1500  107.14%     +100
  c3                   2     1   50.0%     1400     1500  107.14%     +100
  no_1pop              1     0    0.0%      700        0    0.00%     -700
  ...
```

追加 CLI オプション:
- `--strategy-report` : 8 strategy 表のみ表示
- `--no-strategy` : 8 strategy 表を省略

---

## 新規追加 API

### `build_strategy_formations(predictions, race_meta) -> dict`
8 strategy の formation dict を返す。

### `compute_strategy_results(strategy_formations, real_top3, trio_payout, bet_cost) -> dict`
8 strategy の hit/pnl を計算して返す。

### `STRATEGY_KEYS`
8 strategy key list (定数): `['actual','c3','c4','c3c4','no_1pop','divergence','ev_filter','odds_filter']`

---

## 既存 API 後方互換性

`log_phase2()` の `predictions` 引数は省略可能 (デフォルト `None`)。
未渡しの場合 `strategy_formations` は `{key: None for key in STRATEGY_KEYS}` として記録される。

`log_phase3()` / `log_phase1()` / `read_phase()` の呼び出し方は完全不変。

---

## paper eval の使い方

1. `race_auto_notify.py` の `log_phase2()` 呼び出しに `predictions=top_horses` を追加する
   (次ステップ: Sub-task D 想定)
2. `daily_results.py` の `log_phase3()` は既存のまま自動で strategy_results を計算
3. 週次で `python tools/race_notify_log_v2_aggregator.py --all --strategy-report` を実行して
   8 strategy の累計 ROI を確認する
