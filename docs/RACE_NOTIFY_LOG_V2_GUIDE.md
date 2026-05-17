# race_notify_log v2 (3 phase) Guide

**作成日**: 2026-05-17
**目的**: 真の race-time formation ROI を 5/18+ から永続記録
**背景**: data-audit-3 で確定。 race_auto_notify.py L300 `bets = generate_trio_bets(df)` は Discord 送信のみで persist されず、 過去 1.5 ヶ月の race-time formation ROI は永久喪失。 cumulative trio_bets_str は AM 8:00 morning predict のため race-time formation ではない。

## 設計

### 3 phase

| phase | 名前 | タイミング | 出力場所 | 内容 |
|-------|------|----------|---------|------|
| 1 | morning_predict | 朝 8:00 daily_predict | `data/race_notify_log_v2/{YYYYMMDD}/phase1/{race_id}.json` | ranking top5 + 朝 odds + 予定 formation |
| 2 | pre_vote | race -5min race_auto_notify | `data/race_notify_log_v2/{YYYYMMDD}/phase2/{race_id}.json` | 投票 formation 確定 + 投票時 odds + 戦略⑦ skip 情報 |
| 3 | post_result | 20:00 daily_results (将来) | `data/race_notify_log_v2/{YYYYMMDD}/phase3/{race_id}.json` | 実 1-3 着 + 実配当 + hit/miss |

### ★ 既存 logic 完全不変保証 ★

- `race_auto_notify.py`: 各 `_p0_5_notify_log(...)` 呼び出しの直後に `_v2_log_phase2_safe(...)` を **1 行** 追加のみ。 既存 predict / Discord 送信 / strategy_7 フィルタの logic は **1 byte も改変なし**。
- `daily_predict.py`: `_append_prediction_to_csv(row, date_str)` の直後に `_v2_log_phase1_safe(...)` 呼び出しを **1 ブロック** 追加のみ。 CSV 書き込み / DataFrame 構築 / 予測 logic 完全不変。
- log fail / import fail / 例外 全て `try/except` で swallow し、 stderr 出力のみ。 既存 logic を絶対に止めない。
- regression test (231 既存 + 18 新規 = 249 PASS) で 動作確認済。

## file 構成

```
tools/
├── race_notify_log_v2.py                          # 新規: 3 phase logger (log_phase1/2/3, read_phase)
├── race_notify_log_v2_aggregator.py               # 新規: aggregator (DAILY 20:30 想定)
├── register_race_notify_log_v2_aggregator_schtask.bat  # 5/18 admin 登録用
├── race_auto_notify.py                            # 修正: _v2_log_phase2_safe wrapper 追加 + 各 log site で呼び出し
└── daily_predict.py                               # 修正: _v2_log_phase1_safe wrapper 追加 + _append_prediction_to_csv 直後で呼び出し

tests/
└── test_race_notify_log_v2.py                     # 新規: 18 tests

docs/
└── RACE_NOTIFY_LOG_V2_GUIDE.md                    # 本ファイル
```

## API

### `tools/race_notify_log_v2.py`

```python
log_phase1(race_id, race_meta, ranking_top5, formation_planned, morning_odds, date_str=None)
log_phase2(race_id, race_meta, formation_actual, vote_time_odds, strategy_7c_skip, strategy_7c_reason, channel, cond_key, bet_type, date_str=None)
log_phase3(race_id, real_top3, real_payouts, hit_miss, date_str=None)
read_phase(race_id, phase, date_str=None) -> dict | None
```

全関数 fail-safe (例外を投げない、 stderr 出力のみ)。
`RACE_NOTIFY_LOG_V2_ROOT` env で log root を override 可能 (test 用)。

### Aggregator (`tools/race_notify_log_v2_aggregator.py`)

```bash
python tools/race_notify_log_v2_aggregator.py                  # 本日分集計
python tools/race_notify_log_v2_aggregator.py --date 20260518  # 指定日
python tools/race_notify_log_v2_aggregator.py --range 20260518:20260524
python tools/race_notify_log_v2_aggregator.py --all            # 全日付
```

出力 example (cond breakdown 付き):
```
=== 20260518 ===
  phase1 / phase2 / phase3 = 26 / 26 / 24
  complete races = 24 (voted 18, skipped 6)
  hits = 5 (27.78%)
  inv = 12600¥  pay = 18900¥  ROI = 150.00%  PnL = +6300¥
  cond breakdown:
    A: n=12 hits=4 inv=8400¥ pay=11200¥ ROI=133.3% hit_rate=33.3%
    C: n=4  hits=1 inv=2800¥ pay=7700¥  ROI=275.0% hit_rate=25.0%
    D: n=2  hits=0 inv=1400¥ pay=0¥     ROI=0.0%   hit_rate=0.0%
```

Summary JSON: `data/race_notify_log_v2_summary/summary_{TIMESTAMP}.json`

## 5/18 admin 登録対象

新規 schtask 追加 → 計 8 task → **9 task**

| schtask | trigger | 内容 |
|---------|--------|------|
| Keiba-RaceNotifyLogV2-Aggregator | DAILY 20:30 | race_notify_log v2 3 phase 集計 |

★ 実 `schtasks /create` はこの guide 内では実行しない、 5/18 admin tasks に追加するのみ ★
登録は `tools/register_race_notify_log_v2_aggregator_schtask.bat` を **admin で手動実行**。

## 動作 verify (手動 dry-run)

```bash
# 1. 単発 log 書き込み
python -c "from tools.race_notify_log_v2 import log_phase1; log_phase1('TEST', {'race_name':'dry'}, [], '5-7-9', {}, '20260518')"

# 2. 出力確認
ls data/race_notify_log_v2/20260518/phase1/

# 3. aggregator 単体実行
python tools/race_notify_log_v2_aggregator.py --date 20260518
```

## V15 production 不変保証

- race_auto_notify.py / daily_predict.py: log hook (1-2 行) 追加のみ、 logic 完全不変
- V15 .pkl.gz / cumulative_results.csv / predict_core / app.py: 改変なし
- v15.2 training / v15_full training 中断なし
- 5/17 G1 day 影響 0% (5/17 終了済、 17:00 結果回収後の commit)
- 既存 regression test 231 件 ALL PASS 維持
