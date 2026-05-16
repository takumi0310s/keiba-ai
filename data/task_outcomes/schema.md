# task_outcomes JSON schema

各 task 実施後、 `tools/outcome_record.py` で `data/task_outcomes/{task_id}.json` を生成する。

## fields

```json
{
  "task_id": "P0-2",
  "task_name": "京都/中京 戦略⑦再除外",
  "phase": "P0",
  "implemented_at": "2026-05-18T08:30:00",
  "before": {
    "roi": 101.33,
    "hit_rate_top3": 0.5382,
    "trio_hit_rate": 0.2185,
    "n": 563,
    "baseline_ref": "data/task_outcomes/baseline_v15.json"
  },
  "after": {
    "roi": 108.50,
    "hit_rate_top3": 0.55,
    "trio_hit_rate": 0.22,
    "n": 120,
    "data_window": "2026-05-18 - 2026-05-24"
  },
  "delta": {
    "roi": 7.17,
    "hit_rate_top3": 0.012,
    "trio_hit_rate": 0.0015
  },
  "expected": {
    "roi": "+5pt (assumption)",
    "notes": "京都/中京 除外で 戦略⑦ ROI +5pt 期待"
  },
  "statistical_test": {
    "type": "Welch's t-test (ROI per race)",
    "p_value": 0.12,
    "ci_95": [-2.5, 8.3],
    "n_before": 563,
    "n_after": 120,
    "significant_at_0_05": false
  },
  "status": "shadow_eval",
  "notes": [
    "shadow 評価期間: 5/18-5/24",
    "正式 採用判定 は 5/25 朝"
  ]
}
```

## status 値

- `pending`: task 未実施
- `shadow_eval`: 実施済、 評価期間中
- `accepted`: 統計的有意 + 改善確認、 本番反映
- `rejected`: 改善なし or 悪化、 不採用
- `rolled_back`: 一度採用後、 悪化検出で revert

## phase 値

- `P0`: ROI 確定 / 緊急 fix
- `P1`: 短期 improve (週単位)
- `P2`: 中期 improve (月単位)
- `P3+`: 長期 構造改革

## 命名 convention

- file 名: `{task_id}.json` (例: `P0-1.json`, `P1-2.json`)
- task_id は 一意、 reissue 禁止
- 再実施は `P0-1_v2.json` の form
