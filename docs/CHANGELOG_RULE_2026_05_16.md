# CHANGELOG ルール (2026-05-16 制定)

memory drift 再発防止のため、 全数値更新で出典明示と timestamp 付与を必須とする。

## 1. 数値変更時の出典記載必須

CLAUDE.md / memory / docs / baseline_v15.json の **数値** を update する際、 以下を必ず添える:

- **数値の source**: どの calculation / どの commit hash / どの doc / どの csv
- **last_verified date**: `YYYY-MM-DD` 形式
- **真値 vs 旧値 の diff**: 差分 (絶対値 + %) を明示
- **統計的有意性**: bootstrap CI 等を併記 (N が小さい場合は "not significant" 明記)

例:
```
ROI: 101.33%
  source: data/cumulative_results.csv (n=563 settled rows, 5/16 evening recompute)
  last_verified: 2026-05-16
  prev_value: 119.2% (4/18 snapshot, 出典不明、 memory drift)
  diff: -17.87pt
  significance: 95% bootstrap CI includes 100% -- not statistically significant
```

## 2. memory entry timestamp 仕様

`.claude/projects/.../memory/*.md` 全エントリに以下を必須:

- `last_verified: YYYY-MM-DD`
- `verified_source: data/cumulative_results.csv (N=XXX rows)` 等
- `expires_after_days: 7` -- 1 週間経過で stale 判定 (daily_cumulative_audit.py で 自動 detect)

## 3. CHANGELOG template (commit message)

数値変更を含む commit には 以下を含める:

```
[update] <metric> <old> -> <new>

source: <csv / calc / commit>
diff: <pt or yen>
significance: <CI or note>
verified_at: YYYY-MM-DD
```

## 4. drift 再発防止 (4/18 snapshot 系)

- 4/18 / 5/6 等の snapshot 系 数値は **「snapshot」 label 必須**、 累計値と混同禁止
- 累計 / ROI 系 の canonical source は **data/cumulative_results.csv** のみ
- 旧 drift 値 (`13,530` / `119.2%` 等) を引用する場合は 必ず "drift" / "旧値" / "snapshot" / "history" annotation を同行に含める (daily_cumulative_audit.py が grep で誤検知抑止する)

## 5. 自動監査 (5/16 sub-task 10)

- `tools/daily_cumulative_audit.py` が毎日 21:00 で:
  - cumulative_results.csv から 真値計算 -> `data/cumulative_truth.json`
  - baseline_v15.json adopted_value との diff > 5pt (ROI) / >¥5,000 (PnL) で Discord 警告
  - CLAUDE.md 内 旧値残存 (annotation なし) を grep -> Discord 警告
- `tools/weekly_report.py` が drift section を最終 report に同梱

## 6. 禁止事項

- 数値を出典なしで CLAUDE.md / memory に書く
- snapshot 値を 累計値と表記する
- 旧値を annotation なしで 残置する
- daily_cumulative_audit.py の threshold (5pt / ¥5,000) を 議論なしに変更する
