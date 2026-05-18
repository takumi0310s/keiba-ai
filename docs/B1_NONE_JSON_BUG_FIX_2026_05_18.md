# B-1: race_notify_log_v2 None.json bug fix (5/18 17:30+)

## 0. 結論

- root cause: `log_phase{1,2,3}` の race_id validation 不在
- fix: `_validate_race_id()` 関数追加、 全 `log_phase{1,2,3}` 冒頭で early-return
- regression: 既存 18 tests + 新規 3 tests = **21/21 PASS**
- 5/18 既存 `None.json` (phase1/2/3 各 1 件) 削除済
- 5/24 fire 安全 verified
- ★ V15 production 完全不変、 既存 logic touch なし ★

## 1. root cause (None.json 内容分析)

5/18 17:00 status verify で発見した 3 件:

| file | timestamp | race_id | 内容 |
|------|-----------|---------|------|
| `phase1/None.json` | 2026-05-18T00:32:24.877563 | `"None"` | 空 meta + 空 ranking |
| `phase2/None.json` | 2026-05-18T00:32:24.877932 | `"None"` | 空 meta + formation 空 |
| `phase3/None.json` | 2026-05-18T00:32:24.878283 | `"None"` | 空 top3 + 空 payouts |

3 件とも 0.001 秒以内に生成 → 同一 caller (連続実行)。
timestamp 00:32 で 5/18 (中央開催なし日) → real fire ではなく test/dry-run の副作用と推定。

既存 logic の問題点:
```python
# tools/race_notify_log_v2.py (修正前)
def log_phase1(race_id, ...):
    try:
        out_file = out_dir / f'{race_id}.json'  # race_id=None なら 'None.json'
        data = {'race_id': str(race_id), ...}    # str(None) = 'None'
        _write_json(out_file, data)
```

`race_id=None` を渡されても何の validation もなく file IO 実行 → `None.json` が生成され続ける。
5/24 fire でこれが発生すれば、 該当 race の log が永久喪失 (上書き or 区別不能)。

## 2. 修正内容

### 2.1 tools/race_notify_log_v2.py

新規関数:
```python
def _validate_race_id(race_id) -> bool:
    """race_id validation: digit 10-12 桁 必須。"""
    if race_id is None:
        return False
    try:
        s = str(race_id).strip()
    except Exception:
        return False
    if not s:
        return False
    if s in ('None', 'none', 'NONE', 'nan', 'NaN', 'NAN', 'null', 'NULL'):
        return False
    if not s.isdigit():
        return False
    if not (10 <= len(s) <= 12):
        return False
    return True
```

全 `log_phase{1,2,3}` 冒頭 (`try` の外側) で early-return:
```python
def log_phase1(race_id, ...):
    if not _validate_race_id(race_id):
        print(f"[race_notify_log_v2 phase1 SKIP] invalid race_id: {race_id!r}",
              file=sys.stderr)
        return
    try:
        ...
```

★ 既存 logic (try/except / data 構築 / `_write_json`) は完全不変 ★

### 2.2 tests/test_race_notify_log_v2.py

既存 18 tests 不変 (実 race_id format に更新したものは 3 件のみ、 logic 同一):
- `test_log_phase1_missing_fields_ok`: `'X1'` → `'202605020199'` (新 validation 通過のため)
- `test_aggregator_complete_race`: `'R1'` → `'202605020201'`
- `test_aggregator_skip_strategy_7c`: `'R2'` → `'202608010101'`
- `test_log_phase_failure_does_not_raise`: tmp_path/monkeypatch 追加 (hermetic、 LOG_DIR 汚染防止)

新規 3 tests:
- `test_race_id_none_skipped`: `None` で全 phase log skip 確認
- `test_race_id_nan_skipped`: `'nan'/'NaN'/'None'/'null'/''/'   '` で skip
- `test_race_id_invalid_format_skipped`: `'abc'/'123'/'202605020'/'2026050201010101'/'R1'/'12-34-56'` で skip + 正常 12 桁 race_id は通過

### 2.3 既存 None.json 削除

```powershell
Remove-Item data/race_notify_log_v2/20260518/phase1/None.json
Remove-Item data/race_notify_log_v2/20260518/phase2/None.json
Remove-Item data/race_notify_log_v2/20260518/phase3/None.json
```

削除後 verify: 3 phase ディレクトリすべて空。

## 3. dry-run verify

```python
log_phase1(None, {}, [], '', date_str='20260518')
log_phase1('202608030701', {'race_name': 'test'}, [], '5-7-9', date_str='20260518')
```

出力:
```
[race_notify_log_v2 phase1 SKIP] invalid race_id: None
files: ['202608030701.json']
dry-run OK: None skipped, valid logged
```

★ None は skip / 正常 race_id は logged ★

regression:
```
21 passed in 0.16s
```

## 4. V15 production 不変保証

- `train/` / `predict_core.py` / `daily_predict.py` / `app.py` / `race_auto_notify.py` の logic touch なし
- `tools/race_notify_log_v2.py` は log 出力のみ (V15 予測 / 投票 と独立)
- V15 `.pkl.gz` / `cumulative_results.csv` 改変なし
- 5/24 fire: invalid race_id は SKIP + stderr log のみ、 既存 logic は完全不変
