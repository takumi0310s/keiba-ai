# Session #68 C: 修復実装

**作成**: 2026-05-09 16:57 (Session #68、 dev/two-stage)
**対象**: `tools/stage2_predict.py` (1 file 修正、 predict_one_race / predict_core 不変)

---

## 修正内容 (5 項目)

### 1. `_probe_netkeiba(race_id)` 追加 (新規 関数)

netkeiba shutuba 接続診断:
- HTTP status_code / response_len / HorseList 存在確認 / Cookie 有無
- 失敗時 `{probe_error: ...}` 返却

### 2. `predict_stage2()` 拡張

戻り値 schema:
```python
{
  "error": "netkeiba HTTP 400 (server block) / len=0",
  "error_kind": "netkeiba_block | netkeiba_other | shutuba_empty | exception | None",
  "diag": {"status_code": 400, "response_len": 0, ...},
  ...
}
```

`predict_one_race` が None 返した場合に `_probe_netkeiba()` で再診断、 `error_kind` を判別。

### 3. `build_message()` 改善

失敗時 Discord body の構造:
- title: `R{n} {course} 1h 前予測 (Stage 2) — Stage 1 fallback 採用`
- body:
  - `### Stage 2 失敗 ({kind})` + 診断行
  - `### 採用予測 = 朝予測 (Stage 1) top3 ★` (= 本 R の最終予測 を明示)
  - `※ retry: 次 fire (30 分後) で自動再試行 (cache に書込み無し)`

### 4. `predict_one()` cache 挙動修正

旧:
```python
cache[race_id] = datetime.now().isoformat()
save_cache(cache)
```

新:
```python
if stage2.get("error") is None:
    cache[race_id] = datetime.now().isoformat()
    save_cache(cache)
```

→ 失敗時は cache に書込まない。 次 fire (30 分後) で同 R 再試行可能。

### 5. `cmd_check_next_1h()` 起動時 probe

fire 開始時 1 回だけ netkeiba probe:
- block 検知 (HTTP 400) なら 全 R skip + 1 通 Discord 警告 (`yellow` channel=bets)
- block 解除されれば 個別 R 試行へ

`--skip-block-alert` flag 追加 (test 用、 警告 skip)。

---

## 動作確認

### Test 1: 失敗 R 単発 manual (--race-id, --force, --no-discord)

```
$ python tools/stage2_predict.py --race-id 202604010312 --no-discord --force

=== Stage 2 predict 202604010312 (新潟 R12) ===
1. 出馬表取得...
[NG] 出馬表取得失敗
[stage2-trace] predict_one_race returned None,
  diag={'status_code': 400, 'response_len': 0, 'has_horse_list': False,
        'cookie_present': True, 'url': '...'}

## R12 新潟 1h 前予測 (Stage 2) — Stage 1 fallback 採用
発走: 16:10 / レース: 4歳以上1勝クラス

### Stage 2 失敗 (netkeiba_block)
error: `netkeiba HTTP 400 (server block) / len=0`
  - netkeiba HTTP 400 (server block 想定、 Session #62/63 既知)

### 採用予測 = 朝予測 (Stage 1) top3 ★
1. 11 ハイクオリティ (score=0.648)
2. 12 マテンロウミラクル
3. 8 カレンラップスター

※ Stage 2 取得不可のため、 朝予測 (Stage 1) を本 R の最終予測として扱う。
※ retry: 次 fire (30 分後) で自動再試行 (cache に書込み無し)

[stage2-trace] skip cache write (error_kind=netkeiba_block) → next fire で再試行
```

→ ✅ 診断情報 captured / fallback 明示 / cache 書込みスキップ 確認。

### Test 2: JSON 出力

`data/v18/pre_race_predict_5_9_R12_新潟_202604010312.json`:
```json
{
  "race_id": "202604010312",
  "stage2": {
    "error": "netkeiba HTTP 400 (server block) / len=0",
    "error_kind": "netkeiba_block",
    "diag": {"status_code": 400, "response_len": 0, "has_horse_list": false,
             "cookie_present": true, "url": "..."}
  },
  "morning": {...},
  "ts": "2026-05-09T16:57:19"
}
```

→ ✅ 診断情報 JSON に保存、 後の analytics で `error_kind` 集計可能。

### Test 3: --check-next-1h (probe 起動)

5/9 16:57 時点 → `candidates=0` (race day 終了) のため probe 経路 invoke せず。
2026-05-10 朝の test fire 待ち、 or `--race-id` で manual 確認可能。

---

## 安全性確認

| 項目 | 結果 |
|---|---|
| `predict_one_race.py` 不変 | ✅ git diff なし |
| `predict_core.py` 不変 | ✅ git diff なし |
| `daily_predict.py` 不変 | ✅ git diff なし |
| `app.py` 不変 | ✅ git diff なし |
| V15 model file 不変 | ✅ git diff なし |
| schtasks 不変 | ✅ Session #65 C 9 件 不変 |
| 修正 file | `tools/stage2_predict.py` のみ |

---

## 5/16 V18 trial への含意

- netkeiba block が 5/16 までに解除されない場合: Stage 2 は全 R fallback、 朝予測のみで運用
- block 解除された場合: probe で検知、 個別 R 予測 自動再開
- どちらの場合も V18 trial の妨害なし (V15 不変)
- block 解除を継続監視するため、 30 分毎の probe ログを 5/10 朝 backfill で確認
