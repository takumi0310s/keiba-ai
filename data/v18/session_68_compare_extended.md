# Session #68 D: stage_compare_5_9 拡張 (失敗 R の handling + 3 系統 hit rate)

**作成**: 2026-05-09 17:03 (Session #68、 dev/two-stage)
**修正 file**: `tools/stage_compare_5_9.py` (1 file 修正、 後方互換 維持)

---

## 拡張内容 (3 項目)

### 1. error_kind 集計 (失敗 R の内訳)

`compare_pair()` で stage2 blob の `error_kind` / `diag` を保存:
```python
{
    ...
    "stage2_error_kind": s2.get("error_kind"),  # netkeiba_block / shutuba_empty / etc
    "stage2_diag": s2.get("diag", {}),
}
```

aggregate 出力に `stage2_error_kinds` dict 追加。 Session #68 C 修復前の 旧 JSON
(error_kind 無し) は `_infer_kind()` で error 文字列から逆引き:
- "HTTP 400" / "server block" → `netkeiba_block`
- "returned None" → `shutuba_empty`
- その他 → `other`

### 2. 3 系統 hit rate 集計

Stage 2 失敗 R が大量にある場合の verdict は、 単純 平均では偏る。
Session #68 D で **3 系統並記** とした:

| 系統 | 内容 | 用途 |
|---|---|---|
| **系統 1**: morning_only | 全 R で朝予測 top1 が trio 入り 率 | baseline (Stage 2 なし環境の参照) |
| **系統 2**: stage2_success_only | Stage 2 成功 R に限った top1 入賞率 (朝 vs s2 比較) | Stage 2 効果測定 |
| **系統 3**: integrated | s2 成功 R は s2 top1、 失敗 R は morning fallback | **実運用方針** (Session #68 修復後) |

aggregate 出力 key:
```python
{
    "hit_rate_morning_only":              <float>  # 系統 1
    "hit_rate_stage2_only_morning_ref":   <float>  # 系統 2 morning ref
    "hit_rate_stage2_only_stage2_ref":    <float>  # 系統 2 stage2 ref
    "hit_rate_integrated":                <float>  # 系統 3
}
```

### 3. summary md レポート 拡張

3 系統 を分けて表示。 失敗 R の error_kind 内訳も明示。
旧 key (`morning_top1_in_trio_rate` 等) は **後方互換** で維持。

---

## 動作確認 (5/9 17:03)

```
$ python tools/stage_compare_5_9.py --summary

# Session #65 D + Session #68 D: 朝 vs 1h 前 比較 summary (20260509)

## 累積 metrics
- 比較対象 R: 15
- Stage 2 成功 R: 0
- Stage 2 失敗 R: 15 (100.0%)
- 失敗 R の error_kind 内訳:
  - shutuba_empty: 14
  - netkeiba_block: 1

## 実結果 と統合 (3 系統 hit rate)
- verdict 取得 R (全体): 0
- verdict 取得 R (Stage 2 成功 のみ): 0
- 実結果未取得 (5/10 朝 backfill 後 再実行)
```

→ ✅ error_kind 内訳 OK。 Session #68 C 修復後の JSON は `netkeiba_block` で正しく分類。
→ verdict 取得後に 3 系統 hit rate が出る。 5/10 朝 backfill 待ち。

---

## 5/10 朝 backfill 想定 出力 (placeholder)

verdict.json 用意後 (Session #61 realtime_5_9 か手動 backfill) で:

```
### 系統 1: 朝予測のみ (全 R)
- 朝 top1 が trio 入り 率 = 33.3%

### 系統 2: Stage 2 成功 R のみ
- 朝 top1 が trio 入り 率 = N/A (5/9 は Stage 2 成功 0、 系統 2 算出不可)
- Stage 2 top1 が trio 入り 率 = N/A
- Stage 2 効果 (差) = N/A

### 系統 3: integrated (s2 成功は s2、 失敗は morning fallback)
- integrated top1 が trio 入り 率 = 33.3% (= 系統 1 と同じ、 全 R fallback のため)
- 朝のみ vs integrated 差 = 0.0 pt
```

→ 5/9 は Stage 2 全 fail のため 系統 2 / 3 は morning と同値。
   block 解除されれば 系統 2 で Stage 2 効果が出る (5/16 V18 trial で再評価)。

---

## 5/16 V18 trial への含意

- block 解除前: 系統 1 / 系統 3 が同値、 Stage 2 system は noop
- block 解除後: 系統 2 で Stage 2 効果測定可能、 系統 3 で実運用 ROI 推計
- どちらでも V15 不変 / V18 trial 妨害なし

5/10 朝 backfill 後の行動:
1. `data/v18/verdicts_5_9_realtime.json` 用意 (Session #61 realtime_5_9 から or 手動)
2. `python tools/stage_compare_5_9.py --summary` で 3 系統 hit rate 確認
3. 系統 2 が算出可能 (Stage 2 成功 R > 0) なら 朝 vs Stage 2 差を verdict
