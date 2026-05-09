# Session #72 D: 自動 test (5 件 + 2 件 = 7 件 PASS)

**作成**: 2026-05-09 18:09 (Session #72、 dev/two-stage)
**file**: `tests/test_stage2_predict.py` (新規、 ~150 行)

---

## test ケース一覧 (7 件)

| # | 名前 | 確認内容 |
|---|---|---|
| 1 | test_load_full_predictions_5_10_success | 5/10 想定: csv あり → 全馬 dict list、 rank 順、 score 範囲 0-1、 必須 column |
| 2 | test_load_full_predictions_5_9_fallback | 5/9 以前: file 不在 → None (top3 fallback へ) |
| 3 | test_load_full_predictions_invalid_race_id | csv あるが race_id 不在 → None |
| 4 | test_build_horse_table_size_within_limit | 18 頭で table が DISCORD_BODY_SAFE_LIMIT (1700) 内 |
| 5 | test_build_horse_table_truncates_when_over_limit | 100 頭 + 異常長名で size 超過時 truncate + 「(以下 N 頭省略)」 |
| 6 | test_build_message_all_horses_success_path | 5/10+ Stage 2 成功 path: title / body / color (blue) 確認 |
| 7 | test_build_message_all_horses_block_path | 5/10+ Stage 2 失敗 path: 「Stage 2 状況 (失敗: netkeiba_block)」 + Stage 1 採用 明示、 color (yellow) |

ユーザー要望は 5 件、 size 上限を 2 case (内 18 頭 / 100 頭超過) と Stage 2 成功/失敗を 2 case に分けて 計 7 件で網羅。

---

## 実行結果

```
$ python -m pytest tests/test_stage2_predict.py -v

tests/test_stage2_predict.py::test_load_full_predictions_5_10_success PASSED
tests/test_stage2_predict.py::test_load_full_predictions_5_9_fallback PASSED
tests/test_stage2_predict.py::test_load_full_predictions_invalid_race_id PASSED
tests/test_stage2_predict.py::test_build_horse_table_size_within_limit PASSED
tests/test_stage2_predict.py::test_build_horse_table_truncates_when_over_limit PASSED
tests/test_stage2_predict.py::test_build_message_all_horses_success_path PASSED
tests/test_stage2_predict.py::test_build_message_all_horses_block_path PASSED

============================== 7 passed in 0.28s ==============================
```

→ 全 7 件 PASS ✅

---

## test 設計の特徴

1. **本番 data に依存しない** — `tmp_path` + `monkeypatch` で `DAILY_PRED_FULL_DIR` を mock
2. **predict_core / V15 model に依存しない** — `predict_one_race` を呼ばず、 純粋な
   `load_full_predictions` / `build_horse_table` / `build_message_all_horses` を test
3. **Discord 2000 char 上限を実検証** — title + body 長さを assert
4. **18 頭は典型 race size、 100 頭は extreme case** — 両端で truncate logic 検証
5. **Stage 2 成功 / 失敗 両 path を独立に test** — color / 状況文言 を確認

---

## 本 test framework の今後の活用

- 5/16 V18 trial 後の通知変更時に同 test を回せば regression 検知可
- Session #71 の daily_predictions_full schema が変更された場合も schema 更新を反映可能
- Discord 上限変更時 (例 4000 char に拡張) は `DISCORD_BODY_SAFE_LIMIT` 1 行のみで対応
