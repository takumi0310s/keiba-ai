# Session #68 A: stage2_predict.py audit

**作成**: 2026-05-09 16:53 (Session #68、 dev/two-stage)
**対象**: `tools/stage2_predict.py` の `predict_one_race returned None` エラー
**log**: `logs/stage2_predict.log` (384 行、 5/9 13:00-15:30 の 7 回 fire)

---

## fire 履歴 (logs/stage2_predict.log)

| fire 時刻 (推定) | candidates | 試行 (dedup skip 除く) | 結果 |
|---|---|---|---|
| ~13:00 | 6 件 | 6 件 試行 | 全て [NG] 出馬表取得失敗 |
| ~13:30 | 5 件 | 2 件 試行 | 全て fail |
| ~14:00 | 5 件 | 3 件 試行 | 全て fail |
| ~14:30 | 5 件 | 3 件 試行 | 全て fail |
| ~15:00 | 4 件 | 2 件 試行 | 全て fail |
| ~15:30 | 2 件 | 1 件 試行 | 全て fail |

合計: **試行 17 件、 全て fail (成功 0 件)**。

## 失敗した R 一覧

```
202605020507  東京 R7  4歳以上1勝クラス
202608030508  京都 R8  …
202604010309  新潟 R9  …
202605020508  東京 R8  …
202608030509  京都 R9  …
202604010310  新潟 R10 …
202605020508  (再試行 dedup skip 後 再試行)
202608030509  …
202604010310  …
202605020509  東京 R9  …
202608030510  京都 R10 …
202605020510  東京 R10
202604010311  新潟 R11 駿風 S
202608030511  京都 R11 京都新聞杯
202605020511  東京 R11 エプソムカップ
202608030512  京都 R12 4歳以上2勝クラス
202605020512  東京 R12 4歳以上2勝クラス
```

→ 場・距離・グレード問わず **全 R 失敗**。 race-specific bug ではない。

## 失敗 path (predict_one_race)

```
1. 出馬表取得...
[NG] 出馬表取得失敗   ← predict_one_race.py L33-34
return None         ← predict_one_race.py L34
```

→ `predict_core.parse_shutuba(race_id)` が `(_, [], _, _)` を返している。
   `horses` 空 list なので predict_one_race は None return。

## stage2_predict.py 側の挙動

`predict_stage2()` (L118-144):
```python
ret = por.predict_one_race(race_id)
if ret is None:
    return {"error": "predict_one_race returned None"}
```

→ 何も診断情報なく `error: predict_one_race returned None` のみ Discord 送信。

## 副次バグ (発見)

`predict_one()` (L235-274):
- 失敗時も cache に書き込まれる (L272-273)
- 結果: 次の fire で dedup skip → 同 R 再試行が永久に発生しない
- 修復方針: stage2.error が None の時のみ cache 書き込み

## 結論

audit 完了。 全 R 失敗の原因は `parse_shutuba` 側 (netkeiba 取得失敗)。
root cause は B で確定。
