---
description: 1レース予測テスト (URL指定)
---

netkeiba の出馬表 URL 1件で予測パイプライン動作確認。

```bash
python predict_and_log.py "https://race.netkeiba.com/race/shutuba.html?race_id=XXXXXXXXXXXX"
```

確認内容:
- 出馬表取得 (頭数、距離、馬場)
- v15 Pattern B モデル読み込み
- 150 特徴量生成 (jrdb_kyi/sed/tyb 含む)
- 条件分類 (A-X)
- 買い目生成 (三連複7点 or 馬連2点)
- ログ記録 (cumulative_results.csv)

エラー検証:
- 特徴量数が 150/150 になるか
- 条件分類が暫定値ではなく実頭数で動くか
- オッズが取得できるか (cookie要)
