---
description: 全スクレイピングジョブ進捗・ファイル鮮度
---

netkeiba/JRDB系の取得ジョブ進捗とファイル鮮度を一覧表示。

```bash
python tools/scrape_progress.py 2>&1 | tail -40
ls -la data/jrdb_*.csv 2>&1
ls -la data/feature_lookups.pkl 2>&1
```

確認項目:
- jrdb_kyi.csv (基本) — 5日以上古いと警告
- jrdb_sed.csv (前走) — 5日以上古いと警告
- jrdb_tyb.csv (当日) — 当日朝は0%、AM6:30以降取得
- jrdb_cyb.csv
- master_index カバレッジ
- training_eval カバレッジ
- v16 学習トリガ閾値達成判定
- Cookie 有効性 (`python tools/refresh_cookie.py --check`)
- SCRAPER-GUARD 状態

Discord通知: data/scrape_progress.json + Discord #updates
