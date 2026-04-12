---
name: scrape-status
description: 全スクレイピング進捗確認 — netkeiba/JRDB系のジョブ進捗・ファイル鮮度・ガード状態を一覧する。
---

# scrape-status — 全スクレイピング進捗確認

## 1. ガード状態
```bash
python -c "from tools.scraper_guard import is_scraping_allowed; print('allowed:', is_scraping_allowed())"
```
- False = 金22:00〜月06:00 でブロック中

## 2. 主要データファイルの鮮度
```bash
ls -lh data/jra_races_full.csv data/training_times.csv data/odds_history.csv \
       data/netkeiba_speed_index.csv data/netkeiba_race_review.csv 2>&1
```

## 3. 進捗JSON
```bash
cat data/scrape_progress.json 2>&1
cat data/_upset_scrape_progress.json 2>&1
```

## 4. 当日プレミアムキャッシュ
```bash
ls data/weekly_premium_cache/ | tail -5
```

## 5. ログ最終行
```bash
ls -t logs/scrape_*.log 2>/dev/null | head -5 | xargs -I{} sh -c 'echo "=={}=="; tail -3 "{}"'
```

## 6. Cookie有効性
```bash
python tools/refresh_cookie.py --check
```

## 7. 当日のオッズ基準CSV
```bash
ls -lh data/odds_base_$(date +%Y%m%d).csv 2>&1
```
