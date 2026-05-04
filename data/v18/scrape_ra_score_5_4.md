# A. ra_score (data/netkeiba_race_analysis.csv) 5/4 再取得 試行結果

生成: 2026-05-04 (Opus xhigh, Session#8)

## 結論: ⚠️ **未完了** — 上流の `jra_races_full.csv` が 38日 stale で 2026年データなし、blocker

## 試行内容

```bash
PYTHONPATH=. python tools/scrape_stable_comment.py --year 2026 --limit 50
```

```
============================================================
  厩舎コメント一括取得 (comment.html)
  Years: [2026]
============================================================
  2026: 0 special/maiden races
  Already scraped: 3744 races
  To scrape: 0 races
  Nothing to scrape!
```

## 原因

`tools/scrape_stable_comment.py` の `get_special_maiden_race_ids(year)` 関数:

```python
csv_path = os.path.join(DATA_DIR, 'jra_races_full.csv')
df = pd.read_csv(csv_path, usecols=['year', 'race_id', 'class_code', 'race_name'])
yr2 = str(year % 100)
df = df[df['year'] == yr2]  # 2026 filter
```

→ `jra_races_full.csv` には 2026年データが ない (3/27 が最終更新)。  
   よって `2026: 0 special/maiden races` となり、新規取得 0件。

## 既存 CSV 状況

```
data/netkeiba_race_analysis.csv
  rows: 52,765 (3/27まで)
  age: 38d stale
  影響: V17 features ra_score = 全0/NaN
```

## 復旧経路

### 経路1: jra_races_full.csv を先に更新 (大規模、~3-6h)

```bash
# 既存 scrape script 探索
grep -l "jra_races_full" tools/*.py

# 該当する scraper を実行 (要確認)
python tools/scrape_master_index.py  # 候補
```

→ 別タスクとして時間枠取り、本セッションでは未実行。

### 経路2: race_id 直接指定で 5/3 のみ取得 (個別、~10min)

```bash
# 5/3 各レース race_id をループで指定
for rid in 202608030411 202608030412 202604010211 ...; do
  python tools/scrape_stable_comment.py --race_id $rid  # 引数追加必要
done
```

→ scrape_stable_comment.py は --race_id 引数未対応、scrape_data_analysis.py は対応 (但し別 CSV)。

### 経路3: jra_races_full.csv を 2026 部分のみ追加更新

別 scraper で 2026 only fetch → existing CSV に append。

## 5/4 セッションでの判断

🔴 **本セッションでは経路1/2/3 共に着手しない**:
- 経路1: 3-6時間、本セッション残予算で完了不能
- 経路2: scrape script に --race_id 引数追加が必要、副作用リスク
- 経路3: 別の scraper 探索、時間制約

→ **次セッション (5/5 以降) で別途着手**。Phase 2.5 第1週の課題として継続。

## 影響範囲

- V17 ULTRA-CLEAN model の `ra_score` feature が引き続き全 NaN
- Phase 1.7 audit の features 充足率 ~62% は本問題の影響
- 5/9 投資判断には影響なし (V15 単独運用、ra_score 不使用)

## TODO (次セッション)

- [ ] `jra_races_full.csv` 2026 部分の更新 scraper 特定 + 実行
- [ ] 完了後 `tools/scrape_stable_comment.py --year 2026` 再実行
- [ ] V17 features 充足率 再 audit
