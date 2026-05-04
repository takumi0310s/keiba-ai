# B. sc_score (data/netkeiba_stable_comments.csv) 5/4 再取得 試行結果

生成: 2026-05-04 (Opus xhigh, Session#8)

## 結論: ⚠️ **未完了** — A と同じ blocker (`jra_races_full.csv` 2026年なし)

## 試行内容

```bash
PYTHONPATH=. python tools/scrape_comments_bulk.py --years 2026
```

```
============================================================
  Stable Comments Bulk Scraper: [2026]
============================================================
  2026: 0 races
  Total: 0 unique race IDs
  Already done: 14067
  Remaining: 0
  Estimated: 0 min
  Nothing to do!
```

## 原因

`tools/scrape_comments_bulk.py` の `get_race_ids(year)` 関数:

```python
csv_path = os.path.join(DATA_DIR, 'jra_races_full.csv')
df = pd.read_csv(csv_path, usecols=['year', 'race_id'], dtype=str)
yr2 = year % 100
nk_ids = set()
for rid in df[df['year_int'] == yr2]['race_id'].dropna().unique():
    nk_ids.add(_target_to_netkeiba(rid))  # 10桁→12桁変換
```

→ `jra_races_full.csv` 2026 = 0 rows (3/27 で停止) → 2026 race_ids 抽出不能。

## 既存 CSV 状況

```
data/netkeiba_stable_comments.csv
  rows: 128,875 (3/27まで)
  age: 23d stale
  影響: V17 features sc_score = 全0/NaN
```

## A と同じく上流 blocker

A (ra_score) と B (sc_score) の **両方が `jra_races_full.csv` の 2026 部分** が無いと進行不能。

## 共通 復旧経路

### 推奨: jra_races_full.csv の 2026 部分追加スクレイパー作成

```python
# tools/update_jra_races_full_2026.py (新規作成必要)
# 1. 4/1-5/3 の race_id を jrdb_kyi.csv から取得
# 2. 各 race_id について netkeiba 結果ページから情報取得
#    (year, month, day, kai, course, nichi, race_num, race_name, class_code, ...)
# 3. jra_races_full.csv に append
```

時間: 4/1-5/3 = 33日、土日のみで ~10日 × 36R = ~360 races × 5sec = **30分**

→ 別タスクとして時間枠取り、本セッションでは未着手 (C/D 優先)。

## 5/4 セッションでの判断

🔴 **本セッションでは未実施**。次セッションで:
1. tools/update_jra_races_full_2026.py 作成
2. 4/1-5/3 races 取得 → jra_races_full.csv 更新
3. 完了後 A (scrape_stable_comment) + B (scrape_comments_bulk) 再実行

## 影響範囲

- V17 ULTRA-CLEAN features sc_score = 全 NaN 継続
- Phase 1.7 audit features 充足率改善せず
- 5/9 投資判断には影響なし (V15 単独運用)

## TODO (次セッション)

- [ ] tools/update_jra_races_full_2026.py 作成
- [ ] jra_races_full.csv 2026 部分更新 (4/1-5/3 races)
- [ ] A + B 再実行
- [ ] V17 features 充足率 再 audit
