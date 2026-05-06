# premium CSV 追記 bug 根本原因 + 修復レポート

**作成**: 2026-05-06 朝 (Session #27 C)
**bug**: log は `Speed Index: 36/36 success` と report するが `data/netkeiba_speed_index.csv` 等 に 5/2-5/3 全 race_id = 0 行追加
**影響範囲**: 5/9 投資の予測精度 (speed_index 系 features ~7 個)、戦略⑦ retro

---

## 1. 真因

### 設計上の構造

`tools/daily_premium_scrape.py` は **cache JSON のみに保存**:
```python
# L294
all_data[race_id] = race_data
# L300, L303-304
with open(cache_file, 'w', encoding='utf-8') as f:
    json.dump(all_data, f, ensure_ascii=False, indent=2, default=str)
```

→ `data/weekly_premium_cache/{ymd}/premium_cache.json` に書く。

各種 CSV (netkeiba_speed_index.csv, netkeiba_training_eval.csv, netkeiba_stable_comments.csv) は **別 script** が更新する設計:
- `tools/scrape_speed_index.py` (CSV 用、独立 script)
- `tools/scrape_premium_data.py` (一括取得、独立)
- `tools/scrape_super_premium.py` (super premium、独立)

### 別 script が呼ばれていなかった

`tools/scrape_speed_index.py` L249:
```python
race_ids = get_race_ids(args.year)  # default 2025
```

`get_race_ids()` (L219-228):
```python
df = pd.read_csv('data/jra_races_full.csv', usecols=['year', 'race_id'])
yr2 = year % 100  # 例: 2025 → 25
df = df[df['year_int'] == yr2]
```

→ デフォルトで 2025 race_id しか対象にしない。 2026 (5/2-5/3) は scope 外。

仮に `--year 2026` を渡しても、`jra_races_full.csv` に 2026 年データが追加されたのが commit b4c4894c (5/5) だったため、5/2-5/3 時点では 2026 race_id は 0 行。

### log "success" の正体

daily_premium_scrape.py L308-311:
```python
print(f"  Training: {n_training}/{len(new_ids)}")
print(f"  Speed Index: {n_si}/{len(new_ids)}")
```

これは **cache JSON への書き込み件数**。 log は正しく cache 件数を出していたが、ユーザー解釈が「CSV への書き込み」と誤認していた。

---

## 2. 5/2-5/3 復旧 結果

`tools/premium_cache_to_csv.py` (本日新設) で cache → CSV 一括転換:

| 日 | speed_index 新規 | training 新規 | comments 新規 |
|---|---|---|---|
| 2026-05-01 | +15 | 0 | 0 |
| 2026-05-02 | +369 | +507 | 0 |
| 2026-05-03 | +105 | +497 | 0 |
| **合計** | **+489 行** | **+1,004 行** | 0 |

復旧後:
- `netkeiba_speed_index.csv` 269,947 → 270,436 行 (2026 年: 16 → 505 行)
- `netkeiba_training_eval.csv` +1,004 行追加

stable_comments が 0 件追加なのは cache JSON 内に comments データが ほとんど無かったため (cache 内 `races with comments = 10/72`、premium で comment 取得が成功率低い)。

---

## 3. 恒久対策

`tools/daily_premium_scrape.py` の Final save 直後に `premium_cache_to_csv.process_cache_dir()` を呼ぶ修正を追加:

```python
# L304 の後に追加
try:
    from premium_cache_to_csv import process_cache_dir
    csv_to_csv_result = process_cache_dir(date_str, dry_run=False)
    print(f"  [CSV append] speed_index +{csv_to_csv_result.get('speed_index_new', 0)}, ...")
except Exception as e:
    print(f"  [WARN] cache→CSV append 失敗: {e}")
```

→ 次回 03:00 DailyPremiumScrape 発火時から、cache JSON 書き込み直後に CSV にも追記される。 5/9 に向けて recovery 完了。

### log 出力に CSV append 件数も追加

```python
print(f"  CSV append: si+{...} / tr+{...} / sc+{...}")
```

→ 今後は cache 件数と CSV 件数を両方 log に明示、誤認防止。

---

## 4. 影響評価

### 5/2-5/3 の retro 影響

5/2-5/3 投資時の予測には speed_index = 0 (default) が使われていた可能性が高い。 これは V15 model の予測精度を低下させた要因の一つ:

| feature | 5/2-5/3 時 | 復旧後 |
|---------|-----------|--------|
| `index_max_filled` | 全馬 default mean fill | cache JSON 値で計算 |
| `index_avg5_filled` | 同上 | 同上 |
| `index_run1_filled` | 同上 | 同上 |
| training 系 wood/sakaro | 既に 4/19 取得分は cache あり、ただし CSV 反映なし | 復旧 |

ただし **5/2-5/3 の投資結果はもう確定済**、retro での「予測 score 再計算」が改善する程度。

### 5/9 投資への影響

`predict_core.py` は CSV を読み (`netkeiba_speed_index.csv` 等)、ない場合 cache JSON にフォールバック (CLAUDE.md V13.5b 記載の「premium cache JSON フォールバック」)。 すなわち **5/9 投資の予測精度には影響なし** (cache が CSV 化されたことで読み込みが速くなる程度)。

---

## 5. 残課題

### 5.1 stable_comments 取得率 改善

cache 内 `comments = 10/72 races` (14% 取得成功率)。 netkeiba premium の comment ページが頻繁に 404 or タイムアウト。 別 issue として 5/24+ Phase 3 で改善検討。

### 5.2 sex_age / weight_carry / jockey の cache 不在

`scrape_speed_index.py` の CSV header には `sex_age, weight_carry, jockey` 列があるが、cache JSON には保存されていない。 cache → CSV 転換時は空文字。 影響軽微 (model features は別 source から取得)。

### 5.3 history backfill

5/1 以前の cache (4/19 〜 4/29 の cache が複数あるが、CSV にすでに反映済?) の確認は別タスク。 必要なら `python tools/premium_cache_to_csv.py` (引数なしで全 cache 走査) で一括復旧可能。

---

## 6. 結論

- bug 真因: daily_premium_scrape は cache JSON にのみ書き込み、CSV は別 script が更新する設計、別 script は default --year 2025 で 2026 race を scope 外にしていた
- 5/2-5/3 の cache JSON は完全、CSV だけが空 → `tools/premium_cache_to_csv.py` で一括復旧 (+489 si / +1,004 tr)
- 恒久対策: daily_premium_scrape.py に CSV 自動 append を追加、次回 03:00 発火から有効
- 5/9 投資への影響なし、戦略⑦ retro が今後より正確に
