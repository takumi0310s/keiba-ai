# 30 年 backtest data pipeline (Session #84)

> TFJV raw → features parquet → model 学習 → backtest 実行 → report 生成 までの一貫 pipeline。
> 作成: 2026-05-09 (Session #84)

---

## 1. 全体構成 (5 stage)

```
[Stage 1] TFJV raw (90 年 6 GB)
    ↓ tfjv_parser.py + 30 年抽出
[Stage 2] 30 年 features parquet (~50 GB)
    ↓ 月別 partition + LEAK 除外 + KKA 統合
[Stage 3] model 学習 (LGB+XGB+FT+IR × 5 strategies)
    ↓ multiprocessing 並列
[Stage 4] walk-forward backtest 実行
    ↓ 5 fold AUC + ROI + Sharpe
[Stage 5] report 生成 (markdown + plot)
```

---

## 2. Stage 1: TFJV → parquet

### 2.1 input
- TFJV raw: `C:/TFJV/` 配下 45,000+ files (Shift-JIS binary)
- 利用 datatype: UM / RA / SE / HR / WD / TM / TF / O1-O6 / W5 / H1

### 2.2 logic
- `tools/tfjv_parser.py` (Session #44 で実装済)
- 30 年抽出: 1995-2024 で filter
- 各 datatype を CSV 化 (中間)
- → parquet 化 (月別 partition、 high cardinality 対応)

### 2.3 output
- `data/backtest_30y/raw_parquet/{datatype}/{yyyy_mm}.parquet`
- 推定 size: ~10 GB

### 2.4 実装方針
```python
# tools/build_30y_raw_parquet.py (将来実装)
import polars as pl
from tools.tfjv_parser import parse_ra, parse_se, parse_hr, ...

for year in range(1995, 2025):
    for month in range(1, 13):
        df_ra = parse_ra(f"C:/TFJV/RA/{year}{month:02d}*.bin")
        df_se = parse_se(f"C:/TFJV/SE/{year}{month:02d}*.bin")
        # ...
        df_ra.write_parquet(f"data/backtest_30y/raw_parquet/RA/{year}_{month:02d}.parquet")
```

---

## 3. Stage 2: features engineering

### 3.1 input
- Stage 1 raw parquet
- KKA (Session #53) 統合 file: `data/jrdb_kka_v2.csv`
- sib_*_exp (Session #38 修正版): expanding window

### 3.2 logic
- 既存 features logic を流用 (predict_core.py の build_features 関数)
- LEAK 除外 12 件:
  - LEAK_FEATURES_A (8 件): odds_log / horse_weight / 関連 7 件
  - SKB_LEAK_FEATURES (10 件): skb_kishi_code 系 (Session #38)
  - 重複考慮で計 18 件除外
- KKA 統合: SED から KYI で代替
- features 200+ 構築

### 3.3 output
- `data/backtest_30y/features_parquet/{yyyy_mm}.parquet`
- 推定 size: ~50 GB

### 3.4 検証
- LEAK 監査: train/test で features の corr_target を比較
- shift 比較: 1着 vs 着外 で features 分布
- カバレッジ: 各 features の non-null率 >= 80%

---

## 4. Stage 3: model 学習

### 4.1 4-model ensemble (V20 仕様)

| model | role | 備考 |
|-------|------|------|
| LGB | base | 最速、 baseline |
| XGB | base | LGB と相補 |
| FT-Transformer | base | tabular DL、 v13.5b で導入 |
| IntraRace Attention | base | レース内相対関係 (重み 0.35) |

→ Grid Ensemble で 重み最適化。

### 4.2 5 戦略 × 5 fold = 25 学習

| fold | strategy | model 数 | 学習時間 (推定) |
|------|----------|---------|--------------|
| 1-5 | V15 | 4 × 5 = 20 | 5-10 h |
| 1-5 | V18 | 4 × 5 = 20 | 5-10 h |
| 1-5 | V20 | 4 × 5 = 20 | 7-15 h |
| 1-5 | hybrid (V20 ベース) | 0 (V20 流用) | 0 h |
| 1-5 | V22 RL | 4 × 5 = 20 | 10-20 h |

→ 合計 **30-55 時間** (single CPU)、 multiprocessing で 1/4 〜 1/8 削減見込。

### 4.3 実装方針
```python
# tools/train_30y_backtest.py (将来実装)
from concurrent.futures import ProcessPoolExecutor

strategies = ["V15", "V18", "V20", "hybrid", "V22_RL"]
folds = range(1, 6)

with ProcessPoolExecutor(max_workers=4) as ex:
    futures = [
        ex.submit(train_strategy_fold, strategy, fold)
        for strategy in strategies
        for fold in folds
    ]
    for f in futures:
        f.result()
```

---

## 5. Stage 4: walk-forward backtest

### 5.1 logic
- 各 fold で valid year に対し inference
- 案B改 strict / 戦略⑦ / hybrid logic を適用して 投票生成
- HR_DATA から 配当 取得 → ROI 計算
- bootstrap CI95 (1000 resample) 算出

### 5.2 monte carlo simulation
- 各 fold で 1000 回 resample
- 累計 P/L 分布 算出
- 破産確率 P(loss > -¥30,000)

### 5.3 output
- `data/backtest_30y/results/{strategy}_{fold}.json`
- KPI 一覧: AUC / ROI / Sharpe / max DD / 投票数 / 的中率

---

## 6. Stage 5: report 生成

### 6.1 markdown report
- `docs/BACKTEST_30Y_REPORT.md` (将来生成)
- 5 戦略 × 5 fold の matrix
- 経年 ROI plot
- bootstrap CI95 plot

### 6.2 plot
- `data/backtest_30y/plots/` 配下
- AUC trend (年別)
- ROI trend (年別)
- 累計 P/L curve
- DD curve

### 6.3 conclusion section
- 各戦略の「最良 fold」「最悪 fold」
- robust ranking (CI95 lower で 順位)
- 採用判定 (V20 / hybrid / V22 RL)

---

## 7. 制約

### 7.1 storage
- 100 GB 必要、 D ドライブ 推奨
- TFJV raw は維持、 backtest 専用 dir 新規作成

### 7.2 学習時間
- 30-55 時間 (single CPU)
- multiprocessing で 1 週間以内 完走見込

### 7.3 model 検証 negative path
- 各 fold で AUC < 0.85 → 学習 data 不足 / LEAK 疑い
- 各 fold で ROI < 100% → 戦略 不適合 (drop)

---

## 8. 関連 doc

- `docs/BACKTEST_30_YEAR_DESIGN.md` — 範囲設計
- `docs/BACKTEST_STRATEGY_COMPARISON.md` — 戦略比較 plan
- `docs/BACKTEST_BUILD_TIMELINE.md` — schedule
- `docs/v18/tfjv_data_inventory_5_8.md` (Session #44 A)
- `tools/tfjv_parser.py` (Session #44 B 実装済)

---

**結論**: 5 stage pipeline で TFJV raw → 30 年 features → 5 戦略 × 5 fold 学習 → walk-forward backtest → report 生成。 storage 100 GB、 学習 30-55 h、 multiprocessing で 1 週間以内 完走見込。 実装は Session #84 では plan のみ、 Sprint 9 (10/1+) 着手。
