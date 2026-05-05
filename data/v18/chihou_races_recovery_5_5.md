# chihou_races_2020_2025.csv 復旧調査結果

**作成**: 2026-05-05 夜 / 緊急 3 件 #3
**結論**: **真の blocker ではなかった**。 5/12 NAR paper には影響なし。

## 1. 経緯

UPDATE_INVENTORY §1 で 🔴緊急として「`chihou_races_2020_2025.csv` 不在で 5/12 NAR paper 開始の唯一最大 blocker」と記述。
HANDOFF_5_5_TO_5_9.md L389 でも「strict OOS 評価不能」と既知問題扱い。

## 2. 実態調査

### 既存 CSV 確認

```
data/chihou_races_full.csv      1.9 MB  17,072 行  3/11 stale (TARGET由来形式)
                                末尾: 2009-09-08 〜 2020-03-19 浦和
data/nar_all_races.csv         13.0 MB  54,160 行  3/11 stale (netkeiba形式)
                                末尾: 2024-01-01 〜 2025-05-31 推定
```

### archive/nar/ 探索

```
archive/nar/
├── backtest_nar_*.json (3件)
├── backtest_nar_*.py (2件)
├── keiba_model_nar_v4.pkl  ← 本物のモデル
├── keiba_model_v10_nar_ref.pkl
├── optimize_nar_conditions.py
├── train_nar_v4.py            ← 学習スクリプト
└── train_nar*.py (他5件)
```

→ **chihou_races_2020_2025.csv の git/archive コピーは存在しない**

### train_nar_v4.py の依存解析

```python
# archive/nar/train_nar_v4.py L20-25
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
SCRAPED_CSV = os.path.join(DATA_DIR, 'nar_all_races.csv')      # ← 実使用
OLD_CSV = os.path.join(DATA_DIR, 'chihou_races_2020_2025.csv') # ← 参照のみ、未使用
MODEL_PATH = os.path.join(BASE_DIR, 'keiba_model_v9_nar.pkl')
```

```python
# L46-54 load_and_prepare_data()
df = pd.read_csv(SCRAPED_CSV, encoding='utf-8')   # ← nar_all_races のみ load
print(f'  Scraped: {len(df)} rows, {df["race_id"].nunique()} races')

# Note: old KDSCOPE data (184 races) has different schema, not worth merging
# with 4800+ new scraped races
```

→ **`OLD_CSV` (= `chihou_races_2020_2025.csv`) は変数定義のみで実際は読み込まれていない**。コメントで「KDSCOPE data not worth merging」と明示。

## 3. 結論

| 観点 | 判定 |
|------|------|
| 5/12 NAR paper 開始の blocker か | **NO** (nar_all_races.csv 54,160 行で v4 学習・推論完結) |
| strict OOS 評価不能か | △ (nar_all_races.csv 自体が 2024-2025 のみで時系列分割は限定的、ただし train_nar_v4 の学習・評価は同 CSV で完結) |
| HANDOFF L389 の記述 | **誤り**。 V20 統合モデル設計 (Phase 3、6 月後半) で JRA + NAR 統合学習する場合に「あれば望ましい」程度 |
| 緊急対応必要性 | **不要** |

## 4. 関連事実

- NAR v4 model `data/nar/models/keiba_model_nar_v4.pkl` (167 KB) は 5/5 柏記念で 0.777 完全再現済 (HANDOFF L109)
- `tools/predict_nar.py` も nar_all_races.csv ベース、chihou_races_2020_2025.csv は参照しない
- Phase 3 V20 構想 (`data/v18/jra_nar_integration_plan.md`) でも nar_all_races.csv 前提

## 5. 残課題 (低優先度、5/24 Phase 3 移行後)

NAR データの真の課題:
- `nar_all_races.csv` が **2024-2025 の 1 年強** しかカバーしない (54,160 行 / 4,821 races)
- 時系列穴あり (2024-02〜12 の連続性) - HANDOFF L96-105 既知
- **2026 年データ 0 行** ← 5/12 paper 開始までに backfill 推奨

→ これらは別タスクで対応 (`HANDOFF_5_5_TO_5_9.md` Section 4 M4 既定義、60min)。

## 6. UPDATE_INVENTORY 訂正

UPDATE_INVENTORY §0 で 🔴緊急 3 件のうち #3 に挙げていた `chihou_races_2020_2025.csv` 不在を **🟢低 (誤情報) に格下げ**。
代わりに 🟠高 として「nar_all_races.csv 2026 年分 backfill (5/12 paper 前)」を追加。

## 7. 5/12 NAR paper 開始 GO/no-go 判定

| 条件 | 状態 |
|------|------|
| nar_all_races.csv 存在 | ✅ (3/11 mtime、再 scrape 推奨) |
| keiba_model_nar_v4.pkl 存在 | ✅ |
| scrape_nar_today.py 実装 | ✅ (commit eeb48e45) |
| scrape_nar_results.py 実装 | ✅ (commit eeb48e45) |
| schtasks 5 件 admin 登録 | ✅ |
| chihou_races_2020_2025.csv | ❌ → **不要と判明、blocker から除外** |

→ **5/12 NAR paper 開始 GO**。 5/8-5/11 の隙間で nar_all_races.csv の 2025-06〜2026-05 分 backfill 推奨。
