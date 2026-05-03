# v18/v19 model 復旧経路評価

生成: 2026-05-03 (調査+評価+結果記録)

## 経路一覧

| 経路 | 概要 | 所要 | リスク | 結果 |
|------|------|-----:|-------|------|
| **1. CRLF→LF 変換** | sed/python 1コマンド | 1分 | 低 | ✅ **採用、全成功** |
| 2. git checkout | 過去 commit 復元 | 5分 | 低 | ❌ **不可** (model 非追跡) |
| 3. 再 train | run_v19_fukusho.py 等 | 1-3時間 | 中 | 不要 (経路1で復旧) |

## 経路1: CRLF→LF 変換 ✅ 採用

### 必要なもの (全て揃っていた)

- Python 3 (標準)
- 破損 model file (binary 安全 read/write)

### 手順

```bash
# 1. backup
mkdir -p data/_model_bak_20260503
cp data/v18/models/*.txt data/_model_bak_20260503/

# 2. CRLF→LF 変換
python -c "
import os
files = [
    'data/v18/models/v18_tansho_lgb.txt',
    'data/v18/models/v19_fukusho_lgb.txt',
    'data/v17/models/v17_morning_lgb_fold5.txt',
    'data/v17/models/v17_ultraclean_lgb_fold5.txt',
    'data/v17/models/v17_leakfree_lgb_fold5.txt',
    'data/v17/models/v17_lgb_fold5.txt',
]
for fp in files:
    with open(fp,'rb') as f: c = f.read()
    c2 = c.replace(b'\r\n', b'\n')
    with open(fp+'.tmp','wb') as f: f.write(c2)
    os.replace(fp+'.tmp', fp)
"

# 3. load test
for fn in data/v18/models/*.txt data/v17/models/*.txt; do
    python -c "import lightgbm as lgb; m = lgb.Booster(model_file='$fn'); print(f'{m.num_feature()}f, {m.num_trees()}trees')" 2>&1 | tail -1
done
```

### 結果 (実施済)

✅ **全 6 モデル正常 load**:
- v18_tansho: 190f / 991 trees
- v19_fukusho: 190f / 689 trees
- v17_morning: 190f / 689 trees
- v17_ultraclean: 196f / 811 trees
- v17_leakfree: 207f / 1050 trees
- v17_lgb_fold5: 219f / 797 trees

predict 動作確認 ✓ (v19 zero-vec → p=[0.00070878])

### リスク評価

| リスク | 発生確率 | 対策 |
|--------|--------|------|
| 内容欠損で復旧不能 | 低 (file 完全性に問題なし) | バックアップから経路2/3 |
| LightGBM パーサーが LF/CRLF 厳格不要 | 既に検証済 | n/a |
| 復旧成功でも performance 劣化 | 低 (binary 内容変わらず、改行のみ) | 簡易 BT で検証可 |

## 経路2: git checkout ❌ 不可能

### 理由

```bash
git ls-files data/v18/models/   # 空 = untracked
git ls-files data/v17/models/   # 空 = untracked
git log --all -- data/v18/models/v19_fukusho_lgb.txt   # 履歴なし
```

→ Model files は `.gitignore` (おそらく) で除外、git history に存在しない。
→ git checkout 経路は使えない。

### もし git に入れたい場合

```bash
# .gitignore から model path を削除した後
git add data/v18/models/*.txt data/v18/models/*.json
git commit -m "track v18/v19 models"
# 注: ファイルサイズ大きい (160MB+) → LFS 推奨
```

ただしリポジトリサイズが膨れる。LFS 設定が無いなら推奨しない。

## 経路3: 再 train (不要だが評価は完了)

### 必要リソース (再 train 時の参考)

#### Train script

| Model | Script | 状態 |
|-------|--------|------|
| v18_tansho | `train/run_v18_tansho.py` | ✓ 存在 |
| v19_fukusho | `train/run_v19_fukusho.py` | ✓ 存在 |
| v17_morning | `train/run_v17_morning.py` | ✓ 存在 |

#### Train data

| File | Size | Status |
|------|-----:|--------|
| `data/v17/_v17_train_df_cache.pkl` | 1.2GB | ✓ 存在 (mtime 4/29) |
| Phase 2 で使用済 (10年データ 2015-2024 → 2025 OOS) | - | - |

#### Library (確認済)

```
lightgbm  >=3.0
xgboost   >=1.7
pandas    >=1.5
numpy     >=1.20
scikit-learn  >=1.0
```

#### GPU

未使用 (LightGBM/XGBoost CPU 実装で十分速い)。

### 想定 train 時間

`run_v19_fukusho.py` が ~8 分 (Phase 2 実績、6 fold WF train)。
`run_v18_tansho.py` も同様 ~7 分。

合計 30 分以内。

### 結論

経路1 で復旧したため再 train **不要**。  
ただし下記いずれかの場合は経路3 採用:
- 経路1 で復旧した model の predict 精度に retrospective 異常 (検証で発見)
- model file が単純 CRLF 化以上の破損 (内容欠損)

## 推奨復旧プラン (本セッションで実施済)

```bash
# Step 1: backup
mkdir -p data/_model_bak_20260503
cp data/v18/models/*.txt data/_model_bak_20260503/
cp data/v17/models/*lgb*.txt data/_model_bak_20260503/

# Step 2: 経路1 (CRLF→LF) 一括変換
python << 'PY'
import os
files = [
    'data/v18/models/v18_tansho_lgb.txt',
    'data/v18/models/v19_fukusho_lgb.txt',
    'data/v17/models/v17_morning_lgb_fold5.txt',
    'data/v17/models/v17_ultraclean_lgb_fold5.txt',
    'data/v17/models/v17_leakfree_lgb_fold5.txt',
    'data/v17/models/v17_lgb_fold5.txt',
]
for fp in files:
    with open(fp,'rb') as f: c = f.read()
    if b'\r\n' not in c: continue
    c2 = c.replace(b'\r\n', b'\n')
    tmp = fp + '.tmp_lf'
    with open(tmp,'wb') as f: f.write(c2)
    os.replace(tmp, fp)
PY

# Step 3: load test
python -c "
import lightgbm as lgb
files = ['data/v18/models/v18_tansho_lgb.txt',
         'data/v18/models/v19_fukusho_lgb.txt']
for fp in files:
    m = lgb.Booster(model_file=fp)
    print(f'{fp}: {m.num_feature()}f, {m.num_trees()}trees')
"

# Step 4: retro 完全版 実行
python tools/v18_v19_retro_full.py
```

## ロールバック手順

経路1 で復旧した model に何か問題が出た場合:

```bash
# CRLF 版に戻す
for f in data/_model_bak_20260503/*.bak_crlf; do
    bn=$(basename "$f" .bak_crlf)
    # 元のパス推定
    if [[ "$bn" == "v18_"* ]] || [[ "$bn" == "v19_"* ]]; then
        cp "$f" "data/v18/models/$bn"
    else
        cp "$f" "data/v17/models/$bn"
    fi
done
```

ただし CRLF 版は load 失敗するため、ロールバックの意義は限定的。
→ ロールバック=経路3 (再 train) のみが現実的。

## 検証手順 (復旧後)

```bash
# 1. 全 model load 確認
for fn in data/v18/models/*.txt data/v17/models/*.txt; do
    python -c "import lightgbm as lgb; m = lgb.Booster(model_file='$fn'); print('$fn',m.num_feature(),m.num_trees())" 2>&1
done

# 2. 5/2-5/3 retro full 実行
python tools/v18_v19_retro_full.py
# → data/v18/v18_v19_retro_full_predictions.csv 生成

# 3. 楽観バイアス確定
# BT 2025 OOS ROI (v18: 295.1%, v19: 149.3%) vs 5/2-5/3 retro ROI
```

## TL;DR

- **経路1 (CRLF→LF) 完全採用**、全 6 LGB model 復旧成功
- 経路2 (git) 不可、経路3 (retrain) 不要
- 後続: `tools/v18_v19_retro_full.py` 実行可能 → 楽観バイアス確定
