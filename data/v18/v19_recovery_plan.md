# v18/v19 model 復旧プラン (実行済)

生成: 2026-05-03

## エグゼクティブサマリー

✅ **経路1 (CRLF→LF) で全 6 LGB model 完全復旧、即実行済み**

| 項目 | 値 |
|------|---|
| 復旧経路 | 経路1 (CRLF→LF 変換) |
| 所要時間 | 1分 |
| 復旧 model 数 | 6 / 6 |
| データ損失 | なし |
| バックアップ | data/_model_bak_20260503/ |
| 再発防止 | .gitattributes 案推奨 (Phase 2.5) |

## 復旧結果 (全モデル)

| モデル | features | trees | load | predict |
|--------|--------:|------:|-----|---------|
| v18_tansho_lgb | 190 | 991 | ✓ | ✓ |
| v19_fukusho_lgb | 190 | 689 | ✓ | ✓ |
| v17_morning_lgb_fold5 | 190 | 689 | ✓ | ✓ |
| v17_ultraclean_lgb_fold5 | 196 | 811 | ✓ | ✓ |
| v17_leakfree_lgb_fold5 | 207 | 1050 | ✓ | ✓ |
| v17_lgb_fold5 | 219 | 797 | ✓ | ✓ |

XGB JSON は単行 JSON で line ending 影響なし、無修正で動作。

## 採用経路: 経路1 (CRLF→LF 変換)

```bash
# 1) Backup (CRLF 版を保存)
mkdir -p data/_model_bak_20260503
cp data/v18/models/*.txt data/_model_bak_20260503/
cp data/v17/models/*lgb*.txt data/_model_bak_20260503/

# 2) Convert (binary safe)
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
    tmp = fp + '.tmp_lf'
    with open(tmp,'wb') as f: f.write(c2)
    os.replace(tmp, fp)
"

# 3) Verify all models load
for f in data/v18/models/*.txt data/v17/models/*lgb*.txt; do
    python -c "import lightgbm as lgb; m = lgb.Booster(model_file='$f'); print(f'{m.num_feature()}f, {m.num_trees()}t')" 2>&1 | tail -1
done
```

実行結果: 全 6/6 OK (上記表参照)。

## 経路2: git checkout (不可)

```bash
# 確認結果
git ls-files data/v18/models/   # 空
git ls-files data/v17/models/   # 空
# → models 全て untracked、git checkout 経路使えず
```

## 経路3: 再 train (不要だが可能)

```bash
# v19 単独 retrain (~8min)
python train/run_v19_fukusho.py

# v18 単独 retrain (~7min)
python train/run_v18_tansho.py

# 両方 retrain (~15min)
python train/run_v18_tansho.py && python train/run_v19_fukusho.py
```

train data: `data/v17/_v17_train_df_cache.pkl` (1.2GB) 既存、再 train 即可能。

## 復旧後の検証

```bash
# 1. 全 model 読み込み確認 ✓ 完了

# 2. retro full 実行
python tools/v18_v19_retro_full.py
# → 5/2-5/3 67 races の v18/v19 inference
# → 楽観バイアス係数 (BT vs 実 retro)

# 3. 5/9 朝 morning script で v17_morning 利用確認 (06:30 自動)
```

## ロールバック手順 (もし問題発生)

```bash
# Option A: backup から CRLF 版に戻す (load は失敗)
cp data/_model_bak_20260503/v19_fukusho_lgb.txt.bak_crlf data/v18/models/v19_fukusho_lgb.txt

# Option B: 再 train (LF 版を新規生成)
python train/run_v19_fukusho.py
```

ただし CRLF 版は load 不可なので、Option A はロールバック意味薄。
問題発生時は **Option B (再 train)** が推奨。

## 再発防止 (Phase 2.5 推奨)

### 1. .gitattributes 設定

```
# .gitattributes (新規作成 推奨)
data/v18/models/*.txt -text
data/v17/models/*.txt -text
data/**/models/*.txt -text
```

→ git の autocrlf が model file に作用しない保証

### 2. Model save 時の確認

LightGBM `save_model` は内部で LF 出力。何かが後段で CRLF 変換していた場合に備え:

```python
# train script の最後で確認
with open(model_path, 'rb') as f:
    if b'\r\n' in f.read():
        print("WARNING: CRLF detected, converting to LF")
        # auto fix
```

### 3. Model file のチェックサム記録

```bash
# train 直後
md5sum data/v18/models/*.txt > data/v18/models/checksums.md5

# 異常検知時
md5sum -c data/v18/models/checksums.md5
```

## 後続作業

### 今すぐ (5/3 中)

- ✅ Path 1 全 model 復旧
- ✅ Documentation 完了
- ▶ `tools/v18_v19_retro_full.py` 実行 (in progress)

### Phase 2.5 (5/4-5/15)

- [ ] retro 結果から楽観バイアス係数算出
- [ ] .gitattributes 設定 (再発防止)
- [ ] model file checksum 記録 (CI 化)
- [ ] DailyPredict watchdog 化 (admin elevation で手動移行)

## TL;DR

- 全 LGB .txt model が CRLF 化で破損していた
- **経路1 (CRLF→LF) 1分で完全復旧、全 6 モデル動作確認済**
- バックアップ `data/_model_bak_20260503/` に CRLF 版保存
- v18_v19_retro_full.py 再実行可能 → 楽観バイアス確定
- 再発防止: `.gitattributes -text` 推奨
