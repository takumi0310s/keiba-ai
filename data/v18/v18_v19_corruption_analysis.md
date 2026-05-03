# v18/v19 + v17 LGB Model 破損調査 + 経路1復旧結果

生成: 2026-05-03 (Opus xhigh)

## 結論: ✅ **経路1 (CRLF→LF 変換) で全 6 モデル完全復旧**

復旧前: 全 LightGBM .txt model `Model format error`
復旧後: 全 6 モデル正常 load + predict 動作確認

## 1. 破損状況

### 検出時点 (15:00 直前)

| File | Size (CRLF) | Format | Load Test |
|------|------------:|--------|-----------|
| v18_tansho_lgb.txt | 27,654,466 | CRLF, 5956 cols | ❌ Fatal: Model format error |
| v19_fukusho_lgb.txt | 19,354,323 | CRLF, 5465 cols | ❌ Fatal: Model format error |
| v17_morning_lgb_fold5.txt | 19,354,323 | CRLF, 5465 cols | ❌ Fatal |
| v17_ultraclean_lgb_fold5.txt | 22,768,048 | CRLF, ? cols | ❌ Fatal |
| v17_leakfree_lgb_fold5.txt | 29,471,390 | CRLF, ? cols | ❌ Fatal |
| v17_lgb_fold5.txt | 22,414,407 | CRLF, ? cols | ❌ Fatal |
| **JSON files (XGB)** | - | no line terminators | ✓ 影響なし |

**全 LGB .txt model が CRLF 化で破損**。XGB JSON は無影響 (single-line JSON で line ending 関係なし)。

## 2. 根本原因

```
data/v18/models/v18_tansho_lgb.txt (CRLF版): 27,654,466 bytes
data/v18/models/v18_tansho_lgb.txt (LF版):    27,635,323 bytes
差分:                                           19,143 bytes
```

→ \r 追加分が file size 増加と一致。**LF→CRLF 変換が起きている**。

### 原因仮説

1. **`core.autocrlf=true`** + **`git reset` 操作** → checkout 時の text auto-conversion
   - `git reflog` 確認: `20:29 / 22:22 reset: moving to HEAD` イベントあり
   - ただし models は `git ls-files` で untracked → 通常の autocrlf 対象外
2. **session#1 / #3 で何らかの text 処理** が CRLF 書き込み
   - 例: Python `open(fp, 'w')` (newline 未指定) で実行された可能性
3. **再 train save 時の偶発**:
   - LightGBM `save_model` は LF 出力するが、何かで再保存されたかも

### 結論

確定原因は不明だが、**復旧自体は CRLF→LF 変換で完全に可能**。

## 3. Load Test (復旧前/後)

```
=== 復旧前 (CRLF) ===
v18_tansho_lgb.txt: ❌ Fatal: Model format error
v19_fukusho_lgb.txt: ❌ Fatal: Model format error
v17_morning_lgb_fold5.txt: ❌ Fatal
v17_ultraclean_lgb_fold5.txt: ❌ Fatal
v17_leakfree_lgb_fold5.txt: ❌ Fatal
v17_lgb_fold5.txt: ❌ Fatal

=== 復旧後 (LF) ===
v18_tansho_lgb.txt: ✓ feats=190 trees=991
v19_fukusho_lgb.txt: ✓ feats=190 trees=689
v17_morning_lgb_fold5.txt: ✓ feats=190 trees=689
v17_ultraclean_lgb_fold5.txt: ✓ feats=196 trees=811
v17_leakfree_lgb_fold5.txt: ✓ feats=207 trees=1050
v17_lgb_fold5.txt: ✓ feats=219 trees=797

predict 動作確認: v19 zero-vec → p=[0.00070878] (正常)
```

## 4. 復旧手順 (実施済)

```python
# Backup → Convert → Load test
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
    with open(fp, 'rb') as f: c = f.read()
    c2 = c.replace(b'\r\n', b'\n')
    with open(fp + '.tmp_lf', 'wb') as f: f.write(c2)
    os.replace(fp + '.tmp_lf', fp)
```

## 5. バックアップ

CRLF 状態を `data/_model_bak_20260503/` に保存:

```
data/_model_bak_20260503/v18_tansho_lgb.txt.bak_crlf      (27,654,466 B)
data/_model_bak_20260503/v19_fukusho_lgb.txt.bak_crlf     (19,354,323 B)
data/_model_bak_20260503/v17_morning_lgb_fold5.txt.bak_crlf
data/_model_bak_20260503/v17_ultraclean_lgb_fold5.txt.bak_crlf
data/_model_bak_20260503/v17_leakfree_lgb_fold5.txt.bak_crlf
data/_model_bak_20260503/v17_lgb_fold5.txt.bak_crlf
```

ロールバック必要時: `cp data/_model_bak_20260503/X.bak_crlf data/v17/models/X` 等。

## 6. 再発防止

### 即時 (5/3 中)

1. ✅ git config から autocrlf 確認 → 既に `true` (このまま)
2. **`.gitattributes` に LightGBM model file の binary 指定追加** (推奨):
   ```
   # .gitattributes に追加
   *.txt -text       # 全 .txt を CRLF 変換しない (or)
   data/v18/models/*.txt -text  # 特定パスのみ
   data/v17/models/*.txt -text
   ```
   ただし既存 .txt (model 以外) も影響受けるため精査必要。

### 中期 (Phase 2.5)

3. **model save 時に explicit LF 書き出し検証**
   - LightGBM `save_model` 単体は LF。後段で何か変換していないか
4. **session 競合監視**:
   - 同時 model file 触る session の検知 (lock file?)

## 7. 後続作業

### 即実行可能 (model 復旧済み)

- ✅ `python tools/v18_v19_retro_full.py` 再実行 → 5/2-5/3 retro 完成
- ✅ midday script の v17_ultraclean 利用 (15:25 京都11R)
- ✅ morning script の v17_morning 利用

### Phase 2.5 (5/4-5/15)

- 楽観バイアス係数算出 (BT 295% / 149% vs 実 retro)
- 5/16 以降 v18/v19 部分実弾投資額確定

## TL;DR

- 全 LGB .txt model が CRLF 化で破損していた (XGB JSON は無影響)
- CRLF→LF 変換 で全 6 モデル完全復旧
- v18_v19_retro_full.py 再実行可能
- バックアップ `data/_model_bak_20260503/` に CRLF 版保存
- 再発防止に `.gitattributes` の `*.txt -text` 推奨
