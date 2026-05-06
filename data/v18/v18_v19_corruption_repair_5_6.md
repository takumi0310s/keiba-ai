# V18/V19 model corruption 修復 (Session #36 A)

**作成**: 2026-05-07 深夜 (Session #36 A、就寝中マラソン)
**結論**: 🟢 **修復完了**、 V18 features 190、 Session #35 sr 拡張 4 features 確認

---

## 1. 真因

V18/V19 lgb model file の **CRLF 再 corruption**:

```
$ head -c 100 data/v18/models/v18_tansho_lgb.txt | od -c
0000000   t   r   e   e  \r  \n   v   e   r   s   i   o   n   =   v   4
0000020  \r  \n   n   u   m   _   c   l   a   s   s   =   1  \r  \n   n
```

LightGBM は LF を期待、CRLF だと `[LightGBM] [Fatal] Model format error, expect a tree here.` で load 失敗。

Session #32 で同様の修復済 (commit 777cc08e) だが、 5/5 16:22 の何らかの操作 (学習 script の write text mode 等) で CRLF に再変換された可能性。

`.gitattributes` 設定 (`data/v18/models/*.txt -text`) は既に存在 → git の自動変換は抑止済、 別経路で CRLF が混入した。

---

## 2. 修復

```python
# data/v18/models/v18_tansho_lgb.txt + v19_fukusho_lgb.txt
import shutil, os

for path in ['data/v18/models/v18_tansho_lgb.txt',
             'data/v18/models/v19_fukusho_lgb.txt']:
    # backup (rollback 用)
    shutil.copy2(path, path + '.bak_session36_pre')
    # CRLF → LF
    with open(path, 'rb') as f:
        data = f.read()
    new_data = data.replace(b'\r\n', b'\n')
    with open(path, 'wb') as f:
        f.write(new_data)
```

### 結果

| file | size before (CRLF) | size after (LF) | reduction |
|------|-------------------|----------------|-----------|
| v18_tansho_lgb.txt | 27,654,466 | 27,635,323 | -19,143 (CRLF→LF 分) |
| v19_fukusho_lgb.txt | 19,354,323 | 19,340,919 | -13,404 |

backup 保存:
- `data/v18/models/v18_tansho_lgb.txt.bak_session36_pre`
- `data/v18/models/v19_fukusho_lgb.txt.bak_session36_pre`

---

## 3. load 確認

```python
import lightgbm as lgb
v18 = lgb.Booster(model_file='data/v18/models/v18_tansho_lgb.txt')
print(f'V18 features: {len(v18.feature_name())}')   # 190 ✅
v19 = lgb.Booster(model_file='data/v18/models/v19_fukusho_lgb.txt')
print(f'V19 features: {len(v19.feature_name())}')   # 190 ✅
```

→ **両 model load 完全成功**、 retro 実行可能状態。

---

## 4. Session #35 sr 拡張 features の確認

V18 features list に sr/bias 関連 13 件確認:

```
prev_race_first3f
jrdb_prev_track_bias
sr_first3f_avg          ← Session #35 追加
sr_bias_homestr         ← Session #35 追加
sr_bias_4corner         ← Session #35 追加
sr_pace_up_pos          ← Session #35 追加
srb_bias_1corner        ← 未生成 (今後の対応)
srb_bias_2corner        ← 未生成
srb_bias_backstr        ← 未生成
srb_bias_3corner        ← 未生成
srb_bias_4corner        ← 未生成
srb_bias_straight       ← 未生成
srb_pace_up_pos         ← 未生成
```

→ **Session #35 で追加した 4 features は V18 model で直接使われる** (設計通り)
→ さらに **srb_*_bias 7 features も V18 で使用、未 merge** (B/C で対応)

---

## 5. CRLF 再発防止 plan

### .gitattributes 確認

```
$ cat .gitattributes
# LightGBM model files: prevent CRLF conversion (causes "Model format error")
data/v17/models/*.txt -text
data/v18/models/*.txt -text       ← 既に設定済
data/v15.1/*.txt -text
data/_model_bak_*/*.txt -text
```

→ git 経由の自動変換は抑止済。

### 再発の真因候補

5/5 16:22 の更新で CRLF が混入。 候補:
- 学習 script の Python `open(file, 'w')` (text mode、 Windows で `\n` → `\r\n`)
- VS Code 等のエディタで save 時に LF → CRLF 自動変換
- Windows tool が write したもの

→ 学習 script を Phase 3 で再学習する際は、明示的に `open(file, 'wb')` or `newline='\n'` 指定が必要。

### 修復プレイブック (再発時)

```bash
# rollback
cp data/v18/models/v18_tansho_lgb.txt.bak_session36_pre data/v18/models/v18_tansho_lgb.txt

# or 修復
python -c "
data = open('data/v18/models/v18_tansho_lgb.txt', 'rb').read()
open('data/v18/models/v18_tansho_lgb.txt', 'wb').write(data.replace(b'\\r\\n', b'\\n'))
"
```

---

## 6. retro 実行可能性

V18/V19 model load OK で `tools/v18_v19_retro_full.py` 実行可能。 5/2-5/3 retro + 4/11-5/3 拡大が **C 領域** で可能。

---

## 7. 結論

🟢 **V18/V19 model corruption 修復完了**。
- V18 190 features / V19 190 features load OK
- Session #35 sr 拡張 4 features が V18 で使用確認 (設計通り)
- srb_*_bias 7 features 未 merge 発見 (B/C で対応)
- backup 保存、rollback 可能

5/13 plan 内の retro blocker 解消、 5/16 GO 確率向上に貢献。
