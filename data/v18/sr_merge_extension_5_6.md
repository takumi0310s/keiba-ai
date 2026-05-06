# sr merge 拡張 (V18/V19 5/13 作業 前倒し) — 統合レポート

**作成**: 2026-05-07 朝 (Session #35)
**対象**: `tools/jrdb_features.py` L864-876 (sr merge logic)
**結論**: 🟢 **拡張完了、V15 動作完全不変 確認、5/13 作業 前倒し完了**

---

## 1. 設計 (A)

### 1.1 jrdb_sr.csv の構造

```
race_id, harlon_count, harlon_1..harlon_18, corner1_pos..corner4_pos,
pace_up_pos, tb_1corner..tb_homestr, race_comment
```

→ race-level data (馬単位ではない)、各 race に 1 行。
→ race_id format = netkeiba 形式 (例: `202605020101`)。

### 1.2 V162_FEATURES['jrdb_sr'] 4 件の合成方法

| feature | 計算 |
|---------|------|
| `sr_first3f_avg` | `mean(harlon_1, harlon_2, harlon_3)` |
| `sr_bias_homestr` | `tb_homestr` の各位の和 (例: "11133" → 1+1+1+3+3=9) |
| `sr_bias_4corner` | `tb_4corner` の各位の和 |
| `sr_pace_up_pos` | `pace_up_pos` 直接 numeric |

→ race-level の値、全馬同値 (broadcast)。

---

## 2. 実装 (B)

### 2.1 修正前 (`tools/jrdb_features.py` L864-876)

```python
_sr_path = os.path.join(DATA_DIR, 'jrdb_sr.csv')
if os.path.exists(_sr_path):
    try:
        _sr = pd.read_csv(_sr_path, encoding='utf-8-sig', dtype=str)
        _sr_race = _sr[_sr['race_id'].astype(str).str.zfill(12) == _rid_str]
        if len(_sr_race) > 0:
            _sr_row = _sr_race.iloc[-1]
            _tb = str(_sr_row.get('tb_homestr', ''))
            _inner = int(_tb[0]) if _tb and len(_tb) >= 1 and _tb[0].isdigit() else 2
            horses_df['jrdb_tb_homestr_inner'] = _inner   # ← 1 feature のみ
    except Exception as e:
        print(f"[WARN] JRDB SR merge failed: {e}")
```

### 2.2 修正後 (Session #35)

```python
# 既存 jrdb_tb_homestr_inner 維持 (V15 動作不変保証)
_tb = str(_sr_row.get('tb_homestr', ''))
_inner = int(_tb[0]) if _tb and len(_tb) >= 1 and _tb[0].isdigit() else 2
horses_df['jrdb_tb_homestr_inner'] = _inner

# Session #35 拡張: 新 4 features (race-level、全馬同値)
# sr_first3f_avg: harlon_1, harlon_2, harlon_3 の平均
_h1 = pd.to_numeric(_sr_row.get('harlon_1', None), errors='coerce')
_h2 = pd.to_numeric(_sr_row.get('harlon_2', None), errors='coerce')
_h3 = pd.to_numeric(_sr_row.get('harlon_3', None), errors='coerce')
_vals = [v for v in [_h1, _h2, _h3] if pd.notna(v)]
horses_df['sr_first3f_avg'] = (sum(_vals) / len(_vals)) if _vals else 0.0

# sr_bias_homestr / sr_bias_4corner: tb_*** 文字列の数値合計
def _tb_to_numeric(s):
    s = str(s or '')
    return sum(int(c) for c in s if c.isdigit())
horses_df['sr_bias_homestr'] = _tb_to_numeric(_sr_row.get('tb_homestr', ''))
horses_df['sr_bias_4corner'] = _tb_to_numeric(_sr_row.get('tb_4corner', ''))

# sr_pace_up_pos: 数値変換
_pup = pd.to_numeric(_sr_row.get('pace_up_pos', None), errors='coerce')
horses_df['sr_pace_up_pos'] = float(_pup) if pd.notna(_pup) else 0.0
```

### 2.3 動作確認

```bash
$ python -c "..."
# 5/2 東京 12R (race_id 202605020112) で test
jrdb_sr match: 1 row
  harlon_1/2/3: 12.5 / 11.6 / 12.0
  → sr_first3f_avg = 12.03   (unique 値 ✅)
  tb_homestr: 'nan'            (5/2 race の tb 未集計)
  → sr_bias_homestr = 0       (default、安全)
  pace_up_pos: nan
  → sr_pace_up_pos = 0.0      (default、安全)
```

→ harlon 系は実値、tb 系は 5/2 時点 'nan' (未集計、後日更新で取得可能)。

### 2.4 backup 保存

```bash
$ ls tools/jrdb_features.py.bak_session35
-rwxr-xr-x  61864 May 7 01:03  # 修正前のコピー
```

→ 5/9 朝に万が一 V15 動作異常が出たら `cp tools/jrdb_features.py.bak_session35 tools/jrdb_features.py` で即 rollback 可能。

---

## 3. V15 動作不変 確認 (D + E)

### 3.1 V15 model features list (read のみ)

```python
import gzip, pickle
with gzip.open('keiba_model_v15_central_live.pkl.gz', 'rb') as f:
    md = pickle.load(f)
features = md['features']
print(f"V15 features count: {len(features)}")  # 150
sr_feats = [f for f in features if 'sr_' in f or 'first3f' in f or 'tb_' in f]
print(f"V15 sr/tb features: {sr_feats}")
# → ['prev_race_first3f', 'jrdb_tb_homestr_inner']  (2 件のみ)
```

→ **V15 model は新 4 features (sr_first3f_avg / sr_bias_homestr / sr_bias_4corner / sr_pace_up_pos) を読まない**。
→ DataFrame に新 column が追加されても V15 は無視、score 完全不変。

### 3.2 5/3 京都 12R 動作確認 (修正前後 比較)

**Session #32 D 記録** (`data/v18/multi_stage_predict_test_5_6.md`):
```
京都 12R 東大路S: 軸 馬6 (TOP1 score 0.634, 馬体重 +2kg)
```

**Session #35 修正後**:
```
$ python tools/predict_one_race.py 202608030412 | tail -10
  6     ルシフェル 0.634409
  5  ハクサンアイリス 0.603517
  7  ゴッドブルービー 0.595612
  ...
```

→ **軸馬 6 ルシフェル、score 0.634 完全一致** ✅
→ V15 動作 完全不変 確認、絶対遵守ライン保護。

---

## 4. V18/V19 retro 再実行 (C) — skip → 5/13 で再対応

V18 lgb model load で error:
```
$ python -c "import lightgbm as lgb; v18 = lgb.Booster(model_file='data/v18/models/v18_tansho_lgb.txt'); print(v18.feature_name())"
[LightGBM] [Fatal] Model format error, expect a tree here.
```

→ Session #32 で CRLF 修復した V18 model file が再度 corruption している可能性。
→ V18/V19 retro は **5/13 で再対応** (Step 1 完了後に retro 拡大)。

代替: 本セッションでは jrdb_features.py 拡張完了 + V15 動作不変保証 のみ達成、 retro 改善測定は 5/13。

---

## 5. 5/9 V15 投資 安全 final check (E)

| 項目 | 状態 |
|------|------|
| `tools/jrdb_features.py` modify | ✅ 完了 (新 4 features 追加) |
| `tools/jrdb_features.py.bak_session35` 保存 | ✅ rollback 可能 |
| `tools/predict_core.py` 変更 | ❌ なし (絶対遵守) |
| `tools/daily_predict.py` 変更 | ❌ なし |
| V15 model file 変更 | ❌ なし (read のみ) |
| schtasks 変更 | ❌ なし |
| V15 features list に新 4 件 | ❌ 含まれず (model は無視) |
| 既存 `jrdb_tb_homestr_inner` 動作 | ✅ 不変 |
| 5/3 京都 12R V15 score | ✅ 0.634 完全一致 |
| **5/9 V15 案B改 投資 影響** | **🟢 完全になし** |

→ 5/9 朝に V15 daily_predict が 完全に同一動作することを保証。

万が一の rollback:
```bash
cp tools/jrdb_features.py.bak_session35 tools/jrdb_features.py
```
で即時復旧可能。

---

## 6. V18/V19 5/16 試行への効果 (期待)

5/13 で V18 model load 修復後 retro:
- 期待 winner_top1 改善: +2-4pt (sr 4 features merge 拡張)
- 当初 Session #34 評価通り

ただし **5/3 race の jrdb_sr が 5/2 取得時点で未公開**だった可能性が判明:
- 5/3 race_ids = 0 rows in jrdb_sr.csv
- 5/2 race_ids = 28 rows のみ (本来 36 races あるはず)

→ jrdb_sr 自体の取得を 5/3 以降の race も含める backfill が 5/13 以前に必要。

---

## 7. 5/13 (火) 朝の作業 (Session #35 で前倒し済)

旧 plan: Step 1 SR merge 拡張 (2h)
**新 plan**: 既に完了 (本セッション)、5/13 は Step 2 premium 強化 から開始

5/13 短縮 plan:
- Step 1 (sr merge): ✅ 本日完了 → 5/13 着手不要 (jrdb_sr backfill のみ)
- Step 2 (premium): 2h → 5/13 朝
- Step 3 (運用フィルタ): 1h → 5/13 PM
- Step 4 (retro 拡大): 4h → 5/14 (V18 model 修復後)
- Step 5 (paper retro): 5/15

→ 5/13 朝の作業負荷 4h → **3h** に減、効率 30% UP。

---

## 8. 結論

🟢 **sr merge 拡張完了**、 V15 動作完全不変保証 確認、5/13 作業 前倒し完了。

5/9 V15 投資への影響: ゼロ (絶対遵守ライン保護)。
5/16 V18/V19 試行への期待: +2-4pt 改善 (5/13 V18 model 修復後 retro で測定)。
副産物: 5/13 plan で Step 1 不要 → 朝の作業 3h に短縮。
