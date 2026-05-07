# SKB POST-RACE LEAK 完全除外 patch (Session #39 C)

**作成**: 2026-05-07 (Session #39 C)
**対象**: V20 (6/9-30) 学習時の SKB 全 10 features 完全除外
**前提**: Session #38 (5/7) で SKB POST-RACE LEAK 確定 (採用 NO-GO)

---

## 1. 背景 (Session #38 確定事項)

### 1.1 SKB LEAK の証拠

| 観測 | 値 |
|------|---|
| skb_kishi_code_3 単独 AUC gain | **+480bp** (異常) |
| corr_target | **0.137** (pre-race feature では物理的に不可能) |
| finish 順位との monotonic relation | 1着 mean=364, 10着 mean=176 (3.2x) |
| 1着馬の 0-rate | 15% |
| 敗者の 0-rate | 49% |
| 真の改善幅 (V15+SKB_no_kishi) | **-13bp** (むしろ悪化) |

### 1.2 ソース仕様

JRDB SKB ファイル仕様:
- ファイル名 SKB = "**成績拡張**" データ
- = レース結果確定後に算出される post-race indicator
- 学習用 pre-race feature として物理的に使用不可

→ V15.1 採用 NO-GO 確定、 Phase 3 plan で SKB 完全削除へ

---

## 2. 除外対象 features (10 個)

```python
SKB_LEAK_FEATURES = [
    'skb_kishi_code_1', 'skb_kishi_code_2', 'skb_kishi_code_3',
    'skb_baba_code_1',  'skb_baba_code_2',  'skb_baba_code_3',
    'skb_kyaku_code_1', 'skb_kyaku_code_2', 'skb_kyaku_code_3',
    'skb_turf_hoof',
]
```

カテゴリ別:
- **kishi (騎手) code 1〜3**: 騎手相性 → kishi_code_3 が最大 culprit (+480bp)
- **baba (馬場) code 1〜3**: 馬場適性
- **kyaku (脚質) code 1〜3**: 脚質
- **turf_hoof**: フラグ (蹄質 × 芝)

---

## 3. patch 内容

### 3.1 train/v15_1_features.py 修正

**追加 (Session #39 C)**:

```python
SKB_LEAK_FEATURES = list(V15_1_SKB_FEATURES)  # 10 features

V20_LEAK_FEATURES = [
    # === V12 から継承 (確定オッズ系) ===
    'odds_log', 'horse_weight', 'condition_enc',
    'weight_change', 'weight_change_abs', 'weight_cat', 'weight_cat_dist',
    'cond_surface',
    # === Session #38 確定 SKB POST-RACE LEAK (10) ===
    *SKB_LEAK_FEATURES,
]

def filter_v15_1_features(features: list, *, skip_skb: bool = False) -> list:
    if not skip_skb:
        return list(features)
    return [f for f in features if f not in SKB_LEAK_FEATURES]
```

**修正 `merge_v15_1_features()` (signature に skip_skb)**:

```python
def merge_v15_1_features(df, kka_path, skb_path, srb_path, *, skip_skb=False):
    ...
    if not skip_skb:
        skb = load_skb(skb_path)
        df_local = df_local.merge(skb, on=['race_part', 'umaban'], how='left')
    ...
    target_features = filter_v15_1_features(V15_1_NEW_FEATURES, skip_skb=skip_skb)
    for col in target_features:
        ...
```

### 3.2 動作確認

```bash
$ python -c "from train.v15_1_features import V15_1_NEW_FEATURES, filter_v15_1_features; \
             print(len(V15_1_NEW_FEATURES), '→', len(filter_v15_1_features(V15_1_NEW_FEATURES, skip_skb=True)))"
34 → 24
```

→ 34 → 24 (SKB 10 features 除外確認)

---

## 4. V20 (6/9-30) 構築時の使い方

### 4.1 train/run_v20_*.py (新規、 Phase 3 後半)

```python
from train.v15_1_features import (
    merge_v15_1_features,
    V15_1_NEW_FEATURES,
    filter_v15_1_features,
    SKB_LEAK_FEATURES,
    V20_LEAK_FEATURES,
)

# V20: SKB 完全除外
df_train = merge_v15_1_features(df_train, skip_skb=True)

# feature_names = V15 base + KKA + SRB (SKB 抜き)
v20_features = (
    V15_BASE_FEATURES                                   # 150
    + filter_v15_1_features(V15_1_NEW_FEATURES, skip_skb=True)  # 24
    + ['sib_top3_rate_exp', 'sib_shinba_wr_exp']        # Session #39 A
)
# 期待 total: 176 features (V15.1 184 - SKB 10 + sib_*_exp 2)
```

### 4.2 LEAK 監査 (V20 学習前)

```python
def assert_no_skb_leak(feature_names):
    leaked = [f for f in feature_names if f in SKB_LEAK_FEATURES]
    assert not leaked, f"SKB LEAK detected: {leaked}"

assert_no_skb_leak(v20_features)  # 必ず学習前に通す
```

### 4.3 production 投入時 (7/1+)

V20 production model を deploy する場合:
- `tools/predict_core_v20.py` を新設、 feature builder で `skip_skb=True` 強制
- LIVE retro でも skb_* 列を NaN/0 で埋める fallback ガード
- daily_predict.py V20 切替時、 V15 fallback 経路を保持

---

## 5. SKB 代替戦略

### 5.1 騎手相性 (kishi)

旧: skb_kishi_code_1〜3 (post-race)
新: 既存 V15 features で代替
- `jockey_wr_calc` (騎手勝率、 expanding)
- `jockey_horse_top3r` (騎手×馬の expanding 集計、 既に V15 内)

### 5.2 馬場適性 (baba)

旧: skb_baba_code_1〜3 (post-race)
新: 既存 V15 features で代替
- `horse_surface_top3r` (馬の馬場別 expanding 複勝率)
- `sire_surface_wr` (父馬産駒馬場別)

### 5.3 脚質 (kyaku)

旧: skb_kyaku_code_1〜3 (post-race)
新: 既存 features + 新設候補
- `prev_pass4` (前走 4 角位置 → pseudo 脚質)
- 6/9-30 で脚質推定 features 検討 (kyi の `running_style_code` を expanding 集計)

→ いずれも pre-race のみで構成、 LEAK FREE。

---

## 6. 5/9 V15 案B改 投資保護

本 patch は:
- ✅ `train/v15_1_features.py` のみ変更 (helper 関数 + skip_skb flag 追加)
- ✅ V15 production (predict_core / daily_predict / V15 model file) **完全不変**
- ✅ V15 features は SKB を **使っていない** (V15.1 が SKB 採用候補だった、 NO-GO 確定済)
- → **5/9 朝 V15 案B改 動作 完全保証**

---

## 7. 結論

✅ SKB 完全除外 patch 完了
✅ `SKB_LEAK_FEATURES` (10) + `V20_LEAK_FEATURES` (18) 定義
✅ `filter_v15_1_features(skip_skb=True)` で 34 → 24 features 動作確認
✅ V20 (6/9-30) 学習時に `merge_v15_1_features(skip_skb=True)` で安全
✅ 代替戦略 (kishi/baba/kyaku) 既存 features で対応可能
✅ V15 動作不変 保証

---

**Session #39 C 完了**
