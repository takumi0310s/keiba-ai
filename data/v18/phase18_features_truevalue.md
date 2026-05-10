# Phase 18 C: Phase 13 stub features 真値化 plan

**作成**: 2026-05-10 (Session #91 Phase 18 C、 ★ Opus 4.7 ★)
**前提**: Phase 13 で predict_core_v18.py に 25 features 追加 (全 default fill)
**目的**: DOM probe で selector 真値化後、 default → 真の lookup に切り替える plan

---

## 1. 現状 (Phase 13 commit f4d813bf)

`tools/predict_core_v18.py` (predict_core 拡張、 V15 不変):

```python
# Phase 13 features 数 = 25
PHASE13_FEATURES = (
    AI_TENKAI_FEATURES        # 7 features
    + AI_HARAN_FEATURES        # 3 features
    + LAP_FEATURES             # 10 features
    + TRACK_BIAS_FEATURES      # 5 features
)
# 全 features 全 race / 全頭 で default 値 (PHASE13_DEFAULTS) で fill 中
# fetch 真値化は selector 確定後 (Phase 18 A 完了後)
```

V18 candidate 累計: V15 150 + Phase 11 (15 JRDB) + Phase 12 (17 DataLab) + Phase 13 (25 master) = **207 features**

---

## 2. 真値化 切り替え 設計

### 2.1 切り替え point

| 段階 | 状態 | 期待 ΔAUC |
|------|------|----------|
| 0 (Phase 13、 現状) | 全 default fill (全 features 同一値) | 0 (ノイズ無し) |
| 1 (Phase 18 A 後) | 1 R 真値、 残 default | +0.001-0.002 (1 R テスト) |
| 2 (Phase 18 B 6 ヶ月後) | 直近 6 ヶ月 真値 (6,000 R)、 過去 default | +0.010-0.020 (近期データ) |
| 3 (Phase 18 B 1 年後) | 直近 1 年 真値 (12,000 R) | +0.020-0.040 |
| 4 (理想形) | 5 年分 真値 + 当日 fetch | +0.028-0.065 (Phase 13 期待値) |

### 2.2 段階 1 (Phase 18 A 完了後) 切り替え方法

`tools/predict_core_v18.py` 既存 stub:

```python
def merge_phase13_features(df: pd.DataFrame, race_id: str) -> pd.DataFrame:
    """Phase 13 master features を df に merge.
    
    現状: 全 race / 全頭 で default fill (kill switch 等価)
    Phase 18 A 後: 当日 R は fetch_master_features() を呼出、 取得失敗時 default
    """
    for col in PHASE13_FEATURES:
        df[col] = df.get(col, PHASE13_DEFAULTS[col])
    return df
```

→ Phase 18 A 完了後 (selector 真値化済) に以下に変更:

```python
def merge_phase13_features(df: pd.DataFrame, race_id: str, fetch: bool = False):
    if not fetch or is_disabled():
        for col in PHASE13_FEATURES:
            df[col] = df.get(col, PHASE13_DEFAULTS[col])
        return df
    # 真値 fetch (1 R 12 sec)
    umaban_list = df['umaban'].astype(int).tolist()
    bundles = fetch_race_master_features(race_id, umaban_list)
    for idx, row in df.iterrows():
        umaban = int(row['umaban'])
        for col in PHASE13_FEATURES:
            df.at[idx, col] = bundles.get(umaban, {}).get(col, PHASE13_DEFAULTS[col])
    return df
```

### 2.3 段階 2-4: backfill cache 連携

Phase 18 B で取得した `data/netkeiba_master/backfill/{year}/{race_id}.json.gz`
を読み込んで過去レースの features を補完:

```python
def merge_phase13_features_with_backfill(df: pd.DataFrame, race_id: str):
    cache_path = Path(f"data/netkeiba_master/backfill/{race_id[:4]}/{race_id}.json.gz")
    if cache_path.exists():
        # cache hit: 真値 features 返す
        with gzip.open(cache_path, 'rt', encoding='utf-8') as f:
            payload = json.load(f)
            for col in PHASE13_FEATURES:
                df[col] = payload['features'].get(col, PHASE13_DEFAULTS[col])
        return df
    # cache miss: default fill (V18 学習時、 該当 R は features 信号なし)
    for col in PHASE13_FEATURES:
        df[col] = PHASE13_DEFAULTS[col]
    return df
```

---

## 3. V18 学習で features 真の寄与を確認 (Phase 18 D)

### 3.1 必要条件

- selector 真値化済 (Phase 18 A 完了)
- 直近 6 ヶ月 backfill 完了 (Phase 18 B、 6,000 R)
- V18 学習 train data に master features 反映 (sib_w5 と同じ merge ロジック)

### 3.2 V18 再学習 schedule (5/24+ Phase 3 後半)

```
5/24-5/26  V18 sib_w5 + Phase 11 (JRDB 15) + Phase 12 (DataLab 17) で 182 features 再学習
5/27-5/30  Phase 13 master (25) cache hit 範囲のみ追加 (選別 学習で v18 vs v18_master 比較)
6/1-6/3   feature importance ranking 取得、 真の寄与判定
6/4-6/7   V20 4-model ensemble に統合 (LGB+XGB+FT+IR)
6/8       V20 GO 判定
```

### 3.3 期待 寄与 ranking (Phase 13 仮定)

| feature category | 期待 重要度 (LGB importance gain ratio) |
|-----------------|----------------------------------------|
| AI 展開予測 (7) | 中-高 (race-level pace 信号) |
| AI 波乱度 (3) | 中 (上位人気依存性、 V15 と相補的) |
| 個別ラップ (10) | 高 (馬個別 真値、 過去走 補完) |
| トラックバイアス (5) | 中 (当日 馬場 信号、 V15 既存と一部重複) |

→ ★ 個別ラップ + AI 展開予測 が有意になる見込 ★

---

## 4. backfill 不要 (default 維持) 判断 条件

V18 候補 model で feature importance を計測し、 以下の場合は backfill 不要:

| feature | 個別 importance < 0.001 (LGB gain) → 削除 |
|---------|------------------------------------------|
| AI 展開予測 7 features | (要 確認) |
| AI 波乱度 3 features | (要 確認) |
| 個別ラップ 10 features | (要 確認) |
| トラックバイアス 5 features | (要 確認) |

→ 5/24+ V18 再学習で各 feature の真の重要度を計測し、 寄与なしは
  V20 投入前に削除 (over-fitting 防止)

---

## 5. V15 投資保護 (絶対遵守)

✅ predict_core_v18.py は predict_core.py 別ファイル、 V15 不変
✅ Phase 13 features 真値化は predict_core_v18.py のみ修正
✅ V18 候補 model 学習も別 dir、 V15 production は影響なし

---

## 6. 結論

✅ Phase 13 stub → 真値化 切り替え 設計 (3 段階) 確立
⚠ 真値化は Phase 18 A 完了 (selector 真値化) 必須
⚠ 過去 backfill は Phase 18 B 段階的 6 ヶ月推奨 (24 年 NG)
⚠ V18 再学習 + feature importance 評価は 5/24+ Phase 3 後半
✅ V15 投資保護完全

---

**Phase 18 C 完了** (Opus 4.7)
