# Phase 3+: V20 統合モデル (JRA + NAR) 構想 草案

**作成**: 2026-05-06 PM (Session #31)
**期間**: 6/9-6/30 (Phase 3 後半、V15.1 採用後)
**目的**: JRA + NAR の統合モデル設計、52+ 共通 features

---

## 1. 構想

V20 は JRA (V15.1 の 150 features) と NAR (v4 の 22 features) を統合:
- 共通 features 52+ (重複あり、別表)
- `is_nar` フラグで切替
- 1 つの model で JRA 案B改 + NAR 試行を両立

詳細設計: `data/v18/jra_nar_integration_plan.md` (Session #13、5/5 作成)

---

## 2. なぜ V20 を作るのか

### 2.1 現状の二重管理コスト

- V15.1 (JRA): 150 features、`keiba_model_v15_central_live.pkl.gz`
- NAR v4: 22 features、`data/nar/models/keiba_model_nar_v4.pkl`
- 学習・推論パイプラインが二重、保守コスト増

### 2.2 共通 features の重複

両モデルで使う features:
- 基本情報 (距離 / 馬場 / 性別 / 馬齢 / 斤量)
- 騎手・調教師 統計
- 過去成績 (前 3 走 着順 / 上がり 3F)
- 血統 (sire encoding)
- 当日情報 (馬体重 / オッズ)

→ 52+ features が共通、統合価値あり。

### 2.3 NAR データの活用

NAR v4 (4,821 races / 49,213 rows) の知見が JRA 予測にも転用可能の可能性:
- ダート専門馬の傾向
- 中距離・距離適性
- 騎手が JRA-NAR 両方乗る場合の統計

→ 統合学習で JRA AUC 改善余地あり。

---

## 3. データ準備

| データ | 状態 | 必要対応 |
|--------|------|---------|
| jra_races_full.csv | 5/3 まで OK | 6 月 1 度 TARGET 再契約で backfill |
| chihou_races_full.csv | 2020-03 末で stale (6 年遅れ) | nar_all_races.csv (54K rows) で代替済 |
| chihou_races_2020_2025.csv | **不在 (Session #24 で誤情報判明)** | 不要 |
| nar_all_races.csv | 2024-05〜2025-05 | 1 年 stale、backfill 推奨 |
| netkeiba premium | 5/2-5/3 復旧済 (本日 9fe8063e) | OK |
| JRDB 26 種 | 5/3 まで OK | 6 月以降 継続 |

---

## 4. 学習計画

### 4.1 features 整理 (6/9-6/12、12h)

```python
# train/v20_features.py 設計
SHARED_FEATURES = [
    # 基本 (12)
    'distance', 'surface_enc', 'sex_enc', 'age', 'weight_carry',
    'horse_num', 'bracket', 'num_horses', 'horse_num_ratio',
    'bracket_pos', 'carry_diff', 'is_nar',  # ← 切替フラグ
    # 過去成績 (10)
    'prev_finish', 'prev_last3f', 'prev_pass4', ...
    # 騎手・調教師 (8)
    'jockey_wr_calc', 'jockey_course_wr_calc', ...
    # 血統 (8)
    'sire_enc', 'bms_enc', 'sire_dist_wr', ...
    # 当日情報 (6)
    'horse_weight', 'odds_log', 'pop_rank', ...
]  # 52 features
```

### 4.2 学習 (6/13-6/16、16h)

```bash
python train/train_v20_jra_nar.py \
    --jra-data data/jra_races_full.csv \
    --nar-data data/nar_all_races.csv \
    --features 52 \
    --ensemble lgb_xgb_ft_intra
```

期待結果:
- V20 JRA subset AUC ≥ V15 0.8939
- V20 NAR subset AUC ≥ NAR v4 0.8145
- 共通学習で互いの精度向上

### 4.3 評価 (6/17-6/20、12h)

A/B テスト:
- V20 JRA subset vs V15 本番
- V20 NAR subset vs NAR v4

GO 条件: 両方で改善 (AUC > 既存)、ROI も改善見込み。

### 4.4 production 統合 (6/21-6/25、12h)

`tools/predict_v20.py` 新規:
- is_nar フラグで JRA / NAR 切替
- 統一 inference

### 4.5 paper trading (6/26-6/30、5 日)

V20 paper + V15.1 本番 を並行。

---

## 5. 6 月 JRA-VAN 一時再契約

| 期間 | 用途 | コスト |
|------|------|--------|
| 6/9-6/30 | V20 学習データ取得 (5-6 月分 backfill) | ¥2,090 (1 ヶ月) |
| 7/1 以降 | 解約 (再 stale OK) | 0 |

→ 1 ヶ月のみ再契約で V20 学習完遂、その後解約で運用継続。

---

## 6. 完成後の運用 (7 月以降)

| モデル | 用途 |
|--------|------|
| V20 JRA subset | 案B改 (12R 1勝クラス) 投資 |
| V20 NAR subset | NAR 試行 500 円/日 |
| V15.1 (fallback) | V20 障害時 |
| V15 (fallback fallback) | V15.1 障害時 |

→ 4 段 fallback で運用安定。

---

## 7. リスク

- 統合学習で個別精度が **悪化** する可能性 (NAR features が JRA 予測の noise になる)
- A/B テストで NO-GO なら V15.1 + NAR v4 個別運用継続
- 12h × 5 step = 60h の学習コスト、6 月 1 ヶ月で完遂可能か

---

## 8. NO-GO 時の代替

V20 NO-GO の場合:
- V15.1 + NAR v4 個別運用継続
- features 共通化のみで保守コスト削減 (実装無し、運用工夫)
- V21 構想 (7 月以降) で再検討

---

## 9. 結論

V20 は 6/9-6/30 で 5 step 完遂、JRA-VAN 1 ヶ月再契約で実現可能。
全 step GO で 7 月から V20 統合運用、NO-GO なら V15.1 + NAR v4 個別運用継続。
取り返し禁止ルール遵守、安易な切替なし。
