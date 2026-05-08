# AUDIT-1 H: NAR (地方) 別 audit (5/8)

**作成**: 2026-05-08 (AUDIT-1 H 領域)
**前提**: NAR (地方競馬) は archive/nar/ 配下、 V4 model (AUC 0.8145、 36 features)
**位置付け**: read-only audit。 V15 (中央) と 別 model

---

## 1. NAR 専用 source

| source | 内容 | 取得状況 |
|--------|------|---------|
| 楽天競馬 (keiba.rakuten.co.jp) | 地方 odds / 出馬表 / 結果 | tools/scrape_nar_*.py で 取得 |
| 南関 (nankan.jp) | 大井・船橋・川崎・浦和 | scrape_nar_today.py で 取得 |
| netkeiba 地方 | 地方競馬 (一部) | netkeiba 経由で 部分カバー |
| nar.go.jp 公式 | 地方競馬 全国情報 | scrape 不明 |
| **JRA-VAN は NAR 不対応** | -- | 取得不可 |
| **JRDB は NAR 不対応** | -- | 取得不可 |

---

## 2. NAR 取得状況

### 2.1 取得済 file

| ファイル | 内容 |
|---------|------|
| `data/chihou_full_enriched.csv` | 地方 全 race + 拡張 |
| `data/chihou_races_full.csv` | 地方 全 race |
| `archive/nar/keiba_model_nar_v4.pkl` | NAR V4 model (36 features、 AUC 0.8145) |

### 2.2 NAR scrape tools

- `tools/scrape_nar_all.py` - 全 NAR scrape
- `tools/scrape_nar_calendar.py` - カレンダー
- `tools/scrape_nar_live_odds.py` - LIVE オッズ
- `tools/scrape_nar_results.py` - 結果
- `tools/scrape_nar_today.py` - 当日

---

## 3. NAR V4 features (36 件)

### 3.1 NAR_FEATURES_BASE (29 件)

```
num_horses, distance, surface_enc, course_enc,
weight_carry, age, sex_enc,
horse_num, bracket, jockey_wr, jockey_place_rate, trainer_wr,
prev_finish, prev2_finish, prev3_finish, avg_finish_3r,
best_finish_3r, top3_count_3r, finish_trend, prev_odds_log,
rest_days, rest_category, dist_cat, age_group,
horse_num_ratio, bracket_pos, carry_diff, dist_change,
dist_change_abs, is_nar
```

### 3.2 NAR_V4_NEW (7 件)

```
horse_dist_top3r, horse_surface_top3r, jockey_course_wr,
frame_course_dist_wr, horse_career_races, horse_career_wr,
horse_career_top3r
```

### 3.3 NAR スペック

- AUC: 0.8145 (V15 中央 0.886 と 0.07 低い)
- 全条件 ROI 100% 超え (条件別 BT)
- 学習 data 行数: 不明 (chihou_full_enriched.csv ベース)

---

## 4. NAR 未活用 features

### 4.1 取得済 だが NAR V4 未使用

| feature 候補 | source | 期待 |
|------------|--------|------|
| 馬体重 / 馬体重変化 | 地方場 公式 (一部公開) | medium-high (NAR は 中央 並み 公開) |
| 確定オッズ (中央 とは別 リーク扱い 不要 場合) | rakuten | リーク扱い |
| 単勝・複勝 直前 odds | 同上 | リーク扱い |
| 騎手 leading 系 | rakuten | medium (内製と重複) |
| 調教師 系 | rakuten | low |
| 距離 別 適性 (馬個別) | 計算 | medium |
| 場 別 適性 (馬個別) | 計算 | medium-high (大井・船橋・川崎・浦和 別) |
| 場 別 統計 (場 × 距離) | 計算 | medium |
| 砂深さ (場 別) | 不明 | low |
| クラス system (NAR 独自 C1/C2/B3 等) | 公開 | medium |

### 4.2 未取得 (中央には ある が NAR では 取得困難)

| feature | NAR 取得難度 |
|---------|------------|
| JRDB 系 (KYI / SED / TYB 等) | 不可 (JRDB NAR 不対応) |
| マスター指数 | 不可 (netkeiba マスター NAR 不対応) |
| 調教 タイム | 部分 (一部 場 のみ) |
| 厩舎コメント | 部分 |
| 種牡馬 詳細 | netkeiba db 経由 (中央と共通) |
| 父系 / 母系 統計 | 計算可能 |

---

## 5. NAR の独自要素

### 5.1 地方 独自 features

| feature | 内容 | 期待 |
|---------|------|------|
| dirt_only_indicator | 地方 はほぼ 全レース ダート | (定数で意味薄い) |
| 場 別 砂質 / 含水率 | 大井 / 船橋 / 川崎 / 浦和 別 | medium |
| ナイター 開催 | 開始時刻 別 | low-medium |
| 中央交流 indicator | 中央馬 出走 | medium-high |
| 地方馬 中央実績 | 集計 | medium |

### 5.2 V4 → V5 候補 (中央 V15 並み に)

| 候補 | 工数 | 期待 |
|------|------|------|
| sib_*_exp 計算 (NAR 母系統計) | 8h | +0.005-0.010 |
| 場 別 適性 features 5-8 件 | 6h | +0.005-0.010 |
| 速度指数 (内製、 distance × time) | 4h | +0.001-0.003 |
| jockey × horse 補完 (NAR jockey limited) | 3h | +0.001 |
| LightGBM + XGB ensemble | 4h | +0.002-0.005 |
| FT-Transformer 追加 | 6h | +0.003-0.005 |

→ NAR V5 候補で 0.8145 → 0.84-0.85 が 妥当な 目標

---

## 6. NAR 投入優先度

V15 (中央) **0.886** vs NAR V4 **0.8145** の差は 0.07。 中央 V15 の 改善 (+0.005-0.010) の 方が 優先度 高い。

| 期間 | NAR 作業 |
|------|----------|
| 5/9-6/8 | (V20 開発に集中、 NAR は据え置き) |
| 6/9+ | NAR V5 候補 開始 (場別 features + sib_*_exp) |

---

## 7. 5/9 V15 投資保護

✅ NAR 関連 一切 触らない (archive/nar/ 配下、 model NAR V4 不変)
✅ 中央 V15 と完全分離

---

## 8. 結論

✅ NAR audit 完了
✅ NAR V4 36 features、 AUC 0.8145、 中央 V15 と 0.07 ギャップ
✅ NAR 未活用: 場別 features / sib_*_exp / 速度指数 / 中央交流 indicator など 約 10 件
✅ 投入優先度: 中央 V20 (6/8) を 優先、 NAR V5 は 6/9 以降

**NAR は 別 system として 並行管理。 中央 V15-V20 集中時には触らない方針**
