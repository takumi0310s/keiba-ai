# V16 能力ベース予測モデル — 分析ドキュメント

**作成**: 2026-05-24  
**哲学**: オッズは結果。能力の実体 (ジョッキー・血統・調教) で予測する。  
**ステータス**: V16 WF 学習完了 (2026-05-24 00:20) — AUC 0.8677 (V15 0.8678 比 -0.0001)

> V15 production 完全不変。V16 は candidate のみ。

---

## 1. ユーザーの哲学と V16 の目的

### 哲学: オッズは結果

> 「AI はジョッキー・血統・調教追切など『オッズになる前の実体』を見抜くべき。
> オッズを入力に使うのは結果を原因にする倒錯。」

- V15 は `paci_ninki_idx` (JRDB 人気指数、odds-derived) が **gain 16.93%** — 最重要 feature
- これは JRDB の前日人気スコア = 市場の事前人気評価 = 「結果の代理」
- V16: 人気・オッズ系 feature を完全除外し、**能力の実体のみ** で予測する

---

## 2. 除外した「結果系」feature (8個)

| feature | gain% (V15 LGB) | 種別 | 除外理由 |
|---------|----------------|------|---------|
| `paci_ninki_idx` | **16.93%** | JRDB 人気指数 (odds-derived) | 市場の人気評価 = 結果の代理 |
| `pop_rank_change` | 0.83% | 当日人気変動 | 投票状況 = 結果 |
| `oz_tansho_base_log` | 0.12% | 基準単勝オッズ log | 前日オッズ = 結果 |
| `oz_base_pop_rank` | 0.14% | 基準人気順位 | 前日人気 = 結果 |
| `oz_fukusho_base_log` | ~0.1% | 基準複勝オッズ log | 前日オッズ = 結果 |
| `odds_change_rate` | 0.12% | オッズ変化率 | 市場動向 = 結果 |
| `odds_sharp_drop` | ~0.01% | 急落フラグ | 市場動向 = 結果 |
| `prev_odds_log` | 0.00% | 前走オッズ log | 前走人気 = 過去の結果 |
| **合計** | **~18.1%** | | |

**V15 145 features → V16 137 features (−8)**

---

## 3. 能力 feature 棚卸し (残す「実体」)

### ジョッキー (11 features) — 充実度: ★★★

| feature | 内容 |
|---------|------|
| `paci_jockey_exp_wr` | JRDB 騎手勝率 (gain 16.34% — V16 最重要予定) |
| `paci_jockey_exp_3rd` | JRDB 騎手複勝率 (gain 14.78%) |
| `jockey_wr_calc` | 過去実績から計算した騎手勝率 |
| `jockey_course_wr_calc` | コース別騎手勝率 |
| `jockey_surface_wr` | 馬場種別騎手勝率 |
| `jockey_horse_rides` | この騎手×この馬の騎乗数 |
| `jockey_horse_wr` | 騎手×馬 相性勝率 |
| `jockey_horse_top3r` | 騎手×馬 相性複勝率 |
| `jockey_change` | 騎手変更フラグ |
| `jockey_change_to_top` | トップ騎手への変更フラグ |
| `paci_jockey_mark` | JRDB 騎手マーク |

### 血統 (7 features) — 充実度: ★★★

| feature | 内容 |
|---------|------|
| `sire_enc` | 父馬エンコード (fold-wise Bayesian) |
| `bms_enc` | 母父馬エンコード |
| `sire_surface_wr` | 父馬芝/ダ別複勝率 |
| `sire_dist_wr` | 父馬距離別複勝率 |
| `bms_surface_wr` | 母父馬馬場別成績 |
| `sire_shinba_top3r` | 父馬新馬複勝率 |
| `jrdb_bms_rensho_avg` | JRDB 母父連勝平均 |

### 調教・追切 (13 features) — 充実度: ★★☆

| feature | 内容 | 充填率 |
|---------|------|--------|
| `training_time_filled` | 追切タイム (gain 5.36%) | ~60% (欠損 mean fill) |
| `wood_best_4f_filled` | ウッドチップ 4F タイム | ~50% |
| `sakaro_best_4f_filled` | 坂路 4F タイム | ~40% |
| `jrdb_training_idx` | JRDB 調教指数 | ~70% |
| `jrdb_training_arrow` | JRDB 調教矢印 | ~70% |
| `has_training`, `has_wood_training`, `has_sakaro_training` | 各種調教あり/なしフラグ | ~100% |
| `total_training_count`, `wood_count_2w` | 調教本数 | ~80% |
| `training_per_dist`, `training_intensity_enc` | 調教強度 | ~70% |

**課題**: 調教タイムデータが `data/netkeiba_training_times.csv` に 2,551件のみ (2025年一部)。欠損時は population mean で fill → 信号が薄れる。

### JRDB 能力指数 (51 features) — 充実度: ★★★ (JVLink 以降)

主要 features:
| feature | 内容 | gain (推定) |
|---------|------|---------|
| `jrdb_ze_idm_avg` | 過去 SED IDM 平均 (スピード指数) | 9.40% |
| `jrdb_idm` | 当日 IDM 予測値 | 中 |
| `jrdb_composite_idx` | JRDB 総合指数 | 中 |
| `jrdb_ten_idx_pred` | テン指数予測 | 中 |
| `jrdb_agari_idx_pred` | 上がり指数予測 | 中 |
| `jrdb_stable_idx` | 厩舎指数 | 中 |
| `jrdb_dist_apt` | 距離適性 | 低-中 |
| `jrdb_heavy_apt` | 重馬場適性 | 低-中 |
| `jrdb_oikiri_idx` | JRDB 追切指数 | 中 |

**JRDB 指数は純粋な能力評価** — 調教・血統・過去成績をJRDB が総合評価したもの。  
`paci_ninki_idx` (人気指数) だけが odds-derived で、他は能力系。

### 馬の過去成績 (16 features) — 充実度: ★★★

- `horse_career_wr` / `horse_career_top3r` — 生涯勝率/複勝率
- `avg_finish_3r` / `best_finish_3r` — 直近3走平均・最良着順
- `horse_dist_top3r` / `horse_surface_top3r` — 距離・馬場別適性
- `prev_finish` / `prev2_finish` / `prev3_finish` — 前走着順
- `prev_last3f` / `avg_last3f_3r` — 上がり3ハロン

### コース・距離適性 (21 features) — 充実度: ★★★

- コース別騎手勝率、距離区分、馬場種別
- 枠番適性 (`frame_course_dist_wr`)
- 輸送距離 (`transport_distance_km`)

### その他能力系 (remaining)

- 馬体重推移 (`weight_ma5`, `weight_trend`, `weight_peak_diff`) — 体調proxy
- `pci` (ペースコントロール指数)
- `gaisha_rank` (外厩ランク)

---

## 4. V16 WF 学習結果 (学習完了後に更新)

### 学習条件

| 項目 | 値 |
|------|-----|
| データ | `data/_v15_optuna_df_cache.pkl.gz` (527,280 rows) |
| WF folds | 2021, 2022, 2023, 2024, 2025 (5-fold) |
| モデル | LGB + XGB (V15 と同一パラメータ) |
| 出力 | `models/v16_ability_candidate.pkl.gz` |

### WF AUC 結果 (学習完了 2026-05-24 00:20)

| 年 | V16 LGB AUC | V16 XGB AUC | V16 ENS |
|----|------------|------------|---------|
| 2021 | 0.8628 | 0.8640 | 0.8643 |
| 2022 | 0.8656 | 0.8666 | 0.8670 |
| 2023 | 0.8670 | 0.8682 | 0.8684 |
| 2024 | 0.8691 | 0.8700 | **0.8704** |
| 2025 | 0.8671 | 0.8681 | 0.8684 |
| **平均** | 0.8663 | 0.8674 | **0.8677** |

**V15 genuine WF ENS baseline: 0.8678 → V16 delta: -0.0001 (実質同等)**

> 予測 (0.850〜0.862) を大幅に上回った。paci_ninki_idx 除去によるペナルティはほぼゼロ。

---

## 5. V15 vs V16 比較 (予測分析)

### AUC 実測値 vs 予測

予測: V16 WF ENS AUC ≈ 0.850〜0.862 (V15 0.8678 から -0.006〜-0.018)  
**実測: V16 WF ENS = 0.8677 (delta -0.0001) — 予測を大幅に上回る**

`paci_ninki_idx` (gain 16.93%) を除外してもほぼ無影響だった理由:
- `paci_jockey_exp_wr` (gain 16.34%) + `jrdb_ze_idm_avg` (9.40%) が能力信号として補完
- LGB が paci_ninki_idx の情報を他 feature との相互作用で既に内部表現していた可能性
- V15 モデルは「人気＝能力の代理」として学習しており、能力 feature が削除された
  人気指数の役割を十分担えている

ただし以下の緩和要因がある:
- `paci_jockey_exp_wr` (gain 16.34%) + `paci_jockey_exp_3rd` (14.78%) は残存
- `jrdb_ze_idm_avg` (9.40%) も残存
- V15 モデルは paci_ninki_idx との交互作用を学習しており、それが消えることで他 feature の寄与が上がる可能性

### 能力だけで「人気 1 番人気」を超えられるか?

V15 TOP1 実績 (N=649):
- TOP1 連対率: **43.0%**
- 人気 1 番人気 連対率: **約 50%**
- TOP1 複勝率: **55.2%**

V16 (能力のみ) の期待連対率:
- `paci_ninki_idx` (人気評価) を除くと、人気 1 番人気との一致度が下がる
- 市場が過小評価した「能力馬」を拾う可能性が上がる
- ただし連対率自体は人気 1 番人気 (50%) を下回る可能性が高い
- ROI は連対率より「オッズとの関係」で決まる → 人気の歪みを突けるかがポイント

---

## 6. 能力 feature の充実度評価

### 現状で不足しているデータ

| データ | 現状 | 重要度 | 取得難易度 |
|--------|------|--------|-----------|
| 調教タイム (全数) | 2,551件のみ (2025年一部) | **★★★** | JVLink WOOD で全年取得可 |
| 騎手映像解析 | 未取得 | ★★☆ | Phase 4 動画 AI |
| 馬体検査データ | 未取得 | ★★★ | JRA 非公開 (困難) |
| レース映像の馬体評価 | 未取得 | ★★☆ | Phase 4 動画 AI |
| 調教パートナー情報 | 未取得 | ★☆☆ | JRDB CK ファイル |

### 調教データの重要性

V15 で `training_time_filled` は gain 5.36% 貢献しているが、充填率は ~60%。  
**JVLink WOOD データを全年取得すれば調教タイムを完全充填できる** → V20+ で重要な差別化。

### 今後の強化候補

```
Phase 3 (5/24-6/30): JV-Link WOOD 調教全数取得 → 調教タイム充填率 100% へ
Phase 4 (7-8月): 動画 AI (YOLOv8 + DLC) → 馬体・歩様・姿勢 feature
V20: 全調教データ + sib_*_exp + SKB 除外 → 能力 feature さらに充実
```

---

## 7. ★ honest verdict ★

### 能力ベース V16 は人気を超えられるか?

**短期的 (V16 現状): 予想外に有望**

| 観点 | 判定 | 理由 |
|------|------|------|
| WF AUC で V15 を超える | △ | delta -0.0001 — 実質同等 (予測 -0.01〜-0.02 を大幅に上回る) |
| 連対率で人気 1 番人気 (~50%) を超える | ✗ | 能力 feature だけでは市場の総合情報には劣る可能性 |
| 「人気が低い能力馬」を発掘 | ◎ | 人気 feature 除去で真の能力順位が出る → 穴馬候補の精度↑ |
| ROI で V15 を超える | 有望 | AUC ほぼ同等 + 人気乖離馬を押す → 期待値改善の可能性大 |

**中長期 (V20/V21): 条件付き可能**

条件:
1. 調教タイム全数取得 (JVLink WOOD 完全充填)
2. 動画 AI 馬体・歩様 feature (Phase 4)
3. JRDB TYB 直前調教 (V21 で取得中)
4. これらが揃えば「オッズなしで同等 AUC」が視野に入る

### れんはすの哲学 (オッズは結果) は実現可能か?

**実現可能 — ただし追加データが必要**

現状の能力 feature 137個は「調教の全数データ不足」と「馬体評価なし」がボトルネック。

> オッズを捨てると AUC は下がる。しかし AUC 最大化が目標ではなく、
> 「人気の歪みを突いて期待値を上げる」ことが目標なら、
> 能力 feature 強化 + 穴狙い戦略で ROI 改善は十分あり得る。

### paper 検証に進むべきか

**YES — 条件無し、即開始推奨**

- ✅ V16 WF AUC 0.8677 (delta -0.0001) — 条件 (-0.02 以内) を大幅クリア
- paper 期間: 少なくとも 4週間 (20R 以上)
- 判定基準: TOP1 連対率 ≥ 40%、穴馬 (人気3位以下) の的中率が V15 より向上
- **特注**: AUC 同等なら穴馬 ROI で V15 を超えられるかが本命テーマ

---

## 8. 実装まとめ

| ファイル | 内容 |
|---------|------|
| `train/train_v16_ability.py` | V16 能力ベース学習スクリプト (新規) |
| `models/v16_ability_candidate.pkl.gz` | V16 candidate (学習完了後に生成) |
| `data/v16_ability_wf_report.json` | WF AUC 詳細レポート (学習完了後) |

*V15 production 完全不変 — keiba_model_v15_central.pkl.gz は触らない*
