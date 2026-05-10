# Phase 13 — 波乱度 + 個別ラップ + トラックバイアス PoC

**date**: 2026-05-10
**target**: netkeiba マスター 18 features (波乱度 3 + ラップ 10 + バイアス 5)

---

## 1. AI 波乱度 (3 features)

| feature | type | range | 説明 |
|---------|------|-------|------|
| `master_haran_score` | float | 0-100 | レース荒れやすさ score |
| `master_top_pop_trust` | float | 0-100 | 上位人気信頼度 |
| `master_haran_meter` | int | 1-5 | 波乱メーター 5 段階 |

### URL (推定)

```
GET https://race.sp.netkeiba.com/race/upset.html?race_id={race_id}
```

### parser 戦略

- `.haran_score` / `.upset_score` / `#upset_score` から数値抽出
- `.haran_meter.lv1`〜`.lv5` の class で 1-5 段階判定
- `.top_pop_trust` 系 selector で上位人気信頼度

### V20 統合 効用

- **race-level 投資判断 filter**: `haran_score > 70` の race は「条件 X 化」 (荒れ前提で買い目調整)
- **戦略⑦ 強化**: `top_pop_trust < 30` ならスキップ候補

---

## 2. 個別ラップ (10 features)

| feature | range | 説明 |
|---------|-------|------|
| `master_horse_lap_avg_first3f` | 30-45 | 前 3 走 前半 3F 平均 |
| `master_horse_lap_avg_last3f` | 30-45 | 前 3 走 後半 3F 平均 |
| `master_horse_lap_best_last3f` | 30-40 | 前 3 走 後半 best |
| `master_horse_lap_consistency` | 0-3 | ラップ std (低い=安定) |
| `master_horse_lap_best_3f` | 30-40 | 全期間 ベスト後半 |
| `master_horse_lap_pos_change_avg` | -8〜+8 | 位置取り変化 |
| `master_horse_lap_finish_speed` | 10-15 | 終速指標 |
| `master_horse_lap_acc_phase` | 0-3 | 加速 phase 数 |
| `master_horse_lap_dec_phase` | 0-3 | 減速 phase 数 |
| `master_horse_lap_distance_factor` | 0-1 | 距離適応 factor |

### URL (推定)

```
GET https://race.sp.netkeiba.com/race/lap.html?race_id={race_id}
```

→ master の特徴: 各馬 過去走 個別ラップ (実況 30 sec 遅延 realtime data)

### parser 戦略

- table tr ループで馬番 row 抽出
- 直近 3 走の前後半 3F 値 抽出 → 平均 / std / best 計算
- 距離適応 factor は同距離出走時の 3F 順位差から算出
- 取れない場合 default fill (`avg_first3f=35.5`、 `consistency=1.0` 等)

### V20 統合 効用

- 既存 features `prev_last3f` / `avg_last3f_3r` の **完全 superset** (前 3 走 vs 直近 1 走)
- ラップ std (consistency) は新規信号、 期待 +0.02-0.04 AUC
- 距離適応 factor は距離 transition race で重要

---

## 3. トラックバイアス (5 features)

| feature | range | 説明 |
|---------|-------|------|
| `master_track_inner_outer_bias` | -1〜+1 | 内外 bias |
| `master_track_front_back_bias` | -1〜+1 | 前後 bias |
| `master_track_corner_bias` | -1〜+1 | コーナー bias |
| `master_track_pace_bias_score` | -1〜+1 | ペース bias |
| `master_track_today_severity` | 0-100 | 当日馬場 severity |

### URL (推定)

```
GET https://race.sp.netkeiba.com/race/track_bias.html?kaisai_id={kaisai_id}
```

→ kaisai_id = race_id 先頭 10 桁 (場 + 開催 + 日次)、 同日同場 R 共通。

### parser 戦略

- text 全体から「内有利」「外有利」「逃げ有利」「差し有利」を pattern match
- `.bias_severity` / `.track_severity` で severity 数値 抽出
- 1 kaisai = 1 fetch (R 単位 fetch ではなく日単位 12 R 共通) → 効率的

### V20 統合 効用

- **race-level features** 5 件、 全馬同値だが レース予測 model に有効
- 既存 V15 にバイアス情報なし → 完全新規信号、 期待 +0.01-0.03 AUC
- 戦略⑦ filter 強化: 「内有利かつ 8 番枠以下なら買い増し」 等 rule 追加可能性

---

## 4. PoC 実装 状態

`tools/netkeiba_master_scraper.py` 完成 (skeleton + parser stub):

```python
# 4 系統 関数 (kill switch + rate limit + default fill)
_parse_ai_tenkai(html, umaban) → AI_TENKAI_FEATURES (7)
_parse_ai_haran(html) → AI_HARAN_FEATURES (3)
_parse_lap(html, umaban) → LAP_FEATURES (10)
_parse_track_bias(html) → TRACK_BIAS_FEATURES (5)

# Top-level
fetch_master_features(race_id, umaban, kaisai_id) → MasterFeatureBundle
fetch_race_master_features(race_id, umaban_list) → {umaban: features}
```

CLI:
```bash
python tools/netkeiba_master_scraper.py --list           # features 一覧
python tools/netkeiba_master_scraper.py --status         # kill switch + cookie
python tools/netkeiba_master_scraper.py --race <ID>      # 単発 fetch
python tools/netkeiba_master_scraper.py --enable/--disable # kill switch
```

---

## 5. 動作 verify

```
$ python tools/netkeiba_master_scraper.py --list
Phase 13 全 25 features:
  B. AI 展開予測 (7)
  C1. AI 波乱度 (3)
  C2. 個別ラップ (10)
  C3. トラックバイアス (5)

$ python tools/netkeiba_master_scraper.py --status
disabled: False
cookie loaded: True
```

→ skeleton 動作 OK。 実 fetch は Phase 13.5 (実 DOM 検証後)。

---

## 6. 期待 V20 寄与

| category | 期待 ΔAUC | 根拠 |
|----------|-----------|------|
| AI 展開予測 7 | +0.005 〜 +0.012 | netkeiba 内製 AI 結果の流用、 既存 features と相関高い可能性 |
| AI 波乱度 3 | +0.003 〜 +0.008 | race-level 投資判断 filter として、 ROI 改善寄与 |
| 個別ラップ 10 | +0.015 〜 +0.030 | 既存 prev_last3f superset、 std / 距離適応は完全新規 |
| トラックバイアス 5 | +0.005 〜 +0.015 | 完全新規 race-level 信号、 V15 に存在しない |
| **合計** | **+0.028 〜 +0.065** | conservative; V18 0.91-0.93 → V20 0.94 視野 |
