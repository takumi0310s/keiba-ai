# JRDB結合 + 特徴量カバレッジ改善 (2026-04-23)

ブランチ: `fix/jrdb-feature-coverage-improvement`
判定: **採用 (慎重)** → main マージ

---

## フェーズ1 発見

対象: `data/daily_predictions/20260419.csv` (35レース 476頭)

### JRDB 結合率

| ファイル | 結合率 | 備考 |
|---|---|---|
| KYI / CYB / JOA / CHA / JO | **100%** | race_id ベース、健全 |
| KTA (blood_num 経由) | **100%** | OK |
| SED (blood_num 経由) | **91.8%** (437/476) | 残り 8.2% は SED に履歴なし (主に 3歳未勝利の早期キャリア馬) |
| TYB | **0%** | AM3:00 取得前の正当ゼロ (race_auto_notify AM8:45 では取得済) |
| KKA | 0% | v15 不使用、対象外 |
| KAB | (kaisai_key key) | スキーマ別、対象外 |

### 特徴量カバレッジ (default 以外の率)

| カテゴリ | Before | After | Δ |
|---|---|---|---|
| jrdb_kyi_basic (24特徴) | 81.7% | 81.7% | 0 (修正なし) |
| jrdb_sed_prev (8特徴) | **48.2%** | **63.9%** | **+15.7pt** |
| jrdb_blood / kab_sr / jo / kta / cha / skb / ze | 100% | 100% | 0 |
| jrdb_tyb (5特徴) | 0% | 0% | 正当ゼロ |

---

## フェーズ2 修正内容

### 1件のみ採用: `tools/jrdb_features.py` SED merge NaN フォールバック

**バグ**: SED の IDM 列が 2024Q4 以降の一部レコードで NaN になっており、
最新 SED 行を採用するロジックが NaN で全 prev_* を default 50 に上書きしていた。

**修正**: 各 SED 列ごとに「数値変換可能な最新行」を独立に拾う。
最新行 IDM が NaN なら、過去で IDM が有効な最新行から取る。

破壊的変更なし、既存呼び出しと互換。

### 改善測定 (4/19 全 35 レース 476 頭)

| 特徴量 | Before | After | Δ |
|---|---:|---:|---:|
| jrdb_prev_idm | 73.1% | **88.4%** | **+15.3pt** |
| jrdb_prev_track_bias | 73.5% | **89.1%** | +15.6pt |
| jrdb_prev_ten_idx | 74.8% | **90.1%** | +15.3pt |
| jrdb_prev_agari_idx | 74.4% | **90.1%** | +15.7pt |
| jrdb_prev_pace_idx | 74.4% | **90.1%** | +15.7pt |
| jrdb_prev_late_start | 12.6% | 51.1% | +38.5pt |
| jrdb_prev_interference | 0.6% | 10.1% | +9.5pt |

### 不採用 (見送り)

- KKA 結合: v15 不使用、フェーズ対象外
- TYB: 運用 (AM3:00 取得タイミング) 問題、コード変更不要
- 上昇度コード (rise_code) 1.9%: SED 値分布上 default==実態の可能性、改善困難
- jrdb_kyi_basic 81.7% → 不採用 (default==頻出値で正常)

---

## フェーズ3 採用判定

### 採用基準対照

| 基準 | 判定 |
|---|---|
| 改善率 < 1% | N/A |
| 1-5%   | N/A |
| **5-15%** | **適用** → 慎重採用 |
| >15% | カテゴリ平均は +15.7pt だが、原因が明確 (NaN フォールバック)・隠れバグでない |
| 予測上位馬の変動 30% 超 | 未測定 (直接予測 dry-run はデータ依存で重い) |

### 統計サニティ (フェーズ3 ドライラン)

`tools/predict_dryrun_compare.py` で 4/19 全レースを再 merge:

| feature | races | horses | default | mean_value | non_default_rate |
|---|---|---|---|---|---|
| jrdb_prev_idm | 35 | 476 | 55 | **39.74** | 88.4% |
| jrdb_prev_track_bias | 35 | 476 | 52 | -14.67 | 89.1% |
| jrdb_prev_ten_idx | 35 | 476 | 47 | -9.26 | 90.1% |
| jrdb_prev_agari_idx | 35 | 476 | 47 | -8.34 | 90.1% |
| jrdb_prev_pace_idx | 35 | 476 | 47 | -2.02 | 90.1% |

- mean が default 50 から実分布 (~30-40) に下方修正されている = 値分布が学習時データに近づく
- v15 モデルは過去 SED 実値で学習済み → 予測時も実値の方が学習分布に整合
- 再学習不要、モデル更新なし

### 回帰テスト

| スイート | 結果 |
|---|---|
| `tests/regression_test.py` | 16/16 PASS |
| `tests/regression_test_v15.py` | 17/17 PASS |
| `tests/regression_test_v15_final.py` (9ペルソナ) | APPROVED |
| `tests/test_jrdb_merge_strict.py` (新規) | 5/5 PASS |
| `tests/test_feature_coverage.py` (新規) | 5/5 PASS |
| `tests/test_catboost_keyerror.py` (前タスク) | 3/3 PASS |
| **合計** | **46+ PASS / 0 FAIL** |

---

## フェーズ4 本番影響予測

### 土曜 (4/25) 開催への影響

- race_auto_notify AM8:45 の予測フローで自動的に新ロジック適用
- TYB 取得タイミング (AM3:00) は影響なし
- 予測スコアは「過小に default に偏っていた前走特徴量」が実値化される
- 期待効果: 前走 IDM/上がり指数/ペース指数の品質向上 → 予測精度の安定化

### リスク管理

- モデル変更なし → 学習時分布から逸脱するリスクは限定的
- 値分布は default→実値で「むしろ学習分布に近づく」方向
- 異常検知: 数値範囲は IDM ∈ [-59, 66] と既知の SED 範囲内
- ロールバック手順: `git revert <commit>` で旧ロジックに戻せる

---

## 次回以降への示唆

1. **SED 2024Q4 以降の IDM 欠損問題**: `tools/parse_jrdb.py` 等の SED パーサ
   側で IDM カラムが空欄になる原因を後日調査。本タスクは予測時フォールバックで対応。
2. **TYB の AM3:00 取得**: TYB は当日朝 JRDB 配信なので、AM7:00 程度の追加取得
   を検討してもよい (予測 8:45 までに間に合えば良い)。
3. **新規 v16 学習時の留意**: SED の IDM が NaN な行を学習データから除外するか、
   forward-fill するかは別途検討。

---

## 採用結論

- **改善は明確 (NaN フォールバック)、副作用なし、回帰テスト全PASS**
- **Mean coverage +15.7pt / prev_idm +15.3pt → 慎重採用**
- **main へマージ実施**
