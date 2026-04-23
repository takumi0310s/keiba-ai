# フェーズ1 調査サマリー (2026-04-23)

対象データ: `data/daily_predictions/20260419.csv` (35レース / 476頭)

## JRDB結合率 (race_id key 直接結合)

| ファイル | 結合率 | 種別 | 評価 |
|---|---|---|---|
| KYI | 35/35 (100%) | PRE_RACE | OK |
| CYB | 100% | PRE_RACE | OK |
| JOA | 100% | PRE_RACE | OK |
| CHA | 100% | PRE_RACE | OK |
| JO  | 100% | PRE_RACE | OK |
| SED | 0% (race_id) / **91.8% (blood_num)** | PREV | **要詳細** |
| TYB | 0% | LIVE | 正当 (4/19 8:00予測時はTYB未取得) |
| KTA | 0% (race_id) → 100% (blood_num経由) | PRE_RACE | OK |
| KKA | 0% (race_id 直接) | PRE_RACE | バグ可能性あり |
| KAB | rid列なし (kaisai_key key) | PRE_RACE | スキーマ違い、別キー |

## 特徴量カバレッジ (default以外の率)

カテゴリ別平均:
- jrdb_kyi_basic: **81.7%** (24特徴)
- jrdb_sed_prev:  **48.2%** (8特徴)
- jrdb_tyb:        0.0% (5特徴) → 正当 (LIVE未取得)
- jrdb_blood/kab_sr/jo/kta/cha/skb/ze: 100% (各2-4特徴)

問題候補 (LOW < 50%):
| 特徴量 | 非default率 | デフォルト | 判定 |
|---|---|---|---|
| jrdb_paddock_idx, odds_idx, live_composite_idx, body_code, demeanor_code | 0% | 50/50/50/4/2 | 正当 (TYB未取得) |
| jrdb_prev_interference | 0.6% | 0 | 正当 (不利なし大半) |
| jrdb_rise_code | 1.9% | 3 | 正当 (B中間が大半) |
| jrdb_prev_rise_code | 1.9% | 3 | 正当 (同上) |
| jrdb_stable_eval | 10.3% | 3 | 正当 (現状維持が大半) |
| jrdb_prev_late_start | 12.6% | 0 | 正当 (出遅なし大半) |
| jrdb_training_arrow | 18.3% | 3 | 正当 (平行が大半) |

REVIEW (50-80%):
| 特徴量 | 非default率 | 解釈 |
|---|---|---|
| jrdb_entry_days_ago | 53.6% | 入厩情報欠落 = 現役馬として正常 |
| jrdb_heavy_apt | 57.1% | デフォルト=2(普通)が多い、データ欠損ではない |
| jrdb_dist_apt | 71.6% | デフォルト=0(不明)、距離適性データ未付与馬 |
| jrdb_prev_idm | 73.1% | SED 91.8%-26.9%≒65% → 25%は値=default 50.0 |
| jrdb_prev_track_bias | 73.5% | 同上 |
| jrdb_prev_agari/pace/ten_idx | 74-75% | 同上 |
| jrdb_stable_rank | 75.8% | 厩舎ランク不明馬 |

## 「バグ由来」と推定される箇所

1. **SED merge: blood_num マッチ率 91.8% に対して prev_idm 非default率 73.1%**
   - blood_num は 437/476 馬で一致しているのに、26.9% が default 50.0 のまま
   - → blood_num マッチした馬の中に SED 値がそのまま 50.0 だったケース、もしくは
     SED merge 後の `_default_map` 上書きロジックで一部馬が default に戻されている可能性
   - **改善余地: 約 18.7pt (73.1% → 91.8% 相当)**

2. **KKA は race_id+umaban で結合できるはずが 0%**
   - merge_jrdb_predict_features に KKA の処理が入っていない可能性
   - 既存パイプラインは KKA を使っていない (現行 v15 は KKA 不使用)
   - → 採用判定: KKA は v16 用、本タスクの対象外

3. **TYB 0% は AM3:00 取得タイミングが原因**
   - 4/19 8:00 予測時は TYB がまだ JRDB から落ちてない (当日朝発表分)
   - race_auto_notify (AM 8:45) は TYB 取得後なので問題なし
   - **改善対象外** (運用上の正当ゼロ)

## 結論: フェーズ2 で改善可能な点

- **SED merge 後の prev_* default 補填ロジックの精査** (主要)
- KKA は別タスクとして保留
- TYB は運用問題、コード変更不要

## 累計 ROI 影響予測

SED prev_* の 18.7pt 改善が達成できれば、関連特徴量 (8つ) の値が改善し、
モデル予測の安定性向上に寄与する見込み。ただし定量的影響は予測ドライランで確認。
