# netkeiba依存 棚卸し + 代替マッピング + 再学習判断（item1/2/4）

netkeiba解約→JRDB+JRA-VAN(JV-Link)で再構築。V15 pkl不変・全paper eval経由。

## item1. 依存棚卸し

### ★重要な前提補正: JRA-VAN(JV-Link)の守備範囲★
調査の一次結論は JRA-VAN を過小評価していた。JV-Link は以下を提供:
- **出馬表/出走馬**（RA/SE レコード: 馬番・馬名・騎手・斤量・馬体重）
- **確定結果・着順**（SE）・★**払戻金(HR レコード)**★・**オッズ(O1-O6)**・**馬マスタ/血統(UM/BLOD)**
→ 「払戻は netkeiba のみ」「出馬表は netkeiba のみ」は**誤り**。JV-Link で代替可能。

### 真に netkeiba専用（JRDB/JV-Link代替が弱い/無い）= 特徴系
| 系統 | 特徴量 | netkeibaソース | 代替 |
|------|--------|--------------|------|
| 調教実タイム | training_time_filled, training_per_dist, wood/sakaro_best_*f, time_1f_last, training_intensity_enc, wood_count_2w, total_training_count, has_* | oikiri.html(premium) | JRDB CYB調教指数(jrdb_training_idx/oikiri_idx/training_arrow)=**指数のみ・実タイム無** |
| タイム指数 | index_max/run1/avg5_filled | speed.html(premium) | ★代替不能★（JRDB相当なし） |
| 厩舎コメント | stable_comment_score | comment.html(premium) | JRDB stable_eval(数値)=**本文なし・粗い** |
| 前走ラップ | prev_race_first3f/last3f/pace_diff, prev_agari_relative | db.netkeiba race | JRDB SED pace_idx/agari_idx(指数) |

### 入力・表示層（JV-Lynkで代替設計）
- 出馬表/オッズ/結果/払戻: `predict_core.parse_shutuba`(shutuba.html) / `_fetch_odds_api`(api_get_jra_odds) / `daily_results.fetch_race_result`(result.html) → **全て JV-Link へ差し替え設計**（item2）。
- 表示層(Discord/Streamlit): race_auto_notify/daily_discord_report/app.py が調教ランク・タイム指数・厩舎コメントを表示 → netkeiba専用特徴の欠落に伴い**表示も簡素化**（別途）。

## item2. 代替マッピング

| netkeiba機能 | 代替 | 実装方針 |
|------|------|---------|
| 出馬表 | JV-Link RA/SE | jvlink_fetcher で出走馬・騎手・斤量・馬体重取得→parse_shutuba差し替え |
| リアルタイムオッズ | JV-Link O1(単複) | odds_log/pop_rank を O1由来に。※Pattern B当日 |
| 結果・着順・払戻 | **JV-Link SE + HR** | daily_results.py の取得元を result.html→JV-Link HR に差し替え（払戻確定はHR） |
| 調教 | JRDB CYB (jrdb_training_idx/oikiri_idx/training_arrow) | 実タイム→指数へ。training_time_filled は CYB指数へマップ |
| トラックバイアス | JRDB SRB | (V15特徴に直接は無・将来) |
| ラップ | JRDB SED/SRB (pace_idx/agari_idx) | prev_race_lap → SED指数へ |
| タイム指数 | **代替不能** | 欠損（既定値）運用 |
| 厩舎コメント本文 | JRDB stable_eval | stable_comment_score → stable_eval数値へ（粗） |
| 血統(父/母の父) | JV-Link BLOD / blood_full.csv | sire_enc/bms_enc は既存blood_full+JV-Linkで維持可 |

**代替不能リスト**: タイム指数(index_*)・厩舎コメント本文。→ 欠損運用（gain影響は下記で軽微）。

## item4. 再学習判断（★データ実測★）

### gain重要度（V15 booster, 位置対応で解決）
- 上位=市場代理: paci_jockey_exp_3rd 17.8% / paci_ninki_idx 16.7% / paci_jockey_exp_wr 15.1%（計~49%）。
- **netkeiba依存20特徴の合計gain = 9.44%**。内訳: training_time_filled 5.45% + training_per_dist 3.03% に集中（=8.5%）。
- **タイム指数/厩舎コメント/前走ラップ = 合計~1%（無視可能）**。

### WF ablation（netkeiba20を既定値化, 直近3fold 2023-25）
| 構成 | WF AUC |
|------|--------|
| full(145) | **0.8408** |
| netkeiba20 既定値化 | **0.8162** |
| **低下** | **−0.0246** |

### 判断（データに基づく提案）
- **代替不能なタイム指数/コメント/ラップ(~1%gain)は欠損運用でOK**（影響ほぼ無）。
- だが**調教実タイム(training_time, 8.5%gain)の喪失で AUC −0.025**。JRDB調教指数は既にモデル内にあるが**完全代替せず**（netkeiba実タイムに固有信号）。
- → ★**純粋な欠損運用(V15 pkl + 既定値入力)は −0.025以上劣化。v15r再学習で 0.816 まで回復可**★。
  - 推奨: **調教を JRDB CYB指数へマップ + 残netkeiba専用は欠損 → v15r 再学習（WF同一fold）→ paper eval**。
  - ただし autopsy の通り V15 の実エッジは既に市場適応で減衰中。**v15r(0.816)は V15(0.841)より弱く、黒字化は期待薄**＝再構築は「運用継続の最低限」であり収益改善策は別途（レース選択/資金管理）。

### 結論
- 欠損運用のみ=不可(−0.025)。**調教のJRDB代替マップ + v15r再学習が妥当**。
- ただし期待は「V15より弱い模型で運用最低限を維持」。収益性回復には別施策が必要（死因解剖の一次要因=エッジ減衰）。
