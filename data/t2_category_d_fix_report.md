# T2: カテゴリD 修正レポート

Date: 2026-04-13

## v15 カテゴリD 内訳と修正

| 特徴量 | 修正前 | 修正後 | 状態 |
|--------|--------|--------|------|
| jrdb_prev_idm | 常に 50.0 | 実値（例: [33,46,49,53,54]） | **FIXED** |
| jrdb_prev_pace_idx | 常に 50.0 | 実値（例: [-24,2.5,-9.4,1.3,12]） | **FIXED** |
| jrdb_prev_rise_code | 常に 3 | SED実値 | **FIXED** |
| jrdb_cid_idx | — | 実値動作中（例: [-3.6,2.6,-0.2,5.0,4.2]） | **元々OK（誤認）** |
| jrdb_ls_idx | — | 実値動作中 | **元々OK（誤認）** |

## 修正内容

`tools/jrdb_features.py` の `merge_jrdb_predict_features()` の SED 前走データマージ処理を拡張。
従来は3列（馬場差/不利/出遅）のみマージしていたが、6列（+IDM/ペース指数/上昇度コード）に拡張。

blood_num 経由で過去 SED 最新行から取得。デフォルト値は JRDB_DEFAULTS（idm=50, pace_idx=50, rise_code=3）を上書き対象に追加。

## 検証

テストレース `202606030511`:
- 修正前: prev_idm/pace_idx/rise_code 全て default 固定
- 修正後: 5/6 特徴量取得 [JRDB] ログ出力、実値がマージされることを確認

## jrdb_cid_idx/ls_idx について

JO CSV 分析結果:
- 総行数 299,201 (2020-2026)
- cid_idx 非ゼロ率: 99.7% (298,332/299,201)
- ls_idx 非ゼロ率: 30.6% (91,441/299,201 — ls_evalが未入力のレース多い)

→ 「常にゼロ」は誤認識。KYI マッチがあれば正しく動作する。
→ ユーザーが「ゼロ」と感じる原因は、おそらく:
  1. 予測時に当該レースが JRDB 未配信（当日分の取得タイミング問題）
  2. 馬名フォールバックが効かない特定ケース
  3. ls_idx 本来ゼロのレース（30% のみ非ゼロ）
→ 運用改善: `daily_predict.py` 実行前に JRDB 日次取得を確実に実施すること

## カテゴリB の改善 (未実施)

以下は v15 で既に各種フォールバック実装済み。追加改善は別途:

| 特徴量 | 現行フォールバック |
|--------|-------------------|
| jrdb_paddock_idx | TYB未配信時 default 50 |
| jrdb_odds_idx | 同上 |
| jrdb_live_composite_idx | 同上 |
| jrdb_body_code | default 4 |
| jrdb_demeanor_code | default 2 |
| stable_comment_score | 未取得時 0.0 |
| gaisha_rank | PACI CSV default |
| weight_ma5 | 前走履歴から計算、欠損時は現在値 |

改善余地: `daily_premium_scrape.py` の取得成功率向上＋失敗時リトライ（T4で一部対応済み）。
