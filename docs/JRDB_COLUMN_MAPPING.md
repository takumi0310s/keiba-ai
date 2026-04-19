# JRDB カラム名マッピング (日本語 → 英語スネークケース)

v2 CSV (`data/jrdb_{sed,tyb,cyb}_v2.csv`) は英語スネークケースに統一。
既存 `data/jrdb_{sed,tyb,cyb}.csv` は日本語のまま残置（後方互換用）。

Single Source of Truth: `tools/jrdb_column_mapping.py`

## 共通レースキー

| 日本語 | 英語 | 型 | 意味 |
|---|---|---|---|
| 場コード | jo_code | str | 競馬場コード(2桁) 01=札幌..10=小倉 |
| 年 | year2 | str | 2桁年(26=2026) |
| 回 | kai | str | 回次 1-9 |
| 日 | nichi | int | 開催日目(1-12, hex) |
| R | race_num | str | レース番号(01-12) |
| 馬番 | umaban | int | 馬番 |
| jra_race_id | jra_race_id | str | JV形式 race_id(10桁) |
| nk_race_id | nk_race_id | str | netkeiba形式 race_id(12桁) |

## SED (成績データ)

`data/jrdb_sed.csv` → `data/jrdb_sed_v2.csv` (105,658行 × 58 cols, 2024-2026)

| 日本語 | 英語 | 型 | 意味 |
|---|---|---|---|
| 血統登録番号 | blood_num | str | 血統登録番号(8桁、馬固有) |
| 年月日 | yyyymmdd | str | レース日 YYYYMMDD |
| 馬名 | horse_name | str | 馬名 |
| 距離 | distance | int | 距離(m) |
| 芝ダ障害コード | surface_code | int | 1:芝 2:ダ 3:障害 |
| 右左 | lr_code | int | 1:右 2:左 3:直 |
| 内外 | inout_code | int | 1:通常 2:外 3:直ダ |
| 馬場状態 | baba_state | int | 10:良 20:稍重 30:重 40:不良 |
| 種別 | shubetsu | int | 種別コード |
| 頭数 | num_horses | int | 出走頭数 |
| 着順 | finish | int | 着順 |
| 異常区分 | abnormal | int | 0:正常 1:取消 2:除外 3:中止 4:失格 5:降着 |
| タイム | time_min_sec | str | 分+秒(0.1s) |
| 斤量 | weight_carry | float | 0.1kg単位 |
| 確定単勝オッズ | odds_final | float | 確定単勝 |
| 確定単勝人気 | popularity | int | 確定人気順 |
| IDM | idm | float | 確定 INDEX MEMORY |
| 素点 | soten | float | IDM 元スコア |
| 馬場差 | baba_sa | float | トラックバイアス補正 |
| ペース | pace | float | ペース値 |
| 出遅 | deokure | float | 出遅れ補正 |
| 位置取 | ichi_dori | float | 位置取り |
| 不利 | furi | float | 不利合計 |
| 前不利 | mae_furi | float | 前半不利 |
| 中不利 | naka_furi | float | 中盤不利 |
| 後不利 | ato_furi | float | 後半不利 |
| レース | race_score | float | レースレベル |
| コース取り | course_dori | int | 1:最内..5:大外 |
| 上昇度コード | josho_code | int | 1:AA 2:A 3:B 4:C 5:? |
| クラスコード | class_code | int | 能力クラス |
| 馬体コード | batai_code | int | 馬体評価 |
| 気配コード | kehai_code | int | 気配評価 |
| レースペース | race_pace | str | H/M/S |
| 馬ペース | horse_pace | str | H/M/S |
| テン指数 | ten_idx | float | 確定前半3F指数 |
| 上がり指数 | agari_idx | float | 確定上がり3F指数 |
| ペース指数 | pace_idx | float | ペース指数 |
| レースP指数 | race_pace_idx | float | レースP指数 |
| 前3Fタイム | first3f_sec | float | 前半3F(0.1s) |
| 後3Fタイム | last3f_sec | float | 後半3F(0.1s) |
| コーナー順位1-4 | corner{1-4}_pos | int | コーナー順位 |
| 馬体重 | horse_weight | int | 馬体重(kg) |
| 馬体重増減 | horse_weight_diff | str | 増減(符号+2桁) |
| 天候コード | weather_code | int | 1:晴 2:曇 3:小雨 4:雨 5:小雪 6:雪 |
| レースペース流れ | race_pace_flow | int | 第4版 |
| 馬ペース流れ | horse_pace_flow | int | 第4版 |
| 4角コース取り | corner4_dori | int | 4角コース取り |

## TYB (直前データ)

`data/jrdb_tyb.csv` → `data/jrdb_tyb_v2.csv` (1,983行 × 29 cols)

| 日本語 | 英語 | 型 | 意味 |
|---|---|---|---|
| IDM | idm | float | 直前 IDM |
| 騎手指数 | jockey_idx | float | 騎手指数 |
| 情報指数 | info_idx | float | 情報指数 |
| オッズ指数 | odds_idx | float | 直前オッズ指数 |
| パドック指数 | padock_idx | float | パドック指数 |
| 予備1 | reserve1 | float | 予備 |
| 総合指数 | sogo_idx | float | 直前総合指数 |
| 馬具変更情報 | bagu_change | int | 0:無 1:変更 2:変更(効果有) |
| 脚元情報 | ashimoto | int | 0:フラット 1:好転 2:疑問 3:悪化 |
| 取消フラグ | cancel_flag | int | 1:取消 |
| 馬場状態コード | baba_state_code | int | 10:良..40:不良 |
| 天候コード | weather_code | int | 1:晴..6:雪 |
| 単勝オッズ | tansho_odds | float | 直前単勝 |
| 複勝オッズ | fukusho_odds | float | 直前複勝 |
| 馬体重 | horse_weight | int | 馬体重 |
| 馬体重増減 | horse_weight_diff | str | 増減 |
| オッズ印 | odds_mark | str | 印 |
| パドック印 | padock_mark | str | 印 |
| 直前総合印 | sogo_mark | str | 印 |
| 馬体コード | batai_code | int | 1:太..7:緩 |
| 気配コード | kehai_code | int | 1:良..8:イレチ |

## CYB (調教分析)

`data/jrdb_cyb.csv` → `data/jrdb_cyb_v2.csv` (1,874行 × 19 cols)

| 日本語 | 英語 | 型 | 意味 |
|---|---|---|---|
| 調教タイプ | train_type | str | 調教タイプ |
| 調教コースタイプ | train_course_type | str | コースタイプ |
| 調教馬場 | train_baba | str | 馬場状態 |
| 追切指標 | train_mark | str | 追切評価 |
| 仕上指標 | train_amount | str | 仕上り |
| 変化指標 | train_change | str | 変化 |
| 調教コメント | train_comment | str | コメント(40byte) |
| コメント年 | comment_year | str | コメント年 |
| コメント回日 | comment_date | str | コメント日 |
| 調教評価 | train_eval | str | 調教評価 |
| 調教コース | train_course | str | コース名 |

## 後方互換ポリシー

- 既存 `data/jrdb_{sed,tyb,cyb}.csv` は**絶対に上書きしない**
- `tools/jrdb_features.py` は v2 を優先読込、無ければ旧CSVにフォールバック
- 既存 `_sed_col_map` / `_tyb_col_map` の英語→日本語マッピングは温存
  （古い英語カラム名での入力も受け入れる保険）

## 再生成手順

```bash
# 全タイプ・全年
python tools/build_jrdb_v2_csv.py

# 年限定 (SED 用、2024-2026)
python tools/build_jrdb_v2_csv.py --types SED --years 24 25 26

# TYB/CYB のみ
python tools/build_jrdb_v2_csv.py --types TYB CYB
```
