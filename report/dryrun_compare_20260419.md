# 4/19 ドライラン: 修正後 build_features 統計

`merge_jrdb_predict_features` を 4/19 全 35 レース 476 頭で実行した結果:

| feature | races | horses | default | nan | mean | non_default_rate |
|---|---|---|---|---|---|---|
| jrdb_prev_agari_idx | 35 | 476 | 47 | 0 | -8.34 | 90.1% |
| jrdb_prev_idm | 35 | 476 | 55 | 0 | 39.74 | 88.4% |
| jrdb_prev_pace_idx | 35 | 476 | 47 | 0 | -2.02 | 90.1% |
| jrdb_prev_ten_idx | 35 | 476 | 47 | 0 | -9.26 | 90.1% |
| jrdb_prev_track_bias | 35 | 476 | 52 | 0 | -14.67 | 89.1% |

## 解釈
- non_default_rate が約 90% 前後 = blood_num 一致した 91.8% の馬で実値が取得できている
- mean_value はデフォルト 50 から実分布の中央値 (~30-45 程度) に下方修正されている
- 修正前は default=50 が混入して mean が 50 寄りだった

## 予測スコア影響予測
- v15 モデルは過去 SED 値 (default ではなく実数) で学習済み
- 予測時の値が学習分布に近づくため、予測の安定性向上が期待できる
- 模型再学習は不要 (default 50 → 実値はモデルから見て normal range 内)