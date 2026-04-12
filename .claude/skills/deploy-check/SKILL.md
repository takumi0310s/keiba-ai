---
name: deploy-check
description: 本番反映前23項目チェックリスト — Streamlit Cloudデプロイ・モデル更新時に必ず通す。
---

# deploy-check — 本番反映前23項目チェック

## モデル整合性 (1-6)
1. `python tests/regression_test.py` 全PASS
2. `python tests/test_predict_core.py` 全PASS
3. モデルファイル `keiba_model_v*_central.pkl.gz` が存在
4. モデルファイル `keiba_model_v*_central_live.pkl.gz` が存在
5. app.py `_discover_latest_model()` が最新版を返す
6. predict_core.py `load_models()` の特徴量数 == モデル `num_feature()`

## 特徴量パイプライン (7-12)
7. predict_core.py で v15新特徴量関数(`features_v15_new`)が呼ばれている
8. Pattern B生成列数 = モデルfeature数（150）
9. dist_cat が5bin (0-4)
10. odds_change_rate / pop_rank_change / odds_sharp_drop の3列が生成
11. transport_distance_km, gaisha_rank が0以外で出る
12. JRDB特徴量取得が失敗してもクラッシュしない

## データ・ファイル (13-18)
13. data/odds_base_YYYYMMDD.csv が当日分存在
14. data/feature_lookups.pkl(.gz) が存在
15. .env の `NETKEIBA_COOKIE` が有効
16. requirements.txt に beautifulsoup4 が記載
17. .gitignore に .env / *.pkl.gz / data/*.csv が記載
18. 不要な巨大デバッグファイルが未コミット

## 動作確認 (19-23)
19. app.py 構文チェックPASS
20. predict_core.py 構文チェックPASS
21. 当日1レースで「予想する」が動く（特徴量150/150）
22. オッズ変動特徴量がリアルタイムで埋まる（≥1馬で値≠0）
23. Discord通知が送信できる（DISCORD_WEBHOOK_BETS設定済み）

1個でもFAILなら反映禁止。
