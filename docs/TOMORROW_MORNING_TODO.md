# 4/28 夜 〜 4/29 朝 の TODO

## 4/28 (火) 残り22分でやること
1. ⏰ 15:30-15:35 v16.1 完了確認
   - bash tools/v161_check_and_act.sh
   
2. v16.1 採用判定:
   - ✅ 採用 (mean_auc > 0.8856): bash tools/v161_deploy_full.sh
   - ❌ 不採用: git push 7本目 (drift fix + master_index fix のみ)

## 4/29 (水) 朝の確認
1. 朝活時間 (06:00-08:00):
   - レース当日予測の自動実行確認 (08:00 タスクスケジューラー)
   - Discord に予測通知届くか確認

2. v16.1 デプロイなら:
   - 当日予測で v16.1 が稼働してるか確認
   - 戦略⑦v1 の 4R 除外が動いてるか確認

3. 5/2 GW初日 (土) 準備:
   - 残り 4日
   - 全システム動作確認

## 5/4-5/6 GW後半 (バグ修正 + v16.2 準備)

### 5/4 (月)
- master_index 2020-2022 取得 (バグ修正済みで動く!)
python tools/scrape_master_index.py --year 2020
python tools/scrape_master_index.py --year 2021
python tools/scrape_master_index.py --year 2022

### 5/5 (火)
- race_id 変換ロジック調査 (docs/V162_RACE_ID_FIX.md)
- features_v16_premium.py の _build_nk_race_id_from_jv() 改善

### 5/6 (水)
- v16 ablation 再実行 (--only-ablation)
- カバレッジ修正後の真の効果評価

### 5/7-5/8 (木金) 平日夜
- v16.2 学習バックグラウンド実行

### 5/9-5/10 (土日) GW明け週末
- v16.2 デビュー!
- 期待 mean AUC: 0.886~0.890 (+6~+11bp)

## 重要ファイル
- docs/MAY_PLAN_V162.md: 5月計画
- docs/V162_RACE_ID_FIX.md: race_id 変換改善
- docs/MASTER_INDEX_BUG_TODO.md: master_index バグ記録
- tools/v161_deploy_full.sh: v16.1 デプロイ
- tools/v161_check_and_act.sh: v16.1 完了確認

## 重要な実績 (4/28)
- ✅ master_index バグ修正完了
- ✅ race_id 変換問題発見 (10.7% → 改善余地大)
- ✅ カバレッジ不足が効果なし原因と判明
- ✅ v16.2 戦略書完成
- ✅ git push 6本目完了 (8178293b)
