# PHASE 5: 過去の確認漏れチェック (2026-04-24 22:55)

## 5-1. odds_base CSV 生成動作
- `tools/test_odds_base_perday_csv.py` ❌ 存在せず
- 実装: `tools/daily_predict.py` line 322 `save_odds_base(race_id, odds_full, date_str=date_str)`
- 本日 4/24 20:09 時点で `data/odds_base_20260425.csv` 作成済み (9レース/京都のみ)
- **初の本格動作確認は 4/25 AM8:00 DailyPredict で実施される**
- 本番前チェック: tools/daily_predict.py の save_odds_base 呼び出しが有効 ✅

## 5-2. 配当文字化け (三→３ 問題)
- `tools/daily_results.py` line 34: `import unicodedata`
- Line 42-43: `unicodedata.normalize('NFKC', text)` で NFKC 正規化実装
- Line 174-176: `三連複/3連複/３連複` 全バリエーション対応の判定ロジック
- **対策済** ✅

## 5-3. Streamlit 多重起動防止
- `run_streamlit.bat` 存在 ✅
- ポート 8501 チェック機能あり
- --force オプションで既存プロセス強制終了可能
- **多重起動防止設計 OK** ✅

## 5-4. 阪神テスト
- 今週末 (4/25-26): 福島(03) + 東京(05) + 京都(08)
- **阪神開催なし** → 本項目は該当せず
- 代わりに京都で代替テスト可能 (dry-run結果で確認)

## 5-5. 新馬戦予測信頼度
- dry-run 完了後に予測スコア分布を確認予定
- 現時点で未確認 (PHASE 2 進行中)

## 5-6. IPバン兆候
- `logs/*20260424*.log` および `logs/*20260423*.log` 検索:
  - 429 (Too Many Requests): **0件** ✅
  - 503 (Service Unavailable): **0件** ✅
  - 504 (Gateway Timeout): **0件** ✅
- SCRAPER-GUARD 動作ログ:
  - Fri 03:00 早朝特例で ALLOW 正常動作
  - 金曜22:00 以降の GUARD 再発動は本日23時以降に確認可能
- **IP制限の兆候なし** ✅

## 追加発見: モデルバージョン
- 現行: **v15 Pattern B (150 特徴量)** (ファイル名: `keiba_model_v15_central_live.pkl.gz`)
- CLAUDE.md 記載の v13.5b (124特徴量) より新しい
- 4/24 AM8:00 ログで「v15 Pattern B 150特徴量 ロード完了」を確認

## 判定: 🟢 OK (5-5 のみ未確認、dry-run 完了待ち)
