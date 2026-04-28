# master_index スクレイピングバグ (4/28 判明)

## 問題
4/27 のスクレイピングログで以下を確認:
- 2020-2024 全ての年で "Already scraped: 20733" と表示
- "New to scrape: 0" → "Nothing to scrape!" で終了
- 実際には 2020-2022 の master_index は 0 行

## 原因 (推定)
tools/scrape_master_index.py の `Already scraped` 判定:
- 全年の race_id を 1つのリストで判定
- year でフィルタしてない
- 全年の取得済み数 (20,733) を全年に適用
- 結果: どの年指定しても新規対象ゼロ

## 影響
- master_index は 2024年以降のみ (139,673行)
- prev_master_index の充填率 14.6% (上限)
- v16 ablation 結果: prev_master_index は +2bp 効果なし
  → 緊急度低

## 修正方針 (5/4-5/6 GW後半)
1. scrape_master_index.py の Already scraped 判定箇所特定
2. year フィルタを追加
3. 2020-2023 を再取得試行
4. もし改善なら → v16.2 で prev_master_index 再評価
5. もし netkeiba 側に過去データなしなら → 諦める

## 優先度
🟡 中 (5/4-5/6 GW後半で対応)
