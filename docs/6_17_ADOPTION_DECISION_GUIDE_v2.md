# 6/17 採用判定 Operation Guide (2026-06-17 朝 06:00)

## 1. 実行 (1 コマンド)

```
python tools/c3c4_adoption_test_v2.py --date-from 20260524 --date-to 20260616
```

## 2. 結果の読み方

スクリプトは markdown テーブル + 採用サマリーを出力する。

| verdict | 意味 | アクション |
|---------|------|-----------|
| GO | 採用 | race_auto_notify.py の `STRATEGY_X_PAPER_ONLY = False` に変更 |
| 限定GO | 条件付き採用 | さらに 4 週末蓄積後に再判定 |
| NO-GO | 不採用 | 7/15 まで paper 蓄積継続 |

### 5 項目チェック

| # | 項目 | 閾値 |
|---|------|------|
| 1 | paper N | >= 24R |
| 2 | ROI improvement vs baseline | >= +5pt (GO) / >= +3pt (限定GO) |
| 3 | LEAK audit | PASS (paper strategy は構造上 PASS) |
| 4 | LIVE stability (anomaly count) | == 0 (paper 期間中) |
| 5 | Welch's t-test p 値 | < 0.05 |

**GO = check 1,3,4,5 全 PASS + ROI delta >= +5pt**
**限定GO = check 1,3,4,5 全 PASS + ROI delta in [+3, +5)**
**NO-GO = それ以外**

## 3. GO の場合の race_auto_notify.py 変更 (1-5 行のみ)

```python
# race_auto_notify.py の先頭付近の flag を変更する
STRATEGY_B1_PAPER_ONLY = False   # B-1 GO の場合
STRATEGY_B2_PAPER_ONLY = False   # B-2 GO の場合
STRATEGY_C1_PAPER_ONLY = False   # C-1 GO の場合
STRATEGY_C2_PAPER_ONLY = False   # C-2 GO の場合
```

★ C3/C4 は既に production active (PAPER_ONLY flag なし) ★
★ actual (戦略⑦案C) は常に active ★

変更後は必ず構文チェック:
```
python -c "import py_compile; py_compile.compile('tools/race_auto_notify.py', doraise=True)"
```

## 4. NO-GO の場合

paper 継続。7/15 に再判定。

7/15 再判定コマンド:
```
python tools/c3c4_adoption_test_v2.py --date-from 20260524 --date-to 20260714
```

## 5. 限定Go の場合

4 週末蓄積後 (6/15 → 7/13) に再判定:
```
python tools/c3c4_adoption_test_v2.py --date-from 20260524 --date-to 20260713
```

## 6. 次 session での判定 support

6/17 結果に基づき Session 起動 →
「6/17 採用判定を実施したい。`python tools/c3c4_adoption_test_v2.py --date-from 20260524 --date-to 20260616` の出力を貼り付けるので next step を提案して」
→ Claude が verdict table を読んで採用/変更手順を提案。

## 7. 注意事項

- V15 .pkl.gz / predict_core / daily_predict / app.py logic は変更禁止
- race_auto_notify.py の変更は PAPER_ONLY flag の True→False のみ (他の logic 変更禁止)
- 撤退ライン (-50,000 円) を超えた場合は採用判定の前に投資停止
- bootstrap seed=42, n_iter=1000 がデフォルト。再現性が必要な場合は `--seed 42` を明示

## 8. スクリプト詳細

ファイル: `tools/c3c4_adoption_test_v2.py`
テスト: `tests/test_c3c4_adoption_test_v2.py` (9 tests PASS)

データソース: `data/race_notify_log_v2_summary/*.json`
- aggregator (`tools/race_notify_log_v2_aggregator.py`) が daily 20:30 に生成
- `strategy_stats` キーに 8 strategy の N/hits/inv/pay/roi_pct/pnl が格納される
- 5/24 以前はデータなし → "No paper data yet." と出力して終了
