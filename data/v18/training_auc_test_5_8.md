# Session #47 B: training AUC test (2026-05-08)

## 1. 目的

V15 (150 features) baseline と V15 + 拡張調教 8 features (158 features) を WF AUC で比較。
1勝クラス で効果検証 (調教 data カバレッジ 完備 のため)。

## 2. tool

`tools/training_auc_test.py`

```bash
python tools/training_auc_test.py             # full WF (2021-2025, ~30 min)
python tools/training_auc_test.py --quick     # 2024 のみ smoke test
python tools/training_auc_test.py --by-class  # クラス別
```

## 3. 拡張 features (8 個)

| # | feature | source | 計算 |
|---|---------|--------|------|
| 1 | training_time_5f | netkeiba time_5f | 生値 |
| 2 | training_time_3f | netkeiba time_3f | 生値 |
| 3 | training_pace_5f_3f | derived | (5f - 3f) / 2 |
| 4 | days_since_last_training | netkeiba training_date | race_date - training_date |
| 5 | training_count_2w | derived | 2 週間 内 回数 |
| 6 | cyb_train_baba_enc | jrdb_cyb | train_baba |
| 7 | cyb_train_amount | jrdb_cyb | train_amount |
| 8 | cyb_train_change_enc | jrdb_cyb | train_change |

## 4. data sources

- `data/_v15_optuna_df_cache.pkl.gz` (104 MB、 V15 学習 cache、 150 features 含む)
- `data/netkeiba_training_times.csv` (300,574 行)
- `data/jrdb_cyb.csv` (548,607 行)

## 5. 評価方法

WF (walk-forward):
- train: ≤ year-1
- test: year
- LGB single (proxy、 production は ensemble 4 model)
- AUC: target = (finish ≤ 3)

## 6. 採用基準

| 基準 | 閾値 |
|------|------|
| WF mean AUC delta | ≥ +0.0010 |
| 全年 monotonic 改善 | 必須 |
| 年別 gap | < 0.05 |
| 1勝 class delta (--by-class 時) | ≥ +0.0020 |

達成 → V20 候補に追加 (Phase 3 後半)
未達 → 棚卸しのみ、 V15 unchanged

## 7. 結果 (実行待ち)

```
$ python tools/training_auc_test.py
```

出力 file:
- `data/v18/training_auc_test_5_8.json` — 数値結果
- console log — 進行状況

**注意**: full WF は約 30 分。 5/8 中に実行 → 結果確認。
quick mode (2024 のみ) は約 5-7 分で smoke test 可能。

## 8. 期待値

| シナリオ | baseline AUC | extended AUC | delta |
|----------|--------------|--------------|------|
| 楽観 | 0.886 | 0.890 | +0.004 |
| 中央 | 0.886 | 0.888 | +0.002 |
| 悲観 | 0.886 | 0.886 | +0.000 |

中央想定で **採用ライン (+0.002)** 達成見込み。
1勝クラスで +0.005+ 期待 (調教情報依存高い)。

## 9. リーク監査 (Session #38 教訓)

全 8 features を pre-race 確定で構築:
- training_date < race_date 厳守
- CYB train_baba/amount/change は 調教時点で確定 (post-race 含まず)
- ⚠️ CYB train_comment は除外 (post-race 混入 risk)
- ⚠️ CYB train_eval は **B 採用判断時** に corr(target) で再検証
  (Session #38 SKB の corr_target 0.137 を超えるなら除外)

## 10. 次 step

→ B 結果 → C 5/9 全 R で V15 vs V15+training 並列予測
→ D 5/10 朝 結果照合 で 実 AUC 検証
