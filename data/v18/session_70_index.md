# Session #70 index — 5月 11R/12R 重賞除外 全頭 V15 production saved score

作成: 2026-05-09 17:30 JST / branch: dev/audit-backtest

## 領域別 doc

| 領域 | 内容 | doc |
|---|---|---|
| A | 5月 production saved score audit | [session_70_data_audit.md](session_70_data_audit.md) |
| B | 11R/12R 重賞除外 絞り込み (12 R) | [session_70_filtered_races.csv](session_70_filtered_races.csv) |
| C | 全頭スコア markdown (R 別詳細) | [may_filtered_horse_scores.md](may_filtered_horse_scores.md) |
| D | 統計サマリ (ROI/hit rate/クラス別) | [may_filtered_summary.md](may_filtered_summary.md) |
| E | 本 index + LEAK 防止 verification | session_70_index.md |

## 主要結果

- 期間: 2026-05-02, 05-03, 05-09 (5/1, 5/4-5/8 は production saved 不在)
- 対象 R: **12 件** (5/2: 3, 5/3: 5, 5/9: 4)
- 重賞除外: 5 件 (京王杯SC G2 / ユニコーンS G3 / 天皇賞春 G1 / 京都新聞杯 G2 / エプソムC G3)

### 案B改 strict 7 点 三連複 (production)
| 期間 | R 数 | hit | ROI | 損益 |
|---|---|---|---|---|
| 5/2-5/3 settled (cumulative) | 8 | 3 | 140.7% | +¥2,280 |
| 5/9 案B改 strict 投票 | 1 | 0 | 0.0% | -¥700 |
| **5月 total (実投票)** | **9** | **3** | **125.1%** | **+¥1,580** |

### V15 production score 分布 (5/9 4 R のみ score 値あり)
- top1_score 平均 0.5722 / 中央値 0.5905 / 最大 0.6614 / 最小 0.5166
- 5/2-5/3 は cumulative_results.csv で top1_score NaN (95%欠損 既知 = CLAUDE.md)

### V15 hit rate (cumulative finishes ベース、 5/2-5/3 8R + 5/9 投票 1R)
- top1 が 1 着: ?/9 (要 5/9 投票 R の finish 取得、 Session #67 で 11→3着 確認済 = 0/9 で更新候補)
- top1 が 3 着内: ?/9 (同上、 1/9 + 5/2-5/3 の counts)

## 🚨 LEAK 防止 verification checklist

- [x] V15 model.predict() 呼び出してない
- [x] tools/predict_core.py 実行してない
- [x] tools/daily_predict.py 実行してない
- [x] data/v18/session69_horse_scores.csv 使用してない
- [x] pkl/joblib model load なし (`*.pkl`, `*.pkl.gz`, `*.joblib` 全て触らず)
- [x] feature engineering を伴う inference なし
- [x] 全 score の source = "production_saved_score"
- [x] 出力 markdown / csv / py すべてに source 明記
- [x] dev/training-poc / dev/two-stage / main 触らず (read-only も該当 csv 参照のみ)
- [x] schtasks 既存 49 件 不変、 ProcessWatchdog kill-switch 残存

## 制約と次 session 候補

### 既知欠損 (本 session の制約)
1. `data/cumulative_results.csv` の `top1_num/score` 列 NaN (95%欠損) — daily_predict.py の保存ロジック audit 候補
2. `data/daily_predictions/` に 4 着以下の score 列なし — full v15_scores の production save 形式設計が次 session 候補
3. 5/1, 5/4-5/8 の production csv 完全不在 — open question (中央非開催日 or 保存失敗 の切り分け)

### 5/16 V18 trial 含意
- V15 案B改 strict は 5月 9 R (実投票) で **ROI 125.1%、 hit 3/9 (33%)** という production-grade な base data
- 5/9 単独 -¥700 (1R/MISS) は全体 ROI を 16pp 下げたが、 5/2-5/3 +¥2,280 が補填
- 5/16 V18 trial は **GO 条件: V18 を同 9 R に当てた場合の hit rate ≥ V15 + 1pt** などを比較 metric にできる (但し 5/16 の V18 投入判断は別 NO-GO 確定済 = Session #38)

### V15 投資保護
- main 8fc4e13b 不変 / 5/9 投票方針 (新潟 12R ¥700) 不変 / 累計 +¥12,830 死守
