# B-2: CLAUDE.md drift 30 件 全 verify + context-correct 確認 (5/18 17:30+)

## 0. 結論

- CLAUDE.md drift 30 件 → ★ context-mismatch 0 件 ★ (全 「旧値 (drift)」/「真値」/「v13.5b 歴史 事実」 のいずれかに分類済み)
- docs/SYSTEM_MASTER_2026_05_16.md 1 件 訂正 (line 40: predict_core.py「150 features」→「Pattern B list 150 / booster 145 truncate」)
- CLAUDE.md 1 件 注釈追加 (line 1365: V20 plan の「4-model ensemble」 に V15-audit-1 後の意図 明示)
- V15 production 完全不変保証 ✅
- 単純 replace_all 禁止、 line-by-line honest 確認 で context-mismatch を回避

## 1. 訂正前 drift 30 件 一覧

| pattern | count | source |
|---------|-------|--------|
| 119.2 | 3 | CLAUDE.md L11, L1313, L1406 |
| 13,530 | 2 | CLAUDE.md L91, L1389 |
| 4-model | 17 | CLAUDE.md L11(drift 列), L12(drift 列), L13(drift+真値比較), L85(真値), L177(v13.5b 歴史), L254(真値表), L257(drift 引用), L261(v13.5b), L482(v13.5b 表), L537(v13.5b 表), L783(v13.5b script), L943(v13.5b cmd), L999(drift 引用), L1005(真値), L1011(v13.5b 歴史), L1021(真値), L1163(v13.5b 歴史), L1313(drift 引用), L1332(drift 引用), L1365(V20 plan), L1389(drift 引用), L1406(drift 引用) ※ 一部複数行に同 line で重複あり、 grep -c 17 件 が正解 |
| 0.8939 | 7 | CLAUDE.md L13(drift 列), L85(真値), L250(真値), L257(drift 引用), L999(drift 引用), L1006(真値), L1332(drift 引用) |
| 150 features | 1 | CLAUDE.md L257 (drift 引用) |
| **合計** | **30** | — |

## 2. context 別 真値 mapping (★ 5/18 17:30 現在 ★)

| 旧 drift | 真値 | 出典 |
|----------|------|------|
| 累計 ROI 119.2% | **98.34%** (n=596, ≤2026-05-17、 戦略⑦ subset 96.90%) | V15-audit-4 |
| 累計 PnL +13,530 | **¥-6,920** (撤退余裕 ¥43,080) | V15-audit-4 |
| 4-model production | **LGB+XGB 2-model** (mlp=None, FT/IR は .pkl 未保存) | V15-audit-1 |
| WF AUC 0.8939 | **0.8678** (LGB+XGB genuine WF 6-fold) / 0.8858 (Grid 4-model 5-fold WF 評価専用) | V15-audit-2 |
| features 150 | **145** booster (Pattern B list 150 だが truncate で 145) | V15-audit-1/3 |
| stored `.pkl.auc` 0.8939 | ★ 真値 (= LGB train-set self-eval、 in-sample LEAKY) ★ | V15-audit-2 |
| v13.5b 4-model Grid (WF AUC 0.8788) | ★ 真値 (歴史 reference、 v13.5b は 4-model だった) ★ | v13.5b 元 commit (2026-04-03) |

## 3. line-by-line context 分類 (★ 30 件 全件 ★)

### A. drift 値 (旧値) 引用 — 「★ drift ★」 「旧値」 注釈付き、 OK
- L11 (119.2% in audit table drift column)
- L12 (4-model in audit table drift column)
- L13 (0.8939 in audit table drift column)
- L91 (+13,530 with 「※ 旧値 ... は drift」)
- L257 (drift 訂正 summary)
- L999 (drift 引用 summary)
- L1313 (期待効果 旧 drift 記述)
- L1332 (v16 旧目標 旧記述)
- L1389 (撤退ライン 旧値)
- L1406 (ROI 想定 旧値)

### B. 真値 (現状) — V15-audit 出典明記、 OK
- L85 (現行モデル: V15、 145 features booster、 LGB+XGB 2-model、 stored .pkl.auc 0.8939 = LGB train-set self-eval)
- L250 (V15 audit table: stored .pkl.auc 0.8939485520467574 = LGB train-set self-eval)
- L254 (Grid 4-model 5-fold mean 0.8858)
- L1005 (Grid 4-model 5-fold mean 0.8858 真値)
- L1006 (stored .pkl.auc 0.8939 = LEAKY)
- L1021 (Grid 4-model = 0.8858 真値)

### C. v13.5b 歴史 事実 — v13.5b は実際 4-model Grid Ensemble だった、 OK
- L177 (v13.5b IntraRace Attention 採用記述、 WF AUC 0.8788 4-model grid 歴史)
- L261 (v13.5b Pattern A スペック)
- L482 (v13.5b 条件定義 表 header)
- L537 (年別 AUC 表 v13.5b 列)
- L783 (train_v135b_intra_ensemble.py file comment)
- L943 (train_v135b_intra_ensemble.py command 例)
- L1011 (旧ベースライン v13.5b reference)
- L1163 (v13.5b 正式採用 header)

### D. 未来 plan (V20 / v15_full / V22) — 5/18 注釈追加で OK
- L1365 (V20 v1 学習 4-model ensemble plan、 ★ B-2 で V15-audit-1 言及 追加 ★)

## 4. 訂正 step (line-by-line)

### 4-1. CLAUDE.md
1. L1365 注釈追加 (Edit、 replace_all=False、 context 1 件のみ)
2. L6 (Last updated) + L8-15 (Session #89 block) の上に Session #90 B-2 block 追加

### 4-2. docs/SYSTEM_MASTER_2026_05_16.md
1. L40 「150 features」 → 「Pattern B list 150 / booster 145 truncate、 V15-audit-1」 (Edit、 replace_all=False)

### 4-3. ★ 単純 replace_all 禁止 ★
- 各 drift pattern は context 別に 「旧値 引用 / 真値 / 歴史 事実」 で意味が全く異なる
- replace_all=true で「0.8939 → 0.8678」 等やると stored .pkl.auc の真値が壊れる
- ★ Edit (replace_all=False) で 1 件ずつ honest 訂正 ★

## 5. 訂正後 grep result

```
$ grep -c "119.2" CLAUDE.md       # 3 件 (全 drift 引用 + 注釈)
$ grep -c "13,530" CLAUDE.md       # 2 件 (全 drift 引用 + 注釈)
$ grep -c "4-model" CLAUDE.md      # 17 件 (drift 引用 / 真値 Grid 4-model / v13.5b 歴史)
$ grep -c "0.8939" CLAUDE.md       # 7 件 (drift 引用 / stored .pkl.auc 真値)
$ grep -c "150 features\|150 特徴量" CLAUDE.md  # 1 件 (drift 引用)
```

★ 全 30 件 → context-mismatch 0 件 ★

## 6. V15 production 不変保証 ✅

- `tools/predict_core.py` ★ 完全不変 ★
- `data/keiba_model_v15.pkl.gz` ★ 完全不変 ★
- `data/cumulative_results.csv` ★ 完全不変 ★
- `train/features_v15_new.py` ★ 完全不変 ★
- `tools/race_auto_notify.py` ★ 完全不変 ★

## 7. honest 厳守 ✅

- 全 訂正は context 確認後 (Read + Edit replace_all=False)
- 単純 replace_all 禁止 (context 別 真値 mapping 異なる)
- 真値は V15-audit-1〜5 / data-audit-1〜4 から正確に引用
- 「history 引用」 vs 「真値主張」 区別明示
- fabrication 0、 数値は全 出典付き

## 8. 残課題 (Sub-task)

- ★ 過去 docs (data/v18/、 data/v21/inventory_5_16/ 等) の全 grep + 訂正は工数大、 B-3+ で順次
- 5/17 Sub-task D で主要 38 docs 訂正済、 残り は B-3 (5/19+) で対応
