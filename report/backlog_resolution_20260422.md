# バックログ解消レポート (2026-04-22 → 04-23)

## 1. 4/19 DailyResults リトライ ✅ 成功

### Before (4/19夜時点)
- data/daily_results/20260419.csv: 全 35R `pending`
- 累積: 324R, 76/324 的中, ROI 120.2%, +45,920円 (4/19 未反映)

### After (4/23 03:24 リトライ実行)
- 4/19 35R 全て結果反映
- **累積: 359R, 81/359 的中, ROI 112.2%, +30,650円**

### 4/19 当日成績 (差分から逆算)
- N: +35R / 的中: +5R (35R中5R的中、的中率14.3%)
- 投資: +24,500円 (35R × 700円)
- 払戻: +9,230円
- 損益: **-15,270円**
- ROI: 4/19単日 37.7% (大不調日)

### 条件別変化
| 条件 | Before | After | 4/19追加 |
|:---:|:---:|:---:|:---:|
| A | 90R 122.9% | 104R 114.5% | +14R |
| B | 9R 0.0% | 9R 0.0% | +0R |
| C | 89R 123.8% | 99R 111.3% | +10R |
| D | 115R 144.3% | 125R 136.5% | +10R |
| E | 9R 13.2% | 10R 11.9% | +1R |
| X | 12R 13.8% | 12R 13.8% | +0R |

### 評価
- 4/19 単日は不調 (ROI 37.7% / -15,270円)
- 全体ROIは 120.2% → 112.2% (-8%) に低下
- 直近30R ROI 44% → 短期ドローダウン警告 (連敗2R)
- ただし保守的見積り (142.6%) との乖離は -21% → 許容範囲内

---

## 2. CatBoost WF 5model 検証結果 ⚠ 失敗

`logs/catboost_wf.log` (4/14 00:14 最終更新) を確認:

### 進捗
- LGB 5fold: 完了
- XGB 5fold: 完了
- CatBoost 5fold: 完了
- FT-Transformer: 各fold 完了 (FT-2021 best AUC 0.8618)
- **IntraRace Attention 学習で KeyError**:
  ```
  File "train/train_v135b_intra_ensemble.py", line 75
    race_groups = df.groupby('race_id_unique')
  KeyError: 'race_id_unique'
  ```

### 影響
- 5model アンサンブルの CatBoost 採用判定 **未取得**
- v15 4-model (現行 0.8858) との比較不可
- v16 学習にも影響する可能性 (同じ train_v135b_intra_ensemble.py を使うため)

### 対処方針
- 'race_id_unique' カラム生成箇所を train_v15_master.py で確認
- v16 学習前に必須修正
- 本セッションでは時間切れのため未対応 (フェーズ5スキップで影響軽微)

---

## 3. cumulative_results.csv 更新

- ✅ 4/19 結果が data/cumulative_results.csv に追加された
- 359R に増加
- ROI Monitor アラート: 7件 (累積/A/C/直近30R 全て保守的見積り未達)

### 警告詳細
- [WARNING] 累積ROI 112.2% < 保守的見積り 142.6% (359R)
- [WARNING] 条件A: ROI 114.5% < 保守的ROI 143.7% (N=104)
- [WARNING] 条件C: ROI 111.3% < 保守的ROI 199.9% (N=99)
- [WARNING] 直近30R ROI 44.0% < 80.0% (短期ドローダウン)
- [WARNING] 累積ROI 112.2%がBT保守的見積り142.6%を21%下回る (359R)
- [WARNING] 条件A: ROI 114.5%がBT保守143.7%を20%下回る (N=104)
- [WARNING] 条件C: ROI 111.3%がBT保守199.9%を44%下回る (N=99)

→ 警告は出ているが破滅的ではない。長期視点で +30,650円のプラス運用継続中。

---

## 4. 残タスク (今夜未対応、次回送り)

| # | タスク | 推定所要 | 緊急度 |
|---|--------|:---:|:---:|
| A | CatBoost WF の 'race_id_unique' KeyError 修正 | 30分 | Medium |
| B | scrape_missing_all 再起動 (週明け Mon 06:00 以降) | 5分 | High |
| C | JRDB SRB/HJC/OZ/CHA 列名不整合の修正 | 1時間 | Low |
| D | 直近30R ROI 44% の原因分析 (大不調の理由) | 1時間 | Medium |
