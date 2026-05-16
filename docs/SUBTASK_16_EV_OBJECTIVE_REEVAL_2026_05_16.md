# Sub-task 16: EV 目的変数 re-eval (★ 部分失敗 ★)

**実施日**: 2026-05-16 evening
**status**: ★ **partial fail** ★ — agent が完了報告したが 結果 file 0 件、 script のみ残存

## 0. 結論

★ **未完** — script (`tools/subtask16/eval_ev_objective.py`、 223 行) は完成、 ただし 実行結果 (data/subtask16_ev_eval.json) が出力されず。 V15 ensemble GPU 再学習で timeout / error 推定。 5/24+ 別 session で 完走必要 ★

## 1. 想定 作業 (script 内 docstring より)

LGB single (GPU 可) で 3 目的変数 WF 6-fold:
- top3 binary (V15 current)
- win binary
- EV weighted (sample_weight = trio_payout)

paper ROI は random baseline 比較で 「真の advantage」 算出。
出力先: `data/subtask16_ev_eval.json`

## 2. ★ honest 失敗 状況 ★

- agent 起動: ✅ (background)
- agent 実行時間: ~13.6 分 (timeout 推定)
- 結果 file: ❌ 0 件
- script ファイル: ✅ tools/subtask16/eval_ev_objective.py (223 行) のみ残存
- 報告: 「Still running. Wait for notification.」 と confused output

## 3. 推定 失敗 原因

- V15 ensemble (LGB+XGB+FT+IR) GPU 再学習 = 数時間規模
- agent timeout (~15 min) で 完走前に halt
- Sub-task 15 (calibration) は GPU 軽量 LGB+XGB only で完走、 Sub-task 16 も同 architecture 想定だが時間切れ

## 4. 5/24+ 再実行 plan

```powershell
# 5/24+ 別 session で 直接実行 (timeout 制約なし)
python tools/subtask16/eval_ev_objective.py
# 出力: data/subtask16_ev_eval.json
```

## 5. 残課題 (★ TODO ★)

- [ ] script 直接実行で 完走確認
- [ ] 3 目的変数 WF 6-fold AUC + paper ROI 計測
- [ ] 過去 ROI 396.2% (3 月 phase 2) の真贋判定 (leak / honest)
- [ ] V15.2 候補 features に EV 重み付け 加味するか判定

## 6. V15 production 不変保証 ✅

- `predict_core.py` / `daily_predict.py` / `race_auto_notify.py` / `app.py` 不変
- `keiba_model_v15_central*.pkl.gz` 不変
- 既存 schtasks 不変
- V15 retrain なし (script は paper eval only)
- 5/17 G1 day 影響 0%

## 7. 関連

- Sub-task 15 (calibration WF 再評価) は完走、 4 method 比較完了
- Sub-task 16 と 15 は GPU 利用想定で並走させたが、 16 のみ timeout
