# 6/17 (Wed) P0-5 採用判定 checklist

> 作成: 2026-05-17 (N3 task)
> 対象: 4 週末 paper shadow eval (5/18-6/16) 完了後の採用判定
> 判定日: 2026-06-17 (Wed)

---

## 1. 採用判定 5 項目 (commit 2646bf9b 確定済)

| # | 項目 | criteria | PASS condition |
|---|------|---------|----------------|
| 1 | AUC 維持 | V15 0.8939 ± 0.001 以内 | `abs(P0-5 AUC - 0.8939) < 0.001` |
| 2 | paper ROI 改善 | V15 production ROI 上回り | `P0-5 ROI > V15 ROI` |
| 3 | LEAK 監査 PASS | T4 leak_audit で 0 件 | `leak_audit exit 0` |
| 4 | LIVE 安定 | 90%+ 成功率 (4 週末 24-32R) | `success_rate >= 0.90` |
| 5 | 統計的有意性 | Welch's t-test、 p<0.05 | `p_value < 0.05` |

---

## 2. 比較対象

### V15 production (baseline)
- 朝 8:00 prediction、 戦略⑦案 C 適用後
- source:
  - `data/cumulative_results.csv` の status=settled
  - `data/daily_predictions/{20260518...20260616}.csv`
  - `data/daily_results/{20260518...20260616}.csv`
- period: 2026-05-18 〜 2026-06-16 (4 週末)

### V15 + P0-5 overlay (challenger)
- -15 min 再計算、 案 B post-hoc
- source: `data/recalc_15min/{20260518...20260616}/{race_id}.json`
  - 各 file: `original_ranking`, `recalc_ranking`, `status`
- period: 同上

---

## 3. data source 一覧

```
production:
  data/cumulative_results.csv          (5/18-6/16 settled rows)
  data/daily_predictions/{date}.csv
  data/daily_results/{date}.csv

P0-5 paper:
  data/recalc_15min/{date}/{race_id}.json
    fields: original_ranking, recalc_ranking, status, ...
```

---

## 4. 統計検定 method

### 4-1. Welch's t-test on R-level ROI
- V15 production R-level ROI samples
- P0-5 paper R-level ROI samples
- `scipy.stats.ttest_ind(equal_var=False)`
- p<0.05 で 「真に異なる」 と判定

### 4-2. paired bootstrap 1000 iter
- V15 と P0-5 を pair で resample
- mean ROI delta の 95% CI 算出

### 4-3. sample size
- 24-32 R (週末 6 R × 4 週末 paper eval)
- 最低 24 R で 検定実施可能、 32 R で 推奨

---

## 5. judgment matrix

| 状況 | 5 項目 PASS 数 | 判定 | next action |
|------|------------|------|------------|
| 全 PASS | 5/5 | ★ GO ★ | 6/18+ production 投入候補 |
| 部分 FAIL | 3-4/5 | ⚠ 蓄積継続 | 7/15 まで蓄積、 7/16 再判定 |
| 多数 FAIL | 0-2/5 | 🔴 NO-GO | 実装見直し or 永久放棄判定 |

---

## 6. GO 後の production 投入 step

1. schtask `Keiba-LiveOrchestrator` の発火対象 R を拡大 (戦略⑦案 C 適用 R 全部)
2. 既存 `race_auto_notify.py` の通知 logic に P0-5 出力 hook 追加
   - ★ 別 sub-task で 設計 ★
   - P0-5 結果を Discord #買い目 に統合
3. 初回 production 投入後 30R で 再 audit (paper vs production 乖離検証)
4. 累計真値 (`data/task_outcomes/baseline_v15.json`) の after 値 update

> ★ 重要 ★: production 投入 = `race_auto_notify.py` を 1 byte でも変更 = 高 risk。
> 6/17 GO 判定後 別 sub-task で 設計レビュー必須。

---

## 7. NO-GO 後の path

- **蓄積継続** (5/18-7/15) で sample N 倍増 → 7/15 再判定
- **data source 見直し** (JV-Link O1 vs TCOV vs netkeiba 直前 比較再評価)
- **永久放棄** → 工数清算 (~10 day)、 v15.2 学習に注力

---

## 8. 6/17 当日の動作

```
06:00 起床
06:30 data/recalc_15min/ 蓄積確認 (24-32 R)
07:00 python tools/p0_5_evaluation.py --start 20260518 --end 20260616
08:00 5 項目評価結果出力 (docs/P0_5_ADOPTION_VERDICT_2026_06_17.md)
09:00 GO/NO-GO 判定
10:00 Discord #アップデート 通知 (判定結果 + 次 action)
```

---

## 9. ★ 統計検定力 不足 risk ★

n=24-32 で Welch's t-test の検出力:

| ROI delta | 標準偏差 | power |
|-----------|---------|-------|
| +5pt | 30pt | ~0.3 (★ 不十分 ★) |
| +10pt | 30pt | ~0.5 |
| +15pt | 30pt | ~0.7 |

→ n=30 では小差検出は困難、 大差 (+10pt+) のみ確実検出。
→ NO-GO 判定が出ても 「差分なし」 とは限らない (★ honest ★)、 蓄積継続推奨。

---

## 10. checklist 実行手順 (6/17 当日)

- [ ] data/recalc_15min/ 配下に 24-32 R 分の json が存在することを確認
- [ ] data/cumulative_results.csv の 5/18-6/16 settled rows 抽出
- [ ] `python tools/p0_5_evaluation.py --start 20260518 --end 20260616` 実行
- [ ] 出力 `docs/P0_5_ADOPTION_VERDICT_2026_06_17.md` を確認
- [ ] 5 項目それぞれ PASS/FAIL 確認
- [ ] judgment matrix で GO/⚠/NO-GO 判定
- [ ] Discord #アップデート に判定結果通知
- [ ] (GO の場合) production 投入 sub-task を起こす
- [ ] (NO-GO の場合) 蓄積継続 or 永久放棄判定 を別 sub-task で実施

---

## 11. 関連 file

- baseline: `data/task_outcomes/baseline_v15.json`
- features: `data/v15_2/features_v152_candidates.txt`
- leak gate: `tools/v15_2_train_gate.py`
- evaluation template: `tools/p0_5_evaluation.py` (本 N3 で作成)
- 戦略⑦案 C: `tools/race_auto_notify.py` (★ 変更禁止 ★)

---

★ honest 厳守、 V15 完全不変、 commit/push 親集中、 6/17 当日 fine-tuning 想定 ★
