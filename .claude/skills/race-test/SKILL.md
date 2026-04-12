---
name: race-test
description: 全クラステスト手順 — 新モデル/特徴量変更後に全条件×全クラスで予測が通るかを検証する。
---

# race-test — 全クラステスト手順

新モデル投入後・predict_core変更後は必ずこれを実行する。

## 手順

1. **構文チェック**
   ```bash
   python -c "import py_compile; py_compile.compile('app.py', doraise=True)"
   python -c "import py_compile; py_compile.compile('tools/predict_core.py', doraise=True)"
   ```

2. **回帰テスト全PASS**
   ```bash
   python tests/regression_test.py
   ```

3. **全クラステスト** (新馬/未勝利/1勝/2勝/3勝/OP/L/G3/G2/G1)
   ```bash
   python tests/fullclass_test_v15.py
   ```

4. **特徴量数チェック**
   - Pattern A: 145個（v15）
   - Pattern B: 150個（v15）
   - 差は8個（odds_log, horse_weight, weight_change, weight_change_abs, weight_cat, weight_cat_dist, condition_enc, cond_surface）

5. **odds変動特徴量がリアルタイム取得で埋まること**
   - data/odds_base_YYYYMMDD.csv が当日存在することを確認
   - app.pyで「予想する」を1回押し、odds_change_rate ≠ 0 が出るか確認

## 失敗時の対応
- 1個でもFAILなら本番反映禁止
- v15モデル特有のバグはCLAUDE.md「過去の失敗教訓」を参照
