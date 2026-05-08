# Session #54 サマリ: NAR V5 改善 (5/12 paper 候補)

**日付**: 2026-05-09 (Session #54)
**branch**: dev/nar-v5 (main 不変、 V15 中央 投資保護)
**目的**: NAR V4 (AUC 0.8145) → V5 改善、 5/12 paper trade 投入候補

---

## 0. 5 領域 完了

| 領域 | 内容 | 出力 |
|------|------|------|
| A | V4 audit (22 features、 0.8188、 49K rows、 Pattern B) | data/v18/session_54_nar_v4_audit.md |
| B | V5 features 候補 15 件 (expanding 7 + 既取得 4 + NAR 独自 4) | data/v18/session_54_nar_features_candidates.md |
| C | V5 学習 (37 features、 AUC 0.8187、 LEAK 1 件 検出) | data/v18/session_54_nar_v5_training.md + tools/train_nar_v5.py + data/nar/models/keiba_model_nar_v5.pkl |
| D | 5/12 投入判断 = NO-GO (V5 同等、 V4 維持推奨) | data/v18/session_54_nar_v5_evaluation.md |
| E | 統合 + push + Discord | 本 doc |

---

## 1. 主結果

| 指標 | V4 | V5 | delta | 判定 |
|------|----|----|------|------|
| n_features | 22 | 37 | +15 | -- |
| Ensemble AUC | **0.8188** | **0.8187** | **-0.0001** | × |
| 学習 rows | 49,213 | 53,407 | +4,194 | data 増分 |
| 学習 races | 4,821 | 4,827 | +6 | -- |

**5/12 paper trade**: ★ V4 維持 NO-GO ★ (V5 archive 扱い)

---

## 2. 重要 発見 (★)

### 2-1. last3f LEAK 検出 + 修正

V5 初版 で AUC 0.867 (+0.048!) を記録。 last3f は 走破後 計測される post-race 指標 → LEAK。
**修正**: prev_last3f (horse 単位 shift(1)) に変更、 修正後 AUC 0.8187。

→ NAR V5 model file は leak-free 検証済。

### 2-2. NAR は odds_log 圧倒的 支配 (V4 飽和)

- odds_log importance 70%+ (V5 全 importance の)
- pop_rank 15%
- 残り 35 features 合計が 15%

→ NAR は **市場効率 高** (odds + pop が ほぼ 全 信号)、 features 追加 効果薄

### 2-3. 中央 V15 vs NAR V5 飽和パターン

| model | features | AUC | 飽和タイプ |
|-------|---------|-----|---------|
| 中央 V15 | 145 | 0.8788 | features **多様性** で 飽和 |
| NAR V5 | 37 | 0.8187 | **odds_log 単独** で 飽和 |

→ NAR の 真の改善は **異質 source** (NAR JV 不対応で困難) または **FT/Attention ensemble**

---

## 3. 5/12 paper trade 戦略

```
5/12 (火) NAR paper 開始:
- 使用 model: V4 (data/nar/models/keiba_model_nar_v4.pkl) ← NO-GO で V5 不採用
- 投入金: 0 円 (paper のみ)
- 場: 大井 / 船橋 / 川崎 / 浦和 全 R
- 期間: 5/12-5/26 (2 週間 200+ races)
- 評価: 条件別 ROI、 AUC live、 hit rate
```

V5 model は 並行 paper で A/B 観察 (option):
- 5/26 評価で V5 archive 確定

---

## 4. NAR 中期 ロードマップ

```
2026-05-09  V5 学習 (本 Session、 NO-GO)
2026-05-12  V4 paper 開始 (2 週間)
2026-05-26  paper 評価
2026-05-30  V20 中央 投入後、 V6 候補 着手
2026-06-30  V6 学習完了
2026-07-15  V6 投入判断 (目標 AUC 0.825-0.835)
```

V6 改善 path:
- FT-Transformer 追加 (+0.005-0.010)
- IntraRace Attention 追加 (+0.005-0.010)
- NAR 独自 sib_*_exp (+0.002-0.005)
- NAR 独自 速度指数 (+0.001-0.003)
- 4-model grid ensemble (V13.5b 中央 と同様)

→ V6 期待 AUC 0.825-0.835

---

## 5. branch 状態

- `dev/nar-v5`: 5 commits (A, B cherry-pick, C cherry-pick, D cherry-pick, E)
- `main` 不変
- `dev/training-poc` (Session #52 並行): 干渉なし、 Session #52 が管理
- `dev/sprint6-kka` (Session #53 並行): 干渉なし、 Session #53 が管理
- `predict_core / daily_predict / app.py`: 不変
- 中央 V15 model file: 不変
- 5/9 朝 V15 動作: 不変

→ 中央 V15 投資保護: ✅

---

## 6. 5 commits 履歴 (dev/nar-v5)

1. `Session #54 A: NAR V4 audit` (aa911e46)
2. `Session #54 B: NAR V5 features 拡張候補 15 件 確定` (538143e4)
3. `Session #54 C: NAR V5 学習 (AUC 0.8187、 V4 並み、 LEAK 1 件 検出)` (6a65e221)
4. `Session #54 D: V5 評価 + 5/12 paper trade 投入判断 = NO-GO` (5da4a29f)
5. `Session #54 E: doc 統合 + summary` (本 commit)

---

## 7. 結論

✅ V5 学習 完了 (AUC 0.8187)
✅ V4 vs V5 比較 完了 (delta -0.0001、 改善なし)
✅ LEAK 1 件 (last3f) 検出 + 修正
✅ V4 飽和 確認 (odds_log 70%+ 支配)
✅ 5/12 投入判断 = NO-GO (V4 維持)

**主結論**:
- V5 は **AUC 改善 達成不可** (audit 期待 +0.005-0.015 vs 実 -0.0001)
- 5/12 paper では **V4 維持** 推奨
- V5 model file 保存済 (archive 扱い、 LEAK 検証 PASS)
- V6 候補 で 真の改善 追求 (FT + Attention + sib + 速度指数、 5/30 以降 着手、 7/15 投入判断)
- **中央 V15 投資保護**: 完全保持、 NAR は完全独立 system
