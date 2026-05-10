# Phase 15 — V20 ensemble weight 最適化 plan

**date**: 2026-05-10
**target**: LGB+XGB+FT+IR の 4-model 重み最適化

## 現状 (Phase 15 quick)

| weight 戦略 | LGB | XGB | FT | IR | val AUC |
|-------------|-----|-----|-----|-----|---------|
| simple 2-model | 0.55 | 0.45 | — | — | 0.8678 |
| LGB+XGB+FT (固定) | 0.50 | 0.40 | 0.10 | — | (FT 完了後追記) |

## Phase 15 完全版 (5/24+ 実装予定)

```python
# Optuna ensemble weight tuning
study = optuna.create_study(direction='maximize')

def objective(trial):
    w_lgb = trial.suggest_float('w_lgb', 0.1, 0.5)
    w_xgb = trial.suggest_float('w_xgb', 0.1, 0.5)
    w_ft = trial.suggest_float('w_ft', 0.0, 0.3)
    w_ir = trial.suggest_float('w_ir', 0.1, 0.5)
    s = w_lgb + w_xgb + w_ft + w_ir
    pred = (w_lgb*lgb + w_xgb*xgb + w_ft*ft + w_ir*ir) / s
    return roc_auc_score(y_val, pred)

study.optimize(objective, n_trials=100)
```

## v13.5b 参考重み (CLAUDE.md より)

| 年 | LGB | XGB | FT | IR |
|----|-----|-----|-----|-----|
| 典型 | 0.25 | 0.27 | 0.13 | 0.35 |
| 2024 best | 0.22 | 0.28 | 0.15 | 0.35 |
| 2025 best | 0.25 | 0.30 | 0.10 | 0.35 |

→ IntraRace Attention が最大寄与 (0.35)、 これは V20 でも踏襲予定。

## V20 で更に改善するため

1. **fold ごとに重み再最適化** (年 × 条件 ×)
2. **stacking** (level-2 model で重み学習)
3. **diversity 確保** — LGB / XGB の hyperparameter を意図的に分散

## 当面の戦略

Phase 15 quick (本 commit): LGB+XGB+FT 固定重み 0.50/0.40/0.10
Phase 15 full (5/24+ Phase 11/12/13 実 data 完備後): Optuna 100 trials
