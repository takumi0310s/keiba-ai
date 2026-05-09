# Session #75: Archive 計画 + V20 素材保持

**実施日**: 2026-05-09
**実行**: 5/15 merge 完了後 即実行

## Archive 対象 (5 branch)

### NO-GO 確定 archive

| Branch | NO-GO 理由 | session 由来 | tag 名 |
|---|---|---|---|
| **dev/sprint4** | V15.5 PoC AUC 0.8685 (V15 0.8688、 -0.0003) → 既存 145 features 既に飽和 | Session #49 | `archive/dev-sprint4-v15.5-no-go` |
| **dev/nar-v5** | NAR V5 audit で評価未達 | Session #54 | `archive/dev-nar-v5-no-go` |
| **dev/v20-expanding** | V20 expanding 化 PoC delta -0.0000 (sib_w5 が特殊例 確定) | Session #55 | `archive/dev-v20-expanding-no-go` |
| **dev/v20-interaction** | V20 interaction PoC -2bp〜+1.8bp、 V15 既に LGB 内部捕捉 | Session #57 | `archive/dev-v20-interaction-no-go` |
| **dev/session-58-audit** | Discord 重複 audit 一時 doc (1 file)、 役目完了 | Session #58 | `archive/dev-session-58-audit` |

### Archive 手順

```bash
# 履歴保持 tag 化
git tag archive/dev-sprint4-v15.5-no-go dev/sprint4
git tag archive/dev-nar-v5-no-go dev/nar-v5
git tag archive/dev-v20-expanding-no-go dev/v20-expanding
git tag archive/dev-v20-interaction-no-go dev/v20-interaction
git tag archive/dev-session-58-audit dev/session-58-audit

# tag を origin に push (履歴保持)
git push origin archive/dev-sprint4-v15.5-no-go
git push origin archive/dev-nar-v5-no-go
git push origin archive/dev-v20-expanding-no-go
git push origin archive/dev-v20-interaction-no-go
git push origin archive/dev-session-58-audit

# branch 削除 (tag で復元可能)
git branch -D dev/sprint4
git branch -D dev/nar-v5
git branch -D dev/v20-expanding
git branch -D dev/v20-interaction
git branch -D dev/session-58-audit

# remote branch 削除
git push origin --delete dev/sprint4
git push origin --delete dev/nar-v5
git push origin --delete dev/v20-expanding
git push origin --delete dev/v20-interaction
git push origin --delete dev/session-58-audit
```

### 復元方法 (必要時のみ)

```bash
git checkout -b dev/sprint4 archive/dev-sprint4-v15.5-no-go
```

## V20 構築素材 保持 (1 branch)

| Branch | 役割 | 保持期限 |
|---|---|---|
| **dev/v20-ensemble** | V20 4-model ensemble (LGB+XGB+FT+IR)、 V20 本命 | **6/8 V20 投入判定まで** (約 30 日) |

### 保持理由

- CLAUDE.md Phase 3 後半 (6/9-6/30) で V20 学習素材として活用予定
- session #56 で V20 4-model ensemble を試作、 FT-Transformer / IntraRace Attention の重み学習結果を保持
- v20-interaction は archive するが、 ensemble の素材は別 branch (dev/v20-ensemble) で保持

### 保持期限後の判断 (6/8 以降)

- **GO**: V20 production 投入 → ensemble 素材を main に merge
- **NO-GO**: V20 投入見送り → archive/dev-v20-ensemble-no-go として tag 化

## Cleanup 後 状態

| 項目 | 数 |
|---|---|
| **merge 済み** main commits | +95 (7 branch 合算) |
| **archive tag** (履歴保持) | 5 |
| **active dev branch** (V20 素材保持) | 1 (dev/v20-ensemble) |
| **削除予定 branch** | 5 (archive 後) |
| **総 dev branch (5/15 後)** | 1 (v20-ensemble のみ) |

## 実行スケジュール

| 日時 | 作業 |
|---|---|
| 5/15 09:00-14:00 | merge 7 branch (Phase 1-4) |
| 5/15 14:00-15:00 | archive 5 branch (tag + delete) |
| 5/15 15:00 | Discord 通知 + main snapshot tag (`session-75-merge-complete`) |
| 6/8 | dev/v20-ensemble GO/no-go 判定 |

## 絶対遵守

- ★ destructive op は archive/dev-* tag 化 **後** にのみ実行 ★
- ★ V15 production logic 完全不変 (predict_core / daily_predict / app.py 変更なし確認済) ★
- ★ 並行 session (#73 / #74 / #76) と干渉する push --force 禁止 ★
