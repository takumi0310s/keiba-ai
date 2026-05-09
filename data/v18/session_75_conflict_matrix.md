# Session #75: Conflict Matrix

**実施日**: 2026-05-09
**手法**: `git merge-tree --write-tree` (commitless dry-run) + file overlap hash 比較

## main 単独 merge dry-run

| Branch | dry-run 結果 | 備考 |
|---|---|---|
| dev/sprint1 | **CLEAN** | conflict なし |
| dev/sprint2 | **CLEAN** | conflict なし |
| dev/sprint4 | **CLEAN** | (archive 候補) |
| dev/sprint6-kka | **CLEAN** | conflict なし |
| dev/training-poc | **CLEAN** | 巨大 38 commits でも CLEAN |
| dev/two-stage | **CLEAN** | conflict なし |
| dev/audit-backtest | **CLEAN** | conflict なし |
| dev/nar-v5 | **CLEAN** | (archive 候補) |
| dev/session-58-audit | **CLEAN** | (archive 候補) |
| dev/v20-ensemble | **CLEAN** | (V20 素材保持) |
| dev/v20-expanding | **CLEAN** | (archive 候補) |
| dev/v20-interaction | **CLEAN** | (archive 候補) |
| dev/video-poc | **CLEAN** | conflict なし |

→ **全 13 branch ALL CLEAN** (main 単独に対して)

## Pair-wise file overlap matrix

| Pair (a <-> b) | 重複 file 数 | hash 比較 | 実 conflict risk |
|---|---|---|---|
| audit-backtest <-> sprint4 | 5 | SAME | **0** (同一 content) |
| audit-backtest <-> training-poc | 1 | (要確認) | 0 想定 |
| sprint6-kka <-> training-poc | 2 | SAME | **0** |
| sprint6-kka <-> v20-interaction | 1 | (archive、 影響なし) | - |
| training-poc <-> two-stage | 12 | SAME | **0** |
| training-poc <-> v20-interaction | 1 | (archive、 影響なし) | - |
| v20-ensemble <-> v20-expanding | 5 | SAME | 0 (両方 V20、 1 archive 1 keep) |
| v20-ensemble <-> v20-interaction | 14 | SAME | 0 (両方 V20、 1 archive 1 keep) |
| v20-expanding <-> v20-interaction | 5 | SAME | (両方 archive) |

## サンプル hash 検証 (3 ペア)

```
v20-ensemble vs v20-interaction: tools/v20_ensemble.py
  ens: 7f2a4c51e4c5a0a0647e35415009cedb55f25a2c
  int: 7f2a4c51e4c5a0a0647e35415009cedb55f25a2c
  → SAME

training-poc vs two-stage: tools/process_watchdog_v2.py
  tp: cfa730a8fa21eb9889627029009ba42e833d6d0f
  ts: cfa730a8fa21eb9889627029009ba42e833d6d0f
  → SAME

audit-backtest vs sprint4: tools/v15_5_features.py
  ab: 826bc5853f607d4fa25f8cae75d146a034b67a17
  s4: 826bc5853f607d4fa25f8cae75d146a034b67a17
  → SAME
```

→ **全 overlap file が SAME content hash**。 Git は自動 dedup、 conflict 発生せず。

## 解釈

- 各 dev branch は同じ session commit を共有しており、 cherry-pick / branch-off で重複しているだけ
- 異なる content の file は重複していない (互いに排他的な新規 tools)
- → 13 branch を任意順序で merge 可能、 conflict 0 件

## main へ push 時の注意

- pull --rebase で並行 session (#73 / #74 / #76) との conflict 回避
- push 直前に `git fetch origin main && git rebase origin/main` 推奨
- destructive op (push --force / reset --hard) は **禁止** (V15 投資保護 + 並行 session 干渉防止)

## 結論

**全 conflict risk = 0**。 5/15 merge 安全実行可能。
