# CHANGELOG 2026-05-16 (5/16 evening session)

## TYB 実装 + honest 評価 + 動画撤回 + 真値確定 path

### 5/16 evening commits

| # | commit | summary |
|---|--------|---------|
| 1 | cea7c2d9 | 4 並行 Terminal A-D 成果 (V21 architecture skeleton + 戦略 v2 + Paddock 12f + Patrol YOLO prep) |
| 2 | f2a60a50 | calibrator v2 retrain (21→315 sample、 isotonic 飽和解消) |
| 3 | d7580488 | strategy_layer_v2 --calibrator v1/v2 option + 5/16 shadow 比較 |
| 4 | 508b4657 | 5/16 evening summary + 京都 ROI 20% 発見 |
| 5 | (master doc commit) | docs/SYSTEM_MASTER_2026_05_16.md (Claude 包括書) + 3 inventory source |
| 6 | b4948d6a | JRDB TYB 直前情報 実装 (+0.143 AUC 5CV) |
| 7 | d3b78683 | TYB honest 評価 + P0-3 leak 監査 統合 plan |

### 主要発見

- ★ **JRDB TYB merge bug**: 548K rows TYB data が V15 で 0% 結合だった (1 年以上 眠っていた)
- ★ **TYB padock_idx +0.44** が 真の signal (LR coef)
- ★ **京都 ROI 20%** (N=58、 戦略⑦再除外推奨)
- ★ **ROI 乖離**: CLAUDE.md 119.2% vs cumulative 93.23% (P0-1 で真値確定中)

### ★ 動画系撤回 判断と根拠 ★

| Source | 規約 | 状況 |
|--------|------|------|
| YouTube | DL 禁止 + AI 学習禁止 | NG |
| JRA レーシングビュアー | 私的使用範囲外不可 | NG |
| netkeiba SP 動画 | 規約 + IP ban 経験 | NG |
| JRA-VAN NEXT | レーシングビュアー同 | NG |
| JRA アプリ | 私的利用範囲明文 | NG |

→ V21 動画 AI (パドック / パトロール / 調教) ★ 永久放棄 ★

### 段階的 path 改訂

旧 (5/16 朝): 動画 features 5/31+ production 化 (plateau 突破唯一 path)
新 (5/16 evening): 動画放棄、 JRDB TYB + netkeiba SP テキスト + 戦略 layer で 動画なし frontier

### P0-3 → P1-0 → P2-1 path

| Phase | task | 条件 |
|-------|------|------|
| P0-3 | TYB leak 監査 | 5/17 21:00+ |
| P1-0 | TYB shadow eval | P0-3 PASS |
| P2-1 | v15.2 再学習 (TYB含) | P1-0 statistical有意 |

### V15 投資保護 完全維持 ✅

5/16 evening session で V15 production file 全部 unchanged、 G1 day 影響 0%。
