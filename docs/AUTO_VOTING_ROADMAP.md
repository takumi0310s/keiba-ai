# 投票 自動化 ロードマップ

Session #81 (2026-05-09 夜)。

## 全体 vision

V15 / V20 / V21 model の score を 自動投票に 接続し、
1 R あたりの 操作工数を 5-15 分 → 30 秒 → 0 秒 に段階的に削減する。

V15 投資保護を **絶対遵守** しつつ、 段階的に自動化レイヤを 追加。

## phase 一覧

| phase | 期間 | 内容 | 工数 / R |
|-------|------|------|----------|
| 0 (現状) | -5/14 | PAT 手動入力 | 5-15 分 |
| 1 | 5/15-5/22 | NEXT trial (手動入力 + 自動分配) | 30 秒 |
| 2 | 5/23+ | NEXT 採用、 継続運用 | 30 秒 |
| 3 | 6/8+ | V20 score → NEXT 投入 | 30 秒 |
| 4 | 7/1+ | V20 production 完全連携 | 30 秒 |
| 5 | 9/2+ | V21 (動画 features) 統合 | 30 秒 |
| 6 | 12 月 | RL 投票最適化 + 完全自動化 | 0 秒 |

## phase 1 (5/15-5/22): NEXT trial

詳細: [JRA_VAN_NEXT_TRIAL_5_15.md](JRA_VAN_NEXT_TRIAL_5_15.md)

- NEXT 1 ヶ月無料 trial
- 案B改 strict + 自動分配 試用
- 5/16 V18 trial と 同時運用
- 5/22 評価 → 5/23 採用判定

## phase 2 (5/23+): NEXT 採用 継続

GO 条件 ([JRA_VAN_NEXT_TRIAL_5_15.md](JRA_VAN_NEXT_TRIAL_5_15.md) 参照) クリア時:
- 月 2,090 円 で NEXT 継続
- V15 案B改 strict + 自動分配 + 1 click PAT
- 工数: 1 R 30 秒
- TARGET frontier カスタム指数 機能 調査 着手

## phase 3 (6/8+): V20 score → NEXT 投入

V20 投入後:
- V20 case 1+4 候補 → NEXT 馬番 入力
- 自動分配 EV モード で 1 click PAT
- 案B改 strict と V20 case 1+4 の 並行運用
- 投資額: V15 案B改 strict 上限 2,100円 / V20 case 1+4 別予算 (要設計)

詳細は Session #80 以降で [PHASE_3_4_5_INTEGRATED_ROADMAP_v3.md](PHASE_3_4_5_INTEGRATED_ROADMAP_v3.md) 更新予定。

## phase 4 (7/1+): V20 production 完全連携

V20 production 投入 (週末上限 5,000 円/日 → 7/15+ 1万円/日):
- V20 score → TARGET frontier カスタム指数 (実装可能なら)
- カスタム指数 → NEXT 自動分配
- 1 click PAT 送信
- V15 並行 archive 判定 (8/1)

## phase 5 (9/2+): V21 (動画 features) 統合

V21 投入後:
- V21 score → カスタム指数 → NEXT
- stride / gait_symmetry / head_bobbing 等 5 features を含む
- 工数 引き続き 30 秒 / R

## phase 6 (12 月): 完全自動化

RL 投票最適化 + 全工程 自動化:

### 構想
1. **RL 投票最適化**: 過去 6 年 backtest で 最適配分 学習
2. **Selenium / Playwright 操作**: NEXT UI 自動操作
3. **V21 score → 自動投票**: ユーザー介在 ゼロ
4. **safety stop**: 単日 ROI < 50% で 自動停止 (撤退ライン 連動)

### 安全性 設計
- ★destructive op 厳禁★
- V21 model 変更 なし
- predict_core / daily_predict / app.py 変更 なし
- ユーザー confirm UI (初期は 1 R ごとに confirm 必須)
- 段階解除: confirm 1 週間 100% 一致 で 自動投票 enable

### risk
- 法令 / JRA 規約 確認 必須
- 自動投票 禁止規定 ある場合は 中止
- 12 月までに 規約 audit 完了

## 月額コスト 推移

| phase | source | 月額 |
|-------|--------|------|
| 0 (現状) | netkeiba Premium + JRDB Advance | 約 6,500 円 |
| 1+ | + JRA-VAN NEXT | + 2,090 円 |
| 4 (7/1+) | + JRA-VAN ネクスト (Phase 4 動画) + Colab Pro | + 2,178 円 |
| **合計 (7/1+)** | — | **約 10,768 円/月** |

ROI 想定:
- V15 (現状): 119.2% (戦略⑦込み 140%) → 月利 約 2-3 万円
- V20 (7/1+): 145-150% 想定 → 月利 5-10 万円
- V21 (9/1+): 145-150% 想定 → 月利 6-11 万円
- → 月額コスト 約 1万円は V20 以降 月利増分で 十分回収

## 投資保護 (絶対遵守)

🔴 NEVER:
- predict_core / daily_predict / app.py 変更
- V15 / V20 / V21 model 変更
- ★destructive git op★
- 自動投票 phase で safety stop 削除

🟢 OK:
- docs/ 追加 / 更新
- NEXT trial / 採用 (UI レイヤーのみ)
- 累計収支 +14,140 円 死守

## 撤退ライン (3 段階、 全 phase で 共通)

- 単日 ROI<50%: 当日 投票 即停止
- 累計 -10,000 円: 翌日 1 日 停止 で 冷却
- 累計 -50,000 円: 完全撤退、 model / phase 全面見直し

## 関連 doc

- [JRA_VAN_NEXT_AUTO_ALLOCATION.md](JRA_VAN_NEXT_AUTO_ALLOCATION.md)
- [V15_TO_JRA_VAN_INTEGRATION.md](V15_TO_JRA_VAN_INTEGRATION.md)
- [JRA_VAN_NEXT_TRIAL_5_15.md](JRA_VAN_NEXT_TRIAL_5_15.md)
- [PHASE_3_4_5_INTEGRATED_ROADMAP_v3.md](PHASE_3_4_5_INTEGRATED_ROADMAP_v3.md)
