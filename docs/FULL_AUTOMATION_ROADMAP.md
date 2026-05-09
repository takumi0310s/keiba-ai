# 完全自動化 ロードマップ (Session #86)

> 2026-05-09 21:00 過ぎ 作成
> 146h+ マラソン (Phase 2.5+ → 5/9 GW 締め) で 大量の発見・doc 蓄積後の 統合 plan
> 最終目標: **2026-12-01 100% 自動化 (V22 RL 投票最適化)**

## 現状 (2026-05-09)

| 項目 | 状態 | source |
|------|------|--------|
| V15 production 朝予測 | ✅ 自動 (DailyPredict_0800) | schtasks |
| 全馬 score 保存 | ✅ 5/10〜 自動 (SaveAllHorseScores_0930) | Session #71 |
| Stage 2 通知 | ⏸ 5/16〜 復活予定 | dev/two-stage |
| 投票 | 🔴 手動 PAT 入力 (5-15 分 / R) | netkeiba IPAT |
| verdict (R 単位) | ✅ 自動 | Session #61 |
| 1 day summary | ✅ 自動 | Session #61 |
| 累計 monitor | ✅ 半自動 (Discord alert) | Session #61 |
| 撤退 alert | 🔴 手動判断 | — |
| morning_weight_check | ✅ 09:30 自動 | Session #62+ |

**自動化率: 約 60%** (投票 + 撤退判断 が 手動)

## Phase 別 ロードマップ

### Phase 1: 5/15-6/8 (V18 trial + JRA-VAN trial) → **80% 自動化**

| 項目 | before | after |
|------|--------|-------|
| 投票 PAT 入力 | 手動 5-15 分 / R | JRA-VAN NEXT 自動分配 + 1 click 送信 → **30 秒 / R** |
| 買い目入力 | 手動 ~10 個 入力 | JRA-VAN NEXT 内 自動 fill |
| 投票漏れ | 発生 risk | ★ ゼロ ★ (一括分配) |
| Stage 2 通知 | 停止中 | 5/16 復活 (dev/two-stage merge) |

**主要実装**:
- `tools/jra_van_next_allocator.py` (新規、 Session #81 設計)
- `tools/auto_pat_fill.py` (新規、 5/15 trial 後)

**期待効果**:
- 投票時間 **5-15 分 → 30 秒 / R** (10-30 倍速)
- 投票漏れ **ゼロ**

### Phase 2: 7/1-9/2 (V20 ensemble + Phase 4 動画) → **90% 自動化**

| 項目 | before | after |
|------|--------|-------|
| Model | V15 (150 features) | **V20** (320K records 学習、 4-model ensemble) |
| AUC | 0.8939 | **0.900-0.905 想定** |
| 動画 features | なし | **JRA-VAN RV (5/15 trial→) 統合** |
| 投票判断 | 戦略⑦ + 案B改 (人手 rule) | **AI score base** (model 直接出力) |
| 買い目選定 | rule 7 点 fix | **AI 提案 + 上限 rule** |

**主要実装**:
- V20 production 投入 (7/1)
- V21 動画 features 統合 (9/2)
- AI score base 投票判断 logic

**期待効果**:
- ROI **140% → 145-150%** (V20 + 動画)
- winner_top1 **30% → 38-41%**

### Phase 3: 10-12月 (V22 RL) → **100% 自動化** ★

| 項目 | before | after |
|------|--------|-------|
| 投票判断 | rule + AI score | **RL agent 全 AI 決定** |
| 投票金額 | 人手 (案 A/B/C) | **RL 動的最適化** (累計 + 残資金 base) |
| 買い目 | 人手 fix | **RL 動的選定** (期待値最大化) |
| 撤退 logic | 人手 alert | **完全自動** (累計 -50,000円 で halt) |
| 月次 evaluation | 人手 review | **自動 report + 提案** |

**主要実装**:
- `train/v22_rl_*.py` (Session #84/85 設計済)
- 30 年 backtest 環境 (Session #84/85 設計済)
- gym-style env + PPO/SAC

**期待効果**:
- user 操作 **完全ゼロ** (Discord で結果 monitor のみ)
- ROI **最適化 (150%+ 想定)**
- 撤退判断 **0 ms** (即時)

## 完全自動化 後の 1 日 (12/1+ 想定)

```
06:00 — JRA カレンダー fetch (自動)
08:00 — DailyPredict + V20/V21 score 計算 (自動)
09:00 — RL agent 投票判断 + 金額決定 (自動)
09:30 — JRA-VAN NEXT に 投票送信 (1 click 自動 or 完全 API)
10:00-15:30 — 各 R verdict 通知 (Discord、 自動)
15:30 — 1 day summary 通知 (Discord、 自動)
20:00 — 累計 monitor (Discord、 自動)
20:00 — 翌日 schtask 健全性 check (自動)
```

→ **user の操作: ゼロ** (Discord でモニタするだけ)

## 月次 evaluation (12/1+ 自動)

| 項目 | 内容 |
|------|------|
| ROI 集計 | 全条件 + 戦略別 |
| 撤退判定 | 累計 -50,000円 で halt 自動 |
| RL re-training | 月次 自動 (前月 data 追加) |
| Model drift check | AUC degrade 検知 → alert |
| 提案 report | Discord に PDF / md 送信 |

## 投資保護 (絶対遵守)

- **V15 production は Phase 3 完了 まで 並行運用**
- V20 (7/1) も V15 並行 1 ヶ月後 archive 判定
- V22 RL は paper trading 1 ヶ月 → GO 判定後 production
- ★ 累計 -50,000円 撤退ライン 厳守 ★
- 現状 累計 +13,530円 (撤退余裕 +63,530円)

## 月額 cost 想定 (12/1+)

| source | 月額 | 開始 |
|--------|------|------|
| netkeiba Premium | 4,500円 | 既存 |
| JRDB Advance | 約 2,000円 | 既存 |
| JV-Link (DataLab) | 2,090円 | 5/24 |
| JRA-VAN ネクスト + RV | 1,430円 | 5/15 (trial 後) |
| Colab Pro (RL training) | 1,178円 | 10/1 |
| **合計** | **約 11,200円/月** | — |

→ V22 RL (12/1+) で月利 5-15 万円 想定 → 月額 cost 完全回収

## 関連 doc

- [JRA_VAN_NEXT_AUTO_ALLOCATION.md](JRA_VAN_NEXT_AUTO_ALLOCATION.md) — JRA-VAN NEXT 自動分配 設計 (Session #81)
- [STRATEGY_HYBRID_DESIGN.md](STRATEGY_HYBRID_DESIGN.md) — hybrid 戦略 (Session #82)
- [V20_BUILD_DETAILED_PLAN.md](V20_BUILD_DETAILED_PLAN.md) — V20 構築 詳細 (Session #79)
- [PHASE_4_VIDEO_REPLAN_v2.md](PHASE_4_VIDEO_REPLAN_v2.md) — Phase 4 動画 plan v2 (Session #80)
- [V22_RL_DESIGN.md](V22_RL_DESIGN.md) — V22 RL 設計 (Session #84/85)
- [BACKTEST_30_YEAR_DESIGN.md](BACKTEST_30_YEAR_DESIGN.md) — 30 年 backtest 設計
- [PHASE_3_4_5_INTEGRATED_ROADMAP_v3.md](PHASE_3_4_5_INTEGRATED_ROADMAP_v3.md) — 統合 roadmap v3
