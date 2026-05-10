# Phase 21H summary: 月額 ¥10,500 ROI 評価 + 1 年後累計推移期待値

作成: 2026-05-11 (Session #87、 Phase 21H)

---

## 1. 結論 (TL;DR)

| 項目 | 値 |
|------|----|
| 月額 cost (4 source) | **¥10,500/月** |
| 5月 (V15 alone) net | +¥2,000 (中位、 ボーダー) |
| 7月 (V20) net | +¥17,000/月 (◎) |
| 12月 (V22 RL) net | +¥29,500/月 (◎◎) |
| **1 年後 期待累計 (中位)** | **+¥350,000-400,000** |
| **1 年後 期待累計 (楽観)** | **+¥800,000-1,000,000** |
| **1 年後 期待累計 (悲観)** | **+¥87,000-150,000** |
| 撤退余裕 (今) | **+¥64,140** |
| 投資保護 ルール | 6 項目 全 維持中 |

→ **★ 月額 ¥10,500、 7月以降 圧倒的黒字、 1 年後 net +¥220K-280K 期待 ★**

---

## 2. 4 source 月額内訳

| source | 月額 | 主用途 | 判定 |
|--------|-----:|--------|------|
| JRA-VAN DataLab | ¥2,090 | V20 学習 (TFJV 6 GB / 6 年) | ★ 必須 ★ |
| JRA-VAN RV | ¥550 | V21 動画 features PoC | trial OK |
| JRDB Advance | ¥2,880 | V15 既統合 + Phase 11 真値化 | ★ 必須 ★ |
| netkeiba マスター | ¥4,980 | UI + V20 features 候補 | プレミアム DG 検討候補 |
| **合計** | **¥10,500** | | |

詳細: data/v18/phase21h_cost_analysis.md

---

## 3. 月利 推移 期待値 (中位)

| 期間 | model | 月利 | net (cost ¥10,500 控除) |
|------|-------|-----:|------------------------:|
| 5月 | V15 | +¥12,500 | +¥2,000 |
| 6月 | V15 + V18 paper | +¥15,000 | +¥4,500 |
| 7-8月 | V20 | +¥27,500 | +¥17,000 |
| 9-11月 | V21 | +¥32,500 | +¥22,000 |
| 12月+ | V22 RL | +¥40,000 | +¥29,500 |

詳細: data/v18/phase21h_roi_simulation.md

---

## 4. 累計推移 (中位 path)

| 時点 | 累計 |
|------|-----:|
| 5/10 (今) | +¥14,140 |
| 5/17 | +¥14,140 (★ 維持目標 ★) |
| 6/10 | +¥22,000 |
| 7/1 | +¥30,000 |
| 9/1 | +¥70,000 |
| 12/1 | +¥150,000 |
| **5/10 (1 年後)** | **+¥350,000-400,000** |

詳細: data/v18/phase21h_cumulative_projection.md

---

## 5. 撤退基準 audit 結果

| 項目 | 結果 |
|------|------|
| 撤退ライン -¥50,000 | 余裕 +¥64,140 (★ 安全 ★) |
| 危険水域 +¥30,000 | 接近 -¥34,140 (非接近) |
| 直近 5/2-5/3 連続損失 | 戦略⑦適用後 5/10 +¥3,290 で改善 |
| 投資保護 6 項目 | 全 維持中 |
| 1 年後 撤退余裕 (中位) | +¥400K (圧倒的余裕) |

詳細: data/v18/phase21h_risk_audit.md

---

## 6. 次 actions (5/12-5/17)

| 日付 | action |
|------|--------|
| 5/12-5/16 | V15 production + V18 paper 並行、 累計 +¥14,140 維持 |
| 5/17 | V18 真値版 GO/no-go 判定、 5/16 GO worksheet (Phase 21A) で確定 |
| 5/24+ | V20 構築着手 (Session #79 V20_BUILD_DETAILED_PLAN.md) |
| 7/1 | V20 投入候補 |
| 9/1 | V21 投入候補 |
| 12/1 | V22 RL 投入候補 |

---

## 7. 投資保護 絶対遵守

🔴 NEVER:
1. predict_core.py / V15 model 一切変更しない
2. 戦略⑦ filter 解除しない
3. 投資単位 700円/レース 上限
4. 単日投資上限 ¥10,000
5. 撤退ライン -¥50K 接近 で 即 投票停止
6. destructive git op 禁止

🟢 GO:
- 7/1 V20 投入 paper 1 週間 ROI 110%+ → production
- 9/1 V21 投入 V20 + 0.005 AUC 改善 → production

---

## 8. 関連 doc

- data/v18/phase21h_cost_analysis.md (A: cost 分析)
- data/v18/phase21h_roi_simulation.md (B: ROI simulation)
- data/v18/phase21h_cumulative_projection.md (C: 累計推移)
- data/v18/phase21h_risk_audit.md (D: 撤退 audit)
- docs/V20_BUILD_DETAILED_PLAN.md (Session #79)
- docs/V22_RL_DESIGN.md (Session #84)
- docs/FULL_AUTOMATION_ROADMAP.md (Session #86)
