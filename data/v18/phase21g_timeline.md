# Phase 21G - 5/11 → 2027/5/10 timeline (1 年計画)

> 作成: 2026-05-11 (Phase 21G)
> 起点: 2026-05-11 (158h+ マラソン直後 リラックス日)
> 着地: 2027-05-10 (1 年後 真の完成形評価)

## 全体俯瞰 (5 phase)

| phase | 期間 | 主要 milestone | model |
|-------|------|--------------|-------|
| 0. リラックス + 微 task | 5/11 | master DOM probe (任意) | V15 のみ |
| 1. 平日集中構築 | 5/12-5/16 | 真値化 + V18 学習 + V21 PoC + V22 500K-1M | V15 + V18 候補 |
| 2. paper trade 並行 | 5/17-6/30 | 5 model 並行 paper trade | V15 + V18-V22 paper |
| 3. V20 投入 | 7/1-8/31 | V20 段階投入、 V15 並行運用 | V15 + V20 |
| 4. V21/V22 投入 | 9/1-12/1 | V21 (動画) → V22 (RL) | V15 → V20 → V21 → V22 |
| 5. 完成形評価 | 〜2027/5/10 | 1 年後 振り返り | full stack |

---

## Phase 0: 5/11 (今日) — リラックス + 任意 master DOM probe

| time | task | 必須? |
|------|------|------|
| 〜午前 | リラックス + Phase 21G doc 集約 (本 commit) | ✅ |
| 午後 (任意) | netkeiba master DOM probe (Phase 13/18 残) | optional |
| 夜 | 5/12 平日 4 並行 task の 環境チェック | ✅ |

→ 158h+ マラソン直後、 心身回復が最優先。

---

## Phase 1: 5/12-5/16 (平日集中構築)

### 5/12 (月、 平日) — 4 並行 task

詳細: [phase21d_5_12_plan.md](phase21d_5_12_plan.md)

| Terminal | task | 時間 |
|---------|------|------|
| A | Phase 11b 残 9 features 真値化 (KYI) | 90-120 min |
| B | Phase 13b netkeiba master 25 features 真値化 | 120-150 min |
| C | V18 真値版 学習 + WF 評価 | 120 min |
| D | V21 動画 PoC + V22 500K-1M (Gymnasium) | 120 min |

完了基準: 真値化 9/9 + LEAK PASS + delta AUC ≥ +0.0010

### 5/13-5/14 (火/水)

詳細: [phase21d_5_13_5_14_plan.md](phase21d_5_13_5_14_plan.md)

- V18 真値版 LIVE retro
- V20 4-model ensemble 真の学習 (TFJV 6 年 + JRDB 真値)
- V21 動画 features 抽出 (50 レース 1500 動画)

### 5/15 (木)

- 完全自動化 80% 目標達成 (5 model paper trade infrastructure 確認)

### 5/16 (金)

- V18 trial 投入 GO/NO-GO 最終判定
  - GO 条件: WF AUC ≥ 0.895 / 6-fold AUC gap < 0.05 / LEAK PASS / LIVE retro winner_top1 ≥ 35%
  - NO-GO なら V20 まで paper trade 継続

---

## Phase 2: 5/17-6/30 (paper trade 並行)

### 5/17 (土) — V15 + V18-V22 paper trade 5 model 並行

- V15: production (戦略⑦込み 案 B 改)
- V18: paper trade (5/16 GO なら本番候補)
- V20: paper trade (TFJV ベース PoC)
- V21: paper trade (動画 PoC)
- V22: paper trade (RL PoC、 Gymnasium env)

### 5/24+ — V20 ensemble 真の学習開始

- JV-Link DataLab 加入 (5/24 +、 月 ¥2,090)
- jvlink_fetcher.py 動作確認
- V20 学習 data spec 確定 (JRA + NAR 統合、 共通 80 features、 SKB 完全除外)
- V20 v1 学習 (4-model ensemble)

### 6/8 (日) — V20 投入候補 1 ヶ月前倒し判定

- WF AUC ≥ 0.880 / LIVE retro winner_top1 ≥ 30% / shift ≤ 12x / paper ROI ≥ 110% / LEAK 監査 PASS
- GO なら 7/1 投入

---

## Phase 3: 7/1-8/31 (V20 投入 + V15 並行運用)

### 7/1 (火) — V20 段階投入

- 週末のみ、 上限 ¥5,000 / 日
- V15 並行運用 (1 ヶ月後 = 8/1 V15 archive 判定)

### 7/15 (火)

- 順調なら投資額増額 (週末 ¥10,000 / 日 + 平日 ¥5,000 / 日)

### 7-8 月 — Phase 4 動画 PoC

- 7/1-7/14: data 蓄積 (JRA-VAN ネクスト + netkeiba 動画、 50 レース 1,500 動画)
- 7/15-7/31: YOLOv8 馬体検出 + DLC SuperAnimal 姿勢推定 動作確認
- 8/1-8/15: 時系列特徴量抽出 (stride / gait_symmetry / head_bobbing / ear_pos / posture)
- 8/16-8/31: V21 学習 (V20 + VIDEO_FEATURES) + WF 検証

### 8/1 (金) — V15 archive 判定

- V20 1 ヶ月並行運用後、 V15 → V20 完全移行候補

---

## Phase 4: 9/1-12/1 (V21/V22 投入)

### 9/1 (月) — V21 投入候補

- WF AUC ≥ V20 + 0.005 / LIVE retro winner_top1 ≥ V20 + 1pt
- 完全自動化 90% 目標達成

### 10-11 月 — V22 RL 真の学習

- PPO 30 年 backtest full
- Gymnasium env 強化
- paper trade 結果蓄積

### 12/1 (水) — V22 RL 投入候補

- 完全自動化 100% 目標達成
- 累計収支 +30-50 万円目標

---

## Phase 5: 12/1 → 2027/5/10 (完成形評価)

### 2026 末 〜 2027 春

- V20 + V21 + V22 安定運用
- 季節調整 (春 G1 / 秋 G1 期間の挙動評価)
- 30 年 backtest 結果と LIVE の整合 確認

### 2027/5/10 (1 年後) — 振り返り評価

評価項目:
- [ ] V15 → V20 → V21 → V22 段階投入 全成功か
- [ ] 累計収支 +50 万 / +100 万 / +200 万 のいずれを達成したか
- [ ] 完全自動化 100% (12/1 目標) は維持されているか
- [ ] 撤退ライン -¥50,000 を一度も触れなかったか
- [ ] V15 廃止 (8/1 候補) は完了したか
- [ ] RV 動画解析の実 ROI 寄与は何 pt だったか
- [ ] V22 RL は production に投入できたか

---

## 月額コスト 推移

| 期間 | コスト | 内訳 |
|------|-------|------|
| 5/11-5/23 | 約 ¥6,500 / 月 | netkeiba ¥4,500 + JRDB ¥2,000 |
| 5/24-6/30 | 約 ¥8,590 / 月 | + JV-Link ¥2,090 |
| 7/1+ | 約 ¥10,768 / 月 | + JRA-VAN NEXT ¥1,000 + Colab Pro ¥1,178 |

→ V20 以降の月利増分 (5-10 万円想定) で十分回収。

---

## 投資保護 (1 年通期 絶対遵守)

- 🔴 V15 → V20 → V21 → V22 段階投入時、 旧 model は 1 ヶ月並行運用
- 🔴 撤退ライン -¥50,000 (現在 +¥14,140 = 余裕 +¥64,140)
- 🔴 取り返し禁止 (損切り後 翌日に持ち越さない)
- 🔴 destructive git op 禁止

---

## 関連

- 158h+ 全 history: [phase21g_158h_marathon_history.md](phase21g_158h_marathon_history.md)
- 5/10 day 22 phases: [phase21g_5_10_day_22_phases.md](phase21g_5_10_day_22_phases.md)
- 真の集大成 list: [phase21g_achievements.md](phase21g_achievements.md)
- 完全自動化 ロードマップ: [../../docs/FULL_AUTOMATION_ROADMAP.md](../../docs/FULL_AUTOMATION_ROADMAP.md)
- Phase 3-4 統合 roadmap: [../../docs/PHASE_3_4_5_INTEGRATED_ROADMAP_v3.md](../../docs/PHASE_3_4_5_INTEGRATED_ROADMAP_v3.md)
