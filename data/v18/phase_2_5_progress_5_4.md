# Phase 2.5 5/4 朝-午前 進捗サマリー

生成: 2026-05-04 (Opus xhigh, Session#8)

## 実行タスク一覧

| # | タスク | 状態 | 所要 | コミット |
|---|--------|------|----:|----------|
| A | ra_score 再取得 (netkeiba_race_analysis) | ⚠️ blocked | 5min | 470a9d90 |
| B | sc_score 再取得 (netkeiba_stable_comments) | ⚠️ blocked | 5min | 48709274 |
| C | TYB publish タイミング観測 自動化 | ✅ 完了 | 30min | 5262e0c0 |
| D | v18/v19 Platt scaling 試作 | ✅ 完了 | 60min | 0e03c55c |

## 主要結果

### A/B: 共通 blocker

```
jra_races_full.csv 38日 stale (3/27 で停止、2026年データなし)
→ scrape_stable_comment.py / scrape_comments_bulk.py が race_id 列挙不能
→ 0 races to scrape

復旧経路: tools/update_jra_races_full_2026.py 新規作成 (~30min, 別タスク)
影響: V17 features ra_score/sc_score 全 NaN 継続 (5/9 投資には影響なし)
```

### C: TYB publish 観測 ✅

```
✓ tools/tyb_publish_monitor.py 作成
✓ tools/tyb_publish_monitor.bat (schtasks wrapper)
✓ schtasks 登録: Keiba-TybPublishMonitor 毎時実行 (5/4 12:30〜)

テスト結果:
  20260504 fetch: HTTP=404 Size=0B (期待通り)
  20260509 fetch: HTTP=404 Size=0B (まだ早い)

5/4-5/10 期間で publish 時刻分布蓄積。
初公開検出時 → Discord (#updates) 自動通知。
結果次第で midday 戦略 維持/廃止 判断。
```

### D: v18/v19 Platt scaling ✅

```
✓ tools/calibrate_v18_v19.py 作成
✓ data/v18/models/v18_tansho_calibrator.pkl (LR object)
✓ data/v18/models/v19_fukusho_calibrator.pkl

OOS 2025 calibration 効果:
  v18: Brier 0.0514→0.0506, LogLoss 0.1787→0.1747
  v19: Brier 0.1066→0.1062, LogLoss 0.3329→0.3319
  reliability gap |≤0.05| に大幅収束

5/2-5/3 retro 再評価:
  v18 raw max p=0.154, cal max p=0.213 (+1.4x)
  bet (p>=0.5, EV>=1.2): 0件 (calibration では解決せず)
  
🔴 真の原因: distribution shift (2026年GW で model out 6x 縮小)
🔧 次対策: race-level normalization, feature distribution audit
```

## 検出された問題

| # | 問題 | 影響 | 対策 |
|---|------|------|------|
| 1 | jra_races_full.csv 38日 stale | A/B blocked | update_jra_races_full_2026.py 新規作成 |
| 2 | 5/2-5/3 v18/v19 distribution shift | 5/16 部分実弾準備 delay | race-level norm + feature audit |
| 3 | TYB publish 時刻不明 | midday 戦略 維持/廃止 判定不能 | 5/4-5/10 連続観測中 ✅ |

## 残タスク (Phase 2.5 第1週)

### 🔴 緊急 (5/4-5/5)

- [x] A. ra_score blocker 報告 → next session
- [x] B. sc_score blocker 報告 → next session
- [x] C. TYB monitor 設置完了
- [x] D. Platt scaling 試作完了
- [ ] **DailyPredict task watchdog 化** (admin manual) ← ユーザー作業

### 🟠 高 (5/5-5/8)

- [ ] **tools/update_jra_races_full_2026.py 作成** + 実行 (4/1-5/3 races 取得)
- [ ] A/B 再実行 (上記完了後)
- [ ] **race-level probability normalization 試作** (D の次対策)
- [ ] **特徴量分布検証** (5/2-5/3 vs 2024)
- [ ] netkeiba premium 拡大 (ai_position, siblings, master_index)
- [ ] netkeiba_speed_index 再起動
- [ ] netkeiba_training_times date NaN 修復

### 🟡 中 (5/9-5/15)

- [ ] 5/9 案B改 実運用観察
- [ ] 5/9-5/10 結果集計
- [ ] 5/10 TYB monitor 結果解析
- [ ] JRDB ot/ov/ow/oz 再取得
- [ ] v15.1 特徴量拡張準備

## 5/9 投資判断への影響

🟢 **影響なし** — 全タスク 5/9 V15 案B改 運用に直接は影響しない:

- A/B blocker → V17 features 引き続き欠損だが V15 単独で運用
- C TYB observer → midday 戦略の生死判定材料、5/9 では使わない
- D Platt scaling → v18/v19 部分実弾は 5/16 以降課題

5/9 案B改 (12R 1勝クラスのみ、上限 2,100円、期待 ROI 161%) を維持。

## コミット系列

```
470a9d90 Phase 2.5 A: ra_score blocker 確認
48709274 Phase 2.5 B: sc_score 同 blocker
5262e0c0 Phase 2.5 C: TYB publish 観測 自動化 完了
0e03c55c Phase 2.5 D: v18/v19 Platt scaling + retro 再評価
(これから) Phase 2.5 5/4 進捗サマリー
```

## 累計損失状況

```
USER 報告: 累計 +14,140円
5/4-5/8 投資なし → 累計変動なし (+14,140円)
5/9-5/10 想定最悪 -4,200円 → 累計 +9,940円
撤退ライン -50,000円 まで余裕 +60,000円超
```

## TL;DR

- ✅ **C/D 完了** (TYB monitor 設置、Platt scaling 試作)
- ⚠️ **A/B blocked** (jra_races_full 上流問題、別タスク)
- 🟢 **5/9 投資判断: 案B改 維持 (V15 単独)**
- 🔧 **次セッション**: jra_races_full 2026 更新 → A/B 再実行 → race-level norm 試作
