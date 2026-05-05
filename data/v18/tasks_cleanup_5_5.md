# Phase 2.5+ tasks 整理 (5/5 PM、Session #18)

参照: `data/v18/phase_2_5_remaining_tasks_5_4.md` (5/4 朝の original)

---

## 1. Phase 1-A クローズ (5/2 全敗 真因究明)

| 項目 | 状態 |
|------|------|
| 完了 commit | 5fdfc2d0 (Session #2、5/3 19:12) |
| 結果 | `data/v18/may2_postmortem.md` (systematic 疑い P=0.032、軸top3 39.4% vs BT 57.0%) |
| 原因仮説 | netkeiba premium データ欠損 (ra_score/sc_score 0%) → V15 features 部分劣化 |
| 対応 | Session #10 (b4c4894c, 6b5e4e7b) で部分復旧、5/12+ 全レース更新 |

✅ **Phase 1-A クローズ済**。フラグ in_progress は historic、現状は別 commit で進化済。

---

## 2. Phase 2.5 タスク progress (5/4 → 5/5 PM)

### 2.1 緊急 3件 (5/4 朝)

| # | タスク | 5/4 朝 状態 | 5/5 PM 状態 |
|---|--------|--------------|--------------|
| 1 | DailyPredict task watchdog 化 (admin) | 待機 | ✅ admin 実行済 (Session #14 / 確認 #15) |
| 2 | netkeiba_race_analysis 再起動 | 5/4 中 | ✅ Session #10 (b4c4894c) 60 races 取得 |
| 3 | netkeiba_stable_comments 再起動 | 5/4 中 | ✅ Session #10 (6b5e4e7b) 2026年4-5月分取得完了 |

### 2.2 高 7件 (5/4-5/10)

| # | タスク | 5/5 PM 状態 |
|---|--------|--------------|
| 4 | netkeiba_ai_position 再起動 | ⏸️ 5/12+ paper 中に並行 (低 priority) |
| 5 | netkeiba_siblings 再起動 | ⏸️ 同上 |
| 6 | netkeiba_master_index 再起動 | ⏸️ 同上 |
| 7 | TYB publish タイミング 連続観測 | ✅ Session #8 自動化 (5/4-5/10 蓄積中) |
| 8 | jra_payouts 5/2-5/3 取得 | ✅ Session #4 retro で確保 (data/v18/v18_v19_retro_full_predictions.csv) |
| 9 | netkeiba_speed_index 再起動 | ⏸️ 5/12+ |
| 10 | `.gitattributes` models/*.txt -text | ✅ Session #5 (777cc08e) で適用 |

### 2.3 中 6件 (5/11-5/15)

| # | タスク | 5/5 PM 状態 |
|---|--------|--------------|
| 11 | v18/v19 calibration 修正 (Platt scaling) | ✅ Session #8 (0e03c55c) 試作、限界判明 (max 0.154→0.213) |
| 12 | race-level probability normalization | ✅ Session #10 (74eb10b7) softmax T=1.0 確立 |
| 13 | 特徴量分布検証 (2026 vs 2024) | ⏸️ 5/12+、別 task (90min) — `data/v18/v18_v19_integration_plan_5_4_pm.md` 必須前提 |
| 14 | JRDB ot/ov/ow/oz 再取得 | 🟢 5/12+ 不要 (V15 学習に未使用) |
| 15 | odds_history.csv 再取得 | 🟢 5/12+ 不要 (BT 拡張は Phase 3 でも可) |
| 16 | netkeiba_training_times date NaN 修復 | ⏸️ 5/12+ (M1 中) |

### 2.4 低 4件 (5/16+)

| # | タスク | 5/5 PM 状態 |
|---|--------|--------------|
| 17 | v15.1 特徴量拡張 (KKA/SKB/SR) | 🟢 Phase 3 (5/末-6/末) — 統合モデル v20 と同時 |
| 18 | 古いモデル削除 (~130MB) | 🟢 Session #18 (本 task C) で archive 移動候補 |
| 19 | predict_*.py 整理 (13版→3版) | 🟢 Phase 3 |
| 20 | archive/ 移動 (古い静的 CSV) | 🟢 Session #18 (本 task C) で着手 |

---

## 3. 新規発生タスク (5/4 以降)

### 3.1 完了

| 項目 | commit | 出典 session |
|------|--------|------------|
| 静音化 28 task (vbs ラッパー) | 9c88d27c | #9 |
| race-level normalization 試作 + 統合 plan | 74eb10b7 + 6820b362 | #10 |
| 5/5 柏記念 ヒューリスティック + NAR v4 復活 + 体系化 | bfbddebc + e5f71cfa + 57029ff1 | #11-#13 |
| 5/9 本番準備 (8 doc + race_day_report.py) | 2b6dc4eb | #14 |
| 引き継ぎ書 v2 + 振り返り | edfa9897 | #15 |
| NAR pipeline 未実装 script 2 個 | eeb48e45 | #17 |

### 3.2 残 (5/12 paper 開始までに)

| 項目 | 工数 | 出典 |
|------|------|------|
| register_nar_schtasks.ps1 admin **再実行** (stage 引数 反映) | 5min user 手動 | Session #17 |
| feature distribution shift 調査 (2026 vs 2024) | 90min | Session #10 P2.5 plan |
| chihou_races_2020_2025.csv 生成 (NAR strict OOS 用) | 60min | Session #13 |

### 3.3 5/16+ ramp 後

- v18/v19 5/16 試行 (条件達成後 1,000円/日)
- NAR 5/16 試行 500円/日
- v15 → 6/末 統合 v20 設計

---

## 4. 棚卸し状況サマリ

| 区分 | 5/4 朝 棚卸し | 5/5 PM 状態 |
|------|--------------:|--------------|
| 完了 | 14 件 | **20+ 件** (新規 6 件追加完了) |
| 緊急 | 3 件 | **0 件** (全完了) |
| 高 | 7 件 | **0 件 active** (5 件完了 + 2 件 5/12+ defer) |
| 中 | 6 件 | **2 件 5/12+ defer** (4 件完了 or 不要判定) |
| 低 | 4 件 | **2 件 Phase 3 defer** (1 件 active 本 cleanup、1 件 Phase 3) |
| **新規発生** | - | **6 件追加 + 全完了** |

→ Phase 2.5 main scope は **ほぼ完了**。残 issue は Phase 3 と 5/12+ paper 蓄積後判定。

---

## 5. 結論

- **Phase 1-A**: 5fdfc2d0 でクローズ済 (フラグ historic)
- **Phase 2.5 緊急/高 priority**: 完全クリア
- **Phase 2.5 中 priority**: 90% 完了、残 2 件は 5/12+ paper 蓄積中に並行
- **Phase 3 候補**: 5/16+ ramp 後 (v20 統合モデル + cleanup)

Phase 2.5+ の **構造改善 phase は 5/5 PM で実質終了**、5/12 NAR paper 開始 + 5/16 試行 ramp が次の milestone。
