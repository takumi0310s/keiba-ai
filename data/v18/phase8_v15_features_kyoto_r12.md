# Phase 8 完了 (Opus 4.7) ★京都R12 体重統合再予想 + V15 150 features★

date: 2026-05-10 15:15
model: V15 Pattern B (live、 当日情報込み)
AUC: 0.8939
feature count: 150

---

## 1. 京都R12 4歳以上2勝C (16:10 発走) Stage 2 体重統合 結果

### case A 確定 (体重公開済 15:00 後)

Stage 2 force fire (15:14): 14 頭 全 score 取得、 体重 + オッズ realtime 反映済

| 順 | 馬番 | 馬名 | V15 score | オッズ |
|---|---|---|---|---|
| 1 | 14 | ヴィリアリート | 0.735 | 5.0 |
| 2 | 10 | レッドダンルース | 0.675 | 3.0 |
| 3 | 2 | メイショウタムシバ | 0.591 | 8.7 |
| 4 | 8 | ランスオブセヘル | 0.560 | 7.5 |
| 5 | 11 | ユウトザレン | 0.422 | 7.9 |
| 6 | 4 | グリプトグラフィ | 0.384 | 26.8 |

- 朝予測 top1: **14 ヴィリアリート (s=0.735)**
- Stage 2 top1: **14 ヴィリアリート (s=0.735)** ★一致★
- ★ top1 不変 確定 ★ → 体重統合後も投票維持
- オッズ realtime 5.0 (朝オッズと同水準)

### 投票
¥700 (3 連複 7 点、 軸 14 / 流し 2-8-10-11-12)
組合せ: 2-8-14 / 2-10-14 / 2-11-14 / 2-12-14 / 8-10-14 / 10-11-14 / 10-12-14

---

## 2. V15 150 features 全 list (categorize)

### ✅ 完全機能 (蓄積データ + 当日情報)
- JRDB (騎手/調教師/血統等): **44** features
- JRDB KKA / Paddock / PACI: **11** features
- 基本 / レース条件: **23** features
- 前走 / レビュー: **12** features
- 通算成績 / 集計: **8** features
- 騎手: **8** features
- 厩舎 / 調教師: **3** features
    - trainer_top3_calc, location_enc, stable_comment_score
- 調教: **12** features
- 血統 (父/母父): **6** features
- タイム指数 / Speed: **4** features
    - index_max_filled, index_run1_filled, index_avg5_filled, pci
  → 計 **131** features

### ⚠ 朝予測時 default fill → Stage 2 で realtime 取得
- オッズ / 人気 (Stage 2 統合): **6** features
    - prev_odds_log, oz_base_pop_rank, odds_change_rate, pop_rank_change, odds_sharp_drop, jrdb_odds_idx
  → 計 **6** features

### その他 / 未分類
- その他: **13** features
  → 計 **13** features

**total: 150 / 150 features ★ 全数一致 ★**

### ❌ 未機能 (V15 で未学習、 5/16+ 改善 plan)
- 動画解析 features (パドック動き / 周回ペース / 馬体動向): JRA-VAN RV trial 待ち
- 当日厩舎コメント詳細スコア: 取得カバレッジ 30% で V12.1 不採用
- 当日パドック写真特徴量: PoC 段階

## 3. 5/16+ 改善 plan
- V18 sib_w5: 過去 5 R sibling 統合 (期待 +0.05 AUC)
- JRA-VAN RV trial: 動画 features PoC
- JRA-VAN NEXT: 自動分配 (Session #81 設計済)
- ★ dynamic 取得化: Phase 7 で完了済 ★ (5/16+ stage2 fire OK)

## V15 投資保護
✅ V15 model 不変
✅ predict_core / daily_predict / app.py 不変
✅ schtask 不変
✅ Phase 8 = read-only audit のみ
✅ 累計 +¥14,140 維持