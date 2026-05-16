# Phase 3-4 統合 roadmap (Session #39 J)

**作成**: 2026-05-07 (Session #39 J)
**期間**: 2026-05-24 (Phase 3 着手) 〜 2026-09-01 (V21 投入判定)
**目的**: Session #38 確定後の修正版 Phase 3 + Phase 4 + V20/V21 を統合した 4 か月計画

---

## 0. 全体像 (一目)

```
2026/  5月  6月  7月  8月  9月
─────────────────────────────────────────────────────────────
5/9   ★ V15 案B改 単独継続 (絶対遵守)
5/16  ★ V15.1 / V18/V19 共に NO-GO (Session #38)
                    │
5/24  ┌── Phase 3 前半 ──┐
      │ JV-Link 加入       │
      │ sib_*_exp 統合     │
      │ V18/V19 v2 6-fold  │
      │ V18/V19 v2 LIVE    │
6/8   └── 6/15+ 投入判定 ──┘
      │
6/9   ┌── Phase 3 後半 ─────────┐
      │ JV-Link parser           │
      │ V20 学習 + WF + LIVE     │
      │ V20 paper trading        │
6/30  └── 7/1+ 投入判定 ──────────┘
      │
7/1   ┌── V20 production deploy ──┐
      │ + Phase 4 PoC 開始         │
      │ 動画蓄積 + 姿勢推定        │
      │ VIDEO_FEATURES 抽出        │
      │ V21 学習                   │
9/1   └── V21 投入判定 ────────────┘
```

---

## 1. Phase 3 前半 (5/24-6/8): 基盤整備 + sib_*_exp

### 1.1 milestone

| milestone | 期日 | 達成基準 |
|----------|------|---------|
| JV-Link 加入完了 | 5/24 | jvlink_fetcher.py 動作確認 (RACE 1 日分取得) |
| sib_*_exp 統合 | 5/27 | predict_core.py / train pipeline 旧 sib 削除、 新版差し替え |
| V18/V19 v2 6-fold WF | 5/30 | WF AUC ≥ 0.880 (旧 V18/V19 0.8954 / 0.8787 想定) |
| V18/V19 v2 LIVE retro | 6/5 | winner_top1 ≥ 30%、 shift ≤ 12x、 3 週分 (5/30,5/31,6/1) 平均 |
| 6/15+ V18/V19 v2 投入判定 | 6/8 | 上記全 PASS なら GO、 1 つでも NG なら NO-GO |

### 1.2 詳細 schedule

```
5/24 (土)
  AM: JRA-VAN DataLab 加入 (2,090円/月)、 JV-Link DLL インストール
  PM: jvlink_fetcher.py 動作確認 (RACE 1 日分)
  夜: tools/parse_jvlink_race.py 試作開始

5/25-5/27 (日-火)
  sib_expanding_features.py を train/features_v15_new.py に統合
    - 旧 sib_top3_rate / sib_shinba_wr 削除
    - 新 sib_top3_rate_exp / sib_shinba_wr_exp に差し替え
    - + sib_total_races_exp / sib_total_offspring_exp 追加 (4 features)
  predict_core.py に同統合 (build_features 段階)
  動作確認: 単体 race 予測で旧 → 新 値変化を確認
  ★ 5/27 V15 prod 構文チェック (本番 V15 は sib 未使用、 V18/V19 系のみ影響)

5/28-5/30 (水-金)
  train/run_v18_tansho.py / run_v19_fukusho.py を sib_*_exp 版で再学習
  6-fold WF (LGB+XGB)、 5/29-5/30 計算 (~6h)
  結果: data/v18/v18_v19_sib_exp_wf_5_30.json

5/31-6/1 (土-日)
  V18/V19 v2 LIVE retro
    - 5/31 当日 race で winner_top1 計測
    - 6/1 当日 race で再計測
    - 旧 V18/V19 (sib抜き、 24.14%) と比較

6/2-6/5 (月-木)
  shift_factor 評価 (BT vs LIVE)
  3 週分 (5/30, 5/31, 6/1) 平均 winner_top1
  GO/no-go 判定 + V18/V19 v2 docs 更新

6/6-6/8 (金-日)
  V18/V19 v2 GO の場合: 6/15+ 段階投入準備 (paper trading 1 週間)
  NG の場合: V20 への直接 jump、 V18/V19 廃止判断
```

### 1.3 GO 条件 (V18/V19 v2 6/15+ 投入)

| # | 条件 | 必要値 |
|---|------|--------|
| 1 | sib_*_exp 6-fold WF AUC | ≥ 0.880 (V18/V19 元 0.8954/0.8787) |
| 2 | LIVE retro winner_top1 (3 週平均) | ≥ 30% (元 24.14%) |
| 3 | shift_factor | ≤ 12x (元 8.3x、 sib_*_exp で増えたら大問題) |
| 4 | feature LEAK 監査 | PASS (旧 sib_top3_rate 不在 assert) |
| 5 | 既存 V15 production 動作不変 | 必須 |

→ 5 PASS で 6/15+ V18/V19 v2 段階投入 (週末のみ、 上限 5,000円/日)

---

## 2. Phase 3 後半 (6/9-6/30): V20 構築

### 2.1 milestone

| milestone | 期日 | 達成基準 |
|----------|------|---------|
| JV-Link parser 完成 | 6/13 | RACE/HR/O1/TCOV/WOOD/BLOD の 6 parser、 1 か月分 bulk parse |
| V20 学習 data spec | 6/15 | JRA + NAR 統合 master、 共通 80 features (SKB 除外、 sib_*_exp 込み) |
| V20 v1 学習完了 | 6/20 | 4-model ensemble (LGB+XGB+FT+IR)、 Grid Search 重み確定 |
| V20 WF 検証完了 | 6/25 | 6-fold (2020-2025)、 全年 AUC > 0.85 |
| V20 LIVE retro | 6/27 | 6/27 当日 race で winner_top1 計測 |
| V20 paper trading | 6/28 | 6/27-28 weekend、 ROI 試算 |
| V20 GO/no-go 最終判定 | 6/30 | 6 条件 PASS で 7/1 投入 |

### 2.2 詳細 schedule

```
6/9-6/13 (月-金)
  tools/parse_jvlink_race.py 完成 (RACE record → DataFrame)
  tools/parse_jvlink_hr.py (払戻 HR record)
  tools/parse_jvlink_odds.py (O1 単勝/複勝)
  tools/parse_jvlink_tcov_wood.py (調教)
  tools/parse_jvlink_blod.py (血統)
  過去 1 年 bulk fetch (5/24/2025 〜 5/23/2026)
  既存 jra_races_full.csv との整合チェック (race_id / 着順 100 races sample)

6/14-6/20 (土-金)
  V20 学習 data spec 確定
    - JRA + NAR 統合 master (combined_v20_train.pkl)
    - 共通 features 80 件
    - JRA-only 50 件 (jrdb_kyi PRE_RACE + speed_index 等)
    - NAR-only 12 件 (NAR specific)
    - SKB 全 10 features 完全除外 (skip_skb=True 強制)
    - sib_top3_rate_exp / sib_shinba_wr_exp 込み
    - sample weight: JRA 70% / NAR 30%
  V20 v1 学習 (4-model ensemble、 Grid Search 重み最適化)
    - LGB Booster: AUC ≥ 0.86
    - XGB Booster: AUC ≥ 0.85
    - FT-Transformer: AUC ≥ 0.84
    - IntraRace Attention: AUC ≥ 0.84
    - Grid Search ensemble: AUC ≥ 0.880
  ローカル GPU で 2-3 時間想定 (CLAUDE.md 既知 spec)

6/21-6/25 (土-水)
  V20 WF 検証 (6-fold、 2020-2025)
  各年 AUC 確認 (全年 > 0.85)
  feature LEAK 監査 (V20_LEAK_FEATURES 全 18 件不在 assert)

6/26-6/28 (木-土)
  V20 LIVE retro (6/27 race)
  paper trading (6/27-28 weekend、 V20 推奨買い目をシミュレーション、 実投資なし)

6/29-6/30 (日-月)
  GO/no-go 最終判定
    - 6 条件 PASS → 7/1 V20 投入準備
    - 1 つでも NG → 7/15 延期 or V18/V19 v2 / V15 単独継続
  V20 model file 作成 (keiba_model_v20_central.pkl.gz / *_live.pkl.gz)
```

### 2.3 GO 条件 (V20 7/1 投入)

| # | 条件 | 必要値 |
|---|------|--------|
| 1 | WF AUC | ≥ 0.880 (V15 0.8939 比 -10〜30bp 想定) |
| 2 | LIVE retro winner_top1 (1-2 日) | ≥ 30% |
| 3 | shift_factor (BT vs LIVE) | ≤ 12x |
| 4 | NAR subset AUC | ≥ 0.83 (NAR v4 0.8145 と同等以上) |
| 5 | paper trading ROI | ≥ 110% (3 日 SUM、 戦略⑦込み) |
| 6 | feature LEAK 監査 | PASS (V20_LEAK_FEATURES 全 18 件不在) |

→ 6 PASS で 7/1 V20 投入、 失敗で 7/15 延期 or V15 単独継続。

---

## 3. V20 production deploy (7/1+)

### 3.1 段階投入

| 期間 | 投資制約 | 評価 |
|------|---------|------|
| 7/1-7/14 | 週末のみ、 上限 5,000円/日 | 2 週末 SUM ROI ≥ 110% で次 step |
| 7/15-7/31 | 週末 1 万円/日 + 平日 5,000円/日 | 月末 SUM ROI ≥ 130% で次 step |
| 8/1-8/31 | 平常運用 (V15 と同水準) | 月末 ROI ≥ 140% で V15 archive 判定 |
| 9/1+ | V21 投入候補 (V20 + 動画) | 別判定 |

### 3.2 fallback 戦略

V20 投入後の問題発生時:
- **軽微 (winner_top1 -3pt 未満)**: 引き続き V20、 監視継続
- **中度 (winner_top1 -3〜10pt)**: V20 投入停止、 V15 単独復帰、 V20 原因調査
- **重度 (winner_top1 -10pt 以上)**: V20 完全 rollback、 V15 production 復帰、 V20 廃止判断

### 3.3 並行運用 (1 か月)

7/1-7/31 は V15 と V20 で並行運用:
- daily_predict.py 両 model で予測実行
- production: V20 採用、 V15 は comparison ログ
- 7/31 時点で V20 ≥ V15 確認後、 V15 production 経路 archive

---

## 4. Phase 4 (7-8 月): 動画解析 PoC

### 4.1 milestone

| milestone | 期日 | 達成基準 |
|----------|------|---------|
| データ蓄積完了 | 7/14 | 50 レース × 30 動画 = 1,500 動画 |
| 馬体検出動作確認 | 7/21 | YOLOv8 で horse class precision ≥ 80% |
| 姿勢推定動作確認 | 7/31 | DLC SuperAnimal zero-shot で keypoint 12 points × 80% 検出 |
| 特徴量抽出完了 | 8/15 | VIDEO_FEATURES 10 列、 全動画から欠損 < 30% |
| V21 学習完了 | 8/31 | WF AUC ≥ V20 + 0.005 (= 0.885+) |
| V21 投入判定 | 9/1 | LIVE retro winner_top1 ≥ V20 + 1pt |

### 4.2 詳細 schedule

```
7/1-7/14 (火-月、 V20 投入 + Phase 4 並行)
  JRA-VAN ネクスト 加入 (+1,000円/月)
  動画蓄積:
    - 直近 4 週分の調教動画 (50 レース × 30 動画 = 1,500)
    - データ source: JRA-VAN ネクスト + netkeiba (Premium)
    - 利用規約 確認 (個人 PoC 範囲) 済前提
  Colab Pro 加入判断 (+1,178円/月)、 ローカル GPU 不足時のみ

7/15-7/31 (火-木)
  YOLOv8 動作確認:
    - tools/video_ai/detect_horse.py 試作
    - frame 5 fps サンプリング
    - horse bbox 検出 (zero-shot, COCO horse class)
    - 精度測定 (sample 100 frames で precision/recall)
  DLC SuperAnimal 動作確認:
    - tools/video_ai/pose_estimation.py 試作
    - keypoint 12 points (鼻 / 耳 / 肩 / 尻 / 4 脚 / 蹄)
    - zero-shot 推論 → 精度 70-85% 想定
    - 70% 未満なら DLC fine-tune (HORSE-10 pretrained + 自前 50 動画)

8/1-8/15 (金-金)
  時系列特徴量抽出:
    - tools/video_ai/extract_features.py
    - VIDEO_FEATURES 10 件:
      1. video_stride_freq (歩幅頻度)
      2. video_gait_symmetry (歩様左右対称性)
      3. video_head_bobbing_amp (頭振り = 跛行 indicator)
      4. video_ear_avg_y (耳の位置 = 集中度)
      5. video_posture_score (姿勢)
      6. video_acceleration_rate (加速性能)
      7. video_muscle_definition_score (筋肉張り)
      8. video_coat_glossiness_score (毛艶)
      9. video_balance_score (バランス)
      10. video_concentration_score (集中度)
    - 馬単位集計 + expanding window (直近 N 日 = 7-14 日 平均)

8/16-8/31 (土-日)
  V21 学習:
    - V20 features (~142) + VIDEO_FEATURES (10) = ~152 features
    - 4-model ensemble (LGB+XGB+FT+IR) 同構成
    - WF 検証 (6-fold)、 全年 AUC ≥ 0.88
  V21 LIVE retro (8/30-31 weekend)
  paper trading

9/1
  V21 投入判定:
    - WF AUC ≥ V20 + 0.005 (= 0.885+) AND
    - LIVE retro winner_top1 ≥ V20 + 1pt
  GO なら 9/2+ 段階投入、 NG なら V20 単独継続 (Phase 4 PoC データ蓄積継続)
```

### 4.3 GO 条件 (V21 9/1 投入)

| # | 条件 | 必要値 |
|---|------|--------|
| 1 | WF AUC | ≥ V20 + 0.005 (= 0.885 想定) |
| 2 | LIVE retro winner_top1 | ≥ V20 + 1pt |
| 3 | VIDEO_FEATURES の信号強度 | importance top 50 入り |
| 4 | feature LEAK 監査 | PASS (pre-race のみ) |
| 5 | paper trading ROI | ≥ V20 + 5pt |

→ 5 PASS で 9/2+ V21 段階投入、 NG なら Phase 4 NO-GO 判定 (PoC 失敗扱い)

---

## 5. 投資保護 + 撤退戦略 (絶対遵守)

### 5.1 撤退ライン (5/9 以降不変)

| 累計収支 | 状態 | アクション |
|---------|------|----------|
| ≥ 0 | 順調 | 計画通り進行 |
| -10,000 ≦ x < 0 | 注意 | 翌週投資停止、 原因調査 |
| -50,000 ≦ x < -10,000 | 警告 | 全停止、 V20/V21 投入延期 |
| < -50,000 | **撤退** | 完全停止、 全 model 廃止判断 |

現状 (5/7): +13,530 円 → 撤退余裕 +63,530 円 ※ 当時 record、 5/16 P0-1 真値: **+¥5,240** / 撤退余裕 **+¥55,240** (docs/ROI_DISCREPANCY_2026_05_16.md)

### 5.2 段階別 投資上限

| 期間 | 主 model | 投資上限/日 | 想定 ROI | 月利 |
|------|---------|------------|---------|------|
| 5/9-5/15 | V15 案B改 | 2,100円 (700 × 3R) | 161% | +400-1,300円 |
| 5/16-6/14 | V15 単独継続 | 平常 (15,000円/日) | 119-140% | +2-3 万 |
| 6/15-6/30 | V15 + V18/V19 v2 (GO 時) | V18/V19 上限 5,000円/日 | 140% (合計) | +3-4 万 |
| 7/1-7/14 | V20 (段階投入) | 5,000円/日 | 130% (paper) | +5-7 万 |
| 7/15-7/31 | V20 (拡大) | 1.5 万円/日 | 140% | +6-9 万 |
| 8/1-8/31 | V20 (平常) | 平常 | 145% | +7-11 万 |
| 9/2+ | V21 (V20 + 動画、 GO 時) | 平常 | 150% | +8-13 万 |

### 5.3 fallback 階層

```
V21 NG (9/1)
  ↓
V20 単独継続 (Phase 4 NO-GO で PoC データ蓄積継続)

V20 NG (6/30)
  ↓
V18/V19 v2 (sib_*_exp、 6/15 GO 時) + V15 並行
  ↓ (V18/V19 v2 も NG なら)
V15 単独継続 (Phase 3 NO-GO、 7 月以降も V15)

V15 重度問題 (winner_top1 -10pt 等)
  ↓
全停止、 撤退判断 + 原因調査
```

→ 全 path で V15 単独 fallback 確保、 撤退余裕 +63,530 円維持。

---

## 6. 月次 milestone + decision point

| 月 | 主要 decision | 影響 |
|----|--------------|------|
| 5月 | 5/9 V15 案B改 投資 (確定) | +400 - +1,300円 想定 |
| 5月末 | JRA-VAN 加入 (5/24) | +月額 2,090円、 V20 構築開始 |
| 6月初 | V18/V19 v2 GO/no-go (6/8) | GO で 6/15+ 段階投入 |
| 6月末 | V20 GO/no-go (6/30) | GO で 7/1 投入 |
| 7月末 | V20 拡大 → 平常運用 (7/31) | V15 archive 判定 |
| 8月末 | V21 学習完了 (8/31) | 9/1 投入判定 |
| 9月初 | V21 GO/no-go (9/1) | GO で 9/2+ 段階投入 |

---

## 7. リスク + mitigation

| # | risk | 確率 | impact | mitigation |
|---|------|-----|--------|----------|
| 1 | JV-Link 加入後の DLL trouble | 中 | Phase 3 後半遅延 | 5/24 即試行、 pywin32 venv 分離、 NG なら netkeiba scraping 継続 |
| 2 | sib_*_exp WF AUC 改善 < 期待 | 中 | V18/V19 v2 NO-GO | V20 で sib_*_exp 維持、 補助 feature 扱い |
| 3 | V20 学習 data 整合性問題 | 中 | V20 6/30 NO-GO | 5/27 早期整合チェック、 不整合検出時 6/9 開始延期 |
| 4 | V20 6 GO 条件 NG | 中 | 7/15 延期 or V15 単独 | V15 単独継続で +月利 2-3 万円維持 |
| 5 | Phase 4 動画 source 規約抵触 | 低 | Phase 4 中止 | 7/1 着手前 規約確認 (個人 PoC 範囲) |
| 6 | Phase 4 姿勢推定精度 < 70% | 中 | V21 NG | DLC fine-tune fallback、 ResNet50 end-to-end fallback |
| 7 | V20 / V21 投入後 winner_top1 -10pt | 中 | 完全 rollback | V15 production 経路保持、 即 fallback |
| 8 | 累計収支 -50,000 円 到達 | 低 | 完全撤退 | 各 phase 段階投入で risk 分散、 累計監視 daily |

→ 全 risk に mitigation あり、 V15 単独 fallback で資産保護確実。

---

## 8. ユーザー (れんはす) 関与 step

| 期日 | アクション |
|------|----------|
| 5/24 (土) 朝 | JRA-VAN DataLab 加入 (1h) |
| 5/24 (土) 夜 | jvlink_fetcher.py 動作確認 (Claude 連携) |
| 6/15 | V18/V19 v2 投入判断 (GO 時) |
| 6/30 | V20 投入判断 |
| 7/1 (火) | V20 production deploy + Phase 4 PoC 着手 |
| 7/15 | V20 拡大 + JRA-VAN ネクスト 加入判断 |
| 9/1 | V21 投入判断 |

→ 仕事の合間で進捗確認、 重要 decision は週末に判断。

---

## 9. Session #39 deliverable

本 roadmap を支える Session #39 (5/7) の 10 領域:

| # | 領域 | deliverable |
|---|------|-----------|
| A | sib expanding window | tools/sib_expanding_features.py + design + PoC 動作 |
| B | JV-Link 統合 | tools/jvlink_fetcher.py + plan |
| C | SKB 完全除外 | train/v15_1_features.py patch (filter_v15_1_features) |
| D | 全 4 source 役割分担 | docs/PHASE_3_DATA_SOURCE_STRATEGY.md |
| E | V20 architecture 更新 | docs/PHASE_3_V20_DETAILED_DESIGN.md §17-18 |
| F | Phase 4 動画解析 PoC | docs/PHASE_4_VIDEO_AI_DESIGN.md |
| G | 馬体検出 + 姿勢推定 技術調査 | docs/PHASE_4_TECH_RESEARCH.md |
| H | CLAUDE.md 全面刷新 | CLAUDE.md (1339 行) |
| I | README.md V20 構想反映 | README.md |
| J | 統合 roadmap | **本 doc** |
| K | 11 commits + push + Discord | Session #39 11 commits |

→ 5/24+ Phase 3 着手、 全 deliverable 即着手可能状態。

---

## 10. 結論

✅ Phase 3-4 統合 roadmap (5/24-9/1) 確定
✅ V20 (7/1) + V21 (9/2+) 投入 schedule 明確
✅ GO 条件 / fallback / 撤退ライン 全完備
✅ 月額コスト約 1万円、 月利 +5-13 万円 想定
✅ V15 単独 fallback で資産保護確実
✅ ユーザー関与 step 7 件 (週末 メイン)

→ **Phase 3-4 即着手可能状態、 5/24 加入で開始**

---

**Session #39 J 完了**
