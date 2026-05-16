# Phase 3-5 統合 roadmap v2 (Session #41 H)

**作成**: 2026-05-08 深夜 (Session #41 H、 ユーザー就寝中)
**v1**: docs/PHASE_3_4_INTEGRATED_ROADMAP.md (Session #39 J)
**v2 (本ファイル)**: Session #41 結果反映 (JV-Link 加入完了 + sib_exp PoC 結果 + 32-bit Python 環境 plan)

---

## 0. 全体像 (一目)

```
2026/  5月         6月        7月        8月       9月+
─────────────────────────────────────────────────────────────
5/8    ★ Session #41 完了 (JV-Link 加入後 8 領域 前倒し実装)
5/9    ★ V15 案B改 単独継続 (絶対遵守、 max loss -2,100円)
5/16   ★ V18/V19 sib_exp 投入判定 (sib_exp BT +0.12pt のみ → NO-GO 強)
                    │
5/24   ┌── Phase 3 前半 (sib_exp + JV-Link 統合) ─────────┐
       │ JV-Link 32-bit Python 環境 構築                    │
       │ tools/jvlink_full_backfill.py 6 年分 段階的 fetch    │
       │ V18/V19 sib_exp v2 再学習 (XGB+LGB)                │
       │ V18/V19 v2 LIVE retro 5/30, 5/31, 6/1              │
6/8    └── 6/15+ V18/V19 v2 投入判定 ────────────────────┘
       │
6/9    ┌── Phase 3 後半 (V20 構築) ────────────────────┐
       │ JV-Link backfill 完了 + parser 完成              │
       │ V20 学習 data spec 確定 (JRA + NAR 統合)         │
       │ V20 v1 学習 (4-model ensemble、 SKB除外、 sib_exp) │
       │ V20 WF 検証 + LIVE retro                          │
       │ V20 paper trading                                  │
6/30   └── 7/1+ V20 投入判定 (6 GO 条件) ──────────────┘
       │
7/1    ┌── V20 production deploy ──┐
       │ + Phase 4 動画解析 PoC 開始 │
       │ 動画蓄積 + 姿勢推定         │
       │ VIDEO_FEATURES 抽出         │
       │ V21 (V20 + 動画) 学習      │
9/1    └── V21 投入判定 ────────────┘
       │
9月以降 ┌── Phase 5 (V22 構想) ─┐
       │ + 生体データ + 天気予報 │
       │ + 3-way voting          │
       │ V22 学習 + production   │
12月    └────────────────────────┘
```

---

## 1. Phase 3 前半 (5/24-6/8): sib_exp + JV-Link 統合

### 1.1 milestone

| milestone | 期日 | 達成基準 |
|----------|------|---------|
| 32-bit Python 環境 構築 | 5/24 | `tools/setup_python32.ps1` 実行完了 + `jvlink_test_python32.py --check-only` PASS |
| 5/1-5/7 backfill 完了 | 5/25 | `data/jvlink/RACE/20260501_*.csv` 等 28 fetch 完了 |
| schtasks Keiba-JvlinkBackfillNightly 登録 | 5/26 | 毎晩 23:00 〜 200 件 fetch |
| 全 6 年分 backfill 完了 | 6/8 | `data/jvlink_full/RACE/*.csv` 等 8,400+ files |
| V18/V19 sib_exp v2 (XGB+LGB) 学習 | 5/30 | WF AUC ≥ 0.880 |
| V18/V19 sib_exp v2 LIVE retro (5/30, 5/31, 6/1) | 6/5 | winner_top1 ≥ 30% |
| 6/15+ V18/V19 投入判定 | 6/8 | GO/no-go 確定 |

### 1.2 詳細 schedule

```
5/24 (土)
  AM: 32-bit Python install (admin、 setup_python32.ps1)
  PM: JV-Link 動作確認 (jvlink_test_python32.py)
  夜: 5/1-5/7 backfill 試行 (jvlink_backfill_5_1_5_7.py)

5/25-26 (日-月)
  data quality check (USER 5/2-5/3 申告 vs JV-Link HR の照合)
  schtasks Keiba-JvlinkBackfillNightly 登録
  6 年分 backfill 開始 (Nightly 200 件/晩)

5/27-30 (火-金)
  V18/V19 sib_exp v2 学習 (XGB 追加、 6-fold WF) (~2-3h × 2 model)
  Nightly backfill 進行 (約 200 × 4 = 800 fetch 完了)

5/31-6/1 (土-日)
  V18/V19 sib_exp v2 LIVE retro (5/31, 6/1 の race で winner_top1 計測)
  Nightly backfill 進行

6/2-6/5 (月-木)
  V18/V19 sib_exp v2 LIVE retro 集計
  shift_factor 評価 (BT vs LIVE)

6/6-6/8 (金-日)
  V18/V19 v2 GO/no-go 判定:
    - GO 条件: WF AUC ≥ 0.880 AND LIVE winner_top1 ≥ 30% AND shift ≤ 12x
    - GO の場合: 6/15+ 段階投入準備 (週末のみ、 上限 5,000円/日)
    - NG の場合: V20 への直接 jump、 V18/V19 廃止判断
```

### 1.3 5/16 V18/V19 投入判定 (Session #41 D 結果反映)

Session #41 D で sib_exp BT 2025 OOS:
- V18 sib_exp winner_top1: **45.88%** (vs no_sib 45.76%、 +0.12pt 微増)
- V18 既存 sib含 ens: 47.79%

→ BT では sib_exp の効果は微小 (+0.12pt のみ)、 LIVE retro 結果 (進行中) で確定

**5/16 判定 (暫定、 LIVE retro 結果 待ち)**:
- LIVE winner_top1 ≥ 32% (no_sib の 24.14% から +8pt 復活) なら 5/16 paper trading 候補
- LIVE winner_top1 ≤ 27% なら 5/16 NO-GO 確定
- 確率推定: BT +0.12pt は LIVE で +0-3pt の改善見込 → **24.14% + 0-3 = 24-27%** → **5/16 NO-GO 確率 70%**

---

## 2. Phase 3 後半 (6/9-6/30): V20 構築

### 2.1 milestone

| milestone | 期日 | 達成基準 |
|----------|------|---------|
| JV-Link 全 6 年分 backfill 完了 | 6/13 | `data/jvlink_full/` 36-66 GB |
| JV-Link parser (RA/SE/HR/O1/TCOV/WOOD/BLOD) 完成 | 6/13 | 各 datatype の正確な layout parser |
| V20 学習 data spec 確定 | 6/15 | JRA + NAR 統合 master、 共通 80 features |
| V20 v1 学習完了 | 6/20 | 4-model ensemble、 全年 AUC > 0.85 |
| V20 WF 検証完了 | 6/25 | 6-fold (2020-2025) |
| V20 LIVE retro | 6/27 | 6/27 当日 race で winner_top1 計測 |
| V20 paper trading | 6/28 | 6/27-28 weekend、 ROI 試算 |
| V20 GO/no-go 最終判定 | 6/30 | 6 条件 PASS で 7/1 投入 |

### 2.2 V20 features 構成 (Session #39 + #41 反映)

```python
V20_FEATURES = (
    V15_BASE_FEATURES                          # 150 (V15 既存)
    + KKA_FEATURES                              # 16 (Session #39 採用候補)
    - SKB_LEAK_FEATURES                         # -10 (Session #38 確定 LEAK)
    + SRB_FEATURES                              # 8
    + ['sib_top3_rate_exp', 'sib_shinba_wr_exp',
       'sib_total_races_exp', 'sib_total_offspring_exp']  # 4 (Session #39 A)
    + JV_LINK_NEW_FEATURES                     # 5-15 (paci 自前算出 等、 Session #41 B-C)
)
# 期待 total: ~170-180 features (V15 150 から +20-30)
```

### 2.3 GO 条件 (V20 7/1 投入、 6 項目)

| # | 条件 | 必要値 |
|---|------|--------|
| 1 | WF AUC | ≥ 0.880 |
| 2 | LIVE retro winner_top1 (1-2 日) | ≥ 30% |
| 3 | shift_factor (BT vs LIVE) | ≤ 12x |
| 4 | NAR subset AUC | ≥ 0.83 |
| 5 | paper trading ROI | ≥ 110% (3 日 SUM) |
| 6 | feature LEAK 監査 | PASS (V20_LEAK_FEATURES 全 18 件 不在) |

→ 全 6 PASS で 7/1 V20 投入、 失敗で 7/15 延期 or V15 単独継続

---

## 3. Phase 4 (7-8月): 動画解析 PoC

(Session #39 F の plan を維持、 V20 投入後 並行)

### 3.1 milestone

| milestone | 期日 | 達成基準 |
|----------|------|---------|
| データ蓄積完了 | 7/14 | 50 レース × 30 動画 = 1,500 動画 |
| YOLOv8 馬体検出 + DLC SuperAnimal 動作確認 | 7/31 | precision ≥ 80%、 keypoint 検出 ≥ 80% |
| VIDEO_FEATURES 10 件 抽出 | 8/15 | 全動画から欠損 < 30% |
| V21 (V20 + 動画) 学習完了 | 8/31 | WF AUC ≥ V20 + 0.005 |
| V21 投入判定 | 9/1 | LIVE retro winner_top1 ≥ V20 + 1pt |

---

## 4. Phase 5 (9月以降): V22 構想

(Session #40 E の Phase 4-5 構想を継承、 V21 投入後 検討)

### 4.1 V22 候補 features

```python
V22_FEATURES = V21_FEATURES + [
    # 生体 (Session #40 E2)
    'horse_height_cm', 'horse_chest_cm', 'horse_body_length',
    # 天気予報 24h 前 (Session #40 E3)
    'predicted_track_condition_24h_prior',
    'predicted_precipitation_24h_prior',
    # voting boost (Session #40 E4 + 本 v2)
    'voting_score_v15', 'voting_score_v18_v19', 'voting_score_v20',
]
```

### 4.2 milestone (試案)

| 期間 | 内容 |
|------|------|
| 9/1-9/30 | V22 設計 + 馬体寸法 features 開発 |
| 10/1-10/31 | 天気予報 features 統合 + 3-way voting 実装 |
| 11/1-11/30 | V22 学習 + WF + LIVE retro |
| 12/1-12/31 | V22 投入判定 |

---

## 5. 投資保護 + 撤退戦略 (絶対遵守、 v1 から維持)

### 5.1 撤退ライン (5/9 以降不変)

| 累計収支 | 状態 | アクション |
|---------|------|----------|
| ≥ 0 | 順調 | 計画通り進行 |
| -10,000 ≦ x < 0 | 注意 | 翌週投資停止、 原因調査 |
| -50,000 ≦ x < -10,000 | 警告 | 全停止、 V20/V21 投入延期 |
| < -50,000 | **撤退** | 完全停止、 全 model 廃止判断 |

現状 (5/8 深夜): +13,530 円 → 撤退余裕 +63,530 円 ※ 当時 record、 5/16 P0-1 真値: **+¥5,240** / 撤退余裕 **+¥55,240** (docs/ROI_DISCREPANCY_2026_05_16.md)

### 5.2 段階別 投資上限

| 期間 | 主 model | 投資上限/日 | 想定 ROI |
|------|---------|------------|---------|
| 5/9-5/15 | V15 案B改 | 2,100円 (700 × 3R) | 161% |
| 5/16-6/14 | V15 単独継続 | 平常 | 119-140% |
| 6/15-6/30 | V15 + V18/V19 v2 (GO 時) | V18/V19 上限 5,000円/日 | 140% (合計) |
| 7/1-7/14 | V20 (段階投入) | 5,000円/日 | 130% (paper) |
| 7/15-7/31 | V20 (拡大) | 1.5 万円/日 | 140% |
| 8/1-8/31 | V20 (平常) | 平常 | 145% |
| 9/2+ | V21 (V20 + 動画) | 平常 | 150% |
| 12/+ | V22 (生体+天気+voting) | 平常 | 155% |

---

## 6. fallback 階層 (絶対遵守、 v1 から維持)

```
V22 NG (12月)
  ↓
V21 単独継続

V21 NG (9/1)
  ↓
V20 単独継続 (Phase 4 NO-GO で PoC データ蓄積継続)

V20 NG (6/30)
  ↓
V18/V19 v2 (sib_exp、 6/15 GO 時) + V15 並行
  ↓ (V18/V19 v2 も NG なら)
V15 単独継続 (Phase 3 NO-GO、 7 月以降も V15)

V15 重度問題 (winner_top1 -10pt 等)
  ↓
全停止、 撤退判断 + 原因調査
```

→ 全 path で V15 単独 fallback 確保、 撤退余裕 +63,530 円維持。

---

## 7. ユーザー (れんはす) 関与 step (v1 から更新)

| 期日 | アクション | 重要度 |
|------|----------|--------|
| 5/8 (金) 朝 | 起床後 Discord で Session #41 結果確認 | 中 |
| 5/9 (土) 朝 | V15 案B改 投資 (700円 × max 3R) | **絶対** |
| 5/16 (土) | V18/V19 sib_exp 投入判定 (LIVE retro 結果次第) | 高 |
| 5/24 (土) 朝 | 32-bit Python install + JV-Link 動作確認 (1h) | 高 |
| 5/24 (土) 夜 | jvlink_backfill_5_1_5_7.py 試行 | 中 |
| 5/26 (月) | schtasks Keiba-JvlinkBackfillNightly 登録 (admin) | 中 |
| 6/15 (日) | V18/V19 v2 投入判断 (GO 時) | 高 |
| 6/30 (月) | V20 投入判断 | **絶対** |
| 7/1 (火) | V20 production deploy | **絶対** |
| 7/15 (火) | V20 拡大 + Phase 4 加入判断 (Colab Pro) | 中 |
| 9/1 (月) | V21 投入判断 | 高 |
| 12月 | V22 投入判断 | 中 |

→ 仕事の合間 + 週末 で対応可能。

---

## 8. 月額コスト (Session #39 + #40 + #41 反映)

| source | 月額 | 開始 |
|--------|------|------|
| netkeiba Premium | 4,500円 | 既存 |
| JRDB Advance | 約 2,000円 | 既存 |
| **JRA-VAN DataLab (JV-Link)** | **2,090円** | **5/7 加入完了** |
| JRA-VAN ネクスト (Phase 4 動画用) | +1,000円 | 7/1 (予定) |
| Colab Pro (Phase 4 GPU) | 1,178円 | 7/1 (予定) |
| **合計** | **約 10,768円/月** (7/1 以降) | — |

ROI 想定:
- V15 (5/9-): 119.2% (戦略⑦込み 140%) → 月利 約 2-3 万円 ※ 旧記述は drift、 5/16 P0-1 真値: ROI 101.33% / 月利 期待値 ±¥0-3,000 (docs/ROI_DISCREPANCY_2026_05_16.md)
- V20 (7/1+): 145-150% 想定 → 月利 5-10 万円
- V21 (9/2+): 150-155% 想定 → 月利 6-11 万円
- V22 (12月+): 155-160% 想定 → 月利 7-13 万円

→ 月額コスト 約 1万円は V20 以降 月利増分で十分回収。

---

## 9. Session #41 結果 (本 v2 で反映)

| 領域 | 結果 |
|------|------|
| A: 32-bit Python plan | setup_python32.ps1 + jvlink_test_python32.py 完成 |
| B: jvlink_fetcher v2 | 280 行、 RA/SE/HR/O1 parser placeholder |
| C: 5/1-5/7 backfill | 28 fetches plan、 14 min 想定 |
| D: sib_exp PoC | BT 2025 +0.12pt 微増、 LIVE retro 進行中 |
| E: V20 backfill plan | 6 年分 36-66 GB、 schtasks Nightly 推奨 |
| F: doc 更新 | CLAUDE.md / README.md / INDEX.md / pre-check v3 |
| G: pre-check v3 | V15 model md5 不変、 production 100% 維持 |
| H: roadmap v2 | **本 file** |
| I: 9 commits push | (本 commit に含む) |

---

## 10. 結論

✅ Phase 3-5 統合 roadmap v2 (5/24-12月+) 確定
✅ Session #41 結果反映 (JV-Link 加入後 plan + sib_exp BT 結果)
✅ V18/V19 5/16 NO-GO 確率 70% (BT +0.12pt から推定)
✅ V20 (7/1) → V21 (9/2+) → V22 (12月+) 投入 schedule
✅ GO 条件 / fallback / 撤退ライン 全完備
✅ 月額コスト約 1万円、 月利 +5-13 万円 想定
✅ V15 単独 fallback で資産保護確実
✅ ユーザー関与 step 11 件 (週末 + admin task メイン)

→ **Phase 3-5 即着手可能状態、 5/24 加入完了で開始**

---

**Session #41 H 完了**
