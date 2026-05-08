# AUDIT-1 I: 推奨 implementation roadmap (5/8)

**作成**: 2026-05-08 (AUDIT-1 I 領域)
**前提**: E (Top 30) を 短期 / 中期 / 長期 に分類
**位置付け**: Sprint 4 / V20 / V21 / Phase 4 への path 整理

---

## 0. 全体 schedule (5/8 起点)

| Phase | 期間 | model | 期待 AUC | features 件数 |
|-------|------|------|---------|------------|
| 現状 | -5/8 | V15 | 0.8858 (BT 0.8939) | 150 |
| Sprint 4 | 5/12-5/22 (週末除く) | V15.5 | 0.892-0.896 | 156-160 |
| V20 候補 | 5/16-6/8 | V20 | 0.890-0.900 | 165-180 |
| 投入 | 6/8 | V20 production | 0.890+ | -- |
| V20.5 | 6/9-6/30 | V20.5 | 0.895-0.905 | 180-200 |
| Phase 4 開始 | 7/1-8/31 | V21 PoC | -- | + 動画 5-7 件 |
| V21 投入 | 9/1 | V21 production | 0.900-0.910 | 185-205 |

---

## 1. 短期 (1 週間以内、 5/12-5/19) — Sprint 4 候補

### 1.1 投入 features (★★★ + ★★ の 上位)

| # | feature | source | 工数 | 期待 AUC |
|---|---------|--------|------|---------|
| 1 | jrdb_srb bias 6 件 | JRDB SRB | 4h | +0.003-0.005 |
| 2 | netkeiba master_index 5 件 | netkeiba マスター | 6h | +0.003-0.005 |
| 3 | jrdb_jo cid_idx / ls_idx 2 件 | JRDB JO | 2h | +0.002-0.003 |
| 4 | netkeiba speed_index dist / course 2 件 | netkeiba | 1h | +0.001-0.003 |
| 5 | netkeiba ai_opinion pace 1 件 | netkeiba マスター | 2h | +0.001-0.002 |
| **合計** | **16 件** | -- | **15h** | **+0.010-0.018** |

### 1.2 検証 step

1. 各 features を v15 base に 追加 (15-30 行 / feature)
2. WF 6-fold (2020-2025) で AUC + gap < 0.05 確認
3. リーク check (corr_target < 0.10、 1着馬 vs 敗者 monotonic check)
4. 採用 features で V15.5 model 学習
5. LIVE retro (5/24 + 5/25 開催の 36 R) で winner_top1 ≥ V15 + 1pt 確認
6. 6/1 次回開催 で 投入 候補

### 1.3 V15.5 投入 判定

採用 基準:
- WF AUC > 0.886 (V15 baseline)
- 全年 AUC > 0.85
- max gap < 0.05
- LIVE retro winner_top1 ≥ V15 + 1pt
- shift ≤ 12x

---

## 2. 中期 (1 ヶ月以内、 5/22-6/8) — V20 統合候補

### 2.1 投入 features (★★ 残り)

| # | feature | source | 工数 | 期待 AUC |
|---|---------|--------|------|---------|
| 6 | jrdb_kka kyori/track/heavy_seiseki | JRDB KKA | 6h | +0.002-0.005 |
| 7 | jrdb_cha oikiri_rank / idx / 3 time | JRDB CHA | 4h | +0.002-0.005 |
| 8 | jrdb_jo gaisha_bb / breeder_bb 6 件 | JRDB JO | 3h | +0.002-0.003 |
| 9 | jrdb_cyb train_mark / eval / amount 5 件 | JRDB CYB | 3h | +0.001-0.003 |
| 10 | TFJV BS_DATA breeder_top3r | TFJV BS | 8h | +0.002-0.004 |
| 11 | TFJV OW_DATA owner_top3r | TFJV BN | 8h | +0.002 |
| 12 | TFJV BR_DATA dam 拡張 (sib_*_ext) | TFJV BR | 6h | +0.001-0.003 |
| 13 | netkeiba ai_position 位置取り | netkeiba マスター | 2h | +0.001-0.002 |
| 14 | jrdb_tyb cancel_flag / bagu_change | JRDB TYB | 2h | +0.001-0.002 |
| 15 | jrdb_sed time_sec / first_3f / last_3f | JRDB SED | 3h | +0.001-0.003 |
| 16 | netkeiba race_analysis 馬別 score | netkeiba マスター | 2h | +0.001-0.002 |
| 17 | jrdb_ukc keito_code / owner_code | JRDB UKC | 3h | +0.001-0.003 |
| **合計** | **30+ 件 (KKA は 12 group ×4 = 48)** | -- | **50h** | **+0.013-0.030** |

### 2.2 V20 学習構築

期待 V20 AUC: 0.890-0.900 (V15 0.886 + 上記 累計 +0.005-0.014)
NAR + JRA 統合 (共通 80 features)、 SKB 完全除外、 sib_*_exp 込み

### 2.3 V20 投入条件 (Phase 3 v3、 6/8)

- WF AUC ≥ 0.880
- LIVE retro ≥ 30%
- shift ≤ 12x
- NAR AUC ≥ 0.83
- paper trading ROI ≥ 110%
- LEAK 監査 PASS
- 段階投入: 週末のみ、 上限 5,000円/日

---

## 3. 長期 (3 ヶ月以内、 6/8-9/1) — V20.5 + Phase 4

### 3.1 V20.5 (6/9-6/30) - 残 ★ features

| # | feature | source | 工数 | 期待 AUC |
|---|---------|--------|------|---------|
| 18 | netkeiba data_analysis / race_tendency | netkeiba マスター | 2h | +0.0005-0.001 |
| 19 | netkeiba upset_level / top_pop_reliability | netkeiba | 1h | +0.0003-0.001 |
| 20 | netkeiba race_review prev_review_score (再検討) | netkeiba | 2h | +0.0001-0.001 |
| 21 | netkeiba stable_comment_score | netkeiba | 1h | +0.0003-0.001 |
| 22 | jrdb_kz/cz year_leading 系 | JRDB | 3h | +0.0005-0.001 |
| 23 | netkeiba training_times rank A/B/C/D | netkeiba | 1h | +0.001 |
| 24 | netkeiba shinba_eval (新馬戦のみ) | netkeiba | 4h | low (新馬戦 のみ) |
| 25 | TFJV WF (WIN5) appearance_count | TFJV W5 | 4h | +0.0005-0.001 |
| 26 | TFJV TM_DATA 直 利用 | TFJV TM | 6h | unknown |
| 27 | jrdb_jo soten_odds / yoso_odds | JRDB JO | 2h | +0.001 (リーク check 必要) |
| 28 | TFJV JG (取消) リアルタイム | TFJV JG | 6h | live 改善 |
| **合計** | **+15 件** | -- | **30h** | **+0.005-0.015** |

V20.5 期待 AUC: V20 (0.895) + 0.005-0.015 = 0.900-0.910

### 3.2 Phase 4 (7/1-8/31) - 動画 PoC

| 期間 | 内容 |
|------|------|
| 7/1-7/14 | データ蓄積 (JRA-VAN ネクスト + netkeiba 動画 50 race × 30 動画) |
| 7/15-7/31 | YOLOv8 動作確認 + DLC SuperAnimal zero-shot |
| 8/1-8/15 | 時系列 features 抽出 (stride / gait / posture / 5 件) |
| 8/16-8/31 | DLC fine-tune (HORSE-10 ベース) + V21 学習 |

V21 候補 features:
- パドック静止画 体格 score (1)
- 歩様 stride length / freq (2)
- gait_symmetry (1)
- head_bobbing / posture (2)
- 緊張度 (1)
合計 +5-7 件、 期待 AUC +0.005-0.010

### 3.3 V21 投入 (9/1)

V21 期待 AUC: V20.5 (0.905) + 0.005-0.010 = **0.910-0.915**

---

## 4. NAR 並行 (V20 後)

| 期間 | 作業 |
|------|------|
| 5/9-6/8 | (NAR 据え置き、 V20 集中) |
| 6/9-7/1 | NAR V5 開発 (sib_*_exp + 場別 features + LightGBM + XGB ensemble) |
| 7/1+ | NAR V5 投入候補 |

NAR V5 期待 AUC: 0.8145 → 0.84-0.85

---

## 5. roadmap 並列 view

```
2026-05-08 (今)
  ├─ 5/9 V15 案B改 維持 (絶対不変、 12R 1勝クラス上限 2,100円)
  ├─ 5/12-5/19 Sprint 4 (V15.5 候補 16 features 検証)
  ├─ 5/16-5/22 V20 学習 data 構築 + WF 6-fold
  ├─ 5/22-6/8 V20 統合 + LIVE retro + paper trading
  ├─ 6/8 V20 投入 候補 (週末のみ 5,000円/日)
  ├─ 6/9-6/30 V20.5 拡張 + NAR V5 開始
  ├─ 7/1 Phase 4 開始 (動画 PoC)
  ├─ 7/1+ JRA-VAN ネクスト + Colab Pro
  ├─ 7/15-8/31 動画 features 学習
  └─ 9/1 V21 投入 候補

撤退ライン:
  累計 -50,000円 (現在 +13,530円、 撤退余裕 +63,530円)
```

---

## 6. 期待 ROI 試算

| Phase | 月利 (推定) |
|-------|----------|
| V15 (現状、 戦略⑦込み) | +2-3 万円 (ROI 119-140%) |
| V15.5 (5/19-6/8) | +3-4 万円 (ROI 130-145%) |
| V20 (6/8+) | +5-10 万円 (ROI 145-150%) |
| V20.5 (6/9+) | +6-11 万円 (ROI 150%+) |
| V21 (9/1+) | +7-13 万円 (ROI 145-155%) |

月額コスト 1万円 (Premium 4,500 + JRDB 2,880 + JV-Link 2,090 + JRA-VAN ネクスト 1,000 + Colab Pro 1,178) は V20 以降の月利増分で十分回収。

---

## 7. 5/9 V15 投資保護

✅ 5/9 V15 案B改 維持 絶対 (12R 1 勝クラス 上限 2,100 円)
✅ 5/12 開始 Sprint 4 は dev branch で 別作業、 V15 production 完全不変
✅ V15.5 投入は 早くても 6/1 開催から (LIVE retro 必須、 段階投入)

---

## 8. 結論

✅ Sprint 4 (短期) 16 features = +0.010-0.018、 工数 15h
✅ V20 (中期) 30+ features = +0.013-0.030、 工数 50h
✅ V20.5 (長期) +15 features = +0.005-0.015、 工数 30h
✅ Phase 4 (動画) +5-7 features = +0.005-0.010、 工数 80h+

**5/19 V15.5 候補 → 6/8 V20 → 9/1 V21 が 段階的 path**
