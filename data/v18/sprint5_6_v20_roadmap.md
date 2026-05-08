# Session #51 D: Sprint 5 / 6 / V20 / Phase 4 roadmap

**作成**: 2026-05-08 (Session #51 D)
**前提**: Session #51 A (分類) + B (一括 backtest) + C (combo search) 完了
**位置付け**: AUDIT-1 Top 27 を Sprint 5/6/V20/Phase 4 へ振り分け

---

## 0. 重要発見 (B + C より)

1. **V15 145 features は 既に 高度 飽和** (combo 全 15 pair で delta ≤ 0)
2. **LEAK 2 件 確定**: #18 jrdb_sed、 #22 race_review_score → V20 LEAK list 必須
3. **JRDB KKA parser 不全**: seiseki_* 全 0% NaN → Sprint 6 で 修復必要
4. **text encode 失敗**: training_rank ('A/B/C/D')、 ai_pace ('H/M/S') → text encode 修正で 評価可能
5. **TFJV 統合 (BS/BN/BR)** が **本命** (大規模、 領域違い、 V20 主軸)

---

## 1. Sprint 5 (5/16-5/22): 軽量補正 + Pattern B 改善

### 着手 (5 件、 工数 8h)

| 順 | # | 内容 | 工数 | 期待 AUC | dependency |
|---|---|------|------|---------|----------|
| 1 | 11 | training_times rank label encode ('A/B/C/D' → 0-3) + WF AUC 再評価 | 2h | +0.001 | none |
| 2 | 5 | ai_opinion pace label encode ('H/M/S' → 0-2) + WF AUC 再評価 | 2h | +0.0005 | 2024-2025 限定 |
| 3 | 19 | jrdb_kyi 印 残 3 (encoding cp932 修正 + WF AUC) | 2h | +0.0003 | none |
| 4 | 17 | jrdb_tyb cancel_flag (Pattern B 専用、 直前 取消検知) | 1h | live 改善 | Pattern B |
| 5 | (NEW) | V20 LEAK_FEATURES_V20 list 確定 (sed_* + review_score + skb 全 10) | 1h | safety | V20 学習前 必須 |

**期待 AUC**: V15 0.8939 → V15.5 0.8945 (+0.0006、 微増)
**着手日**: 5/16 (V18 trial 後)
**完了日**: 5/22

---

## 2. Sprint 6 (5/23-5/30): KKA parser 修復 + TFJV 部分統合

### 着手 (3 件、 工数 16h)

| 順 | # | 内容 | 工数 | 期待 AUC | dependency |
|---|---|------|------|---------|----------|
| 1 | 4 | JRDB KKA parser 修復 (seiseki_* 12 group × 4) + WF AUC | 6h | +0.002-0.005 | jrdb_kka.csv |
| 2 | 14 | TFJV BS_DATA breeder_top3r expanding 化 + WF AUC | 8h | +0.002-0.004 | tfjv_parser.py |
| 3 | 13 | jrdb_ukc 取得 + keito_code/owner_code 統合 | 2h+取得 | +0.001-0.003 | UKC 取得 |

**期待 AUC**: V15.5 0.8945 → V15.6 0.8975 (+0.003)
**着手日**: 5/23
**完了日**: 5/30

---

## 3. V20 統合 (5/22-6/8): TFJV 大規模 投入 (本命)

### 着手 (4 件、 工数 30h)

| 順 | # | 内容 | 工数 | 期待 AUC | dependency |
|---|---|------|------|---------|----------|
| 1 | 14 | TFJV BS breeder_top3r (expanding) | 8h | +0.002-0.004 | Sprint 6 完了 |
| 2 | 15 | TFJV BN owner_top3r (expanding) | 8h | +0.002 | Sprint 6 完了 |
| 3 | 16 | TFJV BR dam 拡張 sib_*_ext (90 年分) | 6h | +0.001-0.003 | sib_*_exp 拡張 |
| 4 | 25 | TFJV WF (WIN5) appearance_count | 4h | +0.0005-0.001 | WF parser |
| 5 | (Sprint 5 移送) | 18 件 のうち 4 件 を V20 ベースで再検証 | 4h | 重複 dedup | Sprint 5 完了 |

**期待 AUC**: V18/V19 sib_*_exp ベース 0.8847 + V20 統合 → 0.8895-0.8935 (V15.6 を 上回る)
**着手日**: 5/22 (Sprint 5 と並行)
**完了日**: 6/8 (Phase 3 v3 投入 plan)

---

## 4. Phase 4 (7-9月): 動画 + リアルタイム features

### 着手 (2 件、 工数 86h+)

| 順 | # | 内容 | 工数 | 期待 AUC | dependency |
|---|---|------|------|---------|----------|
| 1 | 30 | パドック画像 解析 (体格/緊張度) | 80h+ | +0.005-0.010 | YOLOv8 + DLC |
| 2 | 29 | TFJV JG (出走取消) リアルタイム | 6h | live 改善 | parser realtime 化 |

**期待 AUC**: V20 0.8935 + 動画 → V21 0.8985-0.9035
**着手日**: 7/1 (V20 投入後)
**完了日**: 9/1 (V21 投入候補)

---

## 5. 全体 タイムライン

```
5/9          5/16          5/22       5/30          6/8          7/1          9/1
 │            │             │          │             │            │            │
 ▼            ▼             ▼          ▼             ▼            ▼            ▼
V15 案B改   V18 trial    Sprint 5  Sprint 6      V20 投入     Phase 4      V21 投入
 (絶対)     (sib_exp)    (軽量)    (KKA + UKC)   (TFJV 大規模) (動画 + JG)  (中位想定)

期待 AUC:
V15 0.8939 → V15.5 0.8945 → V15.6 0.8975 → V20 0.8935 → V21 0.8985
                                         (sib + TFJV)
```

---

## 6. 失敗 落とし穴 (Session #38/51 学び)

| 落とし穴 | 学習内容 | 対策 |
|---------|--------|------|
| dam_top3r LEAK (Session #38) | 静的 CSV を そのまま使用 → リーク | expanding 化 必須 (cumsum - current) |
| SKB POST-RACE LEAK | skb_*_code が finish と monotonic | V20 で SKB 全 10 features 完全除外 |
| **SED LEAK (Session #51 B)** | jrdb_sed が finish/time_sec 含む | V20 で **sed_time_sec、 sed_first_3f、 sed_last_3f、 sed_finish、 sed_abnormal** 完全除外 |
| **race_review_score LEAK (Session #51 B)** | review_score 自体が POST-RACE | V20 で review_score 完全除外 |
| V15 飽和 (Session #51 C) | 145 features の 単純追加は 効果薄 | 領域違い (TFJV BS/BN) + 異 modal (画像) を 投入 |
| KKA parser 不全 | seiseki_* 全 0% NaN | Sprint 6 で 修復、 Sprint 5 では 着手不可 |

---

## 7. V20 LEAK_FEATURES_V20 確定 list (Session #51 B 反映)

```python
# train/v15_1_features.py V20_LEAK_FEATURES (Sprint 5 確定)

LEAK_FEATURES_A = {  # Pattern A 既存 8
    'odds_log', 'horse_weight', 'condition_enc',
    'weight_change', 'weight_change_abs',
    'weight_cat', 'weight_cat_dist', 'cond_surface',
}

SKB_LEAK_FEATURES = {  # Session #38 確定 10
    'skb_kishi_code_1', 'skb_kishi_code_2', 'skb_kishi_code_3',
    'skb_baba_code_1',  'skb_baba_code_2',  'skb_baba_code_3',
    'skb_kyaku_code_1', 'skb_kyaku_code_2', 'skb_kyaku_code_3',
    'skb_turf_hoof',
}

SED_LEAK_FEATURES = {  # ★ Session #51 B 新規 確定 ★
    'sed_time_sec', 'sed_first_3f', 'sed_last_3f',
    'sed_finish', 'sed_abnormal', 'sed_pass1', 'sed_pass2',
    'sed_pass3', 'sed_pass4', 'sed_agari_3f',
}

REVIEW_LEAK_FEATURES = {  # ★ Session #51 B 新規 確定 ★
    'review_score', 'prev_review_score',  # 自体が POST-RACE
}

V20_LEAK_FEATURES = (LEAK_FEATURES_A | SKB_LEAK_FEATURES
                     | SED_LEAK_FEATURES | REVIEW_LEAK_FEATURES)
# 合計 30 features 完全除外
```

---

## 8. 結論

✅ Sprint 5 (5/16-5/22): 軽量補正 5 件、 期待 +0.0006 (V15 0.8939 → 0.8945)
✅ Sprint 6 (5/23-5/30): KKA 修復 + TFJV 部分統合 3 件、 期待 +0.003 (→ 0.8975)
✅ V20 (5/22-6/8): TFJV 大規模 4 件、 期待 +0.001-0.005 (→ 0.8935-0.8985)
✅ Phase 4 (7-9月): 動画 + JG 2 件、 期待 +0.005-0.010 (→ 0.8985-0.9035)

**主結論**:
1. V15 飽和 検出 → **大規模 領域違い** features (TFJV、 動画) が 必須
2. **LEAK 30 features list 確定** (Sprint 5 で V20_LEAK_FEATURES 実装、 V20 学習前 必須)
3. AUDIT-1 全 27 件 を 適切な Sprint に 割り振り完了
4. 5/9 V15 案B改 維持 (絶対遵守)
5. V15 投資保護: ✅ 不変

5/16 V18 trial 後 の plan 完成。
