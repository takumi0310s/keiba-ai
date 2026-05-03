# 改良提案 試作 + retro 検証 結果

生成: 2026-05-04 (Opus xhigh, Session#6)

## 試作した改良 (3案)

### 1. Formation 拡張 retro (実装+retro完了)

**Tool**: `tools/formation_retro_5_2_5_3.py`  
**実行結果**: `data/v18/formation_retro_summary.csv`

#### 結果 (healthy 4日 137R)

| Formation | n_bets | 投資 | 払戻 | 利益 | ROI | hit_rate |
|-----------|------:|----:|----:|-----:|----:|---------:|
| **V15_baseline_7 (現行)** | 7 | 94,700 | 37,780 | -56,920 | **39.9%** | 16.1% |
| 2axis_4 | 4 | 94,850 | 40,232 | -54,617 | **42.4%** | 11.7% |
| V15+T1-T4-T5 (8) | 8 | 108,200 | 37,780 | -70,420 | 34.9% | 16.1% |
| V15+T14_T15_T16 (10) | 10 | 135,200 | 37,780 | -97,420 | 27.9% | 19.7% |
| V15+T2-T3-T4 (8) | 8 | 108,200 | 37,780 | -70,420 | 34.9% | 16.8% |
| Box5 (10) | 10 | 135,200 | 28,000 | -107,200 | 20.7% | 16.8% |
| Box6 (20) | 20 | 270,200 | 37,780 | -232,420 | 14.0% | 33.6% |

#### 11R/12R のみ (主要レース)

| Formation | n_bets | ROI |
|-----------|------:|----:|
| **V15_baseline_7** | 7 | **86.6%** |
| 2axis_4 | 4 | 85.5% |
| V15+T1-T4-T5+T1-T4-T6 (9) | 9 | 67.3% |
| V15+T14_T15_T16 (10) | 10 | 60.6% |

#### 結論

🔴 **既存 V15 7点が ROI ベスト** — formation 拡張で hit 率↑だが投資額↑が相殺、ROI 悪化方向。
   2axis_4 が ROI で +2.5pt 上回るが分散大 (n=4 で variance 大)、5/9 推奨せず。

→ **5/9 採用: V15_baseline_7 のまま**。

---

### 2. V18/V19 完全 retro (5/2-5/3, model 復旧後)

**Tool**: `tools/v18_v19_retro_full.py` (Session#5 で修正)  
**実行結果**: `data/v18/v18_v19_retro_full_result.md`

#### 結果

```
全 horses: 932 (5/2 + 5/3)
winner_known horses: 387 (41.5%)

=== v18 単勝 retro (winner_known races only) ===
| prob_min | ev_min | bet | win | inv | pay | ROI |
|---:|---:|---:|---:|---:|---:|---:|
| 0.3 | 1.0 | 0 | - | - | - | - |
| 0.4 | 1.2 | 0 | - | - | - | - |
| 0.5 | 1.2 | 0 | - | - | - | - |

=== v19 複勝 retro ===
| prob_min | ev_min | bet | hit | inv | pay~ | ROI~ |
|---:|---:|---:|---:|---:|---:|---:|
| 0.5 | 1.0 | 0 | - | - | - | - |
| 0.6 | 1.1 | 0 | - | - | - | - |
| 0.7 | 1.1 | 0 | - | - | - | - |
```

#### 結論

🔴 **v18/v19 全フィルタで bet=0** — 5/2-5/3 で predict probability 過小 (max 0.001-0.017)

#### 推定原因

| 仮説 | 確度 |
|------|----:|
| Distribution shift (2026 vs 2015-2024 train data) | 高 |
| Feature pipeline 不整合 (live shutuba vs cache) | 高 |
| 過小 calibration (BT で under-confidence 既知) | 中 |

#### Action items

- ❌ **5/9 v18/v19 直接採用 不可** (Phase 2.5 計画通り 5/16以降は retro 結果を見て判断)
- 🔧 **Phase 2.5 calibration 修正必要**:
  - Platt scaling (sklearn LogisticRegression)
  - Race-level probability normalization (Σp = N)
  - 特徴量分布検証 (2026 vs 2024)
  - データソース整合性 (cache features vs live features)

→ **5/9 v18/v19 部分実弾投入は延期**。Phase 2.5 で calibration 修正完了まで待機。

---

### 3. EV>1 フィルター実装 ★ 試作スキップ

**理由**: v18/v19 が全 filter で bet=0 → EV filter 単独では bet 候補ゼロ。  
calibration 修正後に再試作。

---

## 試作完了状況サマリー

| 提案 | 実装 | retro | 5/9 採用 | 後続アクション |
|------|------|------|---------|---------------|
| formation 拡張 | ✅ | ✅ | ❌ | 採用しない (現行で OK) |
| V18/V19 retro | ✅ | ✅ | ❌ | calibration 修正 (5/16-) |
| EV>1 filter | ⏸️ | ⏸️ | ❌ | calibration 後 (5/24-) |
| V15+V17 ensemble | ⏸️ | n/a | ❌ | retro 効果限定的 (実装せず) |
| 特徴量拡張 v15.1 | ⏸️ | n/a | ❌ | 中期 (5/16-5/24) |

## 5/9 投資判断への影響

🟢 **案B改 (12R 1勝クラスのみ) を維持**:
- V15 batch 軸 (現行)
- V15 trio_bets 7点 formation (現行)
- 投資 700円/R × 採用R数 (上限 2,100円)
- 期待 ROI 161% (healthy 4日 bootstrap CI [135.9%, 222.4%])

→ 改良提案は全て 5/9 後 Phase 2.5 で実装、5/16 以降の運用に反映。

## 撤退ライン (再確認)

| トリガ | 5/9 単日 | 5/9-5/10 累計 | 累計 (5/2-) |
|--------|---------:|--------------:|-----------:|
| ROI < 50% | 警戒 | 撤退検討 | - |
| ROI < 30% | 撤退 | **撤退** | - |
| 損失額 > 5,000円 | 警戒 | - | - |
| 損失額 > 10,000円 | - | 撤退 | - |
| 累計損失 > 50,000円 | - | - | **完全撤退** |
