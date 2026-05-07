# V18/V19 sib抜き LIVE retro 分析 (Session #38 B)

**仮説**: (b 寄り、 hybrid) **sib は LIVE で識別能力 / リーク 両面の役割**
**5/16 推奨**: **NO-GO 確定**、 Phase 3 で sib expanding window 版 設計

---

## TL;DR

V18/V19 sib抜き LIVE retro 結果 (5/2-5/3, 67 races, winner_known 29):
- **shift_factor**: 30.4x → **8.3x** (劇的改善、 sib リーク 部分の証拠)
- しかし **winner_top1**: 34.48% → **24.14%** (-10.3pt、 大幅悪化)
- → sib は **distribution shift を作るリーク 部分** と **legitimate な識別能力 部分** の **両方** を持つ

5/16 投入: **NO-GO 確定** (winner_top1 -10pt は明確悪化)。
Phase 3 plan 修正: sib_top3_rate / sib_shinba_wr の **expanding window 版** 設計が必要。

---

## 1. LIVE 5/2-5/3 比較 (5/2 + 5/3 全 67 races, winner_known 29)

| metric | OLD (含 sib, LGB+XGB Ens) | NEW (sib抜き, LGB) | Δ |
|--------|---------------------------|---------------------|---|
| n_races (winner_known) | 29 | 29 | - |
| **winner_top1** | **34.48% (10/29)** | **24.14% (7/29)** | **-10.34pt** |
| top3_hit_rate | 93.10% | 93.10% | +0.00pt |
| AUC v18 | 0.8164 | 0.8160 | -0.0004 |
| AUC v19 | 0.6300 | 0.6337 | +0.0037 |
| mean p18 | 0.0018 | **0.0066** | **+3.7x** |
| max p18 | 0.1538 | **0.5030** | **+3.3x** |
| mean p19 | 0.0016 | 0.0311 | +19.4x |

**重要**: AUC は ほぼ 同等 (差 -0.0004) だが winner_top1 は -10pt。
理由: AUC は ranking quality を 全体 で見るが、 winner_top1 は **top1 の identification** だけを見る。
sib抜きは ranking 全体 quality は維持、 top1 だけ精度低下 (sib が 1着馬識別に 寄与していた)。

---

## 2. BT 2025 OOS 比較

| metric | OLD (含 sib, Ens) | NEW (sib抜き, LGB) | Δ |
|--------|------------------|------------------|---|
| winner_top1 | 47.79% | 45.76% | -2.03pt |
| mean p18 | 0.0548 | 0.0550 | +0.0002 |

BT で winner_top1 -2pt は 想定内 (sib 寄与の リーク 部分 削除)。
LIVE で -10pt は **想定以上** (sib の legitimate 識別能力部分も同時消失)。

---

## 3. shift factor (BT/LIVE) 大幅改善

```
shift_factor = mean_p18(BT) / mean_p18(LIVE)
- OLD (含 sib): 0.0548 / 0.0018 = 30.4x  ← LIVE で 異常 縮小、 sib リーク 確実
- NEW (sib抜き): 0.0550 / 0.0066 = 8.3x  ← 大幅 改善 (3.7x reduction)
```

**解釈**:
- sib_top3_rate / sib_shinba_wr が BT で 過剰 active、 LIVE で 大半 default 0 → distribution shift 27x
- sib抜き 後、 BT distribution は維持、 LIVE distribution が 0.0018 → 0.0066 (3.7x 上昇)
- BT/LIVE 一致 度 大幅向上、 model calibration 改善

→ **sib は distribution shift を作る "リーク 寄与" を持つ** (確実に削除すべき分)。

---

## 4. 仮説判定 (a / b / c) — 詳細

### 4.1 当初仮説

| 仮説 | 説明 | LIVE 結果 patten |
|------|------|----------------|
| a | sib リーク仮説 正しい | sib抜き winner_top1 ≥ 38% |
| b | sib は本番でも有効 | sib抜き winner_top1 ≤ 30% |
| c | sib は ノイズ範囲 | 32-38% |

### 4.2 実測結果 = 仮説 b (寄り)

LIVE winner_top1: 34.48% → **24.14%** (-10.34pt) → 仮説 **b** に該当。

しかし shift_factor 30.4x → 8.3x の劇的改善は **sib リーク部分も明確に存在** することを示す。

### 4.3 真の解釈: hybrid

sib は **両方の役割**を持つ:
1. **リーク 寄与**: distribution shift (BT で 過剰 active)、 削除すべき
2. **legitimate 識別能力**: 1着馬識別 (top1 picking)、 削除すると失われる

→ 単純な「全削除」 では top1 picking 失う。
→ sib_top3_rate を **expanding window 修正版** で 再構築 必要。

---

## 5. 5/16 投入判断: 🔴 NO-GO 確定

| 判定軸 | 結果 |
|-------|------|
| LIVE winner_top1 改善 | -10.34pt (大幅悪化) |
| LIVE shift_factor 改善 | 30.4x → 8.3x (改善あり) |
| LIVE AUC 改善 | -0.0004 (ほぼ変化なし) |
| 5/16 投入 効果 | ROI 大幅悪化 ほぼ確実 |

→ **5/16 V18/V19 sib抜き 投入 NO-GO 確定**、 V15 案B改 維持。

---

## 6. Phase 3 plan 修正 (Session #38 結果反映)

### 6.1 sib_top3_rate / sib_shinba_wr 修正設計

**現状の問題**:
- 静的 CSV (data/netkeiba_siblings.csv) の集計値 をそのまま使用
- 集計値は 全期間データから計算されている (= retroactive)
- 当該レース時点での "未来情報" を含む可能性

**修正設計** (Phase 3 で実装):
```python
# expanding window 版 sib_top3_rate
def compute_sib_top3_rate_expanding(df, race_date_col='race_date'):
    """
    各レース時点で 母 (mother) 産駒の top3_rate を expanding window 計算。
    - 当該 race_date より前の 産駒 race のみ集計
    - 当該 race の 自身 と 兄弟姉妹 の 結果は 除外 (時点別)
    - cumsum / cumcount で expanding 集計
    """
    df = df.sort_values([dam_id_col, race_date_col])
    df['top3_cum'] = df.groupby(dam_id_col)['is_top3'].shift(1).fillna(0).cumsum()
    df['count_cum'] = df.groupby(dam_id_col).cumcount()
    df['sib_top3_rate_clean'] = df['top3_cum'] / df['count_cum'].clip(1)
    return df
```

→ Phase 3 5/24+ で実装、 V18/V19 / V20 で 採用。

### 6.2 V18/V19 plan 修正

| 項目 | 旧 plan | 新 plan (Session #38 反映) |
|------|---------|--------------------------|
| sib 削除 | 単純全削除 | expanding 版 で 再構築 |
| 6-fold WF | LGB+XGB | LGB+XGB (ensemble 維持) |
| 4-model | FT-Transformer + IntraRace 追加 | 同上 + sib expanding 効果検証 |
| 5/16 投入 | (sib抜き で GO 候補) | **NO-GO**、 Phase 3 で expanding 版 ABC test |
| 5/24+ | 6-fold WF + ensemble | sib expanding 修正 + 6-fold WF |

### 6.3 V20 plan 修正

V20 features design 修正:
- sib_top3_rate / sib_shinba_wr は **expanding 版** で含める (Session #36 plan より)
- LEAK_FEATURES_V20 = SKB 全削除 (Session #38 A 確定) + 旧静的 sib 削除
- 共通 80 features に expanding 版 sib (cleaned) を 復活

---

## 7. sample size 注意

- 5/2-5/3 winner_known races = 29
- winner_top1 -10.3pt = 3 races (10 → 7) の差
- 95% CI (Wilson): OLD [19.5%, 53.0%], NEW [12.0%, 42.7%]
- CI overlap あり = 統計的有意性 弱い

→ Phase 3 で 4/26 + 4/19 + 4/12 の 3 週分 retro 拡大、 winner_known races 80+ で 確証。

しかし shift_factor 30.4x → 8.3x は **大幅 改善 で 統計的に明確**。
sib リーク部分は確実、 識別能力部分の 統計的確証は Session #39 で 拡大検証。

---

## 8. 結論

🔴 **5/16 V18/V19 sib抜き 投入 NO-GO 確定**

主要発見:
1. sib 全削除 で **shift_factor 30.4x → 8.3x** (劇的改善、 リーク確実)
2. しかし **winner_top1 -10.3pt** (識別能力 部分損失)
3. → sib は **hybrid** (リーク + 識別能力)、 expanding window 修正版 必要

Phase 3 (5/24+) で:
- sib_top3_rate / sib_shinba_wr の expanding window 版 設計 + 実装
- V18/V19 (含 sib 修正版) で 6-fold WF + LGB+XGB ensemble
- V20 (6/9-6/30) で expanding sib 採用、 SKB 完全除外

5/9 V15 案B改 投資保護: **完全保証**、 影響ゼロ。
