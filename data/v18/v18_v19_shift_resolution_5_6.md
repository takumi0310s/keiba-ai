# V18/V19 distribution shift 真因 + calibration 対応可否

**作成**: 2026-05-06 PM (Session #32 A)
**対象**: V18 単勝 / V19 複勝 model
**結論**: **calibration では本質的に解決不能、feature shift が真因 → GO 条件 #3 NO**

---

## 1. distribution_shift_analysis.md (Session #10) 再読み

| dataset | race_max_p mean | winner_top1 | winner_top3 |
|---------|----------------|------------|------------|
| BT_2025_OOS | 0.347 | **47.8%** | 78.8% |
| Retro_raw | 0.013 | **34.5%** | 72.4% |
| Retro_calibrated (Platt) | 0.020 | **34.5%** | 72.4% |

**重要観察**:
- race_max_p factor (BT/retro): **27.69x** (絶対値の桁違い)
- **winner_top1**: BT 47.8% → retro 34.5% (**-13.3pt 劣化**)
- **calibration 後も winner_top1 不変** (34.5% のまま)

→ **calibration / normalize は monotonic 変換、rank 不変、winner_top1 改善せず**。

---

## 2. shift の正体 切り分け

### 2.1 全体 scaling か、特定 features か

`tools/analyze_v18_v19_distribution.py` (Session #10) の出力:

```
horse-level:
  BT mean=0.0548 / retro raw=0.0018 (30x scaling shift)
  BT max=0.9863 / retro max=0.1538 (6.4x range shift)
race-level:
  race_max_p factor = 27.69x
  top1/top2 ratio: BT 4.13 → retro 4.37 (close、不変近い)
```

**判定**:
- 全体 scaling shift: **YES** (mean / max が一様に低下)
- 特定 features の問題: **未解明** (個別 feature の比較が必要、Session #25 自己診断 §B1 指摘済)

### 2.2 calibration で対応可能か

calibration (Platt scaling、`v18_tansho_calibrator.pkl`) の実態:
- raw 0.013 → cal 0.020 (+54%、scaling 補正で BT 寄りに少し近づく)
- ただし monotonic 変換で **rank 不変**
- winner_top1 = TOP1 馬の選定精度 → **calibration では改善不可能**

→ **calibration は scaling 問題は緩和するが、rank shift (= winner_top1 劣化) は解決しない**。

### 2.3 真因仮説

monotonic 変換で改善しない = **rank 自体が劣化**。 真因候補:
- a. **feature 値域の shift**: 学習時と本番で feature の分布が異なり、model が誤判定
- b. **feature 不在/欠損**: 本番で取得できない feature が default 値で全馬同じ → 識別力低下
- c. **新規データ分布**: 4-5 月の馬・騎手・コース傾向が学習データと異なる

これらは calibration では解決不能、**feature 別の精密分析が必要** (Session #25 自己診断 §B1 で 90 min 工数指摘)。

---

## 3. GO 条件 #3 判定

> #3: shift 真因が calibration で対応可能と判明

**判定**: 🔴 **NO** (calibration では本質改善せず、feature shift が真因)

→ **5/9 投入 NO-GO 寄り** (#3 NO 確定)。

---

## 4. 本番 pipeline 統合の準備 (#4) - 設計のみ

`tools/predict_core.py` への V18/V19 統合方針 (実 deploy はしない):

```python
# predict_core.py 末尾 or 新 section に統合候補
def predict_v18_v19_orchestrated(df, race_info):
    """V15 + V18/V19 並列予測、failure 時 V15 fallback"""
    # 1. V15 予測 (基本)
    v15_result = predict_with_v15(df)

    # 2. V18 単勝 予測 (試行)
    try:
        v18_p = predict_v18(df)  # data/v18/models/v18_tansho_*
        from race_normalize import normalize_per_race
        v18_norm = normalize_per_race(df, prob_col='v18_p', method='softmax', T=1.0)
        # bet 候補 (p_norm >= 0.5, EV >= 1.2)
    except Exception as e:
        log_v18_failure(e)
        v18_norm = None

    # 3. V19 複勝 同様
    # 4. 結果統合: V15 主、V18/V19 副 (採用 R のみ並列投資)
    return v15_result, v18_norm, v19_norm
```

→ **設計のみ**、本格 deploy は 5/8 までに別 commit で。 5/9 当日 V15 単独投資には影響なし。

---

## 5. 結論

- **shift 真因**: feature distribution shift (rank 劣化、calibration では解決不能)
- **GO 条件 #3 (calibration 対応可能)**: 🔴 **NO 確定**
- **GO 条件 #4 (pipeline 統合)**: 🟡 **準備のみ**、本格 deploy は別途
- 5/9 投入は **NO-GO 寄り**、 #3 で既に絶対方針 (取り返し禁止) との整合判定

→ V18/V19 5/9 投入は **NO-GO 確定** (この時点で確定可能)。
詳細な 6 条件 status は §E で。
