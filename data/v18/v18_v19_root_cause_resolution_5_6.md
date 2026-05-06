# V18/V19 真因 + 解決策 + 5/16 暫定判定

**作成**: 2026-05-06 PM (Session #33 E、A-D 統合)
**結論**: 🟢 **Pattern A 確定** (predict 側 pipeline 修正で復活可能、5/16 GO 期待度 +50pt)

---

## 1. A-D 統合: 真因 確定

| 候補 | 検証結果 | 主因 / 副因 / 否定 |
|------|---------|------------------|
| 1. features 分布差 (A) | **12 features が retro で破綻** (gain share 16.7%) | **主因** ✓ |
| 2. ラベル分布差 (B) | 機械的に同じ (1着率 7.46% vs 7.52%、0.06pt 差) | 否定 (false hypothesis) |
| 3. data leakage (C) | 明確な leakage 不検出、PACI 取得停止が真因 | 否定 (別問題) |
| 4. sample size 偏り (D) | Niigata 0%→28%、Kyoto top1_p3 -22.3pt | 副因 (環境要因) |
| 5. PACI 取得停止 (C) | jrdb_paci.csv 4/4 更新停止、gain top3 default 同値 | **主因** ✓ |

→ **真因は Pattern A (features pipeline 破綻)**、 leakage / class imbalance / sample 偏り は **副因 or 否定**。

---

## 2. Pattern A 詳細: 12 features + PACI 7 features = 19 features 破綻

### 2.1 retro pipeline 破綻 (Session #33 A 発見)

| group | features | rank | gain | 破綻 type |
|-------|---------|------|------|----------|
| sib_* | sib_top3_rate, sib_shinba_wr | 6, 11 | 53,735 + 39,914 | **column 不在** |
| sr_* | sr_first3f_avg | 15 | 33,329 | **column 不在** |
| sire/bms | bms_surface_wr, sire_dist_wr, sire_surface_wr | (top 30) | - | **constant 0.100** (Bayesian prior default) |
| rest/trend | rest_days, weight_trend, pop_rank_change | (top 30) | - | **constant 0** |
| 上がり 3F | avg_last3f_3r, prev_last3f, prev2_last3f | (top 30) | - | **constant** |
| jrdb_dam | jrdb_dam_rensho_avg | (top 30) | - | **constant** |
| premium | training_time_filled | 5 | 70,826 | **92.9% が 0** (premium 取得失敗) |

### 2.2 PACI 取得停止 (Session #33 C 発見)

| feature | rank | gain | 状態 |
|---------|------|------|------|
| paci_jockey_exp_3rd | 1 | (top1) | jrdb_paci.csv 4/4 更新停止、default 同値 |
| paci_jockey_exp_wr | 2 | (top2) | 同上 |
| paci_ninki_idx | 3 | (top3) | 同上 |

V18 全 gain の **~30%** + V19 の **~35%** を占める top features が **default 同値で識別力 ゼロ**。

### 2.3 LightGBM の動作

split が `feature <= 0.100` の時、retro 全馬が同値なら全馬 同 leaf、 識別力 = 0。
4,000+ split のうち 12+ features が動かない → tree forest 全体で予測 confidence 低下、 winner_top1 -13.3pt 劣化に直接寄与。

### 2.4 sample 構成シフト (Session #33 D 副因)

- Niigata 0% → 28.4% (5/2 春開催替わり)
- Hanshin/Nakayama 56.6% → 0% (学習で多く見た会場 消失)
- Kyoto top1_p3 51.5% → 29.2% (course_renovated 1.3% 充填問題と整合)

これは Pattern A の features 破綻と複合、shift を増幅。

---

## 3. monotonic 変換で改善しない理由 (再確認)

calibration / softmax T=1.0 / 正規化は rank 不変。
12+ features が default 同値で **rank 自体が崩壊** → どんな monotonic 変換でも winner_top1 改善せず。

→ **真因対策は features pipeline 修正のみ**。 calibration では絶対に解決不能。

---

## 4. 解決策: 4 グループ patch (5/13-15 で対応可能)

### 4.1 Group 1: PACI 取得復旧 (最優先、3-5h)

- `jrdb_paci.csv` 4/4 更新停止 の経路調査 + 復旧
- `tools/scrape_jrdb.py` の paci type 取得ロジック再確認
- DailyJrdbKyi (06:00) で paci type を再取得
- 影響: V18 gain ~30% / V19 ~35% 復活、winner_top1 +5-8pt 期待

### 4.2 Group 2: sib_* / sr_* 生成追加 (4-6h)

- predict_core.py の build_features 拡張で sib_top3_rate, sib_shinba_wr, sr_first3f_avg を生成
- 既存 CSV: data/netkeiba_siblings.csv (17,441 母馬), data/sire_shinba_stats.csv (449 種牡馬)
- expanding window で計算 (V12 dam_top3r リーク事故 から学ぶ、必ず train_only で計算)
- 影響: gain 16.7% 復活、winner_top1 +3-5pt 期待

### 4.3 Group 3: sire/bms lookup table fallback (2-3h)

- 現状 retro で全馬 0.100 (Bayesian prior default) → BT 学習時の sire/bms encoding を本番に流用
- features_v15_new.py 等に既存実装がある可能性、predict_core.py への merge 確認
- 影響: 補正レベル、winner_top1 +1-3pt 期待

### 4.4 Group 4: premium fallback 強化 (2-4h)

- training_time_filled 92.9% が 0 → mean fill から実値復旧
- daily_premium_scrape の cache JSON → CSV 自動転換 (Session #27 で実装済) を本番 pipeline に組込
- 影響: rank 5 features 復活、winner_top1 +2-4pt 期待

### 4.5 Group 5: sample 構成シフト 対応 (運用層、1h)

- Niigata / Kyoto / 重〜不良 を 5/16 で **採用外 filter** (運用層、model 触らない)
- 5/24+ Phase 3 で V20 統合 model 学習時に開催替わり対応

---

## 5. 5/16 GO 期待度 試算

### 5.1 修正前 (現状) winner_top1 = 34.5%

GO 基準: ≥ 45% (5pt 余裕含む)
不足: 10.5pt

### 5.2 修正後 expected winner_top1 (+11-20pt 期待)

| 修正 | 期待改善 |
|------|--------|
| Group 1 (PACI) | +5-8pt |
| Group 2 (sib/sr) | +3-5pt |
| Group 3 (sire/bms) | +1-3pt |
| Group 4 (premium) | +2-4pt |
| **合計** | **+11-20pt** |

→ winner_top1 推定 **45-55%** (45% 基準クリア、BT 47.8% に近い)

### 5.3 5/16 GO 確率

| 修正範囲 | 工数 | 期待 winner_top1 | 5/16 GO 確率 |
|---------|------|---------------|------------|
| Group 1 only | 3-5h | 39-43% | 30% |
| Group 1+2 | 7-11h | 42-48% | 50% |
| Group 1+2+3 | 9-14h | 43-51% | 65% |
| **Group 1+2+3+4** | **11-18h** | **45-55%** | **75%** |
| 全 + sample 5 | 12-19h | 同上 | 80% |

→ **5/13-15 で 11-18h 確保すれば 5/16 GO 確率 75%** (Pattern A 修正範囲)。

---

## 6. 5/16 試行 投資 plan (GO 時)

```
V15 案B改 (主、12R 1勝): 700 円 × 採用 R
V18 単勝 試行: 500 円/日 × 採用 R 数 (上限 1,000 円)
V19 複勝 試行: 500 円/日 × 採用 R 数 (上限 1,000 円)
合計上限: 4,100 円/日 (V15 2,100 + V18 1,000 + V19 1,000)
```

最悪: -4,100 円 → 累計 +9,430 円 (依然プラス維持、撤退余裕 +59,430 円)

---

## 7. Phase 3 (5/24+) plan 更新

| 期間 | task | 状態 |
|------|------|------|
| 5/13-15 | Pattern A 修正 (Group 1-4) | **新計画** |
| 5/16 (土) | V18/V19 試行 (条件達成時) | 70-75% GO |
| 5/17-23 | paper trading 蓄積 | 継続 |
| 5/24 | Phase 3 移行判定 | V15.1 + V18/V19 並行運用候補 |
| 5/25-6/8 | V15.1 SKB 本格採用 | (`PHASE_3_V15_1_PLAN.md`) |

→ **V18/V19 廃止 不要**、Pattern A 修正で復活可能。 V15.1 + V18/V19 + V20 の **3 路線並行** が現実的。

---

## 8. Session #31 V18_V19_5_16_GO_NOGO 更新点

旧判定 (Session #31 B):
- 5 条件 0/5 達成、暫定 NO-GO
- 暫定 GO 確率 < 30%

新判定 (本セッション):
- **Pattern A 確定** で 5/13-15 修正 → 5/16 GO 確率 **75%**
- 修正範囲は predict 側 pipeline (4 group)、 model 再学習 **不要**
- 廃止候補は transport_distance_km 1 件のみ

---

## 9. 5/13-15 必須作業 (3 日間 11-18h)

### 5/13 (火) 4-6h

- [ ] Group 1 PACI: jrdb_paci.csv 取得経路調査 + 復旧 (3-5h)
- [ ] Group 1 検証: retro で paci values 復活確認 (1h)

### 5/14 (水) 4-7h

- [ ] Group 2 sib_*/sr_*: predict_core 生成 logic 追加 (4-6h)
- [ ] Group 4 premium: training_time_filled 復旧 (2-4h、並行可)

### 5/15 (木) 3-5h

- [ ] Group 3 sire/bms lookup table fallback (2-3h)
- [ ] 4/11-5/15 paper retro 再実行 (4 group 修正後、2h)
- [ ] 5/15 22:00 5/16 GO/no-go 判定

---

## 10. 結論

**真因 Pattern A 確定**: V18/V19 model は健全、predict 側 pipeline で 12+ features が破綻 (PACI 停止 + sib/sr 不在 + sire/bms default + premium 取得失敗)。

**5/16 GO 確率 75%** (Pattern A 修正 11-18h で達成可能)、 model 廃止 不要、再学習 不要。

5/13-15 で 4 group patch 完遂 → 5/16 V18/V19 試行 (1,000 円/日 × 2 = 2,000 円) 投入候補。
撤退余裕 +59,430 円維持で、取り返し禁止ルール下で投入可能な範囲。
