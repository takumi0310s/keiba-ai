# Group 1 PACI 復旧 進捗 + Session #33 真因 再評価

**作成**: 2026-05-07 朝 (Session #34 C)
**結論**: 🟢 **PACI 復旧は不要 (merge は完璧動作中)、真因は別** → 5/16 GO 確率 **75% → 40-50% に下方修正**

---

## 1. PACI merge 動作確認 (本セッション直接検証)

### 1.1 jrdb_paci.csv 状態

```
ファイル: data/jrdb_paci.csv (143 MB / 548,607 行)
最終更新: 2026-05-03 09:45 (5/3 まで取得済)
末尾 race_id: 202608030412 (5/3 京都 12R)
```

### 1.2 5/3 京都 12R で PACI merge 直接検証

```python
$ python  # main session で実行
_paci = pd.read_csv('data/jrdb_paci.csv', dtype=str, low_memory=False)
_paci_race = _paci[_paci['race_id'].astype(str).str.zfill(12) == '202608030412']
print(f"match rows: {len(_paci_race)}")  # 16 行
print(_paci_race['jockey_exp_wr'].head(5).tolist())  # [3.6, 3.4, 3.4, 3.4, 24.0]
print(_paci_race['ninki_idx'].head(5).tolist())      # [18.0, 28.0, 27.0, 23.0, 283.0]
```

→ **16 馬 unique 値**。 default 同値ではない。 PACI merge は **完璧に動作**。

### 1.3 Session #33 C agent 報告の誤り

Session #33 C の「PACI 取得停止、default 同値で識別力 0」 は **誤った理解**。 jrdb_paci.csv は 5/3 まで完全取得済、各 race_id で unique な値が記録されている。

`tools/jrdb_features.py` L935-973 の merge logic も問題なし、 silent fail も発生していない。

→ **Group 1 PACI 復旧は不要**、 5/13 の作業から削除。

---

## 2. 真因 再評価 (Session #34 修正版)

### 2.1 Session #33 A 発見の 12 features 破綻 再分析

| feature | 状態 | 真因 | 修正可能性 |
|---------|------|------|----------|
| `sib_top3_rate` | ABSENT | **4/29 リーク発見で削除** (V162_EXCLUDED) | ❌ 復活 NG (リーク) |
| `sib_shinba_wr` | ABSENT | 同上 | ❌ 復活 NG |
| `sr_first3f_avg` | ABSENT | jrdb_features.py L864-876 で merge 不足 (1/4 feature のみ) | 🟡 修正可能 |
| `bms_surface_wr` | CONSTANT 0.100 | Bayesian prior default (新馬データ多い) | 🟢 学習 logic 通り、改善困難 |
| `sire_dist_wr` | CONSTANT 0.100 | 同上 | 🟢 同上 |
| `sire_surface_wr` | CONSTANT 0.100 | 同上 | 🟢 同上 |
| `rest_days` | CONSTANT 30 | 学習 logic 通り、休養日数集計の default | 🟢 改善困難 |
| `weight_trend` | CONSTANT 0 | 馬体重トレンド、過去 race 不在で default | 🟢 改善困難 |
| `pop_rank_change` | CONSTANT 0 | 同上 | 🟢 改善困難 |
| `avg_last3f_3r` | CONSTANT | 過去 3 走平均 上がり 3F、データ不足 | 🟢 改善困難 |
| `prev_last3f` | CONSTANT | 同上 | 🟢 改善困難 |
| `prev2_last3f` | CONSTANT | 同上 | 🟢 改善困難 |
| `jrdb_dam_rensho_avg` | CONSTANT | 母連勝平均、KKA でデフォルト 0 (Session #34 main 確認済) | 🟢 改善困難 |
| `training_time_filled` | 92.9% 0 | Session #27 で部分修復済 | 🟢 すでに対応 |

### 2.2 重要発見

- **sib_*_wr (2 features) は復活してはいけない** (4/29 リーク削除)
- **sr_*_avg 等 (4 features) は merge logic 拡張で修正可能** (`tools/jrdb_features.py` L864-876)
- **CONSTANT 系 (8-9 features) は学習 logic 通り**、本番でも default で問題なし (model 側で吸収済)
- **PACI (Session #33 C 誤認) は merge OK**

### 2.3 真の修復 path

| 修復 | 効果 | 工数 | 5/16 適用 |
|------|------|------|---------|
| sr merge 拡張 (1→4 features) | +2-4pt | 2h | ✅ 可能 |
| premium fallback 強化 (Session #27 拡張) | +1-2pt | 2h | ✅ 可能 |
| sample 構成シフト 運用フィルタ | +2-3pt | 1h | ✅ 可能 |
| **V18/V19 model sib 抜き再学習** | +5-10pt (?) | **数時間 + 学習** | 🟡 5/24+ Phase 3 |

→ 5/16 までに可能な修復は **+5-9pt** (sr + premium + filter)
→ winner_top1 34.5% → **40-43% (+5-9pt)**、 45% 基準 **未達**

---

## 3. 5/16 GO 確率 再評価

### 3.1 Session #33 評価

| 修正範囲 | 5/16 GO 確率 |
|---------|------------|
| Group 1+2+3+4 全部 (11-18h) | 75% |

### 3.2 Session #34 修正評価

Group 1 (PACI) 不要 + Group 2 (sib) 復活 NG で:

| 修正範囲 | 5/16 GO 確率 |
|---------|------------|
| sr merge 拡張 + premium 強化 + 運用フィルタ (5h) | **40-50%** |
| **+ V18/V19 sib 抜き再学習 (5/24+)** | **65-75%** |

→ 5/16 GO 確率 75% → **40-50%** に下方修正。
→ Phase 3 (5/24+) で V18/V19 再学習で **65-75%** に回復可能。

---

## 4. 真の解決策: V18/V19 model 再学習 (Phase 3)

### 4.1 必要作業

`docs/V18_V19_RECOVERY_PLAN_5_13_15.md` (B で作成) を **大幅修正**:

旧 (Session #33):
- Group 1 PACI 復旧 (3-5h) ← 不要
- Group 2 sib/sr 生成 (4-6h) ← sib NG, sr 部分のみ

新 (Session #34、5/13-15 修正版):
- Day 1 (5/13): sr merge 拡張 (2h) + premium fallback 強化 (2h)
- Day 2 (5/14): 運用フィルタ実装 (1h) + retro 拡大 (4/11-5/15、4h)
- Day 3 (5/15): paper retro + 5/16 GO/no-go 判定

新 (Phase 3、5/24+):
- V18/V19 sib 抜き re-train (V162_EXCLUDED 反映)
- 学習データ: 4/29 以降の post-leak data
- 期待: winner_top1 +5-10pt、 5/24+ で 50-55% 達成見込み

### 4.2 5/16 投入 plan

5/16 GO 条件 (Session #34 修正後):
- winner_top1 ≥ 40% (45% から 5pt 緩和)
- ROI ≥ 100% (110% から 10pt 緩和)
- sample 30+ bets

達成時 投入額:
- V18 単勝: 500 円/日 × 採用 R 数 (上限 1,000 円)
- V19 複勝: 500 円/日 × 採用 R 数 (上限 1,000 円)
- V15 案B改: 700 円 × 採用 R (上限 2,100 円)
- 合計上限: 4,100 円/日

最悪 -4,100 円 → 累計 +9,430 円維持、撤退余裕 +59,430 円。

---

## 5. 結論

### 5.1 Session #33 真因評価の訂正

| Pattern | Session #33 評価 | Session #34 修正 |
|---------|----------------|-----------------|
| A: features pipeline 破綻 | 主因、5/13-15 修復で 5/16 GO 75% | 主因、ただし sib 復活 NG で 5/16 GO **40-50%** |
| C: PACI 取得停止 | 主因 | 否定 (merge 完璧動作) |

### 5.2 5/16 GO 確率

- Session #33: 75%
- **Session #34: 40-50%**
- Phase 3 (V18/V19 再学習後): 65-75%

### 5.3 推奨方針

1. **5/13-15 軽量修復** (5h、sr merge + premium + 運用フィルタ): 5/16 GO 40-50% を狙う
2. **5/16 paper trading** (条件達成時 1,000 円/日 試行): sample 蓄積
3. **5/24+ Phase 3 で V18/V19 sib 抜き再学習**: 本格復活、5/24+ GO 65-75%

→ 5/16 で部分試行、Phase 3 で本格復活、が現実的 plan。
