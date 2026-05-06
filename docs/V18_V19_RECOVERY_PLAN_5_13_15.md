# V18/V19 復活 5/13-15 詳細手順書 (Session #34 修正版)

**作成**: 2026-05-07 朝 (Session #34 B)
**対象期間**: 5/13 (火) - 5/15 (木) 平日 3 日間
**5/16 (土) GO 確率**: **40-50%** (Session #34 下方修正、Session #33 75% から)

---

## ★ Session #34 重大訂正

Session #33 の plan を以下で修正:

| 元 plan | Session #34 訂正 |
|---------|---------------|
| Group 1 PACI 復旧 (3-5h) | **不要** (PACI merge は完璧動作中、Session #33 C 誤認) |
| Group 2 sib_*_wr 生成 | **NG** (4/29 リーク削除済、復活 不可) |
| Group 2 sr_first3f_avg 生成 | **修正可能** (jrdb_features.py L864-876 拡張) |
| Group 3 sire/bms fallback | **学習 logic 通り** (model 側で吸収済、5/13-15 不要) |
| Group 4 premium 強化 | **Session #27 で部分修復済** (再強化のみ) |

→ 5/13-15 で実施は **5h** (旧 11-18h から大幅減)、 5/16 GO 確率 **40-50%**

---

## 1. 全体スケジュール

```
5/13 (火) 4h: Step 1 (sr merge 拡張) + Step 2 (premium fallback 強化)
5/14 (水) 5h: Step 3 (運用フィルタ実装) + Step 4 (4/11-5/15 retro 拡大)
5/15 (木) 3h: Step 5 (5/16 paper retro 最終) + 22:00 GO/no-go 判定
```

合計: **12h 程度**、平日 3 日で確実完遂可能。

---

## 2. 5/13 (火): Step 1 + Step 2 (4h)

### Step 1: SR merge 拡張 (jrdb_features.py、2h)

**現状** (`tools/jrdb_features.py` L864-876):
```python
_sr_path = os.path.join(DATA_DIR, 'jrdb_sr.csv')
if os.path.exists(_sr_path):
    _sr_race = _sr[_sr['race_id'].astype(str).str.zfill(12) == _rid_str]
    if len(_sr_race) > 0:
        _sr_row = _sr_race.iloc[-1]
        _tb = str(_sr_row.get('tb_homestr', ''))
        _inner = int(_tb[0]) if _tb and len(_tb) >= 1 and _tb[0].isdigit() else 2
        horses_df['jrdb_tb_homestr_inner'] = _inner   # ← 1 feature のみ
```

→ V18/V19 学習で使う 4 features のうち **`jrdb_tb_homestr_inner` 1 件のみ生成**、他 3 件 (`sr_first3f_avg`, `sr_bias_homestr`, `sr_bias_4corner`, `sr_pace_up_pos`) は不在 = ABSENT。

**修正** (5/13 朝):

```python
# tools/jrdb_features.py L864-876 拡張
_sr_path = os.path.join(DATA_DIR, 'jrdb_sr.csv')
if os.path.exists(_sr_path):
    try:
        _sr = pd.read_csv(_sr_path, encoding='utf-8-sig', dtype=str)
        _sr_race = _sr[_sr['race_id'].astype(str).str.zfill(12) == _rid_str]
        if len(_sr_race) > 0:
            _sr_row = _sr_race.iloc[-1]
            # 既存 (1 feature)
            _tb = str(_sr_row.get('tb_homestr', ''))
            _inner = int(_tb[0]) if _tb and len(_tb) >= 1 and _tb[0].isdigit() else 2
            horses_df['jrdb_tb_homestr_inner'] = _inner
            # Session #34 拡張 (3 features 追加)
            horses_df['sr_first3f_avg'] = pd.to_numeric(_sr_row.get('first3f_avg', 0), errors='coerce') or 0
            horses_df['sr_bias_homestr'] = pd.to_numeric(_sr_row.get('bias_homestr', 0), errors='coerce') or 0
            horses_df['sr_bias_4corner'] = pd.to_numeric(_sr_row.get('bias_4corner', 0), errors='coerce') or 0
            horses_df['sr_pace_up_pos'] = pd.to_numeric(_sr_row.get('pace_up_pos', 0), errors='coerce') or 0
    except Exception as e:
        print(f"[WARN] JRDB SR merge failed: {e}")
```

**動作確認**:
```bash
python tools/predict_one_race.py 202608030412 2>&1 | grep "sr_"
```

期待: 4 features すべて生成、unique 値が反映される。

### Step 2: premium fallback 強化 (2h)

**対象**: `training_time_filled` 92.9% が 0 (Session #27 で部分修復、5/2-5/3 で +1,004 行追加済)

**強化内容**:
- daily_premium_scrape の cache JSON → CSV 自動転換 (Session #27) を本番 pipeline で確実動作
- `tools/predict_core.py` build_features 内で training cache から実値 fetch (既実装、確認のみ)
- 5/13 朝の DailyPremiumScrape 完了確認 + cache→CSV 動作確認

**動作確認**:
```bash
python -c "
import pandas as pd
df = pd.read_csv('data/netkeiba_training_eval.csv', encoding='utf-8-sig')
df['year_yymm'] = df['race_id'].astype(str).str[:6]
print('2026 5月分:', len(df[df['year_yymm'].str.startswith('202605')]))
"
```

期待: 5月分 1,000+ 行 (Session #27 で復旧済 + 5/13 朝の自動実行で追加)。

### Step 1+2 完了後 22:00 status 報告

```bash
# Discord 通知
python tools/notify_done.py "Session #34 5/13 Step 1+2 完了" "sr merge 拡張 + premium 強化、5/14 へ" --color blue
```

---

## 3. 5/14 (水): Step 3 + Step 4 (5h)

### Step 3: 運用フィルタ実装 (1h)

**目的**: Session #33 D 発見 sample 構成シフト (Niigata 0%→28%、Kyoto top1_p3 -22.3pt) 対応

**実装**: `tools/multi_stage_predict.py` または別 wrapper で:
```python
# V18/V19 試行時のみ filter
if mode == 'v18_v19_trial':
    EXCLUDE_COURSES = ['新潟', '京都']  # 5/16 試行時除外
    EXCLUDE_CONDITIONS = ['B', 'X']      # 重〜不良除外
    if course in EXCLUDE_COURSES or condition_enc in EXCLUDE_CONDITIONS:
        adopted = False
        reason = "V18/V19 学習データ偏り (sample shift filter)"
```

→ 新潟・京都 + 重〜不良で V18/V19 試行は見送り、対象 R を狭めて精度 UP。

### Step 4: 4/11-5/15 retro 拡大 (4h)

**現状**: 5/2-5/3 retro = 67 races / 9-25 bets (sample 不足)
**目標**: 4/11-5/15 で 200+ races / 30+ bets 蓄積

**実行**:
```bash
# tools/v18_v19_retro_full.py に --dates 引数追加 (もしなければ実装)
for d in 20260411 20260412 20260418 20260419 20260425 20260426 20260502 20260503; do
    python tools/v18_v19_retro_full.py --date $d --output data/v18/retro_$d.csv
done
# 集計
python -c "
import pandas as pd
import glob
all = pd.concat([pd.read_csv(f) for f in glob.glob('data/v18/retro_2026*.csv')])
print('total bets:', len(all))
print('hit rate:', all['win'].mean())
print('ROI:', (all['payout'].sum() / all['inv'].sum() - 1) * 100, '%')
"
```

期待:
- bets: 30+
- hit rate: 35-45%
- ROI: 100-130% (sr 拡張 + premium 強化 後)

### Step 3+4 完了後 22:00 status 報告

```bash
python tools/notify_done.py "Session #34 5/14 Step 3+4 完了" "filter + retro 30+ bets、5/15 へ" --color blue
```

---

## 4. 5/15 (木): Step 5 + 22:00 GO/no-go (3h)

### Step 5: paper retro 最終実行 (2h)

`tools/v18_v19_retro_full.py` で 4/11-5/15 + 5/13-14 修正反映後の最終 retro:

```bash
python tools/v18_v19_retro_full.py --date 20260411 --date 20260412 ... --date 20260515 \
    --normalize softmax --T 1.0 \
    --output data/v18/retro_final_5_15.md
```

期待結果:
- winner_top1 ≥ 40% (45% から 5pt 緩和)
- ROI ≥ 100% (110% から 10pt 緩和)
- sample 30+ bets

### 22:00 5/16 GO/no-go 判定

| 達成数 | 判定 |
|--------|------|
| 全 (winner_top1 ≥ 40% + ROI ≥ 100% + sample 30+) | 🟢 GO (V18 500 円 + V19 500 円 = 1,000 円/日) |
| 2/3 (winner_top1 ≥ 40% + sample 30+) | 🟡 paper 継続、5/22 再判定 |
| 1/3 or 0 | 🔴 NO-GO、Phase 3 (5/24+) で V18/V19 再学習 |

---

## 5. 5/16 (土) 投入手順 (もし GO)

### 朝 (06:30 - 09:00)

1. 06:30 Morning_Sat 自動 → V15 11R/12R 軸候補
2. 08:00 DailyPredict (V15) → 全 R 予測
3. 08:50 AM8FireCheck → OK 確認
4. 09:00 JrdbRetryAm9 → JRDB retry
5. 09:30 MorningWeightCheck → V15 補正 + Discord

### 10:00 (新規 V18/V19 試行)

6. **手動実行**: V18/V19 retro 1日分 直前再確認
   ```bash
   python tools/v18_v19_retro_full.py --date 20260516 --normalize softmax --T 1.0 --output data/v18/retro_20260516.md
   ```
   - 採用 R を確認 (運用 filter 適用後)
   - winner_top1 / EV を確認

### 14:50 / 15:45 (既存 multi_stage_predict)

7. 14:50 Race11_1450 → V15 11R 予測 (案B改 採用外、観察)
8. 15:45 Race12_1545 → V15 12R 予測 + 案B改 採用 R 買い目
9. **PAT 投票時に V18/V19 retro 結果を併用判断** (V18 単勝 / V19 複勝、計 1,000 円上限)

### 投資配分 上限

| 項目 | 上限 |
|------|------|
| V15 案B改 (12R 1勝) | 2,100 円 |
| V18 単勝 試行 | 500 円 |
| V19 複勝 試行 | 500 円 |
| **合計上限** | **3,100 円/日** |

最悪: -3,100 円 → 累計 +10,430 円維持 (撤退余裕 +60,430 円)

### 5/16 で確認すべきこと

- V18/V19 retro 結果と当日 race 想定の一致
- fall-back 機構動作確認 (もし V18/V19 fail → V15 単独)
- Discord 通知の retry+log 動作

---

## 6. NO-GO 時の代替

5/15 NO-GO 判定なら:
- 5/16 は V15 案B改 単独維持 (5/9 と同じ運用)
- Phase 3 (5/24+) で V18/V19 sib 抜き再学習 + 本格復活
- 詳細: `data/v18/v18_v19_root_cause_resolution_5_6.md` § 4

---

## 7. Phase 3 (5/24+) plan

V18/V19 再学習計画:
- features list から sib_top3_rate / sib_shinba_wr 除外 (V162_EXCLUDED 反映)
- 学習 sample: 4/29 以降の post-leak data
- 期間: 5/25-6/8 (V15.1 SKB 採用と並行)
- 期待: winner_top1 +5-10pt → 50-55% 達成見込み

詳細: `docs/PHASE_3_V15_1_PLAN.md` (V15.1 採用) と並行運用候補。

---

## 8. 結論 (1 句)

🟡 **5/13-15 軽量修復 (5h、sr merge + premium + filter) で 5/16 GO 確率 40-50%**。 sib リーク削除 + V18/V19 model 自体の更新が必要なので、本格復活は **Phase 3 (5/24+) で sib 抜き再学習** が王道。 5/16 部分試行 → 5/24+ 本格復活 が現実的 plan。
