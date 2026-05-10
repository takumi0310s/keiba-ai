# Phase 12 C: ハロンタイム + 天候馬場 features (6 件) 実装 (5/10)

> Session #87 Phase 12 C 領域 (2026-05-10 18:00+)
> 出力: tools/predict_core_v18.py の HARONTIME_FEATURES (3) + WEATHER_BABA_FEATURES (3)

---

## 1. user 指示 vs 実 JV-Link record mapping

| user 指示 | 実 JV-Link record | 用途 |
|----------|-------------------|------|
| 「HS ハロンタイム」 | SE record の区間タイム埋め込み | 3 features (前走前半/後半 3F + ペース) |
| 「WF 天候馬場」 | WE (馬場) + WH (天候) records | 3 features (含水率 + 馬場差 + 天候変化) |

★ 注: JV-Link 仕様上、 HS = 生産者 record / WF = WIN5 record。 「ハロンタイム」 は SE record 内部の区間タイム fields、 「天候馬場」 は WE / WH records が正規 source。 user の functional intent に基づき正規 source で実装。 ★

---

## 2. 実装 features

### 2.1 ハロンタイム (3 features) — SE record

| feature | source | default |
|---------|--------|---------|
| jv_lap_first3f_pred | SE record 前半 3F (確定) | 36.0 (1200m 平均) |
| jv_lap_last3f_pred | SE record 後半 3F = 上がり 3F | 36.0 |
| jv_race_pace_index | first3f / last3f 比 (1.0 標準) | 1.0 |

V15 既統合: prev_last3f, avg_last3f_3r, prev_race_first3f, prev_race_last3f, pci

Phase 12 C 差別化:
- ★ jv_race_pace_index ★ は V15 の pci (後半 3F / 前半 3F) と類似だが、 SE record 確定値ベース (V15 は netkeiba ラップ)

### 2.2 天候馬場 (3 features) — WE + WH records

| feature | source | default |
|---------|--------|---------|
| jv_baba_moisture | WE record 馬場含水率 (%) | -1.0 (不明) |
| jv_baba_difference | WE record 馬場差 (内有利 = +、 外有利 = -) | 0.0 (中央) |
| jv_weather_change_score | WH 履歴差分 (急変 = 1) | 0 (安定) |

V15 既統合: cushion_value, moisture_rate (JRA 公式 scrape 経由)、 weather_enc, temperature, humidity, precipitation, wind_speed (気象庁 API)

Phase 12 C 差別化:
- ★ jv_baba_moisture ★ は JRA 公式 vs JV-Link 両 source で取得 (信頼性 ↑)
- ★ jv_baba_difference ★ は JV-Link 独自 (内/外 有利度、 V15 未統合)
- ★ jv_weather_change_score ★ は 履歴比較 (V15 では 単時点のみ)

---

## 3. live activation 設計

### 3.1 Phase 12 (本日)
- skeleton 実装、 default fill のみ
- prev_race_id を caller から受け取る設計
- prev_race_id None → ハロン default 36.0

### 3.2 5/24+ Phase 3 後半
- tools/jvlink_fetcher_v2.py の SE parser 拡張 (区間タイム fields 抽出)
- WE / WH parser 新規実装
- `data/jvlink/{SE, WE, WH}/<race_id>_parsed.csv` 経由

---

## 4. 動作 test (Phase 12 全体)

```
C. ハロンタイム (3): ['jv_lap_first3f_pred', 'jv_lap_last3f_pred', 'jv_race_pace_index']
D. 天候馬場 (3): ['jv_baba_moisture', 'jv_baba_difference', 'jv_weather_change_score']
default fill 動作確認: 全 6 features OK
```

---

## 5. V15 投資保護

✅ tools/predict_core.py 不変
✅ V15 model 不変
✅ predict_core_v18.py 新規、 V15 と完全独立
✅ JRA 公式 scrape (cushion / moisture) 既存 経路 不変
✅ 気象庁 API 経路 不変

---

## 6. 結論

✅ C1: ハロンタイム 3 features (SE record 確定値)
✅ C2: 天候馬場 3 features (WE + WH records)
✅ C3: V15 既統合 (pci / moisture_rate / weather_enc) と差別化
✅ C4: skeleton 動作 OK (default fill 6 features)
✅ C5: V15 完全保護
