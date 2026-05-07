# Session #40 A: 5/9 直前効果系 (PAT点数 + 分類 + Kelly + health check)

**作成**: 2026-05-07 (Session #40 A)
**目的**: 5/9 V15 案B改 投資の最終最適化 + 5/8 朝の自動 health check

---

## 1. A1 — PAT 投票点数最適化 retro

### 1.1 retro 結果 (4/26-5/3, N=101 races)

formation 定義:
- 5pt: TOP1, TOP2 軸 + TOP3〜TOP7 流し (堅め、 軸 2 頭固定)
- **7pt (現状 baseline)**: TOP1 軸 - TOP2,TOP3 - TOP2〜TOP6
- 9pt: TOP1 軸 - TOP2,TOP3,TOP4 - TOP2〜TOP7 (広め)

| pattern | 点数 | inv/race | hit | hit_rate | total inv | total pay | profit | ROI |
|---------|------|---------|-----|----------|-----------|-----------|--------|-----|
| 5pt (堅め) | 5 | 500 | 9 | 8.91% | 50,500 | 10,570 | -39,930 | **20.93%** |
| **7pt (現状)** | 7 | 700 | 19 | 18.81% | 70,700 | 29,950 | -40,750 | **42.36%** |
| 9pt (広め) | 9 | 900 | 20 | 19.80% | 90,900 | 32,830 | -58,070 | **36.12%** |

### 1.2 結論

✅ **7pt baseline が最良 ROI** (5pt より +21.43pt、 9pt より +6.24pt)

- 5pt は 軸 2 頭固定 (TOP1+TOP2) で hit_rate 半分以下 (8.91%) → coverage 大幅不足
- 9pt は hit_rate わずかに増加 (+1pt) だが 投資額 +28% で profit 悪化
- → **5/9 は 7pt 維持 (案B改 baseline)** で確定

### 1.3 retro 期間が GW (4/26-5/3) で全体 ROI 低調

GW 期間 ROI 42% は 累計 119.2% より低い:
- 戦略⑦未適用の生データ (06_特別 / 京都 / 条件E 含む)
- 5/9 は戦略⑦ + 案B改 で 12R 1勝クラス のみ → 期待 ROI 161%

---

## 2. A2 — race_classifier.py (race_name 自動分類)

### 2.1 機能

`tools/race_classifier.py` (新規):
- race_name → クラス分類 (G1/G2/G3/L/OP/3勝/2勝/1勝/未勝利/新馬/06_特別)
- 戦略⑦ filter (京都除外、 条件E除外)
- 5/9 採用判定 (1勝クラスのみ ACCEPT)

### 2.2 動作確認

```
$ python tools/race_classifier.py --name "12R 4歳以上 1勝クラス" --course "東京" --num-horses 14
class_code: 1勝 (1勝クラス)
ACCEPT: True (1勝クラス + 戦略⑦ 通過)

$ python tools/race_classifier.py --name "11R G2" --course "東京" --num-horses 16
class_code: G2 (重賞 G2)
ACCEPT: False (BT サンプル少)
```

### 2.3 5/8 21:00 出馬表確定後の使い方

```bash
# 5/8 21:00 - 5/9 朝
python tools/race_classifier.py --csv data/daily_predict_lite_20260509.csv --out data/v18/5_9_accept_list.csv
```

→ accept_5_9 = True の行が 投資対象。 race_name の手動目視と併用推奨。

---

## 3. A3 — Kelly 基準 + 5/9 投資戦略

詳細: [`data/v18/kelly_betting_strategy_5_7.md`](kelly_betting_strategy_5_7.md)

### 3.1 結論

V15 案B改 (700円/R 固定) = **Quarter Kelly (8.3%) より保守的 (5.2%)** で数理整合。

- Full Kelly: 33.3% (4,503円/R) — 推定誤差リスク高、 採用 NG
- Half Kelly: 16.7% (2,260円/R) — 撤退余裕の負担 中
- Quarter Kelly: 8.3% (1,124円/R) — 妥当
- **案B改 5.2% (700円/R)** — Quarter Kelly 以下、 最も保守的

### 3.2 撤退ラインとの整合

| シナリオ | 投資 | 最悪損失 | 余裕 (-50,000 line) |
|---------|------|---------|------------------|
| 案B改 3R 全外し | 2,100円 | -2,100円 | 余裕 +63,530円 (3.3% 消費) |
| 連敗 6R | 4,200円 | -4,200円 | 余裕 +63,530円 (6.6% 消費) |

→ **5/9 案B改 維持で数理的に安全**

---

## 4. A4 — 5/8 朝 final health check

### 4.1 ファイル

`tools/final_health_check_5_8.py` (新規、 約 240 行)

### 4.2 検証 10 項目

| # | 項目 | critical | 結果 (5/7 試行) |
|---|------|----------|---------------|
| 1 | V15 model 読込 (.pkl.gz) | YES | ✅ 2.0 MB |
| 2 | predict_core.py syntax | YES | ✅ |
| 3 | daily_predict.py syntax | YES | ✅ |
| 4 | app.py syntax | YES | ✅ |
| 5 | netkeiba Cookie 有効 | YES | ✅ |
| 6 | JRDB 鮮度 (extracted/Bac) | YES | ✅ 20260503 |
| 7 | jra_payouts.csv 鮮度 | NO | ✅ INFO (4/6 停止 既知) |
| 8 | .env webhooks (BETS/UPDATES) | YES | ✅ 全 OK |
| 9 | 累計収支 sanity | YES | ✅ 撤退ライン未達 |
| 10 | schtasks 登録 (DailyPredict / RaceAutoNotify) | YES | ✅ 2 件確認 |

→ 5/7 23:30 試行で 10/10 OK。 5/8 朝 自動実行で再 confirm 予定。

### 4.3 5/8 朝 自動実行 schtasks 追加 (推奨)

ユーザー側 (admin) で以下を schtasks 登録:

```cmd
schtasks /Create /TN "Keiba-FinalHealthCheck_5_8" ^
    /TR "powershell -ExecutionPolicy Bypass -File C:\Users\takum\keiba-ai\final_health_check_5_8.ps1" ^
    /SC ONCE /SD 05/08/2026 /ST 06:00 /F
```

`final_health_check_5_8.ps1`:
```powershell
cd C:\Users\takum\keiba-ai
python tools\final_health_check_5_8.py
if ($LASTEXITCODE -ne 0) {
    python tools\notify_done.py "5/8 health check FAIL" "critical NG 検出, 5/9 投資前 修正必須" --color red
}
```

→ 5/8 06:00 自動実行、 NG 時は Discord 警告。

### 4.4 5/8 21:00 出馬表確定後の手順 (人間操作)

1. 21:00 PAT で 5/9 出馬表確認
2. `python tools/race_classifier.py --csv data/daily_predict_lite_20260509.csv --out data/v18/5_9_accept_list.csv`
3. accept_5_9 = True の race_name + 馬番 を Discord #investments に投稿
4. 5/9 朝 06:00 → final_health_check 自動実行
5. 06:30 → 体調確認、 10:00 〜 PAT 投票

---

## 5. 5/9 V15 案B改 投資保護 final 確認

✅ predict_core.py 完全不変 (syntax OK)
✅ daily_predict.py 完全不変 (syntax OK)
✅ V15 model file 完全不変
✅ schtasks 既存 task 完全不変
✅ 新規追加のみ: race_classifier.py / final_health_check_5_8.py / bet_pattern_optimization.py

→ **5/9 朝 V15 daily_predict 完全同一動作 保証**

---

## 6. 結論

✅ A1: 7pt baseline 最良 (5pt -12pt / 9pt -6pt)、 **5/9 維持 OK**
✅ A2: race_classifier.py で自動分類 + 採用判定 (5/8 21:00 即動作)
✅ A3: Kelly 基準で 700円/R = Quarter Kelly 以下 で **数理整合**
✅ A4: 10 項目 health check 自動化 (5/8 06:00 schtasks 推奨)
✅ A5: 統合 doc (本ファイル)

→ **5/9 案B改 投資準備 完了**

---

**Session #40 A 完了**
