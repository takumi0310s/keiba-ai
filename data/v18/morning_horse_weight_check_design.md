# 当日朝 馬体重 → 予測補正 機構 設計

**作成**: 2026-05-06 朝 (Session #26)
**目的**: 発走 ~1 時間前に公開される馬体重を反映した予測 vs 朝 08:00 予測を比較、変化大きい場合 Discord アラート
**狙い**: 5/9 朝 試運転 → 5/16 以降 本格運用

---

## 1. 機構の概要

```
06:30 morning_top_races (Keiba-Morning_Sat) — 11R/12R 軸候補通知 (馬体重なし)
08:00 daily_predict (DailyPredict)            — 全 R 予測 (馬体重デフォルト 480kg)
                                                 → data/daily_predictions/{ymd}.csv
[本機構] 09:30 morning_weight_check            — 主要 R で再予測 (馬体重 公開済み)
                                                 → 朝予測と diff、変化大なら Discord
14:00 PAT 投票
18:00 結果集計
```

公開タイミング (経験則 + netkeiba 仕様):
- 馬体重: 発走 約 70 分前〜 (出馬表ページに表示)
- 確定オッズ: 発走 約 1 分前
- レース当日朝 09:00-10:00 の段階で 1R-12R すべて馬体重利用可能なケースが多い

---

## 2. 馬体重 features (V15 既実装)

`tools/predict_core.py` 内で既に実装済み:

| feature | 出所 | 効果 |
|---------|------|------|
| `horse_weight` | 出馬表 `(\d{3,})\(([\+\-]?\d+)\)` (L713-721) | 体重絶対値 (350-600kg clip) |
| `weight_change` | `'場体重増減'`.fillna(0) (L1837) | 前走比 ±diff (kg) |
| `weight_change_abs` | `weight_change.abs()` (L1838) | 絶対値 |
| `weight_cat` | pd.cut 4 段階 (≤440 / ≤480 / ≤520 / >520) (L1547) | 体重カテゴリ |
| `weight_cat_dist` | weight_cat × distance category (L1548 周辺) | 体重×距離カテゴリ |
| `体重カテゴリ` | L1501 4 段階 | 既存版 |
| `weight_dist` | `馬体重 * 距離 / 10000` (L1809) | スケール化体重 |
| `carry_per_weight` | `斤量 / 馬体重 * 100` (L1811) | 斤量比 |

→ **新規 features 不要**、馬体重を取得さえできれば V15 の予測 確率が自動で再計算される。

---

## 3. 予測補正の判断基準

朝 08:00 予測 vs 09:30 補正後予測を比較:

| 指標 | 閾値 | アクション |
|------|------|----------|
| **軸馬の確率** | ±5% 以下 | 朝の買い目維持 |
| 〃 | ±5-10% | 注意、軽量で投資 |
| 〃 | ±10% 以上 | 修正検討、Discord アラート |
| **TOP1-3 構成変化** | 1 馬入替 | 注意 |
| 〃 | 2 馬以上入替 | 修正検討、Discord アラート |
| **馬体重 ±15kg 以上** | 軸馬対象 | 軸変更検討、Discord アラート |
| 〃 | 相手馬対象 | 相手選定見直し、Discord 注意 |
| **馬体重 ±10kg かつ 確率 ±5%** | 同時条件 | Discord 注意 |

判断はあくまでユーザーの最終裁量、機構は **情報提供** のみ。

---

## 4. アーキテクチャ

```
┌─────────────────────────────────────────────────────┐
│ tools/morning_weight_check.py (新規)                 │
│                                                       │
│ 1. data/daily_predictions/{ymd}.csv 読込             │
│    (朝 08:00 の予測、top1-6 + スコア)                │
│                                                       │
│ 2. レース選定                                         │
│    - --races: 特定レースのみ                          │
│    - default: 案B改 採用候補 R (12R 1勝クラス)       │
│                                                       │
│ 3. for race_id in races:                            │
│      result = predict_one_race(race_id) ←既存 module │
│      → 馬体重 公開済みなら自動で新値                  │
│                                                       │
│ 4. 朝予測 と 比較:                                    │
│    - top1 確率 diff                                   │
│    - top1-3 メンバー入替                              │
│    - 馬体重 ±15kg の馬 detect                        │
│                                                       │
│ 5. 閾値超えで Discord 通知                            │
│                                                       │
│ 6. 結果 CSV 保存:                                     │
│    data/morning_weight_check/{ymd}.csv               │
└─────────────────────────────────────────────────────┘
```

---

## 5. 自動化

- schtasks 登録: 09:30 (土曜) を初期値
- silent_runner.vbs 経由
- `tools/morning_weight_check.bat` wrapper
- `tools/register_morning_weight_check_schtasks.ps1` (admin で 1 回)

実行コマンド:
```powershell
wscript.exe tools\silent_runner.vbs tools\morning_weight_check.bat
```

5/9 朝の試運転後、5/16 から本格運用。

---

## 6. 通知フォーマット (例)

### case 1: 安定 (alert なし)

```
🐴 馬体重チェック 5/9 09:30
東京12R: 軸馬 #5 馬体重 506(+2)、TOP1-3 維持
京都12R: 軸馬 #3 馬体重 484(-4)、TOP1-3 維持
新潟12R: 軸馬 #7 馬体重 470(0)、TOP1-3 維持
→ 全 R 朝の買い目維持で OK
```

### case 2: 変化大 (alert あり)

```
🚨 馬体重チェック ALERT 5/9 09:30
東京12R: 軸馬 #5 馬体重 522 (+18kg) 急増
   朝予測 確率 0.42 → 補正後 0.31 (-11%)
   TOP1 が #5 → #2 に入替
   → 修正検討 / Discord #bets で再通知済
京都12R: 馬体重変化なし、買い目維持
新潟12R: ...
```

---

## 7. テスト計画

### 過去レース dry-run

5/3 (日) のレース (`data/daily_predictions/20260503.csv`) を使い、現時点で 馬体重 公開済み のはずなので比較が成立。

```bash
python tools/morning_weight_check.py --date 20260503 --dry-run
```

期待:
- 朝予測と現時点予測で 馬体重差分が反映される
- 通知形式の動作確認

### 5/9 当日

```bash
# 09:30 自動発火 (silent_runner 経由)
python tools/morning_weight_check.py --date 20260509
```

期待:
- 案B改 12R (採用候補) のみで実行
- 馬体重 取得成功率 100%
- Discord 通知 1 件 (alert あり/なし問わず)

---

## 8. 想定する外し方

| ケース | 対応 |
|--------|------|
| 馬体重未公開 (出馬表に未掲載) | scrape 結果 `馬体重 = 480 (default)` で予測継続、通知に "馬体重未取得" 注記 |
| netkeiba ban | scrape 失敗で skip、Discord 警告 |
| Cookie 切れ | refresh_cookie --auto で自動復旧、失敗なら警告 |
| 朝予測 CSV 欠落 (DailyPredict 未完了) | `09:30 まだ DailyPredict 完了せず` と通知してスキップ |
| predict_core 不整合 | try/except で個別 R skip、他 R 継続 |

---

## 9. 5/9 採用判定

5/9 の試運転で:
- 馬体重 取得成功率 ≥ 90%
- 通知形式が読みやすい
- 朝予測 vs 補正後の diff が解釈可能

→ 5/16 から本格運用。 案B改 採用 R + 11R (将来 G3 検討時) で実行。

---

## 10. 結論

V15 には馬体重 features 既実装、新規モデル不要。 09:30 で predict_one_race 再実行 → 朝予測と比較 → 閾値超え Discord アラート、というシンプル機構。 工数: 設計 10min / 実装 30min / ps1 5min / テスト 15min = 60min。
