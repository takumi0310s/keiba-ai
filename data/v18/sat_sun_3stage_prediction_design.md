# 当日 3 段階自動予測機構 設計 (5/9 本番)

**作成**: 2026-05-06 朝活 (Session #28)
**目的**: 土日本番で 10:00 / 14:50 / 15:45 の 3 段階で全開催場を予測 + Discord 通知
**狙い**: 馬体重補正 を最大限活用、案B改 フィルタを段階別に精密化

---

## 1. 機構の概要

| 時刻 | 対象 R | 予測タイプ | 通知範囲 | 用途 | 投票影響 |
|------|--------|-----------|----------|------|---------|
| **10:00** | 2R (実値) + 3R-12R (朝予測) | 馬体重補正 (2R) + 朝予測 (他) | 全 R 状態 | 情報取得正常性、機構動作確認 | なし (情報提供のみ) |
| **14:50** | 11R (実値、重賞含む) | 馬体重補正 | 全 11R 表示 (重賞含む、買い目なし) | 11R 観察 + 投票判断 | 該当時のみ買い目 |
| **15:45** | 12R (実値) | 馬体重補正 | 全 12R 表示 + 採用 R の買い目 | 12R 投票 ★主戦場 | 案B改 採用なら買い目 |

各 stage で `tools/predict_one_race.py` を再実行し、馬体重 公開済み の R では実値で V15 予測を再計算。

---

## 2. 各タイミングの馬体重公開状況

netkeiba 出馬表ページの公開タイミング (経験則):
- **馬体重**: 発走 約 70 分前〜
- **確定オッズ**: 発走 約 1 分前

| 時刻 | 確実に公開済 | 未公開 / 終了済 |
|------|-------------|---------------|
| 10:00 | 2R (発走 10:30 想定) | 1R 直前 / 3R 以降未公開 |
| 14:50 | 11R (発走 15:25 想定) | 12R 未公開、1-10R 終了済 |
| 15:45 | 12R (発走 16:20 想定) | 11R 終了済、1-10R 終了済 |

**重要**: 10:00 stage で 3R-12R を「実値で再予測」するのは netkeiba 仕様で不可能。 朝予測 (08:00 DailyPredict) のまま表示する。

---

## 3. 採用フィルタ (案B改 ベース)

### 3.1 stage='test10' (10:00)

R フィルタ: 全 R (2R + 3R-12R)
案B改 フィルタ: **適用しない** (情報提供のみ、投票判断なし)
表示:
- 2R: 馬体重補正後予測 (実値)
- 3R-12R: 朝予測 そのまま (馬体重デフォルト)

### 3.2 stage='race11_1450' (14:50)

R フィルタ: 11R のみ (全開催場)
案B改 フィルタ:
- G1/G2/G3 (重賞) → **採用外 (重賞)**、予測のみ表示、買い目なし
- 駿風 S 等の距離不一致 OP → **採用外 (条件)**
- 1勝クラスがあれば → **採用**、買い目あり (ほぼ該当なし、11R は重賞 or OP が定石)
- その他 (2勝/3勝/OP/特別/S) → **採用外 (条件)**

判定ロジック (race_name から):
```python
def classify_11r(race_name: str) -> tuple[bool, str]:
    """戻り: (採用?, 理由)"""
    rn = race_name.replace(' ', '')
    # 重賞 (G1/G2/G3)
    if any(g in rn for g in ['(G1)', '(G2)', '(G3)', 'G1', 'G2', 'G3']) or any(rn.endswith(s) for s in ['杯', 'カップ', '記念']):
        return False, "重賞"
    # 1勝クラス
    if '1勝' in rn:
        return True, "1勝クラス"
    # その他
    return False, "条件"
```

### 3.3 stage='race12_1545' (15:45)

R フィルタ: 12R のみ (全開催場)
案B改 フィルタ:
- 1勝クラス → **採用**、買い目あり (案B改 の主戦場)
- その他 (2勝/3勝/OP/特別/重賞) → **採用外 (条件)**、予測のみ表示

```python
def classify_12r(race_name: str) -> tuple[bool, str]:
    if '1勝' in race_name:
        return True, "1勝クラス"
    if '未勝利' in race_name:
        return False, "未勝利"
    if '2勝' in race_name or '3勝' in race_name:
        return False, "2勝/3勝"
    if any(s in race_name for s in ['特別', 'S', 'OP', 'リステッド', 'L']):
        return False, "特別/OP"
    return False, "その他"
```

---

## 4. Discord 通知 format

### 4.1 stage='test10' (10:00)

```
🔍 [10:00 テスト予測] 5/9 (土)
開催: 東京/京都/新潟 (3場 36R)

★ 2R 馬体重補正後 (実値):
東京 2R: TOP1 馬5 score 0.615 (朝0.580 +0.035) 馬体重 +6kg
京都 2R: TOP1 馬3 score 0.498 (朝0.521 -0.023) 馬体重 -2kg
新潟 2R: TOP1 馬8 score 0.703 (朝0.682 +0.021) 馬体重 +1kg

★ 3R-12R 朝予測 (馬体重未公開、参考):
全 30R 朝予測通り

機構: 動作正常
次回: 14:50 (11R 一括) / 15:45 (12R 一括)
```

### 4.2 stage='race11_1450' (14:50)

```
🏇 [14:50 11R 一括予測] 5/9 (土)
全 3場 11R 予測 (重賞含む):

★ 東京 11R G2 (NHKマイルC):
   軸 馬7 (TOP1 score 0.523, 馬体重 +2kg)
   採用外 (重賞)、観察用予測

★ 京都 11R G3 (京都新聞杯):
   軸 馬12 (TOP1 score 0.481, 馬体重 -1kg)
   採用外 (重賞)、観察用予測

★ 新潟 11R OP (駿風S):
   軸 馬3 (TOP1 score 0.502, 馬体重 +4kg)
   採用外 (距離不一致)、観察用予測

投資合計: 0円 (案B改 フィルタ全 NG)
```

### 4.3 stage='race12_1545' (15:45)

```
🏇 [15:45 12R 一括予測] 5/9 (土)
採用 R: 2/3

★ 東京 12R 1勝クラス:
   軸 馬5 (TOP1 score 0.615, 馬体重 +6kg)
   買い目: 三連複 7点 5-{2,8}-{2,8,1,7,3} = 700円

★ 京都 12R 1勝クラス:
   軸 馬9 (TOP1 score 0.589, 馬体重 -3kg)
   買い目: 三連複 7点 9-{4,11}-{4,11,2,6,8} = 700円

・新潟 12R OP特別:
   軸 馬2 (score 0.461)
   採用外 (OP特別)、観察用予測

投資合計: 1,400円
累計余裕: +62,830円 (撤退ライン超まで)
```

---

## 5. アーキテクチャ

```
tools/multi_stage_predict.py
  ├── load morning predictions (data/daily_predictions/{ymd}.csv)
  ├── select target races (stage 別 R フィルタ)
  ├── for each target:
  │     if 馬体重対象R:
  │         predict_one_race(race_id) ←既存
  │         compare with morning
  │     else:
  │         use morning prediction as-is
  ├── apply 案B改 フィルタ (stage 別)
  ├── generate buying pattern (採用 R のみ)
  ├── format discord message (stage 別 template)
  └── send to Discord #updates

tools/multi_stage_predict.bat (silent vbs wrapper)
tools/multi_stage_predict_config.json (stage 別設定)
tools/register_multi_stage_predict_schtasks.ps1 (admin で 6 task 登録)
```

---

## 6. 既存 morning_weight_check.py との関係

| 機構 | 時刻 | 対象 | 役割 |
|------|------|-----|------|
| morning_weight_check | 09:30 | 案B改 採用候補のみ (12R 1勝) | 早朝補正、買い目変化 alert |
| **multi_stage_predict (本書)** | 10:00/14:50/15:45 | 全開催場 stage 別 | 段階別 全予測、本番投票判断 |

**役割分担**: 09:30 は早朝の concept 確認 (案B改 採用候補のみ)、10:00/14:50/15:45 は本番投票直前の包括予測 (全開催場、stage 別)。 重複しないが補完関係。

---

## 7. schtasks 設計

| TaskName | 曜日 | 時刻 | bat |
|----------|------|------|-----|
| Keiba-MultiStagePredict_Test10_Sat | SAT | 10:00 | multi_stage_predict.bat test10 |
| Keiba-MultiStagePredict_Test10_Sun | SUN | 10:00 | 同上 |
| Keiba-MultiStagePredict_Race11_1450_Sat | SAT | 14:50 | multi_stage_predict.bat race11_1450 |
| Keiba-MultiStagePredict_Race11_1450_Sun | SUN | 14:50 | 同上 |
| Keiba-MultiStagePredict_Race12_1545_Sat | SAT | 15:45 | multi_stage_predict.bat race12_1545 |
| Keiba-MultiStagePredict_Race12_1545_Sun | SUN | 15:45 | 同上 |

silent_runner.vbs 経由、admin 1 コマンド (register ps1) で登録完了。

---

## 8. 失敗パターンの handling

| ケース | 対応 |
|--------|------|
| 馬体重未公開 race | 朝予測のまま表示、note 「馬体重未公開」 |
| netkeiba ban | scrape 失敗、Discord 警告 + 朝予測のまま表示 |
| Cookie 切れ | refresh_cookie --auto で復旧、失敗なら Discord 警告 |
| 朝予測 CSV 欠落 | 「DailyPredict 未完了」と通知してスキップ |
| 全 R 取得失敗 | Discord critical 通知 |
| race_name 判定不能 | 採用外 (判定不能) として表示 |

---

## 9. 5/9 リハーサル

5/8 (金) 22:00: 5/3 (日) データで dry-run、Discord format 確認
5/9 (土) 本番: 自動発火 (10:00/14:50/15:45)

詳細は `docs/sat_sun_5_9_rehearsal_plan.md`。

---

## 10. 結論

3 段階予測機構を新設し、 09:30 (案B改 候補早朝確認) と 10:00/14:50/15:45 (全開催場段階別) で 4 段階監視体制を構築。 投票主戦場は 15:45 12R、観察用は 10:00 全 R + 14:50 11R。 案B改 フィルタを段階別に精密化し、重賞は予測実行 (買い目なし)、1勝クラスのみ買い目生成。
