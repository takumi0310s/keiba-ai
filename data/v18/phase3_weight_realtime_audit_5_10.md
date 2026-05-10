# Phase 3 / C: 体重・直前情報 統合 audit (5/10) ★最重要★

## 結論
**case A: 朝予測 = 体重統合 なし (V15 設計通り)**
- 5/10 投票判断 影響なし (V15 は朝予測 base で運用、 +¥14,140 累計実績)
- ⏸ 1h 前 Stage 2 (`PreRacePredict_Watchdog_5_9`) **disable のまま** = 5/10 体重統合なし
- 📅 5/15 V18 trial 直前 PreRacePredict re-enable で Stage 2 体重統合 復活 plan

## V15 features 中の 体重 関連

`tools/predict_core.py` で使用:
- `馬体重` (default 480)
- `場体重増減` (default 0)
- `weight_cat` (体重カテゴリ 0-3)
- `weight_dist` (体重 × 距離)
- `weight_change`, `weight_change_abs`
- `carry_per_weight` (斤量 / 体重)
- `weight_cat_dist` (体重カテゴリ × 距離カテゴリ)
- `horse_weight` (英語 alias)

→ **V15 model には 体重 features 8 個 込み**。 但し default 全馬同値 (480/0) なら 非情報。

## timing 別 体重統合 status

| 時刻 | 機構 | 体重取得 | 5/10 status |
|------|------|---------|------------|
| 8:00 | `daily_predict.py` (朝予測) | ❌ shutuba.html → 70 分前公開のため 8:00 では default 480 | ✅ 動作 (体重 default) |
| 9:30 | `save_all_horse_scores.py` (Phase 2 緊急) | ❌ 10:00 R のみ 70 分前条件成立、 残 35 R は default 480 | ✅ 動作 (体重 default 大半) |
| 9:30 | `morning_weight_check.py` (Phase 2 緊急) | ❌ 同上 (5/10 csv は 1 row、 weight=480, diff=0) | ✅ 動作 (体重未公開と判定) |
| 発走 1h 前 | `PreRacePredict_Watchdog_5_9` (Stage 2) | ⭕ 70 分前条件成立、 体重統合可 | ⏸ **DISABLED** |

## 5/10 朝の真の予測 status

```
- 朝予測 (8:00):     ❌ 体重未統合 (default 480)        → V15 設計通り
- 9:30 全馬 score:    ❌ 体重未統合 (default 480)        → 9:30 時点で 70 分前条件不足
- 1h 前 Stage 2:     ⏸ 動作なし (PreRacePredict disable) → 5/15 re-enable
- 投票判断時:         ❌ 体重情報なし                    → V15 設計通り (累計 +¥14,140 実績)
```

→ ★ **5/10 投票 = 朝予測 + 9:30 score (両方 体重 default) で判断** ★

## case 判定

### case A: 朝予測が 体重含まない → 設計通り ✅
- ✅ V15 は朝予測 base 運用 (累計 +¥14,140 / +¥45,920 / 全期間 ROI 120.2% 実績)
- ✅ 5/10 投票 影響なし
- ⏸ 5/15 PreRacePredict re-enable で Stage 2 体重統合 復活

→ **本 case A 確定**

### case B: 朝予測も 体重統合 期待 だが 取得失敗
- ❌ 該当せず (出馬表 70 分前公開 = 8:00 取得不可は仕様)

### case C: stage2_predict.py 5/9 hardcode 残存
- ❌ Stage 2 自体 disable のため 該当せず
- 📝 5/15 re-enable 時 hardcode 確認必要

## 5/10 投票 影響 evaluation

**結論**: **影響なし**

理由:
1. V15 model 学習時から 朝予測 base で AUC 0.8939 達成
2. 体重 8 features は default 480/0 で 全馬同値 → 非情報化 → 他 features (騎手 / JRDB / 前走) で予測
3. 累計 +¥14,140 (5/9 朝時点) も同 logic で実績
4. 案B改 strict 12R 1 勝クラス ¥2,100 投票 推奨 (Phase 2 緊急 確定済)

## 5/15 V18 trial 直前 改善 plan

### step 1: PreRacePredict_Watchdog re-enable
```powershell
schtasks /change /tn "Keiba-PreRacePredict_Watchdog_5_9" /enable
```

### step 2: Stage 2 体重統合 動作確認
- `tools/multi_stage_predict.py --stage race12_1545 --date 20260515 --dry-run` 実行
- 体重 default 480 → 実体重 反映確認
- weight_change = 0 → 実増減 反映確認

### step 3: hardcode date 確認
- `tools/multi_stage_predict.py` / `tools/pre_race_predict.py` 内
- "20260509" / "5/9" などの hardcode 残存 grep
- Session #78 で修正済 だが 再確認

### step 4: 5/16 V18 trial で初稼働
- Stage 2 体重統合 で 通知精度向上
- 体重 急変 (±10kg) アラート追加検討

## 修正 (今回)
✅ **なし** (case A = 設計通り、 5/15 plan 化)
