# Session #49 D: 5/10 verdict framework 拡張

**作成**: 2026-05-08 23:XX (Session #49 D、 dev/training-poc)
**目的**: 5/10 朝 重賞 5 system verdict 拡張 (Session #46 result_verification + 重賞専用)

---

## 1. 構成

`tools/verify_majors_5system_5_10.py` (190 行):
- 5 system 個別 hit_rate 計算
- 動画 features (System 5) 貢献度
- 「もし重賞 buy したら ROI ○○%」 算出
- 5/5 一致 / 4/5 一致 など 高信頼時 hit_rate

---

## 2. 5/10 朝 timeline

```
05:00 起床
06:00 NightlySanity (5/9 23:00 起動分)
08:00 result_verification_5_10.py (Session #46 D、 V15 案B改 verdict)
08:30 verify_majors_5system_5_10.py 実行 (本 Session D)
       → 重賞 3 R verdict
       → 5 system 個別 hit_rate
       → 「もし重賞 buy したら」 estimated ROI
       → Discord #investments 通知
09:00 ユーザー手動 confirm + 5/16 plan 検討
```

---

## 3. 動作確認 (5/8 23:XX)

```
$ python tools/verify_majors_5system_5_10.py --date 20260509
[Predictions] available: True (Session #49 C 出力 load OK)
  n_majors: 3
[Actual results] available: False (5/9 18:00 後 利用可)
[System evaluation] results_unavailable
[Consensus high confidence] deferred
[Estimated ROI if bought] deferred
```

→ 5/9 結果照合後、 5/10 朝に 全 verdict 自動算出可能

---

## 4. 5/10 朝 出力 例 (5/10 朝 実 run 後)

```
[Predictions] n_majors: 3
[Actual results] n_majors: 3 (5/9 18:00 DailyResults 自動 update)

[System evaluation]:
  system_1 (V15): top1_rate 33.33% (1/3)
  system_2 (拡張調教): top1_rate ?? (deferred)
  system_3 (TM): top1_rate ?? (deferred)
  system_4 (当日体重): top1_rate ?? (deferred)
  system_5 (動画): top1_rate ?? (deferred、 5/9 朝 動画 DL 後)

[Consensus high confidence]:
  n_high_confidence: 1 (5/3 一致 R)
  → 高信頼時 hit_rate ?% (sample 少、 累積で評価)

[Estimated ROI if bought]:
  scenario: 全 3 重賞 高信頼時 馬連 軸 1 頭流し 2 点 (200 円/R)
  max_loss: 600 円 (3 R × 200 円)
  actual_payout: ¥??? (5/10 朝 集計)
  ROI: ?? %
  ★ ただし 5/9 重賞 投票なし、 計算のみ (絶対遵守) ★
```

---

## 5. 5/9 + 5/16 + 5/22-23 累積評価

```
3 週間 × 約 9-12 R 重賞:
- system 1-5 個別 top1_rate
- 高信頼時 hit_rate
- 累積 ROI 想定 (もし買ってたら)
- 「重賞も勝てるか」 結論 → 6/8+ V20 投入後 trial 検討材料
```

---

## 6. ★ 投票方針 (絶対遵守) ★

```
5/9 + 5/16 + 5/22-23: 重賞 投票しない
6/8+ V20 投入後: 5/9-5/23 累積 verdict 良好なら trial 候補
```

→ Session #49 D の verdict は **学習用、 投票推奨ではない**

---

## 7. V15 投資保護

✅ V15 model md5: 842b9a5f... 不変
✅ predict_core / daily_predict 完全不変
✅ main 不変、 dev/training-poc 専用
✅ 重賞 投票なし (絶対遵守)

→ **5/9 朝 V15 案B改 完全保証**

---

## 8. 結論

✅ D1: tools/verify_majors_5system_5_10.py (190 行)
✅ D2: 5 system 個別 hit_rate + 合議高信頼 + ROI 推定 (★ 投票しない ★)
✅ D3: 5/8 23:XX 動作確認 (5/9 結果未確定で deferred、 5/10 朝 自動 verdict)
✅ D4: 累積評価 plan (5/9 + 5/16 + 5/22-23)
✅ V15 投資保護

→ **5/10 朝 重賞 verdict tool 完成、 5/9 結果反映で全 evaluation 自動**

---

**Session #49 D 完了 (dev/training-poc)**
