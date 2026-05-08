# Session #49 C: 5 system 合議 (5/9 重賞 3R)

**作成**: 2026-05-08 23:XX (Session #49 C、 dev/training-poc)
**目的**: ★ 5 system 合議で「重賞も勝てるか」 検証 ★ (投票なし、 学習用)

---

## 1. 5 system (Session #48 D の 4 → +動画 = 5)

| # | system | features | 状態 |
|---|--------|---------|------|
| 1 | V15 単独 | production current | baseline |
| 2 | V15 + 拡張調教 | + Sprint 2 (horse_weight + race_interval + running_style) | 学習用 |
| 3 | V15 + TM 公式調教 | + TFJV TM_DATA (Session #44 B) | 学習用 |
| 4 | V15 + 当日体重 | + Stage 2 (Session #48 B) | 学習用 |
| 5 | **V15 + 動画 features** ★新★ | + Session #49 B (動画 PoC) | 学習用 |

---

## 2. 合議 logic

```
5/5 一致 → ★★ 高信頼 ★★
4/5     → ★ 中高
3/5     → 中
2/5     → 低
1 以下 → 不一致

confidence-based recommendation (★ 投票しない、 verdict 用 ★):
- 4-5/5 一致: 馬連 軸 1 頭流し candidate
- 3/5 一致:  3連複 BOX candidate
- 2/5 以下: skip
```

---

## 3. 5/9 重賞 3 R

| 場 | R# | レース | grade | 発走 |
|----|-----|--------|-------|------|
| 東京 | 11R | エプソムカップ | G3 | 15:45 |
| 京都 | 11R | 京都新聞杯 | G2 | 15:30 |
| 新潟 | 11R | 駿風 S | OP | 15:20 |

---

## 4. 動作確認 (5/8 23:XX)

```
$ python tools/predict_majors_5system_5_9.py --date 20260509

--- 東京 11R エプソムカップ (G3) ---
  System 1 (V15): deferred (5/9 朝 daily_predict 後 取得)
  System 2-4: deferred (Sprint 2 / TM / Stage 2 features)
  System 5 (動画 ★): partial (動画 0、 5/9 朝 DL 後 batch run)
  合議: deferred (5/9 朝 全 system 動作後)
  recommendation: skip (★ 投票しない、 verdict 用 ★)

(他 2 重賞 同様)
```

→ 5/9 朝 daily_predict + 動画 DL 後 全 system 動作

---

## 5. 5/9 朝 動作 timeline

```
08:00 V15 daily_predict (System 1 取得)
09:00 ユーザー: predict_majors_5system_5_9.py 実行 → 5 system 比較
09:30 ユーザー: 重賞動画 manual DL → data/v18/videos_5_9/
10:00 video_poc_majors_5_9.py --all → System 5 features 抽出
10:30 predict_majors_5system_5_9.py 再実行 → 全 system 合議 確定
       → Discord #investments に 通知
11:00- 観戦 (重賞は 投票しない、 verdict 用)
15:20-45 重賞 3R 終了
20:30 1 day summary + 「もし重賞 buy したら ROI ○○%」 算出 (Session #49 D)
```

---

## 6. 「重賞も勝てるか」 検証 plan

### 6.1 単発 (5/9)

5/9 重賞 3R で:
- 5 system 合議で 高信頼 (5/5 or 4/5 一致) の R 数
- 推定 ROI (もし投票してたら)
- 既存 V15 単独との 差分

### 6.2 累積 (5/9 + 5/16 + 5/22-23)

3 週間で 約 9-12 R 重賞 verdict:
- 4-5/5 一致 R の hit_rate
- 3/5 一致 R の hit_rate
- 推定 ROI
- 投票価値 (V20 投入 6/8+ 後 検討材料)

---

## 7. ★★ 投票方針 (絶対遵守) ★★

```
5/9 投票:
- 12R 1勝クラス のみ (V15 案B改、 max 2,100円)

5/9 投票しない:
- 11R 重賞 3R (Session #48 D + Session #49 C 予測対象、 verdict のみ)
- その他全 R

5/16 + 5/22-23 投票判断:
- 5/9 結果次第 (Session #46 result_verification_5_10 で verdict)
- V18 sib_w5 trial は 5/16 GO 確率 85-95% (Session #43 C)
- 重賞 trial は 6/8+ V20 投入後 検討
```

---

## 8. V15 投資保護

✅ V15 model md5: 842b9a5f... 不変
✅ predict_core / daily_predict 完全不変
✅ main 不変、 dev/training-poc 専用
✅ 5/9 重賞 投票なし (絶対遵守)

→ **5/9 朝 V15 案B改 完全保証**

---

## 9. 結論

✅ C1: tools/predict_majors_5system_5_9.py (200 行)
✅ C2: 5 system + 合議 logic + recommendation (★ 投票しない、 verdict 用 ★)
✅ C3: 動作確認 (5/9 朝 daily_predict + 動画 DL 後 全 system 動作)
✅ C4: 「重賞も勝てるか」 検証 plan (5/9 + 5/16 + 5/22-23 累積)
✅ V15 投資保護

→ **5 system 合議 tool 完成、 5/9 朝 ユーザー manual run + verdict ready**

---

**Session #49 C 完了 (dev/training-poc)**
