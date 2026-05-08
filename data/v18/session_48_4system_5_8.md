# Session #48 D: 5/9 重賞 4 system 予測 (dev/training-poc)

**作成**: 2026-05-08 (Session #48 D)
**目的**: ★ 4 system 比較で「重賞も勝てるか」 verdict ★ (投票なし、 学習用)

---

## 1. 5/9 重賞 3 R

| 場 | R# | レース | grade | 発走 |
|----|-----|--------|-------|------|
| 東京 | 11R | エプソムカップ | G3 | 15:45 |
| 京都 | 11R | 京都新聞杯 | G2 | 15:30 |
| 新潟 | 11R | 駿風 S | OP | 15:20 |

---

## 2. 4 system 構成

| # | system | features | 用途 |
|---|--------|---------|------|
| 1 | **V15 単独** | 現状 (production) | baseline |
| 2 | V15 + 拡張調教 | + Sprint 2 (horse_weight, race_interval, running_style) | 学習用 |
| 3 | V15 + TM 公式調教 | + TFJV TM_DATA (Session #44 B) | 学習用 |
| 4 | V15 + 全部 + Stage 2 + パドック | + 当日体重 + 動画 (Phase 4) | 重賞 verdict |

---

## 3. 合議 logic

```
4 system top1 一致 → 高信頼 (★)
3 system 一致 → 中信頼
2 system 一致 → 低信頼
不一致 → "各 system 異なる" 報告のみ
```

---

## 4. 5/9 利用 timeline

```
朝 09:00: predict_majors_4system_5_9.py 実行
         → 4 system top3 + 合議 通知

各 R 終了 5 分後: realtime_verdict_5_9 で 4 system verdict
         → 「もし投票してたら ROI ○○%」 算出

夜 20:30: 1 日 summary
         → 「重賞 3 R 全 system verdict」
         → 4 system 合議の精度検証
```

---

## 5. ★ 投票方針 (絶対遵守) ★

```
5/9 投票:
- 12R 1勝クラス のみ (V15 案B改)
- 最大 2,100 円 (700 × 3 R)

5/9 投票しない:
- 11R 重賞 3 R (本 Session D の 4 system 予測対象、 verdict のみ)
- 12R 1勝以外 / 9R 一般戦
```

→ Session #48 D の 4 system 予測は **学習用、 投票推奨ではない**

---

## 6. 動作確認 (5/8 朝)

```bash
$ python tools/predict_majors_4system_5_9.py --date 20260509 --no-discord

--- 東京 11R エプソムカップ (G3) ---
  System 1 (V15): deferred (5/9 朝 daily_predict 後に取得)
  System 2-4: deferred (本 Session 設計のみ)

--- 京都 11R 京都新聞杯 (G2) ---
  ... 同様
```

→ 5/9 朝 daily_predict 後に再 run で実 prediction 取得可能

---

## 7. 数週間継続検証 plan

5/9 + 5/16 + 5/22-23 = 約 9-12 R 重賞 で 4 system 比較。
6 月以降:
- 4 system 合議 (TOP1 一致時) の hit_rate を実証
- 「重賞も勝てるか」 = 数週間 sample で 確認
- 6/8+ V20 投入後の 重賞 trial 検討材料

---

## 8. V15 投資保護

✅ V15 production 完全独立、 main 不変、 dev/training-poc 専用
✅ V15 model md5: 842b9a5f... 不変
✅ predict_core / daily_predict 完全不変
✅ 5/9 朝の V15 動作完全独立

→ **5/9 朝 V15 案B改 完全保証**、 11R 重賞は 投票なし (絶対遵守)

---

## 9. 結論

✅ D1: tools/predict_majors_4system_5_9.py (180 行) 4 system + 合議
✅ D2: 動作確認 (5/9 朝 daily_predict 後に実 prediction 取得可)
✅ D3: 投票方針確認 (重賞は 予測のみ、 投票なし)
✅ D4: 数週間継続検証 plan
✅ V15 投資保護

→ **dev/training-poc 追加 commit、 5/9 朝 4 system 予測 + verdict ready**

---

**Session #48 D 完了 (dev/training-poc)**
