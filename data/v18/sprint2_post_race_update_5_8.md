# Sprint 2 H: post_race_features_update (Session #47 H)

**作成**: 2026-05-08 (Session #47 H、 dev/sprint2)

---

## 1. 構成

`tools/post_race_features_update.py` (140 行):
- 5/9 結果確定後、 即時に features 更新
- sib_top3_rate_exp_w5 (Session #43 C) を 5/9 race を含めて recompute
- 馬・騎手・調教師 expanding stats update
- `tools/result_verification_5_10.py` (Session #42 D) と integrate

---

## 2. 5/9-5/10 timeline (本 Session H で contributing)

```
5/9 18:00 DailyResults 自動 結果照合
5/9 21:00 ユーザー manual: post_race_features_update.py --date 20260509
            → sib_w5 csv 更新
5/10 06:00 NightlySanity 翌日 task pre-check
5/10 08:00 result_verification_5_10.py 自動
            → V15 ROI 集計 + 5/16 verdict
            → Discord #investments 通知
5/10 09:00 ユーザー manual confirm
```

---

## 3. dry-run 動作確認

```
[sib_w5] update for date 20260509
[sib_w5] {'status': 'dry_run', 'would_run': 'sib_expanding_variants.py --variant a (window=5)'}
[horse career] V15 build_features の expanding 計算 と同じ logic
[result_verification integration] tools/result_verification_5_10.py
```

→ deferred 各 step、 5/15 merge 後 ユーザー manual 実行 (or schtasks 自動化)

---

## 4. 統合先

```python
# 5/9 21:00 (or 翌朝 06:00) に実行
python tools/post_race_features_update.py --date 20260509

# 5/10 朝 自動実行
python tools/result_verification_5_10.py --date 20260509
# → V15 ROI、 5/16 GO 確率 verdict
```

---

## 5. V15 投資保護

✅ V15 model md5 不変、 main 不変、 dev/sprint2 only
✅ 5/9 朝の V15 動作は完全独立 (本 tool は 5/9 18:00 後 / 5/10 朝 のみ実行)

→ **5/9 朝 V15 完全保証**

---

**Session #47 H 完了 (dev/sprint2)**
