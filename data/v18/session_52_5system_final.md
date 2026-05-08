# Session #52 D: 5 system 合議 v2 (実動画統合)

**作成**: 2026-05-09 00:XX (Session #52 D、 dev/training-poc)

## 1. v2 改善 (vs Session #49 C)

System 5 を simulate → **実動画 features 読み込み** に置換:
- `data/v18/horse_motion_5_9.csv` (Session #52 B+C 出力) を load
- race_name fragment match で 該当 race の features 集計

## 2. 動作確認 (5/8 23:XX、 全 deferred)

- System 1 (V15): 5/9 朝 daily_predict 後 取得
- System 2-4: 設計のみ (Sprint 2/TM/Stage 2)
- System 5 (★ 実動画): horse_motion_5_9.csv 不在 で deferred、 5/9 朝 動画 DL + features 抽出 後 利用可

## 3. 5/9 朝 timeline

```
08:00 V15 daily_predict (System 1 取得)
09:00 ユーザー: 動画 DL (Premium Cookie 必要)
       python tools/video_downloader.py --majors
10:00 motion features 抽出
       python tools/horse_motion_features.py --batch
       → horse_motion_5_9.csv 出力
10:30 5 system v2 実行
       python tools/predict_majors_5system_5_9_v2.py
       → predictions_majors_5system_5_9_FINAL.json
11:00 Discord #investments 通知 (★ verdict 用、 投票なし ★)
```

## 4. ★ 投票方針 (絶対遵守) ★

5/9 重賞 投票なし、 12R 1勝のみ V15 案B改 max 2,100円。

## 5. V15 投資保護

✅ V15 model md5: 842b9a5f... 不変、 main 不変、 dev/training-poc 専用

→ **5/9 朝 V15 完全保証**

---

**Session #52 D 完了 (dev/training-poc)**
