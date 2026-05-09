# Session #61 B: scoring logic

## input
- horse_motion_5_9.csv (3 race × 3 頭 simulate)
- predictions_majors_5system_5_9_FINAL.json (race meta + V15 top3)
- predictions_5_9_all.json (V15 全馬 score)

## scoring 方針
- 動画 motion = simulate のみ × 3 頭 → 全馬 unique scoring 不可
- 代替: V15 score を race 内 percentile 化 → integrated_score
- motion 3 頭は race 代表値 (stride/body/stab/tens 平均) として補助情報
- 東京 11R エプソム C は v15_scores 0 件 → fallback top3 のみ

## 出力
- horse_video_scores_5_9.csv (35 rows、 race_name 併記)

## 制約事項
- 真の動画解析 score ではない (Session #60 動画 DL 失敗のため)
- 全馬 score = V15 ベース。 motion features は補助
- 次回: 動画 DL 経路修正後に true motion-based scoring へ
