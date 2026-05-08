# Session #47 A: 調教 features audit (2026-05-08)

## 1. 現状 V15 で使用中の調教 features (12 個)

V15 (150 features) のうち調教関連:

| # | feature | 出所 | 説明 | リーク |
|---|---------|------|------|------|
| 1 | training_time_filled | netkeiba time_4f / TARGET CK | 調教 4F タイム (mean fill ~52.0s) | safe (pre-race) |
| 2 | has_training | derived | 調教データ有無 (0/1) | safe |
| 3 | training_per_dist | derived | training_time / (dist/200) | safe |
| 4 | wood_best_4f_filled | TARGET CK | 木馬場 14日 best 4F | safe |
| 5 | has_wood_training | derived | 木馬場 data 有無 | safe |
| 6 | wood_count_2w | derived | 木馬場 2 週内回数 | safe |
| 7 | sakaro_best_4f_filled | TARGET CK | 坂路 14日 best 4F | safe |
| 8 | sakaro_best_3f_filled | TARGET CK | 坂路 14日 best 3F | safe |
| 9 | has_sakaro_training | derived | 坂路 data 有無 | safe |
| 10 | total_training_count | derived | 木+坂路 合計 | safe |
| 11 | training_intensity_enc | netkeiba intensity | 0=不明, 1=馬なり, 2=強め, 3=一杯 | safe |
| 12 | time_1f_last_filled | netkeiba time_1f | 追切ラスト 1F タイム | safe |

JRDB CYB (PACI Tier B) 関連:
| # | feature | 出所 | 説明 |
|---|---------|------|------|
| 13 | paci_train_mark | jrdb_paci.csv | 調教評価マーク encoding (◎=5...) |

**stable_comment_score**: コメント -3〜+3 score 化、 V15 で使用 (新馬戦は shinba_comment_score 別系統)

## 2. 利用可能な調教 data 全量

### 2.1 netkeiba (Premium Cookie 経由)
- **netkeiba_training_times.csv**: 300,574 行
  - cols: race_id, race_date, umaban, horse_name, course, condition, rider,
          time_6f, **time_5f**, time_4f, **time_3f**, time_1f, intensity,
          rank, evaluation, **training_date**
- **netkeiba_training_eval.csv**: 302,204 行
- カバレッジ: 2025 部分 (full coverage 拡大中)

### 2.2 JRDB CYB (調教分析)
- **jrdb_cyb.csv**: 548,607 行
  - cols: umaban, train_type, train_course_type, **train_baba**,
          train_mark, **train_amount**, **train_change**, train_comment,
          comment_year, comment_date, train_eval, **train_course**, race_id
- 2015 年〜 累積、 ほぼ全 R 全頭

### 2.3 TARGET JV CK_DATA
- 955,580 行 (training_times.csv)、 木/坂路 raw ハロン timing

## 3. 未使用 / 未活用 data 棚卸し

| ファイル | 未使用 column | 拡張余地 |
|----------|---------------|---------|
| netkeiba_training_times.csv | time_6f, time_5f, time_3f, training_date, course (詳細), condition | ハロン別/間隔/馬場 features 化 |
| netkeiba_training_eval.csv | rank (A/B/C/D), evaluation 詳細 | rank encoding 拡張 |
| jrdb_cyb.csv | train_baba, train_amount, train_change, train_eval, train_course | 5 features 直接活用可 |
| TARGET CK | 直近 3 走の調教 progression | 時系列特徴量化 |

## 4. 拡張候補 features (15 個 → 7-8 個 採用想定)

### 4.1 ハロン別 / ハロン差分 (4 個)
| feature | 計算 | 期待 |
|---------|------|------|
| training_time_5f | time_5f (生値) | 5F も model に与える |
| training_time_3f | time_3f (生値) | 後半 3F |
| training_pace_5f_3f | (time_5f - time_3f) / 2 | 中間 2F lap |
| training_acceleration | time_3f / 3 - time_1f | 加速度 (低ければ良) |

### 4.2 追切間隔 / 直近性 (3 個)
| feature | 計算 | 期待 |
|---------|------|------|
| days_since_last_training | race_date - max(training_date) | 直近性 |
| training_count_2w | 2 週間 内 回数 | 仕上げ量 |
| training_count_4w | 4 週間 内 回数 | 中期仕上げ |

### 4.3 JRDB CYB 直接活用 (4 個)
| feature | 計算 | 期待 |
|---------|------|------|
| cyb_train_baba_enc | train_baba encoding (良=0...) | 調教馬場 |
| cyb_train_amount | 追切量 (raw) | 仕上げ量 |
| cyb_train_change_enc | 調子変化 encoding | 状態 |
| cyb_train_eval_enc | 評価 encoding | 評価 |

### 4.4 偏差 / 厩舎比較 (4 個)
| feature | 計算 | 期待 |
|---------|------|------|
| training_time_vs_3m_avg | 今回 - 過去 3 ヶ月 avg (馬単位 expanding) | 自己比較 |
| training_time_vs_stable_avg | 今回 - 同一厩舎当週 avg | 厩舎比較 |
| training_time_vs_course_avg | 今回 - 同 course 同 condition avg | 馬場比較 |
| training_intensity_progression | 直近 3 走 intensity trend | 仕上げ進捗 |

## 5. リーク検証 (重要)

全候補 features は **pre-race** で確定。
- 調教は レース前 数日 〜 1 週間 完了
- 全 expanding window 使用 (該当馬の過去のみ)
- 厩舎 / course 平均 は **当該馬除く** で計算

⚠️ **CYB train_comment** は post-race 含む可能性 → 採用見送り
   (Session #38 SKB POST-RACE LEAK と同根の reflection)

## 6. 採用優先順位 (B で AUC 計測)

優先 A (まず計測):
- training_time_5f, training_time_3f, training_pace_5f_3f
- days_since_last_training, training_count_2w
- cyb_train_baba_enc, cyb_train_amount, cyb_train_change_enc

優先 B (A で改善あれば追加):
- training_acceleration, training_count_4w
- cyb_train_eval_enc
- training_time_vs_3m_avg, training_intensity_progression

採用判定: B (training_auc_test.py) で
- WF AUC (LGB single fold で proxy) 改善 ≥ 0.0010
- 1勝クラス で +0.002+
- 全年 monotonic 改善 (年別 gap < 0.05)

## 7. 効果検証 plan (B section)

```
V15 baseline (150 features) WF AUC: 0.886 想定
V15 + 拡張 8 features (158 features) WF AUC: ?

クラス別比較:
- 新馬: 調教情報依存 高 → 期待大
- 1勝: 中庸
- 重賞: 騎手・血統依存高 → 期待小
```

採用基準: 上記達成すれば **V20 候補に追加**、 未達なら **棚卸しのみ**。

## 8. 関連 file

- predict_core.py: 1593-1980 行 で training features 生成
- features_v15_new.py: V15 新 features 統合
- train_v15_master.py: V15 学習 entry point
- jrdb_features.py: PACI / KYI features

## 9. 次 step

→ Session #47 B: tools/training_auc_test.py で AUC 比較
→ Session #47 C: 5/9 全 R で V15 vs V15+training 予測
→ Session #47 D: 5/10 朝 verdict (実結果照合)
