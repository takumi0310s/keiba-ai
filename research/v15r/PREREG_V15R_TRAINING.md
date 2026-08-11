# v15r 学習・検証 事前登録（2026-08-12・学習実行前にコミット・変更禁止）

## 学習セット（v15r 定義）
ベース = V15の145 (leak-free v2 キャッシュ) から:
- 除去: premium系16（JV置換12 + 代替なし4 = index_max/run1/avg5 + stable_comment_score）
- 除去: 前走レースラップ4（prev_race_first3f/last3f/pace_diff, prev_agari_relative）= netkeiba race頁由来・供給消滅
- 除去: **S1リーク3**（odds_change_rate/pop_rank_change/odds_sharp_drop）= 6/11監査の確定オッズ混入を v15r で根治
- 追加: JV調教8（jv_slope_best_4f/3f/1f_14d, accel, slope_count, wood_best_4f, wood_count, train_count。14日窓・レース前日まで）
- 追加: SRB前開催日バイアス6（同場の**前開催日**集約のみ。当該レースSRBはPOST-RACEのため不使用）
- 追加: KKA条件別成績4（track/kyori/heavy/class の top3率・Bayesian α=5）
- **計 122 + 8 + 6 + 4 = 140特徴**（確定リストは v15r_features.json）

## プロトコル（V15と同一）
- WF fold: test year Y ∈ {2021..2025}, train = year<Y, early-stop valid = year Y−1
- LGB+XGB（V15と同一ハイパーパラメータ・50/50平均）
- 目的変数 target = finish≤3

## Ablation（必須・同一fold）
| 構成 | 内容 |
|------|------|
| (a) full | 140特徴 |
| (b) −JV | 132特徴（JV調教8を除く） |
| (c) −第2弾 | 130特徴（SRB6+KKA4を除く） |
| (d) V15参照 | 現行145を同一foldで再学習（比較基準） |

## 合格基準（事前固定）
1. **(a) WF mean AUC ≥ 0.860** で合格（V15比 −0.005 以内の趣旨。(d) が 0.865 を下回る場合は (d)−0.005 を代用基準とする）
2. **第2弾ブロック（SRB6+KKA4）**: 寄与 = AUC(a) − AUC(c)。**寄与 ≤ 0 なら容赦なく落とす**（参考で SRB/KKA サブ寄与も報告するが判定はブロック一括）
3. **JVブロック**: 供給代替（死亡premiumの置換）のため寄与の正負では落とさない。ただし寄与 < −0.002 なら特徴設計を見直してから確定
4. 最終モデル = 判定適用後のセットで year≤2025 全期間学習 → `research/v15r/v15r_model.pkl.gz`（research専用・本番不変）

## リーク監査（学習前ゲート・T4同等）
- A. JV調教: 使用 workout 日 < レース日 を **assert**（構築時に強制・違反で例外）
- B. SRB: venue内 shift(1) 構造で前開催日のみ。構築時に元日付 < レース日を保証
- C. KKA: as-of 性（同一馬の合計出走数が時系列非減少）をサンプル検証
- D. 充足率: jv_slope_best_4f の 2021+ 充足 ≥ 60% を要求（未達なら JV バックフィル不足として学習を中止）

## 完成後（本書の範囲外・別途）
- 8/15〜 paper 並走（V15と同一レースで両記録・別カウント）
- v15r ゲートを PRE_REGISTRATION_RESUMPTION_GATE に追記（V15と同一閾値構造）
- 主従宣言は WF+ablation 提示後にユーザーと合意
