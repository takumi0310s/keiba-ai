# v15r-B 学習・検証 事前登録（2026-08-12・学習前コミット・変更禁止）

## ラウンド1の教訓（V15R_RESULTS.md）
- gain源 `training_time_filled` の正体 = ソースDB `training_4f`（最終追い切り・コース混合）。
  JV配信は坂路(HC)/ウッド(WC)のみで芝/P追いを含まず、**全写像候補 corr<0.05 = 再構成不能**。
- 方針転換: 模倣写像ではなく**ライブ供給が実在する調教信号**を正面採用する。

## v15r-B 学習セット
- ラウンド1の140 (base122+JV8+SRB6+KKA4) に **CYB調教7** を追加 = **147特徴**
  - cyb_train_type / cyb_course_type / cyb_baba / cyb_mark / cyb_amount / cyb_change / cyb_eval
  - 源 = jrdb_cyb.csv（前日発表・2015-2026全履歴562k行・leak-free as-of）
  - join = (nk_race_id, umaban)。数値コード化（非数値→NaN）
- 除去系はラウンド1と同一（premium16 / lap4 / S1リーク3）

## プロトコル・合格基準（ラウンド1と同一・変更なし）
- WF fold 2021-25、LGB+XGB 同一パラメータ
- Ablation: (aB) full147 / (bB) −CYB7 / 参照 = ラウンド1の (a)(d)(e) 実測値を使用
- 合格: **(aB) ≥ 0.860**（(d)=0.8407<0.865 のため代用基準 = (d)−0.005 = **0.8357**）
- CYBブロック: 寄与 = aB − bB。**≤0 なら落とす**
- 主従は既合意（V15=主）。v15r-B は合格時のみ「従(paper並走・別カウント)」に昇格、
  不合格なら観察保存のみ
