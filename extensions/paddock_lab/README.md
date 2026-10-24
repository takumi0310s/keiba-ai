# パドック比較・精度検証機構 v0.1

当日映像と同一馬の過去映像を証拠付きで記録し、パドックを加える前後の予測を同じレース群で検証する独立モジュール。Python 3.10以上、追加依存なし。

**実装済みは記録・比較条件の判定・予測固定・検証・収集タスク生成。無人映像取得、映像からの自動特徴抽出、既存予測モデルへの接続、精度改善の実証は未完。** 現段階は独立して予測を並行記録する検証用。数値確率を主観点数で補正しない。

## なぜ必要か

これまでの比較は短い動画・終端画像・異なる角度が混在し、全馬を均等に確認できていない。これでは「変わらない」と「比較できない」を区別できず、パドックを見た効果を測れない。本機構では未観察はnull、比較不可はholdとし、0点や据え置きへ変換しない。

## 実行

```bash
cd extensions/paddock_lab
python3 -m unittest discover -s tests -v
python3 examples/run_demo.py /tmp/paddock-demo-new
python3 paddock_lab.py --db /tmp/paddock-demo-new/synthetic.sqlite verify
python3 paddock_lab.py --db /tmp/paddock-demo-new/synthetic.sqlite evaluate
```

デモの馬・確率・結果・証拠URLはすべて架空で、実レースの分析・成績ではない。同じ出力先を上書きしない。

## 収集から結果まで

1. 出走表から全出走馬の永続horse_id・当日馬番・馬名・発走時刻・出典を登録。パドックアイUUIDとnetkeiba race_idは相互推測せず別に管理。
2. 能力・適性・調教・馬場・市場による全頭3着内確率を`before`として、映像・専門家短評を見る前に固定。確率が未計算なら空欄を守り、比較記録だけを作成する。
3. `acquisition_plan`で全馬の収集タスクを作る。リンクを実画面で確認できた過去走から直近3走を選ぶ。着順は選定に使わない。好走映像だけを選ばない。
4. 当日と過去の馬名・horse_id・開催日・レースを確認。動画内の過去馬番が違っても同一馬で照合。動画開始・進行・実際に見た区間を記録。動画の全長を観察時間にしない。シーク直後の古い画面や黒画面を観察済みにしない。
5. 輪郭・首の使い方・歩行リズム・歩幅のばらつき・発汗・落ち着きを事実と解釈に分ける。画角・遮蔽・天候・馬具・連続観察時間を残す。左右差の測定は未実装として常に保留。
6. 同一馬・条件の揃う項目だけ比較する。品質不足なら無理に上げ下げしない。全馬の欠測理由も残す。
7. 独立学習・検証済みの数値モデルが用意できた場合だけ、全頭の`after`確率とモデル情報を固定。現実の記録時刻は書き換えない。afterモデルは利用するfeature_namesを宣言し、各馬で必要な比較項目が揃うか検査する。10分前を基本締切、5分前は取消・馬体重・オッズの再確認。終了時刻は収集計画であり、実際のスケジューラは含まない。
8. 速報を除き確定結果を全頭分登録。`evaluate`で同じ対象群の前後を評価する。発走後の保存、専門家・結果混入、全頭比較不足、モデル学習時刻不正は除外理由付きで集計する。

## APIとCLI

```python
from paddock_lab import Store, compare, acquisition_plan, evaluate
s = Store('paddock.sqlite')
s.register(race)  # examples/input_templates.json参照
s.forecast(race_id, 'before', probabilities, baseline_model)
s.observation(race_id, observation)
comparison = compare(current_observation, past_observation)
s.forecast(race_id, 'after', probabilities_after, paddock_model)
s.result(race_id, finish_by_horse_id, confirmed_at, official_result_url)
report = evaluate(s)
s.close()
```

CLIは`register / observe / forecast / exposure / result / invalidate / plan`にJSONファイル、`audit`にrace_id、`evaluate / verify`は引数なし。JSON形状はサンプルを参照。`plan`はrace/catalog/as_ofを受け取る。

途中取消・出走表変更は`invalidate`で旧比較を無効化する。固定済みの確率を無言で正規化し直さない。改めた対象馬群の実験は別の台帳で開始し、旧台帳との区別を保つ。同一台帳の同一レースはbefore/after/resultそれぞれ一度のみ。結果訂正も旧実験を無効化し別途記録する。

## 項目の意味

- `facts`: 観察できた内容。馬体重は出典表示値。体重変化から良化を断定しない。
- `interpretations`: 例「腹回りに厚みが増したように見える」。医学的診断ではない。
- `features`: 人の順序尺度0〜4またはnull。`feature_evidence`でhuman_ordinalを明記。歩幅cm・左右差%等の実測を装わない。
- `outline`: 輪郭の厚み（小さい〜大きい）。`neck_use`: 首の上下動の程度。`walk_rhythm`: リズムのばらつき。`stride_consistency`: 歩幅のばらつき。`sweat`: 発汗の見え方。`settledness`: 落ち着きの見え方。大小は一律の良い/悪いではない。尺度は暫定で、同一評価者・同一尺度を守る。
- `ordinal_delta`: 順序尺度の差。確率変化に変換しない。歩様の比較は両映像とも連続8秒以上を暫定条件とする。古い過去映像180日超、角度違い、馬具不明・変更、遮蔽は保留。閾値は工学上の初期値で精度保証ではない。
- 同じ馬に評価者2名を割り当て再観察し、意見の不一致を記録することを推奨。observer_agreementで同一馬・同一映像・同一項目を独立評価したペアの重み付きκを計算できる。自動追跡は含まない。

## 予測と検証の契約

- 対象は**3着以内の周辺確率**。全馬合計3を検査する（同着なし近似）。単勝確率や複勝払戻対象との混同を避ける。ワイド・3連複の同時的中確率は積で計算しない。
- 前後でbaseline model versionとbaseline_input_sha256（馬場・オッズを含む元入力のハッシュ）を一致させ、同一レース・同一馬群で比較。パドックモデルの学習終了と固定がbeforeより前であること、artifact_sha256形式を検査する。ただしモデル実体のハッシュ一致・学習データ監査は接続側で行う必要がある。自己申告だけでモデルの妥当性を保証しない。
- 指標はレースを等重みとしたBrierと対数損失。値が小さいほど予測誤差が小さい。校正表は出走馬単位で別表示。Brier改善だけで確率校正完了とはしない。
- 前後差の95%区間は開催日単位のペア・ブートストラップ。負の差が改善方向。同日相関を扱うが、開催日をまたぐ同一馬の依存は残る。
- `chronological_split`でレースを丸ごと学習→検証→未使用テスト期間に分ける。特徴量取得時点・ラベル確定時点を検査。分割境界は学習前に固定し、テスト結果を見て選び直さない。全レース単位の完全なデータセットが必要。
- 暫定レビュー下限100Rかつ20開催日。両指標の区間上限が負の場合だけreview_signal。これは本番採用条件の一部であり、自動昇格なし。100Rで十分という科学的保証でもない。未使用期間の再検証・校正・市場価格の時刻整合を別途確認する。
- 実購入・配分・確定払戻の台帳未接続。実ROIは常にnull。候補的中と実収支を混同しない。

## 記録の信頼範囲

SQLiteへの追加時刻・前後予測・観察をハッシュ鎖で記録し、APIからの上書きを禁止する。改変検出は可能だが、管理者が台帳全体とハッシュを置き換えることや過去のバックアップへの巻き戻しは防げない。head_hashを独立保存するまで外部時刻証明ではない。映像確認・専門家閲覧・モデル学習経緯の自己申告の正しさも機械だけでは保証できない。

ブラウザー認証情報・Cookie・動画の配信URLトークンは保存しない。公開用ページのURLと観察区間だけ保存する。会員ログインと映像視聴は通常のブラウザーで行う。会員画面の自動取得・再生監視とこの台帳の接続は別途必要。

## 実データの扱い

会話内の限定比較は、この公開コードの実績に含めない。正確な連続観察時間、馬具条件、全頭比較、発走前保存が揃う記録だけを評価対象にする。本パッケージの実行例はすべて架空の検査用データ。

## 既存AIとの接続

既存リポジトリtakumi0310s/keiba-aiは確認済み。ただしREADME記載の過去のAUC/ROIを本機構の実績には使わない。独立モジュールを既存予測の後段で並行稼働させる。まず既存出力のhorse_id・モデル版・予測時刻・全頭top3確率を渡すアダプターを接続する必要がある。今回、本番predict_core・自動購入・通知・既存スケジュールは変更していない。

## 参照

- scikit-learn公式: https://scikit-learn.org/stable/modules/calibration.html
- scikit-learn公式ソース: https://github.com/scikit-learn/scikit-learn/blob/main/doc/modules/model_evaluation.rst
- 当日映像表示: https://prc.jp/jraracingviewer/contents/new/calender/rv_live.html
