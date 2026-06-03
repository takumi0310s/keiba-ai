# 前向き paper trading: s2b(穴特化候補) — 運用設計

作成 2026-06-03。**究極の検証(リーク構造的に不可能)**。 本番 V15/V16・predict_core・daily_predict・app.py・race_auto_notify は**一切不変**。 投票・買い目通知には**一切出さない**(記録のみ)。

## 目的
leak-free バックテストで s2b は全券種で V15 超(単勝111.6%/三連複top4box194.3%/三連単1-2-5 229%/馬連top3box146%)。 だが過去データの楽観が残る(三連複は本番で下がる可能性)。 **これからのレースで発走前に s2b 予測を記録 → 結果照合 → ROIを貯める**。 未来データはリーク不可能 = ここでプラスなら本物。 本番liveデータ(paci/ZE live-gap 含む)での真の挙動も同時検証。

## スクリプト `tools/paper_trade_s2b.py`(完全分離・新規)
| モード | 内容 | 実行タイミング |
|--------|------|----------------|
| `predict --date YYYYMMDD` | 当日全レースを s2b で予測 → top6+人気を `data/paper_s2b/{date}_pred.jsonl` に記録 | ★発走前★(朝、DailyPredict 8:00 の後 ~8:30) |
| `results --date YYYYMMDD` | JRA払戻(jra_payouts.csv, key=date_course_race_num)と照合 → 券種別 return/points を `{date}_results.jsonl` | 結果確定後(DailyResults 20:00 の後 ~20:30) |
| `report` | 累積ROI/的中率/N を券種別表示(leak-freeバックテストと乖離確認) | れんはすが任意に実行 |
| `from-oof` | (検証専用) leak-free OOF から擬似pred生成し results/report 配線テスト | 開発時のみ |

- 予測経路: `predict_core.build_features`(本番と同じ発走前特徴) → `merge_jrdb_predict_features` → s2b特徴(人気代理族13除去 + レース相対特徴 one-hot脚質/距離適性・n_front・front_advantage・脚質×バイアス×枠) → s2b候補 `models/v16_anaba_s2b_candidate.pkl.gz`(LGB+XGB)。
- ze4特徴は live では過去ZEDのみ = **元々 leak-free**。
- 集計券種: 単勝(top1) / 複勝top1 / 馬連top3box(3点) / 三連複top4box(4点) / 三連単form1-2-5。
- ログは `data/paper_s2b/`(**.gitignore対象**)。 Discord(買い目)・投票・本番CSVには一切書かない。

## スケジューラ組み込み(本番と分離・新タスク)
```
# 発走前予測(土日 8:30、DailyPredict の後)
schtasks /Create /TN "keiba-ai\PaperS2BPredict" /SC WEEKLY /D SAT,SUN /ST 08:30 ^
  /TR "<silent_runner で> python tools\paper_trade_s2b.py predict" /RL HIGHEST
# 結果照合(土日 20:30、DailyResults の後)
schtasks /Create /TN "keiba-ai\PaperS2BResults" /SC WEEKLY /D SAT,SUN /ST 20:30 ^
  /TR "python tools\paper_trade_s2b.py results"
```
- 既存 daily_predict/race_auto_notify とは別プロセス・別ログ。 失敗しても本番に無影響。
- 注: `results` は `jra_payouts.csv` が当該日まで更新されている前提。 未更新なら未照合レースはskip(後日再実行で補完可)。 payout自動更新が止まっている場合は別途取得経路が必要(既知課題)。

## 確認方法(れんはす)
```
python tools\paper_trade_s2b.py report
```
→ 「今 何レース貯まって、各券種の paper ROI はいくつか」を即表示。

## 判定基準(N と分散)
- **単勝(低分散)**: ~300-500R で ROI が安定(±数%)。 1開催 ≈ 24-36R なので **約10-15開催(2-3ヶ月の土日)** で目安が立つ。
- **三連複/三連単(高分散・配当歪み)**: **1000R+**(約6-12ヶ月)必要。 それ未満の高ROIは偶然を疑う。
- **本番投入検討の水準**: paper 単勝 ROI が 300R+ で >100% を維持 ★かつ★ 三連複が 1000R+ で >100% を維持 ★かつ★ paper がleak-freeバックテスト(単勝111%/三連複194%)に概ね追従。 paper << backtest なら「backtest楽観が本物」=投入見送り。
- ★ いずれも paper のみ。 本番投票への昇格は別途 れんはす判断(このスクリプトは投票しない)。 ★

## 検証済み(2026-06-03)
- `score_s2b` を leak-free cache の実レース(阪神16頭)で動作確認 → top6算出OK。
- `from-oof`(leak-free OOF)→`results`→`report` が leak-free バックテストROIを**完全再現**(単勝111.6%/複勝top1 102.3%/馬連top3box146.0%/三連複top4box194.3%/三連単1-2-5 229.0%、N=10350)= ROI計算・照合配線の正しさを実データ+実払戻で確認。
- `predict`(live)は daily_predict の発走前構築を mirror。 次の開催日(土)に実地検証。
