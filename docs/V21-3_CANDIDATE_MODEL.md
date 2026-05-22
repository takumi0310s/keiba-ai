# V21-3 Candidate Model — Final Verdict

**作成日**: 2026-05-22
**担当**: V21-3 agent (claude-sonnet-4-6)
**git push**: なし (明示)
**V15 production 変更**: なし (確認済み)

---

## 1. Verdict: NO-GO (TIMEOUT → エビデンス精査により NO-GO 確定)

`data/v21_verdict.json` は存在しない (V21-2 ablation スクリプトが未完了 または未実行)。
ただし既存の TYB 評価エビデンス (`data/v21/` 配下) を精査した結果、
V21 候補モデルの作成は **NO-GO** と判断する。

---

## 2. 判断根拠

### 2-1. V21 WF AUC が V15 baseline を大きく下回る

`models/v21/wf_summary.json` (3-fold, 100 features, include_video=True):

| fold (val_year) | AUC ens |
|-----------------|---------|
| 2023 | 0.7857 |
| 2024 | 0.7955 |
| 2025 | 0.7942 |
| **mean** | **0.7918** |

V15 genuine WF mean (6-fold LGB+XGB): **0.8678**
V21 WF mean (3-fold): **0.7918** → **△ -0.0760 (大幅劣後)**

### 2-2. TYB 単体信号はあるが「V15 への genuine 上乗せ」が確認できない

`data/v21/jrdb_tyb_eval_report.json` 結果 (n=348, 後付き evaluation):

| 評価 | AUC |
|------|-----|
| V15 top1_score 単独 | 0.509 |
| TYB features 単独 (5CV) | 0.583 |
| V15 + TYB 結合 logistic (5CV) | 0.601 |

- **問題1 — タイミング LEAK**: TYB ZIP 配信は 17:00 JST (post-race)。`odds_idx` は bet-close 前オッズ基準 (corr=0.42 with target)、`jrdb_paddock_idx` は paddock 観察 (corr=0.35)。live 運用では **delivery timing mismatch** により使用不可。`data/v21/tyb_leak_audit.json` の `verdict` が明示: "Live deployment blocked by delivery timing until publish-monitor reconnect."
- **問題2 — training-time merge コードが存在しない**: `build_v134_dataframe` に TYB merge step がなく、`fill_v141_defaults` が定数デフォルト (50.0) で埋めている。format mismatch (cache=10桁 JV ID / TYB=12桁 NK ID) も未解決。
- **問題3 — V15 は TYB 未学習**: V15 LGB booster の num_feature=145、TYB columns は Pattern B features list (150) に含まれるが `predict_core` が `X[:, :145]` でスライスするため **推論時に TYB 値が届かない**。TYB を活かすには完全 retrain 必須。
- **問題4 — n=348 の小サンプル評価**: WF ではなく後付き 5CV (n=348) での AUC 0.601 は leaky upper bound の可能性が高い。

### 2-3. Calibration WF では AUC 変化なし

`data/v21/calibration_wf/metrics.json` (6-fold, 145 features, LGB+XGB):
- mean AUC: **0.8673** (V15 genuine 0.8678 と同水準)
- Platt / isotonic / beta / temperature: AUC delta ≈ 0.000 (calibration は AUC を変えない)
- **TYB を加えた genuine WF での AUC 向上ゼロが独立確認された**

### 2-4. Paper sim はポジティブだが信頼性に課題

`data/v21/calibration_wf/paper_sim_summary.json` (n=347 live samples, 2026-03-14 〜 05-16):
- baseline ROI 1.091 / threshold=0.5 で ROI 1.292 (n=189)
- ただしこれは V15 calibration overlay の paper sim であり **TYB 追加効果の評価ではない**
- n=189 (threshold=0.5) は統計的有意性なし (CI 幅が大きい)

---

## 3. 推奨: TYB を Discord 補完表示のみに使用

TYB data (特に `padock_idx`, `odds_idx`, `jockey_idx`) は **情報としての信号は存在する** (individual AUC 0.617-0.644)。
ただしモデル組み込みではなく **Discord buy-ticket 通知への補完表示** として活用する:

```
[TYB 直前情報]
パドック指数: 72 (A-rank相当)
オッズ指数: 68 (市場支持)
騎手指数: 65
総合指数: 69
```

- `tools/tyb_shadow_fetcher.py` (既存) をそのまま使用
- V15 スコア本体は変更しない
- TYB 値は「参考情報」として rider・購買判断の補助に使用

---

## 4. V21 Retrain 必要条件 (6/1+ 観測後判断)

V21 retrain が正当化される条件:

| 条件 | 目標値 |
|------|--------|
| TYB merge code 実装 (training + inference, race_id format 統一) | 完了 |
| TYB live timing confirmed (JRDB delivery < race time) | 確認済み |
| Genuine WF AUC (6-fold) | V15 0.8678 + 0.002 = **0.8698+** |
| LIVE retro winner_top1 | ≥ 30% |
| Shift | ≤ 12x |
| LEAK 監査 PASS | SKB 除外 確認 |

上記が揃う前は V21 retrain は行わない。**次の判断タイミング: 6/17 観測結果確認**。

---

## 5. V15 Regression Test 結果

```python
import gzip, pickle
with gzip.open('keiba_model_v15_central.pkl.gz', 'rb') as f:
    v15 = pickle.load(f)
lgb = v15['model']
assert len(lgb.feature_name()) == 145  # PASS
print('V15 version:', v15['version'])       # 'v15'
print('V15 auc stored:', v15['auc'])        # 0.8939485520467574
print('V15 trained_at:', v15['trained_at']) # 2026-04-08T23:32:37.533143
print('V15 features:', len(lgb.feature_name())) # 145
```

**結果: PASS — V15 production (keiba_model_v15_central.pkl.gz) は一切変更なし**
- LGB booster features: **145** (変更なし)
- stored AUC: **0.8939** (V15-audit-2 確認済: train-set self-eval, in-sample)
- genuine WF LGB+XGB: **0.8678** (V15-audit-2 真値)

---

## 6. まとめ

| 項目 | 結果 |
|------|------|
| v21_verdict.json | 存在しない (TIMEOUT) |
| V21-2 ablation 実行状況 | 未完了 |
| TYB 信号 (単体) | 中程度 (AUC 0.617-0.644) だが LEAK リスク大 |
| V21 genuine WF AUC | 0.7918 (V15 0.8678 より -0.076 劣後) |
| V21 candidate モデル作成 | **NO-GO** |
| v21_candidate.pkl.gz | **作成せず** |
| TYB 活用推奨 | Discord 補完表示のみ |
| 次回判断 | 6/17 LIVE retro 結果次第 |
| V15 production | **不変確認済み** |
| git push | **なし** |

---

★ git push なし 明示 ★
このドキュメントはローカル commit のみ。Streamlit Cloud への反映なし。
