# JRDB TYB (直前情報) retrospective evaluation report

**実施日**: 2026-05-16
**source**: `tools/v21/jrdb_tyb_evaluate.py`
**V15 production 影響**: ★ 0% (評価のみ、 model 不変) ★

## 1. data 概要

- n_samples: **348**
- pos / neg: 204 / 144
- pos_rate: 0.586

## 2. AUC 比較

| model | AUC |
|-------|-----|
| V15 top1_score 単独 | 0.5091
| TYB features 単独 (5CV) | 0.5831
| V15 + TYB 結合 (5CV) | 0.6008

**V15 + TYB delta: +0.0917** (採用候補)

## 3. TYB feature 単独 correlation + AUC (top3 hit)

| feature | n | corr | AUC best (反転考慮) |
|---------|---|------|---------------------|
| tansho_odds | 348 | -0.242 | 0.644 |
| fukusho_odds | 348 | -0.229 | 0.627 |
| odds_idx | 348 | 0.220 | 0.622 |
| padock_idx | 348 | 0.229 | 0.619 |
| jockey_idx | 348 | 0.213 | 0.618 |
| info_idx | 348 | 0.189 | 0.609 |
| padock_mark | 320 | -0.129 | 0.583 |
| weight_diff | 105 | -0.134 | 0.558 |
| idm | 348 | -0.027 | 0.525 |
| ashimoto | 348 | -0.076 | 0.523 |
| kehai_code | 348 | 0.012 | 0.523 |
| sogo_idx | 348 | 0.056 | 0.517 |
| baba_code | 348 | -0.038 | 0.516 |
| bagu_change | 348 | -0.020 | 0.508 |
| horse_weight | 348 | 0.003 | 0.504 |
| weather_code | 348 | 0.012 | 0.503 |
| cancel_flag | 348 | nan | 0.500 |

## 4. V15 + TYB logistic regression coefficients (standardized)

| feature | coef |
|---------|-----:|
| tansho_odds | -0.677 |
| padock_idx | +0.332 |
| top1_score | +0.231 |
| ashimoto | -0.179 |
| weight_diff | -0.171 |
| idm | -0.148 |
| sogo_idx | -0.098 |
| baba_code | -0.095 |
| fukusho_odds | +0.092 |
| info_idx | +0.078 |
| padock_mark | +0.078 |
| jockey_idx | -0.041 |
| weather_code | -0.032 |
| horse_weight | +0.027 |
| kehai_code | +0.025 |
| bagu_change | +0.024 |
| odds_idx | -0.015 |
| cancel_flag | +0.000 |

## 5. honest 結論

★ **採用候補** ★ — V15 + TYB で AUC +0.0917、 V21 retrain 候補に追加。

## 6. 次 action 候補

- TYB は **5/9 から fetch 停止** (extract dir 確認)。 fetch 復旧必要
- TYB を merge した V15 cache を 構築 → V21/V22 retrain 候補
- 直前 (-15 min) TYB 更新版 取得の schtask は別途検討