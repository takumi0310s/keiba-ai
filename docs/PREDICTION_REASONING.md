# 予測理由 言語化 設計書

**作成**: 2026-05-23  
**目的**: 「なぜこの馬が上位に選ばれたか」を Discord 通知に表示する

---

## 1. 概要

V15/V21 のスコアは LGB+XGB の出力値 (0〜1) で、人間が直感的に理由を理解しにくい。
各馬のスコアが高い理由を「feature 値 vs レース平均」で言語化する。

---

## 2. V15 Top 特徴量 (5/23 audit 結果)

| rank | feature | gain寄与 | 意味 |
|------|---------|---------|------|
| 1 | `paci_ninki_idx` | 16.93% | JRDB PACI 人気指数 |
| 2 | `paci_jockey_exp_wr` | 16.34% | 騎手勝率 (経験加重) |
| 3 | `paci_jockey_exp_3rd` | 14.78% | 騎手複勝率 (経験加重) |
| 4 | `jrdb_ze_idm_avg` | 9.40% | JRDB スピード指数 (直近平均) |
| 5 | `training_time_filled` | 5.36% | 調教タイム (4F) |
| 6 | `training_per_dist` | 2.92% | 調教距離あたりタイム |
| 7 | `jrdb_ze_ten_avg` | 2.13% | JRDB テン指数平均 |
| 8 | `jrdb_idm` | 1.95% | JRDB IDM (当日速度指数) |
| 9 | `jrdb_class_code` | 1.63% | クラスコード |
| 10 | `horse_career_wr` | 1.21% | 馬通算勝率 |
| 11 | `pop_rank_change` | 0.83% | オッズ変動 (基準比) |

**直接オッズ系**: 1.2% のみ (odds_log 系は Pattern B 未含有、pop_rank_change 0.83% のみ)

---

## 3. 言語化ルール

各馬の feature 値をレース内 z-score で評価:
- z ≥ 1.0 → ◎ (良い)
- z ≥ 0.5 → ○ (やや良い)
- z ≤ -1.0 → × (悪い)

### feature ごとの言語テンプレート

| feature | 高値=良い? | 表示テキスト |
|---------|-----------|-------------|
| `paci_ninki_idx` | ✅ 高い | `PACI人気◎({val:.0f})` |
| `paci_jockey_exp_wr` | ✅ 高い | `騎手勝率◎({val:.0%})` |
| `paci_jockey_exp_3rd` | ✅ 高い | `騎手複勝率◎({val:.0%})` |
| `jrdb_ze_idm_avg` | ✅ 高い | `速度指数◎({val:.0f})` |
| `training_time_filled` | ❌ 低い(速い) | `調教◎({val:.1f}秒)` |
| `horse_career_wr` | ✅ 高い | `勝率◎({val:.0%})` |

---

## 4. 実装設計

```python
TOP_REASON_FEATURES = [
    ("paci_ninki_idx",     True,  "PACI人気",   "{:.0f}"),
    ("paci_jockey_exp_wr", True,  "騎手勝率",   "{:.1%}"),
    ("jrdb_ze_idm_avg",    True,  "速度指数",   "{:.0f}"),
    ("training_time_filled", False, "調教タイム", "{:.1f}秒"),
    ("horse_career_wr",    True,  "馬勝率",     "{:.1%}"),
]

def generate_horse_reason(horse_row: pd.Series, race_df: pd.DataFrame) -> str:
    """
    horse_row: 対象馬の feature 行
    race_df: 同一レース全馬の feature DataFrame
    Returns: "騎手勝率◎(35%) + 速度指数◎(95)" のような文字列
    """
    reasons = []
    for feat, high_is_good, label, fmt in TOP_REASON_FEATURES:
        if feat not in horse_row.index or feat not in race_df.columns:
            continue
        val = horse_row[feat]
        mean = race_df[feat].mean()
        std = race_df[feat].std()
        if std < 1e-6:
            continue
        z = (val - mean) / std
        if not high_is_good:
            z = -z  # 低いほど良い場合は反転
        if z >= 0.8:
            val_str = fmt.format(val)
            reasons.append(f"{label}◎({val_str})")
    return " / ".join(reasons[:3]) if reasons else "データ標準範囲"
```

### Discord 表示例

```
| 順 | 馬番 | 馬名 | V15score | 単勝 | 理由 |
|----|-----|------|---------|------|------|
| 1  |  5  | ホワイトオーキッド | 0.721 | 3.5 | 騎手勝率◎(38%) / 速度指数◎(95) |
| 2  |  3  | フィールオーサム | 0.683 | 5.2 | PACI人気◎(82) / 調教◎(51.2秒) |
| 3  |  7  | サクセスカラー | 0.651 | 7.8 | 速度指数◎(91) / 馬勝率◎(31%) |
```

---

## 5. 実装ステップ (5/24+)

1. `tools/v21_per_race_paper.py` の `predict_v21` 関数で prediction 後に `df` から `reason_list` 生成
2. `build_discord_message` の table に `reason` 列追加
3. V15 (race_auto_notify → stage2_predict) にも同じ関数を適用

---

## 6. オッズ由来スコアについての補足

V15 は **直接オッズ系が 1.2% のみ** — スコアの大半は騎手成績・速度指数・調教から来ている。
→ 「オッズが高いから上位」ではなく「実力・騎手・調教」が主因。

ただし `paci_ninki_idx` (16.9%) は JRDB の人気指数で、間接的に公衆評価 (オッズ傾向) を反映する。
→ 「PACI人気◎」の理由を表示する際は「市場人気」ではなく「JRDB予測指数」と表現推奨。

*V15 production 完全不変 — 本設計は表示のみ*
