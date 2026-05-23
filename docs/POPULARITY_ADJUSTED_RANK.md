# 人気除外順位 (Popularity-Adjusted Rank) 設計ドキュメント

**作成日**: 2026-05-23  
**対象モデル**: V15 production (LGB+XGB 2-model, 145 features)  
**ステータス**: 設計のみ / V15 inference 完全不変

---

## 1. 目的と背景

### 問題

V15 の最重要特徴量は `paci_ninki_idx` (JRDB PACI 人気指数、LGB gain 16.93%)。
この特徴量は JRDB が前日配信する総合人気評価であり、市場オッズと高い相関を持つ。

Discord 通知で「V15 #1 オオタニサーン」と表示されても、それが **JRDB 人気も #1 だから** なのか **能力評価で純粋に #1** なのかがわからない。

### ユーザー要求

通知に「**人気除外順位**」を並記したい。例:

```
V15 #1 オオタニサーン (スコア0.85) [能力:#1 / 人気:#3]
V15 #2 ペッパーミル   (スコア0.72) [能力:#5 / 人気:#1]  ← 人気頼み
V15 #3 サムライブルー (スコア0.68) [能力:#2 / 人気:#7]  ← 能力で上位
```

### 絶対制約

- **V15 production の inference logic は完全不変**
- `predict_core.py` / `app.py` のコア予測ロジックは触らない
- 「表示・通知のみ」の後処理として実装する

---

## 2. paci_ninki_idx の性質 (分析済み)

| 項目 | 値 | 出典 |
|------|----|----|
| V15 LGB gain 寄与 | **16.93%** | docs/ODDS_DEPENDENCY_ANALYSIS.md |
| 種別 | JRDB PACI ファイル (前日配信、PRE-RACE 確認済) | V15-audit-3 |
| 直接オッズ系 (odds_log 等) | 合計 1.21% のみ (別集計) | 同上 |
| paci_ninki_idx を除外した場合の AUC 変化 | 推定 -0.015 | 同上 |
| 実際のスコア確認 (`fullclass_test_v15.json`) | 欠損時 default 159.0 (全馬同値) | data/fullclass_test_v15.json |

**重要**: `paci_ninki_idx` は JRDB 独自の人気総合指数であり、当日確定オッズそのものではない。しかし市場と高相関の「事前人気評価」であるため、「人気が影響しているか」の代理指標として使える。

---

## 3. 設計方針: 方法 A vs 方法 B

### 方法 A: 再予測法 (paci_ninki_idx を置換して再推論)

```
手順:
1. predict_race() 実行後、df (スコアソート済み) を取得 → V15 順位確定
2. df のコピーを作り paci_ninki_idx を全馬 population mean (or レース内平均) に置換
3. predict_race() を再度呼び出す (model_data 同一)
4. 再予測スコアでランク付け → 「能力順位」
5. V15 順位と能力順位の差を通知に付記
```

**長所**:
- 厳密。モデルが実際に paci_ninki_idx をどう使っているかを正確に反映
- 交互作用特徴量 (paci_ninki_idx × 他 feature のツリー分岐) も含めて除去できる

**短所**:
- `predict_race()` を 2 回呼ぶ → 推論時間 x2 (1 レース 0.1-0.5 秒程度、許容範囲内)
- `predict_core.py` から `predict_race` を import して呼ぶだけで、中身は変更しない → 制約上セーフ
- paci_ninki_idx が 0 (デフォルト値 = 全馬同値 159.0 のケース多数、5/23 確認済) のとき、再予測はほぼ V15 と同じになり差が出ない

### 方法 B: 近似 SHAP 法 (スコアから寄与分を引いてリランク)

```
手順:
1. predict_race() 実行後の ai_scores (保存不可、df['スコア'] のみ) を使用
2. 各馬の paci_ninki_idx の z-score を計算 (レース内の相対値)
3. 近似寄与 = 固定係数 × z-score
   (固定係数 = gain 16.93% に基づく経験値)
4. df['スコア'] から近似寄与を引いてリランク → 「能力順位」
```

**長所**:
- 推論 1 回のみ (高速)
- 既存の df['スコア'] だけで計算できる (predict_core の内部は見なくていい)

**短所**:
- 近似精度が低い。LGB/XGB のツリーは非線形なので SHAP の近似は不正確
- `gain 16.93%` はモデル全体の重要度であり 1 馬ごとの線形係数ではない
- 「引くべき係数」が不明確なため、ユーザーに誤解を与えるリスクがある
- 再度 df['スコア'] が `ai_scores` (LGB+XGB 生出力) ではなく、pace_scores / apt_scores 等を含む複合スコアなので、さらに近似精度が落ちる

### 推奨: **方法 A を採用**

理由:
1. 正確性が高い (交互作用含む)
2. 推論 2 回のオーバーヘッドは軽微 (1 レース数百ミリ秒)
3. 実装がシンプル — `predict_race()` をそのまま再利用するだけ
4. 「predict_core.py を変更しない」制約を満たせる

---

## 4. 推奨実装案

### 4-1. 新規ヘルパー関数 (tools/popularity_rank.py として新規作成)

```python
"""
人気除外順位 (Popularity-Adjusted Rank) 計算モジュール

V15 production の inference は完全不変。
predict_race() を 2 回呼ぶことで「人気指数を除いた場合のスコア順位」を算出する。
通知・表示用の後処理のみ。モデル学習・評価には一切使わない。
"""
import numpy as np
import pandas as pd


NINKI_FEATURE = 'paci_ninki_idx'


def calc_popularity_adjusted_rank(df_sorted, model_data, odds_available=False, race_info=None):
    """
    V15 スコアソート済み df を受け取り、人気除外順位を付与して返す。

    Parameters
    ----------
    df_sorted : pd.DataFrame
        predict_race() が返した、スコアでソート済みの DataFrame。
        columns に 'スコア', '馬番', NINKI_FEATURE が含まれること。
    model_data : dict
        load_models() が返すモデルデータ。predict_race() に渡すものと同一。
    odds_available : bool
        オッズ利用可否。predict_race() と同じ値を渡す。
    race_info : dict or None
        レース情報。predict_race() と同じ値を渡す。

    Returns
    -------
    pd.DataFrame
        入力 df に 'PAR_rank' (int) および 'PAR_label' (str) 列を追加したもの。
        PAR_rank  : 人気除外スコアでの順位 (1=最上位)
        PAR_label : 表示用文字列 例 "#2" or "#2-3" (近接スコアの場合)
    """
    try:
        from predict_core import predict_race as _predict_race
    except ImportError:
        # ランクなしで返す (graceful fallback)
        df_sorted['PAR_rank'] = list(range(1, len(df_sorted) + 1))
        df_sorted['PAR_label'] = [f"#{i}" for i in range(1, len(df_sorted) + 1)]
        return df_sorted

    if NINKI_FEATURE not in df_sorted.columns:
        # paci_ninki_idx がない場合は V15 順位をそのまま使う
        df_sorted['PAR_rank'] = list(range(1, len(df_sorted) + 1))
        df_sorted['PAR_label'] = [f"#{i}" for i in range(1, len(df_sorted) + 1)]
        return df_sorted

    # --- 再予測用 df を作成 (元 df は変更しない) ---
    df_copy = df_sorted.copy()

    # paci_ninki_idx をレース内平均値に置換 (全馬を「平均的な人気」と仮定)
    original_vals = df_copy[NINKI_FEATURE].copy()
    race_mean = original_vals.replace(0, np.nan).mean()  # 0 はデフォルト (欠損扱い)
    if np.isnan(race_mean):
        race_mean = 100.0  # JRDB 人気指数の典型的中間値
    df_copy[NINKI_FEATURE] = race_mean

    # --- 再予測 (predict_race は df['スコア'] / 'AI順位' を上書きする) ---
    try:
        df_reranked = _predict_race(df_copy, model_data, odds_available, race_info)
    except Exception:
        # 再予測失敗 → V15 順位をそのまま使う
        df_sorted['PAR_rank'] = list(range(1, len(df_sorted) + 1))
        df_sorted['PAR_label'] = [f"#{i}" for i in range(1, len(df_sorted) + 1)]
        return df_sorted

    # --- 再予測スコアを元 df に結合 ---
    par_score_map = dict(
        zip(df_reranked['馬番'].astype(int),
            df_reranked['スコア'])
    )
    df_sorted = df_sorted.copy()  # 元 df 保護
    df_sorted['PAR_score'] = df_sorted['馬番'].astype(int).map(par_score_map)

    # 順位付け (スコア降順)
    df_sorted['PAR_rank'] = df_sorted['PAR_score'].rank(
        ascending=False, method='min').fillna(99).astype(int)

    # ラベル生成: 近接スコア (差 < 0.005) は範囲表示
    scores_sorted = df_sorted['PAR_score'].sort_values(ascending=False).values
    score_gaps = np.diff(scores_sorted)

    def _make_label(rank_val):
        r = int(rank_val)
        # 前後が近接している場合は "2-3" 表示
        prev_close = (r > 1 and len(score_gaps) >= r - 1
                      and abs(score_gaps[r - 2]) < 0.005)
        next_close = (r < len(df_sorted) and len(score_gaps) >= r
                      and abs(score_gaps[r - 1]) < 0.005)
        if prev_close:
            return f"#{r-1}-{r}"
        if next_close:
            return f"#{r}-{r+1}"
        return f"#{r}"

    df_sorted['PAR_label'] = df_sorted['PAR_rank'].apply(_make_label)

    # 一時列削除
    df_sorted = df_sorted.drop(columns=['PAR_score'], errors='ignore')

    return df_sorted


def format_par_line(row, v15_rank):
    """
    1 馬分の通知行を生成。

    Parameters
    ----------
    row : pd.Series
        df_sorted の 1 行 (PAR_rank, PAR_label, 馬番, 馬名, スコア 等を含む)
    v15_rank : int
        V15 での順位 (1始まり)

    Returns
    -------
    str
        例: "軸: 5 オオタニサーン (スコア0.85) [能力:1位 / 人気:3位]"
    """
    uma = int(row.get('馬番', 0))
    name = row.get('馬名', '?')
    score = float(row.get('スコア', 0))
    par_label = row.get('PAR_label', f"#{v15_rank}")
    pop_rank = int(row.get('pop_rank', 0) or 0)

    rank_labels = {1: '軸', 2: '2位', 3: '3位'}
    prefix = rank_labels.get(v15_rank, f"{v15_rank}位")

    # 人気括弧
    pop_str = f"人気:{pop_rank}位" if pop_rank > 0 else "人気:不明"
    par_str = f"能力:{par_label}"

    # 乖離強調: PAR 順位が V15 より 3 以上低い場合 (= 人気頼み)
    par_rank_num = int(row.get('PAR_rank', v15_rank))
    if par_rank_num - v15_rank >= 3:
        flag = " ★人気"
    elif v15_rank - par_rank_num >= 3:
        flag = " ★能力"
    else:
        flag = ""

    return f"{prefix}: {uma} {name} (スコア{score:.2f}) [{par_str} / {pop_str}]{flag}"
```

### 4-2. tools/notify.py の build_rich_bet_message() への追加 (オプション引数)

```python
# 既存シグネチャに par_df=None を追加 (後方互換)
def build_rich_bet_message(df, race_name, race_info, cond_key, cond_profile,
                           bets, odds_dict=None, horses=None, date_str=None,
                           upset_data=None, newspaper_data=None,
                           pp_stars=0, pp_matched=None,
                           par_df=None):          # ← NEW (optional)
    ...

    # TOP3 表示部分 (既存 L272-280) を以下に差し替え:
    for i in range(min(3, len(df))):
        row = df.iloc[i]
        if par_df is not None:
            # PAR 情報付き行
            par_row = par_df[par_df['馬番'] == row['馬番']].iloc[0] if len(
                par_df[par_df['馬番'] == row['馬番']]) > 0 else row
            from tools.popularity_rank import format_par_line
            lines.append(format_par_line(par_row, i + 1))
        else:
            # 既存フォーマット (PAR 計算なし)
            num = int(row['馬番'])
            name = row.get('馬名', '?')
            score = row.get('スコア', 0)
            rank_label = ['軸', '2位', '3位'][i]
            lines.append(f"{rank_label}: {num} {name} (スコア{score:.2f})")
```

### 4-3. tools/race_auto_notify.py への呼び出し追加

```python
# predict_and_notify() 内、df = predict_race() の直後に追加

# === 人気除外順位計算 (表示のみ、V15 production 不変) ===
par_df = None
try:
    from tools.popularity_rank import calc_popularity_adjusted_rank
    par_df = calc_popularity_adjusted_rank(
        df.copy(), model_data,
        odds_available=odds_available,
        race_info=rinfo
    )
    print(f"    [PAR] 人気除外順位計算完了")
except Exception as _par_err:
    print(f"    [PAR] 計算失敗 (通常フォーマットで通知): {_par_err}")
    par_df = None

# build_rich_bet_message に par_df を渡す
title, msg, color = build_rich_bet_message(
    df, race_name, rinfo, cond_key, cond_profile,
    bets, odds_dict=odds_dict, horses=horses, date_str=date_str,
    upset_data=_upset_data, newspaper_data=_newspaper_data,
    pp_stars=_pp_stars, pp_matched=_pp_matched,
    par_df=par_df)      # ← PAR 追加
```

---

## 5. 通知フォーマット案

### 現行フォーマット

```
軸: 5 オオタニサーン (スコア0.85)
2位: 2 ペッパーミル (スコア0.72)
3位: 8 サムライブルー (スコア0.68)
```

### PAR 追加後フォーマット

```
軸: 5 オオタニサーン (スコア0.85) [能力:#1 / 人気:3位]
2位: 2 ペッパーミル (スコア0.72) [能力:#5 / 人気:1位] ★人気
3位: 8 サムライブルー (スコア0.68) [能力:#2 / 人気:7位] ★能力
```

### フラグ定義

| フラグ | 条件 | 意味 |
|--------|------|------|
| `★人気` | PAR 順位 - V15 順位 >= 3 | V15 で高評価だが、人気を除くと大きく下がる → 人気頼みの可能性 |
| `★能力` | V15 順位 - PAR 順位 >= 3 | 人気を除いても上位 → 純粋な能力評価が高い |
| (なし) | 差 < 3 | V15 と人気除外順位がほぼ一致 |

### 近接スコアの表示例

```
2位: 12 ホープフルキング (スコア0.69) [能力:#3-4 / 人気:5位]
  ↑ 人気除外スコアが 3位と 4位の間にある (差 < 0.005)
```

### 通知全体サンプル (三連複の場合)

```
🏇 5/24(土) 中山11R 15:45発走

**アネモネS** 芝1600m 良 14頭
条件A ★★★ ROI 205.3% (的中44.5%)

三連複フォーメーション 7点
1列目: 5
2列目: 2, 8
3列目: 2, 3, 8, 11, 14

軸: 5 オオタニサーン (スコア0.85) [能力:#1 / 人気:3位]
2位: 2 ペッパーミル (スコア0.72) [能力:#5 / 人気:1位] ★人気
3位: 8 サムライブルー (スコア0.68) [能力:#2 / 人気:7位] ★能力

💰 配当レンジ: 1,200円〜15,600円
投資額: 700円
```

---

## 6. 変更が必要なファイル一覧

| ファイル | 変更種別 | 変更内容 |
|---------|---------|---------|
| `tools/popularity_rank.py` | **新規作成** | `calc_popularity_adjusted_rank()` + `format_par_line()` |
| `tools/notify.py` | 追加 (後方互換) | `build_rich_bet_message()` に `par_df=None` 引数を追加、TOP3 表示部分を拡張 |
| `tools/race_auto_notify.py` | 追加のみ | `predict_and_notify()` 内で PAR 計算を呼び出し、`build_rich_bet_message` に `par_df` を渡す |

### 変更禁止ファイル

| ファイル | 理由 |
|---------|------|
| `tools/predict_core.py` | V15 inference の本体。完全不変 |
| `app.py` | Streamlit メイン。変更後は必ず構文チェックが必要なため別タスク化 |
| `train/train_v15_master.py` | 学習ロジック。変更不要 |
| `*.pkl.gz` (V15 モデルファイル) | モデル本体。変更禁止 |

---

## 7. V15 production 不変の保証

### なぜ production が変わらないか

1. `calc_popularity_adjusted_rank()` は **predict_race() の戻り値の後処理** にすぎない
2. V15 の `df['スコア']` は predict_race() が計算したままの値を保持 (上書きしない)
3. `par_df` は別の変数に保存し、元 `df` は変更しない
4. bets (買い目) は `df` (V15 スコアソート済み) から生成 — `par_df` から生成しない
5. 例外発生時は `par_df = None` にフォールバック → 既存フォーマットで通知

### テスト手順

```bash
# 1. 構文チェック
python -m py_compile tools/popularity_rank.py
python -m py_compile tools/notify.py
python -m py_compile tools/race_auto_notify.py

# 2. PAR 計算単体テスト
python -c "
from tools.popularity_rank import calc_popularity_adjusted_rank
import pandas as pd
# ダミー df (paci_ninki_idx 列あり)
df = pd.DataFrame({
    '馬番': [1, 2, 3],
    '馬名': ['A', 'B', 'C'],
    'スコア': [0.85, 0.72, 0.68],
    'paci_ninki_idx': [200, 100, 150],
    'pop_rank': [3, 1, 7],
})
print('PAR テスト: model_data なし → graceful fallback 確認')
"

# 3. 既存レース予測テスト (V15 スコアが変化しないことを確認)
# /race-test スキルを使用
```

---

## 8. 既知の限界と注意事項

### paci_ninki_idx が 0 のケース

`data/fullclass_test_v15.json` を確認すると、全馬が `mean=159.0 / min=159.0 / max=159.0` のケース (= JRDB データ未取得のため全馬デフォルト) が多い。この場合:

- V15 スコアには paci_ninki_idx の影響がほぼない (全馬同値のため)
- 再予測スコアも V15 とほぼ同じになる
- PAR 順位 = V15 順位 (または近接) → 「能力:#1 / 人気:3位」のような大きな乖離は出ない

**通知方針**: paci_ninki_idx が全馬ほぼ同値の場合、PAR ラベルを表示しない (または「(JRDB人気データなし)」と表示) ことを推奨。

```python
# calc_popularity_adjusted_rank() 内の追加チェック
std_ninki = df_sorted[NINKI_FEATURE].std()
if std_ninki < 10.0:  # 全馬ほぼ同値 → PAR に意味なし
    df_sorted['PAR_rank'] = list(range(1, len(df_sorted) + 1))
    df_sorted['PAR_label'] = [f"#{i}" for i in range(1, len(df_sorted) + 1)]
    df_sorted['PAR_valid'] = False
    return df_sorted
df_sorted['PAR_valid'] = True
```

### 計算コスト

- 追加推論 1 回 ≒ 0.1〜0.5 秒 (レース馬数 8〜18 頭、CPU inference)
- 発走 5 分前の通知ロジック全体 (scraping + stats + build_features) が 20〜60 秒かかる中では誤差範囲

### 解釈上の注意

- 「能力順位」は **paci_ninki_idx を除いた場合の V15 予測** であり、真の能力評価ではない
- paci_ninki_idx 以外の間接的な人気系特徴量 (pop_rank_change 等) は除去していない
- あくまで **参考情報** として表示する

---

## 9. 実装優先度と推奨タイミング

| ステップ | 内容 | 推奨タイミング |
|---------|------|--------------|
| Step 1 | `tools/popularity_rank.py` 新規作成 + 単体テスト | 次セッション (30 分) |
| Step 2 | `tools/notify.py` に `par_df` 引数追加 | Step 1 完了後 |
| Step 3 | `tools/race_auto_notify.py` に PAR 呼び出し追加 | Step 2 完了後 |
| Step 4 | 5/24 週末 1 レースで Discord 通知フォーマット確認 | Step 3 完了後 |

**実装工数**: 合計 1〜2 時間 (コード + テスト)

---

*V15 production 完全不変 — 本ドキュメントは表示・通知の拡張設計のみ*
