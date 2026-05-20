# race_notify_log_v2 hook verify — rerank / calibration paper 蓄積 5/23 SAT fire 判定

作成: 2026-05-21 (c タスク — read-only audit)

---

## 1. v15_top5_rerank.py hook 状態

### 関数・API

| 項目 | 状態 | 詳細 |
|------|------|------|
| `get_paper_formation()` | **存在しない** | 公開エントリは `get_top5_reranked(scores, method, ...)` のみ |
| `simulate_rerank_hit_rate()` | 存在する | 過去データ batch eval 用 (paper backtest) |
| race_notify_log_v2 への hook | **存在しない** | race_auto_notify.py / race_notify_log_v2.py との接点ゼロ |
| 出力 format | N/A | 馬番 `list[int]` を返すのみ (trio 文字列 "1-3-5" 形式は未生成) |
| 5/23 SAT phase 2 から呼び出せるか | **No** | race_auto_notify.py 内に import / 呼び出し箇所なし |

### 欠損している hook

```python
# 現状の race_auto_notify.py → _v2_log_phase2_safe() の呼び出し (line 460)
_v2_log_phase2_safe(
    race_id, race_name, rinfo, bets, odds_dict, cond_key, bet_type,
    channel='bets', strategy_7c_skip=False, strategy_7c_reason=None
)
# ★ predictions 引数が渡されていない → log_phase2() 内の 8 strategy 計算が空になる
# ★ rerank 結果を計算する箇所が皆無
```

`log_phase2()` の `predictions` 引数は **オプション** (省略時 `{k: None}` になる) なので現状動作はするが、rerank 専用 strategy の formation は一切記録されない。

---

## 2. calibrator_overlay_v2.py hook 状態

### 関数・API

| 項目 | 状態 | 詳細 |
|------|------|------|
| `get_paper_formation()` | **存在しない** | calibration は スコア変換器であり formation 生成関数ではない |
| race_notify_log_v2 への hook | **存在しない** | race_auto_notify.py との接点ゼロ |
| calibrated formation の出力 | N/A | calibrate_v2() は float[] → float[] 変換のみ |
| 5/23 SAT phase 2 から呼び出せるか | **No** | import / 呼び出し箇所なし |
| calibrator .pkl 保存先 | `data/v21/calibrator_v2_isotonic.pkl` | **現時点でファイル未生成 (fit 未実行)** |

### calibration の役割整理

`calibrator_overlay_v2.py` は "calibrated スコア → formation" の変換ではなく、  
"raw score → calibrated prob" の変換器。formation 生成には別途 rerank (または baseline) を通す必要がある。

- paper 蓄積するには `calibrate_v2(scores)` で補正後 `get_top5_reranked()` を呼ぶ 2-step が必要。
- その組み合わせを race_notify_log_v2 に渡す仲介コードは現状存在しない。

---

## 3. race_notify_log_v2.py — 拡張可能性

### STRATEGY_KEYS (現行 8 keys)

```python
STRATEGY_KEYS = [
    'actual', 'c3', 'c4', 'c3c4',
    'no_1pop', 'divergence', 'ev_filter', 'odds_filter',
]
```

### 新 key (rerank / calibration) を追加する場合

`build_strategy_formations()` は `STRATEGY_KEYS` をリテラルで参照しているため、  
**リストに追加するだけでは不十分**。以下の 3 箇所を同時変更が必要:

1. `STRATEGY_KEYS` リストに `'rerank_cond_aware'` / `'calibrated'` 等を追加
2. `build_strategy_formations()` 内に formation 計算ロジックを追加
3. `compute_strategy_results()` は `STRATEGY_KEYS` イテレーションなので **変更不要**

aggregator (`race_notify_log_v2_aggregator.py`) は `STRATEGY_KEYS` を直接 import して動的にイテレートするため、**aggregator 側は変更不要**。新 key が `race_notify_log_v2.py` の `STRATEGY_KEYS` に追加されれば自動集計される。

---

## 4. race_auto_notify.py — phase 2 呼び出し箇所

### `_v2_log_phase2_safe()` 呼び出し一覧

| 場所 | channel | predictions 渡し状況 |
|------|---------|---------------------|
| no_horse_data skip (line 164) | 'skip' | なし |
| obstacle_race skip (line 171) | 'skip' | なし |
| distance<=1000 skip (line 181) | 'skip' | なし |
| strategy_7_06_tokubetsu skip (line 195) | 'skip' | なし |
| strategy_7_kyoto skip (line 204) | 'skip' | なし |
| strategy_c4 skip (line 296) | 'skip' | なし |
| cond_E skip (line 302) | 'skip' | なし |
| cond_B skip (line 307) | 'skip' | なし |
| cond_X skip (line 315) | 'skip' | なし |
| **bets 送信成功 (line 460)** | **'bets'** | **なし ← hook point** |
| exception (line 476) | 'error' | なし |

**hook point** は line 460 (bets 送信成功時) のみ。  
この時点で `df` (scores DataFrame) と `odds_dict` は揃っているため、rerank 計算は可能。

### predictions 形式 (log_phase2 が期待する形式)

```python
# log_phase2() の predictions 引数 — 各要素 dict:
predictions = [
    {'horse_num': int, 'pop_rank': int, 'odds': float, ...},
    ...
]
# race_auto_notify.py 内の df は '馬番', 'スコア', 'pop_rank', '単勝オッズ' 等を持つ
```

---

## 5. 5/23 SAT fire ready 判定

| 対象 | 判定 | 理由 |
|------|------|------|
| **rerank paper 蓄積** | ❌ **not ready** | race_auto_notify → _v2_log_phase2_safe に predictions 未渡し / rerank 計算ロジックなし |
| **calibration paper 蓄積** | ❌ **not ready** | calibrator .pkl 未 fit / race_auto_notify に hook なし |
| 既存 8 strategy paper 蓄積 | ⚠️ **部分 ready** | predictions 未渡しのため strategy_formations が全 None になっている (phase 2 のみ) |

### 既存 8 strategy の状態補足

`_v2_log_phase2_safe()` が `predictions=None` で `log_phase2()` を呼ぶため、  
`strategy_formations = {k: None for k in STRATEGY_KEYS}` になっている (log_phase2 line 403)。  
→ phase 3 で `strategy_results` も全 skipped=True となり、aggregator の strategy 集計は **全 N=0**。

---

## 6. 要修正箇所と変更量

### 最小変更 — 既存 8 strategy + rerank paper 蓄積を有効化する

#### 変更 A: `_v2_log_phase2_safe()` に predictions を渡す (race_auto_notify.py)

変更量: **~15 行** (既存ロジック不変、wrapper 内部のみ)

```python
# race_auto_notify.py — _v2_log_phase2_safe シグネチャ変更 (line 641)
# Before:
def _v2_log_phase2_safe(race_id, race_name, rinfo, bets, odds_dict, cond_key, bet_type,
                         channel='bets', strategy_7c_skip=False, strategy_7c_reason=None):

# After:
def _v2_log_phase2_safe(race_id, race_name, rinfo, bets, odds_dict, cond_key, bet_type,
                         channel='bets', strategy_7c_skip=False, strategy_7c_reason=None,
                         df=None):  # +1 arg
```

wrapper 内部で predictions を構築して `log_phase2` に渡す:

```python
        # _v2_log_phase2_safe 内部 (log_phase2 呼び出し前) に追加
        predictions_list = []
        if df is not None:
            try:
                for _, row in df.iterrows():
                    predictions_list.append({
                        'horse_num': int(row.get('馬番', 0)),
                        'pop_rank': int(row.get('pop_rank', 99)),
                        'odds': float(row.get('単勝オッズ', 0.0) or row.get('odds', 0.0)),
                    })
            except Exception:
                predictions_list = []

        _v2_log_phase2(
            ...,
            predictions=predictions_list or None,  # ← 追加
        )
```

bets 送信成功時の呼び出し (line 460) を変更:

```python
# Before (line 460):
_v2_log_phase2_safe(race_id, race_name, rinfo, bets, odds_dict, cond_key, bet_type,
                    channel='bets', strategy_7c_skip=False, strategy_7c_reason=None)

# After:
_v2_log_phase2_safe(race_id, race_name, rinfo, bets, odds_dict, cond_key, bet_type,
                    channel='bets', strategy_7c_skip=False, strategy_7c_reason=None,
                    df=df)  # ← df は predict_and_notify() スコープで利用可能
```

#### 変更 B: rerank strategy を `STRATEGY_KEYS` + `build_strategy_formations()` に追加 (race_notify_log_v2.py)

変更量: **~30 行**

```python
# race_notify_log_v2.py — STRATEGY_KEYS に追加
STRATEGY_KEYS = [
    'actual', 'c3', 'c4', 'c3c4',
    'no_1pop', 'divergence', 'ev_filter', 'odds_filter',
    'rerank_cond_aware',  # ← 追加: condition-aware rerank で top5 を選び直した formation
    'rerank_odds_div',    # ← 追加: odds divergence boost rerank
]
```

`build_strategy_formations()` 末尾に追加:

```python
        # rerank_cond_aware
        try:
            from tools.v15_top5_rerank import get_top5_reranked
            import pandas as pd
            scores_ser = pd.Series(
                {h.get('horse_num') or h.get('umaban'): float(h.get('score', h.get('スコア', 0.0)))
                 for h in predictions if h.get('horse_num') or h.get('umaban')}
            )
            if len(scores_ser) >= 3:
                reranked5 = get_top5_reranked(scores_ser, method='condition_aware', condition=cond_key)
                result['rerank_cond_aware'] = None if base_skip else make_trio_bets(reranked5)
            else:
                result['rerank_cond_aware'] = None
        except Exception:
            result['rerank_cond_aware'] = None
```

ただし predictions の各 dict に `score` / `スコア` キーが含まれている必要がある。  
変更 A で predictions を構築する際に `'score': float(row.get('スコア', 0.0))` を追加する。

#### 変更 C: calibration paper (追加で必要な前提作業)

1. `python tools/calibrator_overlay_v2.py --fit-isotonic` を一度実行して `data/v21/calibrator_v2_isotonic.pkl` を生成
2. `build_strategy_formations()` 内に `rerank_cond_aware` と同様のパターンで `calibrated` strategy を追加

変更量: **~20 行** (fit 実行後)

---

## 7. まとめ

| 確認項目 | 結果 |
|----------|------|
| rerank `get_paper_formation()` 関数 | 存在しない (設計外) |
| calibration `get_paper_formation()` 関数 | 存在しない (設計外) |
| 既存 8 strategy の predictions 渡し | **欠落** — strategy_formations 全 None |
| aggregator の動的 key 対応 | **OK** — STRATEGY_KEYS import で自動集計 |
| 5/23 SAT rerank paper ready | **❌ not ready** |
| 5/23 SAT calibration paper ready | **❌ not ready** |
| 最小修正で 5/23 SAT に間に合わせるか | 変更 A (~15行) のみで既存 8 strategy を救済可能 / rerank 追加は変更 A+B (~45行) |

### 推奨アクション (5/23 SAT まで)

1. **変更 A のみ実施** — predictions を `_v2_log_phase2_safe` に渡し、既存 8 strategy の paper 蓄積を有効化。rerank / calibration は今週末はスキップ (6/1 以降)。
2. 変更対象: `tools/race_auto_notify.py` のみ (race_notify_log_v2.py は不変)
3. 変更量: ~15 行、テスト: `python -c "import py_compile; py_compile.compile('tools/race_auto_notify.py', doraise=True)"`

> **注意**: race_auto_notify.py は 🔴 NEVER 改変対象のため、本 verify は read-only。  
> 実施する場合は担当者判断でこの doc の diff を適用すること。
