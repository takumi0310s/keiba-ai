# Phase D-1 — V15 inference flow review (read-only)

date: 2026-05-16
session: Terminal D (Phase D, V21 architecture)
status: read-only review、 V15 file は 1 行も触らない

---

## 1. scope と read-only 確約

| file | LOC | 状態 | 取扱 |
|---|---|---|---|
| `tools/predict_core.py` | 2,680 | ★ V15 production core ★ | 100% read-only |
| `tools/daily_predict.py` | (頭 60 行のみ確認) | 朝 8:00 cron | 100% read-only |
| `tools/race_auto_notify.py` | 未読 | 5 分前 cron | 100% read-only |
| `train/features_v15_new.py` | 517 | V15 features 定義 | 100% read-only |
| `keiba_model_v135_central*.pkl.gz` | — | V15 model | 100% read-only |

→ 本 Phase D で V15 inference に 1 文字も改変は加えない。

---

## 2. V15 inference flow (ASCII diagram)

```
┌──────────────────────────────────────────────────────────────┐
│ daily_predict.py / race_auto_notify.py (caller)              │
└──────────────────────────────────────────────────────────────┘
                              │
                              ▼
              ┌─────────────────────────────────┐
              │ predict_core.load_models()      │  (line 471-518)
              │  - Pattern B 優先 / Pattern A   │
              │  - keiba_model_v135_*.pkl.gz    │
              │  - returns dict:                │
              │    {model, features, sire_map,  │
              │     bms_map, ensemble_weights,  │
              │     xgb_model, ft_*, ir_*}      │
              └─────────────────────────────────┘
                              │
                              ▼
              ┌─────────────────────────────────┐
              │ predict_core.parse_shutuba()    │  (line 569+)
              │  - netkeiba 出馬表 scrape       │
              │  - 馬名 / 馬番 / 斤量 / 馬齢   │
              │  - returns: horses[], race_info │
              └─────────────────────────────────┘
                              │
                              ▼
              ┌─────────────────────────────────┐
              │ predict_core.get_horse_stats()  │  (line 757+)
              │  - 個別 馬 expanding stats       │
              │  - 騎手勝率 / 父産駒勝率 等      │
              └─────────────────────────────────┘
                              │
                              ▼
              ┌─────────────────────────────────┐
              │ predict_core.apply_horse_stats()│  (line 1419)
              │  - horses[i] に stats merge      │
              └─────────────────────────────────┘
                              │
                              ▼
              ┌─────────────────────────────────┐
              │ predict_core.build_features()   │  (line 1469+)
              │  - 全 150 features 構築         │
              │  - V15 new features 結合        │
              │    (jockey_horse_* /            │
              │     transport_* /               │
              │     course_renovated 等)        │
              │  - returns: DataFrame[150]      │
              └─────────────────────────────────┘
                              │
                              ▼
              ┌─────────────────────────────────┐
              │ predict_core.predict_race()     │  (line 2149+)
              │  - LGB.predict(X)               │
              │  - XGB.predict(X)               │
              │  - FT-Transformer (optional)    │
              │  - IntraRace Attention (★ 0.35) │
              │  - ensemble: weighted sum       │
              │    ai_scores = w_lgb·LGB        │
              │              + w_xgb·XGB        │
              │              + w_ft·FT          │
              │              + w_ir·IR          │
              │  - final_scores = 0.65·ai_score │
              │                 + 0.08·odds +.. │
              │  - df['スコア'] = final_scores  │
              │  - df['AI順位'] = rank          │
              │  - returns: sorted DataFrame    │
              └─────────────────────────────────┘
                              │
                              ▼
              ┌─────────────────────────────────┐
              │ predict_core.classify_race_*    │  (line 387)
              │  + generate_trio_bets / umaren  │  (line 415, 444)
              │  → 買い目 7 点 (umaren 2 点)    │
              └─────────────────────────────────┘
                              │
                              ▼
                       output: CSV / Discord
```

---

## 3. 入力 / 出力 schema

### 入力 (predict_race の前提)

`build_features` の出力 DataFrame は 以下を含む:

| 列 | 型 | 由来 |
|---|---|---|
| 150 features (`model_data['features']`) | float | V15 学習時の features list |
| `人気傾向` | float | 補助 score |
| `距離適性`, `馬場適性` | float | 補助 score |
| `脚質` | int (1-4) | 逃 / 先 / 差 / 追 |
| `上がり3F` | float | 前走 上がり |
| `コース適性`, `血統スコア`, `複勝率` | float | 補助 |
| `単勝オッズ` | float | live mode のみ |
| `馬番`, `馬名`, `斤量`, `馬齢` | mixed | UI 表示 |

### model_data dict (load_models 返却値、★ 不変 contract ★)

```python
{
    'model': lgb.Booster,                # LGB main
    'features': List[str],               # ★ 150 features (V15) ★
    'sire_map': Dict[str, int],          # 父馬 encoding
    'bms_map': Dict[str, int],           # 母父 encoding
    'version': 'v15' | 'v13.5b' | ...,
    'n_top_encode': 80,
    'is_live': bool,
    'ensemble_weights': {                # ★ V15 grid 重み ★
        'lgb': 0.25, 'xgb': 0.25,
        'ft': 0.15,  'ir': 0.35,
    },
    'xgb_model': xgb.Booster | None,
    'ft_model_state': dict | None,       # PyTorch state_dict
    'ft_model_config': dict | None,
    'ft_scaler_mean': np.ndarray | None,
    'ft_scaler_scale': np.ndarray | None,
    'ir_model_state': dict | None,
    'ir_model_config': dict | None,
    'ir_scaler_mean': np.ndarray | None,
    'ir_scaler_scale': np.ndarray | None,
}
```

### predict_race の出力

入力 df に以下を追加:
- `スコア` (float, final_scores)
- `AI順位` (int, rank by スコア desc)

ai_scores (ensemble combined) は **DataFrame に格納されず返却もされない**。
→ V21 stacking 用には `predict_race` を呼ぶ前に **同じ計算を再現** か、 hook 設計が必要。

---

## 4. V21 統合 で 触れない こと

### 4.1 V15 production code は完全 read-only

`predict_core.py` の `load_models / build_features / predict_race` 等の関数を **import で再利用するだけ**。
追加 wrap や hook を入れない (V15 inference を 1 byte も変えない)。

### 4.2 schtasks

- DailyPredict (08:00) / RaceAutoNotify (5 分前) は V15 path のみ
- V21 path は paper trade engine の 独立 schtasks を 別途新規登録 (6/1+)

### 4.3 V15 model file

- `keiba_model_v135_central_live.pkl.gz` / `*_central.pkl.gz` は touch しない
- V21 meta model は **新規 file** (`models/v21/v21_meta_lgb.pkl.gz`)

---

## 5. V21 統合 で 必要な V15 hook

V21 = V15 score + 動画 30 features → meta stacking。

V15 inference の中間結果 (LGB / XGB / FT / IR の **個別** スコア、 ensemble 前) を取りたい場合:

### 案 1 (推奨): predict_race を call 後の **final_scores のみ** 利用
- pros: V15 inference path 完全不変、 import で再利用するだけ
- cons: 中間 raw を取れず、 stacking 入力は (final_score, video_30) の 31 dim
- 採用: ★ 案 1 を採用 ★ (V15 不変保証 最優先)

### 案 2 (不採用): predict_race を model 別に分解
- pros: stacking 入力に LGB / XGB / FT / IR の個別 score を渡せる
- cons: V15 inference logic を再実装する必要、 不変保証 違反のリスク
- → 採用 NG

→ ★ V21 = stacking(V15_final_score, video_30_features) で確定 ★

---

## 6. 既存 `tools/predict_core_v21.py` との関係

Phase 16 (Session #87, 5/10) で 既に `tools/predict_core_v21.py` (top-level) が作成済:
- 動画 30 features の **registry + default fill + skeleton fetcher** のみ
- inference orchestrator (V15 score と stacking する logic) は未実装

Phase D で追加する `tools/v21/predict_core_v21.py` は **inference orchestrator** 役割。
既存 Phase 16 file の registry / fetcher を import で再利用する。

| file | 役割 | 状態 |
|---|---|---|
| `tools/predict_core.py` | V15 production inference | ★ 不変 ★ |
| `tools/predict_core_v21.py` (Phase 16) | 動画 30 features registry / fetcher skeleton | 既存、 不変 |
| `tools/v21/predict_core_v21.py` (Phase D 新規) | V21 inference orchestrator (V15 + 動画 stacking) | ★ 本 Phase で新規 ★ |

---

## 7. fabrication 防止 注記

- 本 review は `tools/predict_core.py` の line 471 (load_models) / 1469 (build_features) / 2149 (predict_race) / 各 ensemble block を実際に read した結果に基づく
- V15 ensemble weights ({lgb: 0.25, xgb: 0.25, ft: 0.15, ir: 0.35}) は CLAUDE.md 記載値、 model_data['ensemble_weights'] dict の実値は learning 時に確定
- V15 inference を 1 byte も変えない 確約

---

## 8. 結論

V21 統合 は 以下 で 確定:
- V15 production inference は predict_race の **出力 final_score** のみ利用
- 動画 30 features は 既存 Phase 16 skeleton で取得 (default fill OK)
- meta-model (LGB) で `[V15_final_score] + [video_30]` → V21 score を予測
- 動画 features 欠損時は fallback で V15 score をそのまま返す (完全互換)
