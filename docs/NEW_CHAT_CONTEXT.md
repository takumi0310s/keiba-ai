# 競馬AI予測システム — 新規チャット用コンテキスト
> このファイルをそのまま貼り付けると新規Claudeチャットでシステム全体を理解できる。
> プロジェクトパス: C:\Users\takum\keiba-ai

---

## ★ 絶対ルール (違反禁止)

```
🔴 NEVER: keiba_model_v15_central.pkl.gz を上書き
🔴 NEVER: predict_core.py / daily_predict.py / app.py のロジック改変
🔴 NEVER: V16 / V16a を本番投票に使う (paper shadow のみ)
🔴 NEVER: destructive git op (push --force, reset --hard 等) を無断実行
🟢 OK:    candidate model 作成・WF 検証・ファイル生成・情報追記
```

**撤退ライン**: 累計 -50,000円 / 現在 ¥-35,280 (n=697, ROI=92.77%, 5/26時点) / 余裕 ¥14,720

---

## モデル一覧

| モデル | ファイル | feats | WF AUC | 状態 |
|--------|---------|-------|--------|------|
| **V15** | `keiba_model_v15_central.pkl.gz` | **145** | **0.8678** | **本番稼働** |
| V16 | `models/v16_ability_candidate.pkl.gz` | **137** | **0.8677** | paper shadow |
| V16a | `models/v16a_candidate.pkl.gz` | **144** | **0.8674** | paper shadow (NO-GO) |
| v15_full_optuna | `models/v15_full_optuna_candidate.pkl.gz` | — | — | paper shadow |
| v15_2 | `models/v15_2_candidate.pkl.gz` | — | — | paper shadow |
| v20_base | `keiba_model_v20_base_central.pkl.gz` | — | — | paper shadow |

**WF AUC 注意**: stored `.pkl.auc = 0.8939` は LGB train-set self-eval (LEAKY)。真値は genuine WF 6-fold = **0.8678**。  
**アーキテクチャ**: 全モデル LGB + XGB 2-model ensemble。MLP/FT/IR は pkl.gz 未保存 → 推論は LGB+XGB のみ。

---

## V15 本番モデル詳細

### pkl.gz 内部構造
```python
{
    'model':            LightGBM Booster,
    'xgb_model':        XGBoost Booster,
    'features':         [145 names],          # 推論時はこのリストを使う
    'ensemble_weights': {'lgb': 0.5036, 'xgb': 0.4964, 'mlp': 0},
    'version':          'v15',
    'auc':              0.8939,               # ★LEAKY (in-sample)。参照禁止★
    'leak_free':        True,
    'leak_pattern':     'A',
    'sire_map':         {...},
    'bms_map':          {...},
    'course_map':       {...},
    'mlp_model':        None,
    'mlp_scaler':       None,
}
```

### 推論コード (tools/predict_core.py)
```python
def predict_race(df, model_data, odds_available=False, race_info=None):
    use_features = model_data['features']          # 145列名
    m_lgb = model_data['model']
    X_full = df.reindex(columns=use_features).fillna(0).values  # (N, 145)
    n_lgb = m_lgb.num_feature()                   # booster の特徴量数でスライス
    X = X_full[:, :n_lgb]
    p_lgb = m_lgb.predict(X)
    if 'xgb_model' in model_data and model_data['xgb_model']:
        p_xgb = model_data['xgb_model'].predict(xgb.DMatrix(X))
        w = model_data['ensemble_weights']         # lgb=0.5036, xgb=0.4964
        ai_scores = w['lgb'] * p_lgb + w['xgb'] * p_xgb
    else:
        ai_scores = p_lgb
    df['スコア'] = ai_scores
    return df.sort_values('スコア', ascending=False).reset_index(drop=True)
```

### V15 特徴量 145個 (pkl.gz から実取得、番号=モデル内インデックス順)
```
  1 weight_carry         斤量
  2 age                  馬齢
  3 distance             距離(m)
  4 course_enc           競馬場コード(int)
  5 surface_enc          芝=0,ダ=1
  6 sex_enc              牡=0,牝=1,セ=2
  7 num_horses_val       頭数
  8 horse_num            馬番(1-18)
  9 bracket              枠番(1-8)
 10 jockey_wr_calc       騎手勝率(expanding)
 11 jockey_course_wr_calc 騎手×コース勝率
 12 trainer_top3_calc    調教師複勝率(expanding)
 13 prev_finish          前走着順
 14 prev_last3f          前走上がり3F
 15 prev_pass4           前走通過4角
 16 prev_prize           前走賞金
 17 prev2_finish         2走前着順
 18 prev3_finish         3走前着順
 19 avg_finish_3r        直近3走平均着順
 20 best_finish_3r       直近3走最高着順
 21 finish_trend         着順トレンド
 22 top3_count_3r        直近3走複勝圏回数
 23 avg_last3f_3r        直近3走平均上がり3F
 24 prev2_last3f         2走前上がり3F
 25 dist_change          前走比距離変化(m)
 26 dist_change_abs      距離変化絶対値
 27 rest_days            休養日数
 28 rest_category        休養カテゴリ(0-4)
 29 sire_enc             父馬エンコード
 30 bms_enc              母父エンコード
 31 dist_cat             距離カテゴリ
 32 age_sex              齢×性別
 33 season               季節(0-3)
 34 age_season           齢×季節
 35 horse_num_ratio      馬番/頭数(外枠率)
 36 bracket_pos          枠位置(inner/mid/outer)
 37 carry_diff           斤量差(平均との差)
 38 age_group            齢グループ(2/3/4/5+)
 39 surface_dist_enc     馬場×距離カテゴリ
 40 course_surface       コース×馬場
 41 location_enc         地域エンコード
 42 is_nar               地方フラグ(V15は常に0)
 43 prev_odds_log        前走オッズlog ← V16で除外
 44 training_time_filled 調教タイム(補完済)
 45 has_training         調教データ有無フラグ
 46 training_per_dist    距離あたり調教タイム
 47 jockey_surface_wr    騎手×馬場勝率
 48 horse_career_races   通算出走数
 49 horse_career_wr      通算勝率
 50 horse_career_top3r   通算複勝率
 51 sire_surface_wr      父馬×馬場勝率
 52 sire_dist_wr         父馬×距離勝率
 53 bms_surface_wr       母父×馬場勝率
 54 wood_best_4f_filled  坂路最高4F
 55 has_wood_training    坂路データ有無
 56 prev_race_first3f    前走前半3F
 57 prev_race_last3f     前走後半3F
 58 prev_race_pace_diff  前走ペース差
 59 prev_agari_relative  前走上がり相対値
 60 wood_count_2w        2週間坂路本数
 61 sakaro_best_4f_filled 坂路4F
 62 sakaro_best_3f_filled 坂路3F
 63 has_sakaro_training  坂路データ有無
 64 total_training_count 総調教本数
 65 horse_dist_top3r     距離別複勝率
 66 horse_surface_top3r  馬場別複勝率
 67 frame_course_dist_wr コース×距離×枠勝率
 68 index_max_filled     調教指数最高値
 69 index_run1_filled    最終追い指数
 70 index_avg5_filled    直近5本平均指数
 71 time_1f_last_filled  最終1F
 72 training_intensity_enc 調教強度カテゴリ
 73 sire_shinba_top3r    父馬新馬複勝率
 74 pci                  PACI総合指数
 75 jrdb_idm             IDM指数
 76 jrdb_training_idx    調教指数
 77 jrdb_stable_idx      厩舎指数
 78 jrdb_composite_idx   総合指数
 79 jrdb_upset_idx       波乱指数
 80 jrdb_ten_idx_pred    テン指数予測
 81 jrdb_pace_idx_pred   ペース指数予測
 82 jrdb_agari_idx_pred  上がり指数予測
 83 jrdb_position_idx_pred ポジション指数予測
 84 jrdb_class_code      クラスコード
 85 jrdb_rise_code       上昇度コード
 86 jrdb_heavy_apt       重馬場適性
 87 jrdb_hoof_code       蹄コード
 88 jrdb_ranch_rank      牧場ランク
 89 jrdb_stable_rank     厩舎ランク
 90 jrdb_training_arrow  調教矢印
 91 jrdb_stable_eval     厩舎評価
 92 jrdb_running_style   脚質コード
 93 jrdb_dist_apt        距離適性
 94 jrdb_prev_idm        前走IDM
 95 jrdb_prev_track_bias 前走馬場バイアス
 96 jrdb_prev_interference 前走不利
 97 jrdb_prev_late_start 前走出遅れ
 98 jrdb_prev_pace_idx   前走ペース指数
 99 jrdb_prev_rise_code  前走上昇度
100 jrdb_oikiri_idx      追い切り指数
101 jrdb_ten_time_idx    テンタイム指数
102 jrdb_shimai_time_idx 終い時間指数
103 jrdb_cid_idx         CID指数
104 jrdb_ls_idx          LS指数
105 jrdb_kta_idm         近走平均IDM
106 jrdb_kta_ten_pred    近走テン予測
107 jrdb_kta_agari_pred  近走上がり予測
108 jrdb_ze_idm_avg      全走平均IDM
109 jrdb_ze_ten_avg      全走平均テン
110 jrdb_ze_agari_avg    全走平均上がり
111 jrdb_ze_furi_count   振り返りカウント
112 jrdb_tb_homestr_inner 内ラチ直線バイアス
113 jrdb_dam_rensho_avg  母馬連勝平均
114 jrdb_bms_rensho_avg  母父連勝平均
115 stable_comment_score 厩舎コメントスコア(-3〜+3)
116 oz_tansho_base_log   基準単勝オッズlog ← V16で除外
117 oz_fukusho_base_log  基準複勝オッズlog ← V16で除外
118 oz_base_pop_rank     基準人気順位 ← V16で除外
119 odds_change_rate     当日オッズ変化率 ← V16で除外
120 pop_rank_change      人気順位変化 ← V16で除外
121 odds_sharp_drop      オッズ急落フラグ ← V16で除外
122 weight_ma5           馬体重5走移動平均
123 weight_trend         体重トレンド
124 weight_peak_diff     ピーク比体重差
125 paci_manken_idx      万券指数
126 paci_goal_rank       ゴール順位予測
127 paci_dochu_rank      道中順位予測
128 paci_goal_diff       ゴール差
129 paci_jockey_exp_wr   騎手経験勝率
130 paci_jockey_exp_3rd  騎手経験複勝率
131 paci_ninki_idx       PACI人気指数 ← V16で除外 (gain 16.93%)
132 jockey_horse_rides   騎手×馬 騎乗数
133 jockey_horse_wr      騎手×馬 勝率
134 jockey_horse_top3r   騎手×馬 複勝率
135 jockey_change        騎手交代フラグ
136 jockey_change_to_top 人気騎手への交代フラグ
137 transport_distance_km 輸送距離(km)
138 is_long_transport    長距離輸送フラグ(>100km)
139 course_renovated     コース改修フラグ(京都等)
140 post_renovation_flag 改修後フラグ
141 gaisha_rank          外厩ランク
142 paci_sogo_mark       PACI総合マーク
143 paci_idm_mark        PACI IDMマーク
144 paci_jockey_mark     PACI騎手マーク
145 paci_train_mark      PACI調教マーク
```

---

## V16 候補モデル

**設計思想**: オッズ・人気系 8 features を完全除外した「能力のみモデル」。  
`paci_ninki_idx` の LGB gain が 16.93% → 実態はオッズの proxy。能力だけで同等性能 → 人気乖離馬を正当評価できる。

### 除外した 8 features (V15 #43/#116〜#121/#131)
```python
ODDS_FEATURES_REMOVE = [
    'paci_ninki_idx',       # #131 gain 16.93%
    'odds_change_rate',     # #119
    'odds_sharp_drop',      # #121
    'oz_base_pop_rank',     # #118
    'oz_fukusho_base_log',  # #117
    'oz_tansho_base_log',   # #116
    'pop_rank_change',      # #120
    'prev_odds_log',        # #43
]
# V15 145 - 8 = V16 137 features
```

### pkl.gz 内部構造
```python
{
    'version':           'v16_ability_candidate',
    'model':             LightGBM Booster,
    'xgb_model':         XGBoost Booster,
    'ensemble_weights':  {'lgb': 0.5, 'xgb': 0.5, 'mlp': 0},
    'features':          [137 names],
    'n_features':        137,
    'wf_auc_mean':       0.8677,      # genuine WF 真値
    'removed_features':  [8 names],
    'leak_free':         True,
    'is_live':           False,
    'is_candidate':      True,
    'v15_baseline_wf_auc': 0.8678,
}
```

### WF AUC (年度別)
```
2021: ENS=0.8643  2022: ENS=0.8670  2023: ENS=0.8684
2024: ENS=0.8704  2025: ENS=0.8684  平均: 0.8677
vs V15: delta = -0.0001 (実質同等)
```

### race_auto_notify.py での V16 スコア計算フロー
```python
# tools/race_auto_notify.py
_cached_v16_model = None

def _load_v16_model():
    global _cached_v16_model
    if _cached_v16_model is not None: return _cached_v16_model
    with gzip.open('models/v16_ability_candidate.pkl.gz', 'rb') as f:
        _cached_v16_model = pickle.load(f)
    return _cached_v16_model

def _get_v16_scores(df, v16_data) -> dict:  # 返却: {馬番(int): score(float)}
    feat16 = v16_data['features']           # 137個
    X = df.reindex(columns=feat16).fillna(0).values
    p_lgb = v16_data['model'].predict(X)
    p_xgb = v16_data['xgb_model'].predict(xgb.DMatrix(X))
    w = v16_data['ensemble_weights']        # lgb=0.5, xgb=0.5
    scores = w['lgb']*p_lgb + w['xgb']*p_xgb
    return {int(row['馬番']): float(scores[i]) for i,(_, row) in enumerate(df.iterrows())}

def _save_v16_paper_log(race_id, date_str, df, v16_scores, odds_dict):
    # → data/v16_paper_log/YYYYMMDD.json に v15_top3/v16_top3/scores 追記

# predict_and_notify 関数内での呼び出し順:
# 1. _get_v16_scores → _save_v16_paper_log (V16専用ログ)
# 2. run_paper_shadow_comparison → log_paper_shadow (全CANDIDATE統合ログ)
# 3. build_rich_bet_message(..., v16_scores=...) → Discord #bets に V16 top3 表示
```

---

## V16a 候補モデル

**設計思想**: V16 (137 feats) に「オッズ非依存の能力特徴量」7個を追加。全て expanding window + Bayesian smoothing でリークフリー。

### 追加した 7 features

#### 1. Jockey × Trainer Combo (2 feats)
```python
# train/features_v16a.py
_JT_PRIOR_TOP3, _JT_ALPHA = 0.33, 30   # N=30 で事前分布に収束

df = df.sort_values('date_num')
jt_key = ['jockey_id', 'trainer_id']
df['_jt_cumcnt']  = df.groupby(jt_key).cumcount()               # 当該レース"前"の回数
df['_jt_cumtop3'] = df.groupby(jt_key)['is_top3'].cumsum() - df['is_top3']  # 当該除外

df['jockey_trainer_n_exp'] = df['_jt_cumcnt'].astype(float)
df['jockey_trainer_top3_rate_exp'] = (df['_jt_cumtop3'] + 0.33*30) / (df['_jt_cumcnt'] + 30)
```

#### 2. 馬体重 Slope & Std 直近5走 (2 feats)
```python
lags = {k: df.groupby('horse_id')['horse_weight'].shift(k) for k in range(1,6)}
hw_mat = pd.concat(lags, axis=1).values  # (N,5): lag1=直前, lag5=5走前

for i, row in enumerate(hw_mat):
    valid = ~np.isnan(row)
    if valid.sum() < 2: continue
    x = np.array([5,4,3,2,1])[valid]    # 古い順に大きい値
    y = row[valid]
    slopes[i] = np.cov(x,y)[0,1] / np.var(x)   # 正=体重増加トレンド
    stds[i]   = y.std()

df['horse_weight_slope_5'] = slopes    # fillna=0.0 (中立)
df['horse_weight_std_5']   = stds     # fillna=5.0 (典型的変動)
```

#### 3. 正規化パフォーマンス ELO proxy (1 feat)
```python
_PERF_PRIOR, _PERF_ALPHA = 0.33, 10   # N=10 で事前分布に収束

denom = (df['num_horses'] - 1).clip(lower=1)
df['_norm_finish'] = ((df['num_horses'] - df['finish']) / denom).clip(0,1)
# 意味: 1位=1.0, 最下位=0.0

g = df.groupby('horse_id')
df['_pf_cumcnt'] = g.cumcount()
df['_pf_cumsum'] = g['_norm_finish'].cumsum() - df['_norm_finish']   # 当該除外

df['horse_perf_score_exp'] = (df['_pf_cumsum'] + 0.33*10) / (df['_pf_cumcnt'] + 10)
```

#### 4. 枠番バイアス (2 feats)
```python
_GATE_PRIOR_TOP3, _GATE_ALPHA = 0.33, 50   # N=50 で事前分布に収束

df['_dist_bin'] = (df['distance'] // 400) * 400   # 400m刻みバケット
gate_key = ['course', '_dist_bin', 'surface', 'umaban']

df['_gate_cumcnt']  = df.groupby(gate_key).cumcount()
df['_gate_cumtop3'] = df.groupby(gate_key)['is_top3'].cumsum() - df['is_top3']

df['gate_bias_n_exp']      = df['_gate_cumcnt'].astype(float)
df['gate_bias_top3_exp']   = (df['_gate_cumtop3'] + 0.33*50) / (df['_gate_cumcnt'] + 50)
```

### デフォルト値 (unknown key 時)
```python
V16A_DEFAULTS = {
    'jockey_trainer_top3_rate_exp': 0.33,
    'jockey_trainer_n_exp':         0.0,
    'horse_weight_slope_5':         0.0,
    'horse_weight_std_5':           5.0,
    'horse_perf_score_exp':         0.33,
    'gate_bias_top3_exp':           0.33,
    'gate_bias_n_exp':              0.0,
}
```

### pkl.gz 内部構造
```python
{
    'version':           'v16a_candidate',
    'model':             LightGBM Booster,
    'xgb_model':         XGBoost Booster,
    'ensemble_weights':  {'lgb': 0.5, 'xgb': 0.5, 'mlp': 0},
    'features':          [144 names],    # V16 137 - 8 odds + 7 V16a new = 144
    'n_features':        144,
    'wf_auc_mean':       0.8674,         # genuine WF 真値
    'new_features':      ['jockey_trainer_top3_rate_exp','jockey_trainer_n_exp',
                          'horse_weight_slope_5','horse_weight_std_5',
                          'horse_perf_score_exp','gate_bias_top3_exp','gate_bias_n_exp'],
    'new_feature_defaults': V16A_DEFAULTS,
    'leak_free':         True,
    'is_live':           False,
    'is_candidate':      True,
    'v15_baseline_wf_auc': 0.8678,
    'v16_baseline_wf_auc': 0.8677,
}
```

### WF AUC (年度別)
```
2021: ENS=0.8638  2022: ENS=0.8666  2023: ENS=0.8681
2024: ENS=0.8703  2025: ENS=0.8682  平均: 0.8674
vs V15: -0.0004 / vs V16: -0.0003   verdict: NO-GO (< 0.8677)
```

### lookup tables (推論時の feature augmentation)

**ファイル**: `data/v16a_lookup_tables.pkl` (5.2 MB)
```python
# 構造
tables = {
    'jt':     {(jockey_id_str, trainer_id_str): {'top3_rate': float, 'n': int}},  # 31,197件
    'perf':   {horse_id_str: float},                                               # 58,921件
    'weight': {horse_id_str: {'slope': float, 'std': float, 'last': float}},      # 58,921件
    'gate':   {(course_str, dist_bin_int, surface_str, umaban_int):
               {'top3_rate': float, 'n': int}},                                    # 1,499件
    'built_at': '2026-05-28T11:55:21',
}
```

**推論時 augment** (`tools/features_v16a_lookup.py`):
```python
@lru_cache(maxsize=1)
def _load_tables():
    with open('data/v16a_lookup_tables.pkl', 'rb') as f: return pickle.load(f)

def augment_features_v16a(features_df, race_info: dict):
    """race_info keys: 'course'(str), 'distance'(int), 'surface'(str)
    features_df の各行に horse_id / jockey_id / trainer_id / umaban が必要"""
    tables = _load_tables()
    dist_bin = (int(race_info.get('distance', 0)) // 400) * 400
    rows = []
    for _, row in features_df.iterrows():
        jt = tables['jt'].get((str(row.get('jockey_id','')),
                               str(row.get('trainer_id',''))), {})
        wt = tables['weight'].get(str(row.get('horse_id','')), {})
        gt = tables['gate'].get((race_info.get('course',''), dist_bin,
                                 race_info.get('surface',''),
                                 int(row.get('umaban', row.get('馬番',0)))), {})
        rows.append({
            'jockey_trainer_top3_rate_exp': jt.get('top3_rate', 0.33),
            'jockey_trainer_n_exp':         float(jt.get('n', 0)),
            'horse_weight_slope_5':         wt.get('slope', 0.0),
            'horse_weight_std_5':           wt.get('std', 5.0),
            'horse_perf_score_exp':         tables['perf'].get(str(row.get('horse_id','')), 0.33),
            'gate_bias_top3_exp':           gt.get('top3_rate', 0.33),
            'gate_bias_n_exp':              float(gt.get('n', 0)),
        })
    aug = pd.DataFrame(rows, index=features_df.index)
    for col in aug.columns: features_df[col] = aug[col]
    return features_df

def invalidate_cache(): _load_tables.cache_clear()
```

---

## Paper Shadow システム

### CANDIDATE_MODELS (`tools/paper_shadow_v15_full.py`)
```python
CANDIDATE_MODELS = {
    'v15_full_optuna': 'models/v15_full_optuna_candidate.pkl.gz',
    'v15_2':           'models/v15_2_candidate.pkl.gz',
    # v22_top100: DISABLED — track_lap POST-RACE leak 確認済み
    'v20_base':        'keiba_model_v20_base_central.pkl.gz',
    'v16_ability':     'models/v16_ability_candidate.pkl.gz',
    'v16a_ability':    'models/v16a_candidate.pkl.gz',
}
```

### 推論・比較・ログの流れ
```python
def predict_paper_shadow(features_df, model_key, race_info=None):
    model = load_candidate_model(model_key)
    if model_key == 'v16a_ability':
        df_aug = _augment_v16a(features_df.copy(), race_info or {})
    else:
        df_aug = features_df
    df_in = df_aug.reindex(columns=model['features']).fillna(0.0)
    X = df_in.values.astype(np.float32)
    p_lgb = model['model'].predict(X)
    w = model.get('ensemble_weights', {'lgb':0.5,'xgb':0.5})
    p_xgb = model['xgb_model'].predict(xgb.DMatrix(X, feature_names=model['features']))
    scores = w['lgb']*p_lgb + w['xgb']*p_xgb
    return {i: float(s) for i,s in enumerate(scores)}

def run_paper_shadow_comparison(race_id, features_df, v15_predictions, race_info=None):
    # v15_predictions: [{'horse_num': int, 'score': float}, ...]
    # 全 CANDIDATE_MODELS に predict_paper_shadow を呼び比較
    # 返却: {'race_id':..., 'v15_top3':[3,7,12], 'paper_shadows':{model_key:{top3,agree}}}

def log_paper_shadow(result):
    # → data/paper_shadow_log/shadow_YYYYMMDD.jsonl に追記

def get_paper_shadow_stats(days=7):
    # → {model_key: {'n_agree': int, 'n_total': int}}
```

### ログ形式 (`data/paper_shadow_log/shadow_YYYYMMDD.jsonl`)
```json
{"race_id":"202605281201","timestamp":"2026-05-28T09:15:00",
 "v15_top3":[3,7,12],
 "paper_shadows":{
   "v16_ability":  {"top3":[3,7,11],"agree_with_v15":false},
   "v16a_ability": {"top3":[3,7,12],"agree_with_v15":true}}}
```

### 呼び出しタイミング
| スクリプト | タイミング | 内容 |
|-----------|-----------|------|
| `tools/daily_predict.py` | 08:00 各レース | paper shadow → JSONL 保存。全完了後に stats → Discord #updates |
| `tools/race_auto_notify.py` | 発走5分前 各レース | V16 paper log + paper shadow JSONL 保存 |

---

## 採用基準

### AUC ゲート (WF 検証)
```
V16a WF AUC >= 0.8677 (V16 baseline) → GO (本番候補へ)
V16a WF AUC <  0.8677               → NO-GO (現在: 0.8674 → NO-GO)
```

### LIVE 採用基準 (AUC GO 通過後、N≥30 蓄積後)
1. **人気乖離 ROI > V15** — V15 と異なる予想馬での配当効率
2. **TOP1 連対率 ≥ 40%** — 1着予想馬の2着内率

両方 PASS → 週末のみ・上限 5,000円/日 から段階投入

### 現在の判定
| モデル | AUC ゲート | 状態 |
|--------|-----------|------|
| V16 | PASS (0.8677) | paper shadow 継続、LIVE 検証中 |
| V16a | FAIL (0.8674) | paper shadow のみ。6/15+ に再評価予定 |

---

## 学習スクリプト・パラメータ

### LGB/XGB 共通 (V15/V16/V16a 全て同じ)
```python
LGB_PARAMS = {
    'objective':'binary','metric':'auc','boosting_type':'gbdt',
    'num_leaves':63,'learning_rate':0.05,
    'feature_fraction':0.8,'bagging_fraction':0.8,'bagging_freq':5,
    'min_child_samples':50,'reg_alpha':0.1,'reg_lambda':0.1,
    'verbose':-1,'seed':42,
}  # early_stopping=50, max=1000

XGB_PARAMS = {
    'objective':'binary:logistic','eval_metric':'auc',
    'max_depth':6,'learning_rate':0.05,
    'subsample':0.8,'colsample_bytree':0.8,
    'min_child_weight':50,'reg_alpha':0.1,'reg_lambda':0.1,
    'seed':42,'tree_method':'hist',
}  # early_stopping=50, max=1000
```

### WF ループ構造
```python
WF_YEARS = range(2021, 2026)  # 5-fold
# 学習データ: data/_v15_optuna_df_cache.pkl.gz (104MB)
#   → {'df': DataFrame(527,280 rows, 232 cols), 'features': [145 names]}
#   → df['year'] は 2桁 (21=2021 … 25=2025)
#   → df['target'] = (finish <= 3).astype(int)

for test_year in WF_YEARS:
    ty = test_year - 2000
    train_mask = df['year'] < ty
    test_mask  = df['year'] == ty
    # LGB + XGB それぞれ early stopping で学習
    # auc_ens = roc_auc_score(y_te, 0.5*p_lgb + 0.5*p_xgb)
```

### 各学習スクリプト
```
train/train_v15_master.py   → keiba_model_v15_central.pkl.gz
train/train_v16_ability.py  → models/v16_ability_candidate.pkl.gz
train/train_v16a_ability.py → models/v16a_candidate.pkl.gz
                               + data/v16a_lookup_tables.pkl (lookup table も同時生成)
```

---

## リークフリー設計原則

```python
# Pattern A で除外 (8個) — V15 / V16 / V16a 全て適用
LEAK_FEATURES_A = [
    'odds_log','horse_weight','condition_enc',
    'weight_change','weight_change_abs','weight_cat','weight_cat_dist','cond_surface',
]

# SKB POST-RACE LEAK (10個) — V20 以降で追加除外。V15/V16/V16a には未含
SKB_LEAK_FEATURES = [
    'skb_kishi_code_1','skb_kishi_code_2','skb_kishi_code_3',
    'skb_baba_code_1','skb_baba_code_2','skb_baba_code_3',
    'skb_kyaku_code_1','skb_kyaku_code_2','skb_kyaku_code_3','skb_turf_hoof',
]

# V16a expanding window の必須ルール:
# 全統計特徴量 = cumsum - current (当該行を除外した累積)
# Bayesian smoothing で低サンプル時の過学習を抑制
```

---

## 主要ファイルパス

```
C:\Users\takum\keiba-ai\
├── keiba_model_v15_central.pkl.gz         # V15 本番
├── models/
│   ├── v16_ability_candidate.pkl.gz       # V16 候補(137 feats)
│   ├── v16a_candidate.pkl.gz              # V16a 候補(144 feats)
│   ├── v15_full_optuna_candidate.pkl.gz
│   ├── v15_2_candidate.pkl.gz
│   └── keiba_model_v20_base_central.pkl.gz
├── data/
│   ├── v16a_lookup_tables.pkl             # V16a 推論用 lookup(5.2MB)
│   ├── v16_paper_log/YYYYMMDD.json        # V16 詳細 paper log
│   ├── paper_shadow_log/shadow_YYYYMMDD.jsonl  # 全CANDIDATE 比較ログ
│   ├── v20/v16a_wf_results.json           # V16a WF 結果
│   ├── _v15_optuna_df_cache.pkl.gz        # 学習キャッシュ(104MB)
│   └── cumulative_results.csv             # 累計予測・収支ログ
├── tools/
│   ├── predict_core.py                    # 特徴量生成・推論エンジン
│   ├── paper_shadow_v15_full.py           # paper shadow 管理
│   ├── features_v16a_lookup.py            # V16a 推論時 augment
│   ├── race_auto_notify.py                # 発走5分前通知(V16+V16a paper)
│   ├── daily_predict.py                   # 08:00 全レース予測
│   ├── daily_results.py                   # 18:00 結果照合
│   └── build_allscores_html.py            # 全馬スコアHTML生成
├── train/
│   ├── features_v16a.py                   # V16a 特徴量計算(学習時)
│   ├── train_v16_ability.py               # V16 学習
│   └── train_v16a_ability.py              # V16a 学習
└── docs/
    ├── NEW_CHAT_CONTEXT.md                # このファイル
    └── V15_V16_V16a_COMPLETE_REFERENCE.md # 詳細リファレンス
```

---

## 自動実行スケジュール

| 時間 | スクリプト | 内容 |
|------|-----------|------|
| 毎日 03:00 | `daily_premium_scrape.py` | プレミアムデータ事前取得 |
| 毎日 08:00 | `daily_predict.py` | 全レース予測 + paper shadow + Discord通知 |
| 土日 08:45 | `race_auto_notify.py` | 発走5分前 V15予測 + V16/V16a paper shadow |
| 毎日 20:00 | `daily_results.py` | 結果照合・ROI計算 + HTML再送 |
| 月曜 08:00 | `weekly_report.py` | 週次レポート |

---

## 今後のロードマップ

| 期限 | 作業 | 基準 |
|------|------|------|
| N=30達成後 | V16 LIVE 採用判定 | 人気乖離ROI>V15 & TOP1連対率≥40% |
| 6/15+ | V16a 再評価 or V17 設計 | AUC NO-GO なら新特徴量探索 |
| 7/1 | V20 投入候補 | WF AUC≥0.880, JV-Link データ活用 |
| 8/1 | V15 archive 判定 | V20 並行1ヶ月後 |

---

*2026-05-28 Session #92 時点。コードとの乖離確認は `python -c "import gzip,pickle; m=pickle.load(gzip.open('models/v16a_candidate.pkl.gz','rb')); print(m['wf_auc_mean'], m['n_features'])"` で即確認可能。*
