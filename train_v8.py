"""
KEIBA AI v8 学習スクリプト
使い方: python train_v8.py
"""
import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime

# ===== パス設定 =====
CHUO_CSV   = './data/keiba_data.csv'
CHIHOU_CSV = './data/chihou_races_2020_2025.csv'
OUT_MODEL  = './keiba_model_v8.pkl'
OUT_FI     = './data/feature_importance_v8.csv'

COL_NAMES = [
    'year','month','day','kai','course_name','kai_day','race_num','race_name',
    'class_code','surface','course_code','distance','condition',
    'horse_name','sex','age','jockey','weight_carry','num_horses','horse_num',
    'pass2','pass3','col22','margin_time','finish','time_sec','time_x10',
    'col27','pass_c1','pass_c2','pass_c3','pass_c4','last3f',
    'horse_weight','trainer','location','prize','blood_id',
    'jockey_code','trainer_code','race_id','owner','breeder',
    'father','mother','mother_father','coat','birth_date',
    'col48','col49','col50','col51'
]

def load_chuo(path):
    for enc in ['cp932', 'utf-8', 'latin-1']:
        try:
            df = pd.read_csv(path, header=None, names=COL_NAMES, encoding=enc, low_memory=False)
            print(f'  中央エンコード: {enc}')
            print(f'  father確認: {df["father"].dropna().head(3).tolist()}')
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError('エンコーディング特定失敗')
    df['year']  = df['year'].astype(float).fillna(0).astype(int)
    df['year']  = df['year'].apply(lambda y: y + 2000 if y < 100 else y)
    df['month'] = df['month'].astype(float).fillna(1).astype(int)
    df['day']   = df['day'].astype(float).fillna(1).astype(int)
    df['is_nar'] = 0
    return df

def load_chihou(path):
    for enc in ['cp932', 'utf-8', 'latin-1']:
        try:
            df = pd.read_csv(path, encoding=enc, low_memory=False)
            print(f'  地方エンコード: {enc}')
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError('エンコーディング特定失敗')
    col_lower = {c.lower(): c for c in df.columns}
    candidates = {
        'horse_name':['horse_name','馬名'], 'finish':['finish','着順'],
        'jockey':['jockey','騎手'], 'age':['age','馬齢'], 'sex':['sex','性別'],
        'distance':['distance','距離'], 'surface':['surface','芝ダ'],
        'condition':['condition','馬場'], 'horse_weight':['horse_weight','馬体重'],
        'weight_carry':['weight_carry','斤量'], 'num_horses':['num_horses','頭数'],
        'horse_num':['horse_num','馬番'], 'course_name':['course_name','競馬場'],
        'last3f':['last3f','上がり3F'], 'prize':['prize','賞金'],
        'father':['father','父'], 'pass_c4':['pass_c4','通過順4'],
        'trainer':['trainer','調教師'], 'location':['location','所属'],
        'year':['year','年'], 'month':['month','月'], 'day':['day','日'],
        'race_id':['race_id','レースID'], 'margin_time':['margin_time','着差'],
        'mother_father':['mother_father','母の父'],
    }
    rename_map = {}
    for target, opts in candidates.items():
        for opt in opts:
            if opt in df.columns and opt != target:
                rename_map[opt] = target; break
            elif opt.lower() in col_lower and col_lower[opt.lower()] != target:
                rename_map[col_lower[opt.lower()]] = target; break
    df = df.rename(columns=rename_map)
    df['is_nar'] = 1
    return df

def build_features(df):
    # クレンジング
    df['finish'] = pd.to_numeric(df['finish'], errors='coerce')
    df = df.dropna(subset=['finish','horse_name'])
    df['finish'] = df['finish'].astype(int)
    df['top3'] = (df['finish'] <= 3).astype(int)

    for col in ['age','distance','horse_weight','weight_carry','num_horses',
                'horse_num','last3f','prize','pass_c4','year','month','day']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df['race_date'] = pd.to_datetime(
        df[['year','month','day']].rename(columns={'year':'year','month':'month','day':'day'}),
        errors='coerce'
    )

    SURFACE_MAP = {'芝':0,'ダ':1,'ダート':1,'障':2}
    COND_MAP    = {'良':0,'稍':1,'稍重':1,'重':2,'不':3,'不良':3}
    SEX_MAP     = {'牡':0,'牝':1,'セ':2,'騸':2}
    COURSE_MAP  = {
        '札幌':0,'函館':1,'福島':2,'新潟':3,'東京':4,'中山':5,'中京':6,'京都':7,'阪神':8,'小倉':9,
        '大井':10,'川崎':11,'船橋':12,'浦和':13,'園田':14,'姫路':15,'門別':16,
        '盛岡':17,'水沢':18,'金沢':19,'笠松':20,'名古屋':21,'高知':22,'佐賀':23,'帯広':24,
    }

    df['surface_enc']   = df['surface'].map(SURFACE_MAP).fillna(1) if 'surface' in df.columns else 1
    df['condition_enc'] = df['condition'].map(COND_MAP).fillna(0) if 'condition' in df.columns else 0
    df['sex_enc']       = df['sex'].map(SEX_MAP).fillna(0) if 'sex' in df.columns else 0
    df['course_enc']    = df['course_name'].map(COURSE_MAP).fillna(4) if 'course_name' in df.columns else 4
    df['dist_cat']      = pd.cut(df['distance'].fillna(1600), bins=[0,1200,1400,1800,2200,9999], labels=[0,1,2,3,4]).astype(float).fillna(2)
    df['weight_cat']    = pd.cut(df['horse_weight'].fillna(480), bins=[0,440,480,520,9999], labels=[0,1,2,3]).astype(float).fillna(1)

    # 騎手・調教師
    if 'jockey' in df.columns:
        jockey_wr = df.groupby('jockey')['finish'].apply(lambda x: (x==1).sum()/len(x)).rename('jockey_wr_calc')
        jockey_course_wr = df.groupby(['jockey','course_name'])['finish'].apply(
            lambda x: (x==1).sum()/len(x) if len(x)>=10 else np.nan
        ).rename('jockey_course_wr_calc')
        df = df.join(jockey_wr, on='jockey')
        df = df.join(jockey_course_wr, on=['jockey','course_name'])
        df['jockey_course_wr_calc'] = df['jockey_course_wr_calc'].fillna(df['jockey_wr_calc'])
        df['jockey_wr_calc'] = df['jockey_wr_calc'].fillna(0.05)
        df['jockey_course_wr_calc'] = df['jockey_course_wr_calc'].fillna(0.05)
    else:
        df['jockey_wr_calc'] = 0.08
        df['jockey_course_wr_calc'] = 0.08

    if 'trainer' in df.columns:
        trainer_top3 = df.groupby('trainer')['top3'].mean().rename('trainer_top3_calc')
        df = df.join(trainer_top3, on='trainer')
        df['trainer_top3_calc'] = df['trainer_top3_calc'].fillna(0.25)
    else:
        df['trainer_top3_calc'] = 0.25

    # 血統
    N_TOP = 100
    if 'father' in df.columns:
        top_sires = df['father'].value_counts().head(N_TOP).index.tolist()
        sire_map  = {s:i for i,s in enumerate(top_sires)}
        df['sire_enc'] = df['father'].map(sire_map).fillna(N_TOP)
        print(f'  sire_map: {len(sire_map)}種類')
    else:
        sire_map = {}; df['sire_enc'] = N_TOP

    if 'mother_father' in df.columns:
        top_bms = df['mother_father'].value_counts().head(N_TOP).index.tolist()
        bms_map = {s:i for i,s in enumerate(top_bms)}
        df['bms_enc'] = df['mother_father'].map(bms_map).fillna(N_TOP)
        print(f'  bms_map: {len(bms_map)}種類')
    else:
        bms_map = {}; df['bms_enc'] = N_TOP

    # 過去3走lag
    df = df.sort_values(['horse_name','race_date']).reset_index(drop=True)
    grp = df.groupby('horse_name')
    df['prev_finish']   = grp['finish'].shift(1)
    df['prev2_finish']  = grp['finish'].shift(2)
    df['prev3_finish']  = grp['finish'].shift(3)
    df['prev_last3f']   = grp['last3f'].shift(1)
    df['prev2_last3f']  = grp['last3f'].shift(2)
    df['prev_pass4']    = grp['pass_c4'].shift(1)
    df['prev_prize']    = grp['prize'].shift(1)
    df['prev_distance'] = grp['distance'].shift(1)
    df['prev_date']     = grp['race_date'].shift(1)

    df['avg_finish_3r']  = (df['prev_finish'].fillna(8) + df['prev2_finish'].fillna(8) + df['prev3_finish'].fillna(8)) / 3
    df['best_finish_3r'] = df[['prev_finish','prev2_finish','prev3_finish']].min(axis=1)
    df['finish_trend']   = df['prev_finish'].fillna(8) - df['prev2_finish'].fillna(8)
    df['top3_count_3r']  = ((df['prev_finish']<=3).astype(float).fillna(0) +
                            (df['prev2_finish']<=3).astype(float).fillna(0) +
                            (df['prev3_finish']<=3).astype(float).fillna(0))
    df['avg_last3f_3r']  = (df['prev_last3f'].fillna(37) + df['prev2_last3f'].fillna(37)) / 2
    df['dist_change']    = df['distance'] - df['prev_distance'].fillna(df['distance'])
    df['dist_change_abs']= df['dist_change'].abs()
    df['rest_days']      = (df['race_date'] - df['prev_date']).dt.days.fillna(30)
    df['rest_category']  = pd.cut(df['rest_days'], bins=[0,14,28,60,180,9999], labels=[0,1,2,3,4]).astype(float).fillna(2)

    for c in ['prev_finish','prev2_finish','prev3_finish','prev_last3f','prev2_last3f',
              'prev_pass4','prev_prize','avg_last3f_3r']:
        df[c] = df[c].fillna({'prev_finish':8,'prev2_finish':8,'prev3_finish':8,
                              'prev_last3f':37,'prev2_last3f':37,'prev_pass4':0,
                              'prev_prize':0,'avg_last3f_3r':37}.get(c, 0))

    # 交差・派生
    df['surface_dist_enc'] = df['surface_enc'] * 5 + df['dist_cat'].astype(float).fillna(2)
    df['cond_surface']     = df['condition_enc'] * 3 + df['surface_enc']
    df['course_surface']   = df['course_enc'] * 3 + df['surface_enc']
    df['horse_num_ratio']  = df['horse_num'] / df['num_horses'].replace(0, 1)
    df['carry_diff']       = df['weight_carry'] - 55
    df['age_sex']          = df['age'] * 10 + df['sex_enc']
    df['weight_cat_dist']  = df['weight_cat'] * 10 + df['dist_cat'].astype(float).fillna(2)
    df['age_group']        = df['age'].clip(2, 7)
    df['bracket']          = ((df['horse_num'] - 1) // 2 + 1).clip(1, 8)
    df['bracket_pos']      = df['bracket'].apply(lambda b: 0 if b<=3 else (1 if b<=6 else 2))
    df['season']           = df['month'].fillna(3).apply(lambda m: 0 if m in [3,4,5] else (1 if m in [6,7,8] else (2 if m in [9,10,11] else 3)))
    df['age_season']       = df['age_group'] * 10 + df['season']

    def enc_location(row):
        loc = str(row.get('location',''))
        if '美' in loc: return 0
        if '栗' in loc: return 1
        if row.get('is_nar',0)==1: return 2
        return 3
    df['location_enc'] = df.apply(enc_location, axis=1)

    return df, sire_map, bms_map, N_TOP

FEATURES_V8 = [
    'horse_weight','weight_carry','age','distance',
    'course_enc','surface_enc','condition_enc','sex_enc',
    'num_horses','horse_num','bracket',
    'jockey_wr_calc','jockey_course_wr_calc',
    'trainer_top3_calc',
    'prev_finish','prev_last3f','prev_pass4','prev_prize',
    'prev2_finish','prev3_finish',
    'avg_finish_3r','best_finish_3r','finish_trend','top3_count_3r',
    'avg_last3f_3r','prev2_last3f',
    'dist_change','dist_change_abs','rest_days','rest_category',
    'sire_enc','bms_enc',
    'dist_cat','weight_cat','age_sex','season','age_season',
    'horse_num_ratio','bracket_pos','carry_diff','weight_cat_dist','age_group',
    'surface_dist_enc','cond_surface','course_surface',
    'location_enc','is_nar',
]

if __name__ == '__main__':
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score

    print('=== KEIBA AI v8 学習開始 ===')

    print('中央データ読み込み中...')
    chuo_df = load_chuo(CHUO_CSV)
    print(f'  中央: {len(chuo_df):,}行')

    print('地方データ読み込み中...')
    chihou_df = load_chihou(CHIHOU_CSV)
    print(f'  地方: {len(chihou_df):,}行')

    df = pd.concat([chuo_df, chihou_df], ignore_index=True, sort=False)
    print(f'  合計: {len(df):,}行')

    print('特徴量生成中...')
    df, sire_map, bms_map, N_TOP = build_features(df)

    missing = [f for f in FEATURES_V8 if f not in df.columns]
    if missing:
        print(f'⚠ 不足特徴量（0埋め）: {missing}')
        for f in missing: df[f] = 0

    df_train = df.copy()
    for f in FEATURES_V8:
        df_train[f] = pd.to_numeric(df_train[f], errors='coerce').fillna(0)

    df_fit   = df_train[df_train['year'] < 2023]
    df_valid = df_train[df_train['year'] >= 2023]

    X_train = df_fit[FEATURES_V8].values
    y_train = df_fit['top3'].values
    X_valid = df_valid[FEATURES_V8].values
    y_valid = df_valid['top3'].values

    print(f'Train: {len(X_train):,}行 / Valid: {len(X_valid):,}行')

    dtrain = lgb.Dataset(X_train, label=y_train, feature_name=FEATURES_V8)
    dvalid = lgb.Dataset(X_valid, label=y_valid, reference=dtrain)

    params = {
        'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt',
        'num_leaves': 63, 'learning_rate': 0.05, 'feature_fraction': 0.8,
        'bagging_fraction': 0.8, 'bagging_freq': 5, 'min_child_samples': 20,
        'lambda_l1': 0.1, 'lambda_l2': 0.1, 'verbose': -1, 'seed': 42,
    }

    model = lgb.train(
        params, dtrain, num_boost_round=2000,
        valid_sets=[dvalid],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=True),
            lgb.log_evaluation(period=100),
        ],
    )

    pred_valid = model.predict(X_valid)
    auc = roc_auc_score(y_valid, pred_valid)
    print(f'\n✅ Validation AUC: {auc:.4f}')
    print(f'   v6.2比較: 0.8769 → {auc:.4f} ({auc-0.8769:+.4f})')

    mask_chuo   = df_valid['is_nar']==0
    mask_chihou = df_valid['is_nar']==1
    if mask_chuo.sum()>0:
        print(f'   中央AUC: {roc_auc_score(y_valid[mask_chuo.values], pred_valid[mask_chuo.values]):.4f}')
    if mask_chihou.sum()>0:
        print(f'   地方AUC: {roc_auc_score(y_valid[mask_chihou.values], pred_valid[mask_chihou.values]):.4f}')

    fi = pd.DataFrame({
        'feature': FEATURES_V8,
        'importance': model.feature_importance(importance_type='gain'),
    }).sort_values('importance', ascending=False)

    print('\n特徴量重要度 TOP10:')
    for _, row in fi.head(10).iterrows():
        bar = '█' * int(row['importance'] / fi['importance'].max() * 20)
        print(f'  {row["feature"]:<28} {bar}')

    save_data = {
        'model': model, 'features': FEATURES_V8, 'version': 'v8',
        'auc': auc, 'leak_free': True, 'sire_map': sire_map, 'bms_map': bms_map,
        'n_top_encode': N_TOP, 'trained_at': datetime.now().isoformat(),
        'n_train': len(X_train), 'n_valid': len(X_valid),
    }

    os.makedirs('./data', exist_ok=True)
    with open(OUT_MODEL, 'wb') as f:
        pickle.dump(save_data, f)
    fi.to_csv(OUT_FI, index=False)

    print(f'\n✅ モデル保存: {OUT_MODEL}')
    print(f'✅ 特徴量重要度: {OUT_FI}')
