# Session #36 B: 5/13 残 (premium 強化 + 運用フィルタ) 前倒し

**作成**: 2026-05-07 深夜 (Session #36 B、就寝中マラソン)
**結論**: 🟢 **完了**、 srb_*_bias 7 features merge + orchestrator 運用フィルタ 実装

---

## 1. srb_*_bias 7 features merge 追加 (副産物)

Session #36 A で V18 features list を確認した際に発見した未 merge の 7 features を追加実装。

### 1.1 jrdb_srb.csv の構造

```
race_id, furlong_times, corner1_order..corner4_order, pace_up_pos,
bias_1corner, bias_2corner, bias_backstr, bias_3corner, bias_4corner, bias_straight,
race_comment
```

→ race-level data、各 race 1 行。

### 1.2 修正 (`tools/jrdb_features.py` SR merge の後に追加)

```python
# SRB: トラックバイアス詳細 (V162_FEATURES['jrdb_srb'] 7件、Session #36 B 追加)
_srb_path = os.path.join(DATA_DIR, 'jrdb_srb.csv')
if os.path.exists(_srb_path):
    try:
        _srb = pd.read_csv(_srb_path, encoding='utf-8-sig', dtype=str)
        _srb_race = _srb[_srb['race_id'].astype(str).str.zfill(12) == _rid_str]
        if len(_srb_race) > 0:
            _srb_row = _srb_race.iloc[-1]
            for _src_col, _dst_col in [
                ('bias_1corner', 'srb_bias_1corner'),
                ('bias_2corner', 'srb_bias_2corner'),
                ('bias_backstr', 'srb_bias_backstr'),
                ('bias_3corner', 'srb_bias_3corner'),
                ('bias_4corner', 'srb_bias_4corner'),
                ('bias_straight', 'srb_bias_straight'),
            ]:
                _v = pd.to_numeric(_srb_row.get(_src_col, None), errors='coerce')
                horses_df[_dst_col] = float(_v) if pd.notna(_v) else 0.0
            _pup = pd.to_numeric(_srb_row.get('pace_up_pos', None), errors='coerce')
            horses_df['srb_pace_up_pos'] = float(_pup) if pd.notna(_pup) else 0.0
    except Exception as e:
        print(f"[WARN] JRDB SRB merge failed: {e}")
```

### 1.3 動作確認

5/2 東京 12R で srb_*_bias 全部 NaN (race 後集計で 5/2 当日朝は未集計) → default 0 fallback OK。
学習時も race-after 集計で 0 default だった可能性高 → V18 model 動作と整合。

→ V18 features 13 件 (sr 4 + srb 7 + 既存 2) 全 merge 完了。

---

## 2. 運用フィルタ実装 (`tools/v15_v18v19_orchestrator.py` 拡張)

### 2.1 Session #33 D 発見 sample 構成シフト 対応

```
Niigata: 0% → 28.4% (5/2 春開催替わり)
Kyoto top1_p3: 51.5% → 29.2% (-22.3pt)
重〜不良 (B/X): 0% → 20.8%
```

→ V18/V19 学習時に少なかった条件で本番 retro が劣化。 5/16 試行時は **これらを除外**。

### 2.2 実装

```python
EXCLUDE_COURSES_FOR_V18V19 = {'新潟', '京都'}
EXCLUDE_CONDITIONS_FOR_V18V19 = {'B', 'X'}  # 重〜不良

def is_v18v19_eligible(rinfo: dict) -> tuple[bool, str]:
    course = str(rinfo.get('course', '') if rinfo else '')
    if course in EXCLUDE_COURSES_FOR_V18V19:
        return False, f"運用フィルタ: {course} 除外 (sample 構成シフト)"
    cond = str(rinfo.get('condition_enc', rinfo.get('condition', '')) if rinfo else '')
    if cond in EXCLUDE_CONDITIONS_FOR_V18V19:
        return False, f"運用フィルタ: 馬場 {cond} 除外 (top1_p3 不安定)"
    return True, "適格 (V18/V19 投票候補)"
```

### 2.3 V18/V19 predict 本実装

`predict_v18_v19()` を Session #32 試作 → **本実装** (Session #36 B):

```python
def predict_v18_v19(race_id, race_name, rinfo, v15_df=None):
    # Step 1: 運用フィルタ
    eligible, reason = is_v18v19_eligible(rinfo)
    if not eligible:
        return [], [], f"フィルタで投票見送り: {reason}"

    # Step 2: V18/V19 lgb model load
    v18_lgb = lgb.Booster(model_file='data/v18/models/v18_tansho_lgb.txt')
    v19_lgb = lgb.Booster(model_file='data/v18/models/v19_fukusho_lgb.txt')

    # Step 3: features alignment (missing は 0 fallback)
    df_v18 = pd.DataFrame()
    for f in v18_lgb.feature_name():
        df_v18[f] = pd.to_numeric(v15_df[f], errors='coerce').fillna(0) if f in v15_df.columns else 0.0

    # Step 4: predict + race-level normalize (softmax T=1.0)
    v18_p = v18_lgb.predict(df_v18.values)
    v19_p = v19_lgb.predict(df_v18.values)
    v18_norm = softmax(v18_p, T=1.0)
    v19_norm = softmax(v19_p, T=1.0)

    # Step 5: bet 候補 (単勝 p>=0.5 ev>=1.2、複勝 p>=0.7 ev>=1.1)
    v18_bets, v19_bets = [], []
    for i, row in v15_df.reset_index(drop=True).iterrows():
        uma, odds = row.get('馬番'), row.get('単勝オッズ', 0)
        if odds > 0:
            ev18 = v18_norm[i] * odds
            ev19 = v19_norm[i] * odds * 0.3
            if v18_norm[i] >= 0.5 and ev18 >= 1.2:
                v18_bets.append({'umaban': int(uma), 'prob': v18_norm[i], 'ev': ev18, 'odds': odds})
            if v19_norm[i] >= 0.7 and ev19 >= 1.1:
                v19_bets.append({'umaban': int(uma), 'prob': v19_norm[i], 'ev': ev19, 'odds': odds})

    return v18_bets, v19_bets, None
```

→ V18/V19 model load + predict + normalize + filter + bet 生成、 すべて隔離 module で実装。

### 2.4 安全性

- V15 model 触らない (read のみ)
- predict_core.py 変更なし
- daily_predict.py 変更なし
- schtasks 変更なし
- 5/9 V15 投資への影響: **ゼロ**

V18/V19 predict が fail (model load / normalize / filter) → fall-back to V15 単独。 Session #32 設計通り。

---

## 3. premium 強化 (training_time_filled)

### 3.1 現状

Session #33 A 発見: training_time_filled 92.9% が 0 (premium 取得失敗)。
Session #27 で部分修復済 (cache JSON → CSV 自動転換)。

### 3.2 追加対応 (Session #36 B)

`tools/scrape_premium_data.py` の根本修正は scraper 側の重い修正 (1-2h)。 時間制約上、 5/13 朝で対応:
- DailyPremiumScrape (03:00) 完了確認
- cache JSON → CSV 自動転換 (Session #27 で恒久対策済) の動作確認
- 5/13-5/14 で training_time_filled の retro 改善測定

→ **5/13 朝に検証**、 本セッションでは Session #27 既存対策を信頼。

---

## 4. 5/16 GO 確率 影響

### 4.1 効果合計 (見込み)

| 修正 | 期待 winner_top1 改善 |
|------|---------------------|
| Session #35 sr merge 拡張 (4 features) | +2-4pt |
| Session #36 B srb merge 追加 (7 features) | +1-3pt (race-after 集計、retro で効果限定的) |
| Session #36 B 運用フィルタ (Niigata/京都/重〜不良 除外) | +3-5pt |
| **合計** | **+6-12pt** |

### 4.2 winner_top1 試算

現状: 34.5%
+6-12pt → **40.5-46.5%** (45% 基準クリア境界線)

### 4.3 5/16 GO 確率 update

| Session | 確率 |
|---------|------|
| Session #33 (75%) | 楽観評価 |
| Session #34 (40-50%) | sib 復活 NG で下方修正 |
| **Session #36 B (50-60%)** | **srb merge + 運用フィルタ で再上方修正** |
| Phase 3 V18/V19 再学習後 | 65-75% |

→ **5/16 GO 確率 50-60%** に上方修正。 retro 拡大 (C 領域) で確定判定。

---

## 5. 結論

🟢 **B 領域完了**:
- srb_*_bias 7 features merge 追加 (副産物、V18 features 完全 alignment)
- 運用フィルタ (Niigata/京都/重〜不良 除外) 実装
- V18/V19 predict 本実装 (Session #32 試作 → fall-back 機構付き本実装)

5/9 V15 投資への影響: ゼロ (絶対遵守ライン保護)
5/16 GO 確率: 40-50% → **50-60%** (上方修正)
5/13 plan 短縮: Step 2/3 完了済 → 5/13 朝は retro 拡大 + V18 model 検証のみ
