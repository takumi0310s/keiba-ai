# V21 status (5/12 朝、 horse_id mapper 発見 + V21 学習 完了)

## 🎯 結論

**V21 < V15**。 V15 production 継続、 V21 は shadow eval / 将来開発用。

| model | WF AUC | 状態 |
|-------|--------|------|
| V15 (production) | **0.8939** | 不変、 継続運用 |
| V20 (旧候補) | 0.8376 (LGB+XGB) | 不採用 |
| **V21 (本 session)** | **0.7918** (mean, 3 folds) | **shadow eval、 production 投入なし** |

## ✅ 本 session 成果

### 1. horse_id mapper 発見 (tools/horse_id_mapper.py)
- V20 (TFJV) 8桁 `19102173` ↔ V21 (netkeiba) 10桁 `2019102173`
- rule: `tfjv_to_netkeiba = '20' + zfill(8)`
- verified: 22/29 paddock horses match (残 7 は 2024 産で V20 未収録)

### 2. V21 training data (data/v21_training_data_full.csv)
- 189,957 rows × 139 cols (V20 base + 33 video features)
- video coverage: **0.03%** (29 horses のみ、 LGB は NaN tolerant)
- horse_id mapping 正常 機能

### 3. V21 trainer (train/train_v21_lgb_xgb.py)
- LGB+XGB ensemble、 6-fold WF
- POST-RACE LEAK 厳密除外 (prize/run_time/agari_3f/pass3/pass4/popularity 等)
- video features 含む 学習 (将来 coverage 拡大時 自動活性化)

### 4. V21 WF results
| fold | n_train | n_val | AUC ENS |
|------|---------|-------|---------|
| 2023 | 47,220 | 47,672 | 0.7857 |
| 2024 | 94,892 | 47,181 | 0.7955 |
| 2025 | 142,073 | 47,884 | 0.7942 |
| **mean** | — | — | **0.7918** |

注: 2020-2022 は v20_training_data_full に未収録のため skip。

### 5. TOP features (LEAK-free 確認済)
1. horse_recent5_top3: 44,062
2. training_4f: 42,207
3. jockey_recent30_top3: 37,222
4. fresh_horse: 21,849
5. **jockey_trainer_combo_top3_exp: 14,799** ← Phase 26 発見 features 機能
6. distance: 11,915
7. sire_no_class_down_top3_rate_exp: 11,512
8. ...
15. **corner_position_delta: 6,076** ← Phase 26 features 機能

video features: importance 0 (sparse coverage で signal なし、 期待通り)

## 🔴 V21 < V15 の 理由

V21 base = v20_training_data_full.csv は **100 usable features** のみ。
V15 = **150 features** (JRDB KYI / siblings_exp / training premium 等 含む)。

V21 を V15 越えするには:
1. V15 training data (150 features) を base に
2. Phase 24 + 26 features 追加 (jockey_trainer_combo 等)
3. 4-model ensemble (LGB + XGB + FT-Transformer + IntraRace)
4. video features (現状 sparse、 1000+ paddock dirs 必要)

→ 1 day では 完成不可。 **V15 production 継続、 V21 は long-term project**。

## 🛡 V15 投資保護 (継続遵守)

- V15 .pkl.gz / predict_core / daily_predict / app.py 完全不変
- V21 model は `keiba_model_v21_central.pkl.gz` 別 file
- production 切り替えは 手動判断のみ (5/16 自動切り替えなし)

## 5/17 (土) 本番運用

- **V15 戦略⑦ 案B改 単独継続** (絶対)
- Strategy 8 sidecar (Jackpot pattern Discord alert) shadow eval
- V21 model は 動作確認のみ、 投票 影響なし

## 5/16 までに 完成 / 未完成

完成:
- ✅ horse_id mapper (TFJV ⇔ netkeiba)
- ✅ V21 training data builder
- ✅ V21 video features pipeline (29 dirs / 33 features 抽出)
- ✅ V21 LGB+XGB WF (clean, no leak)
- ✅ Strategy 8 sidecar (V15 完全保護)
- ✅ Phase 24 / 26 features 統合 (jockey_trainer_combo +21.3pt 等)
- ✅ live_features_5_17.py (5/17 動的 features)

未完成 (5/24+ 着手):
- ⚠️ V21 AUC > V15 達成 (V15 training data base + 4-ensemble 必要)
- ⚠️ video features 実用 (1000+ paddock dirs 蓄積必要)
- ⚠️ JV-Link 32-bit COM 実行 (user manual task)
- ⚠️ netkeiba 2026 catchup (auto-block、 user 認可必要)

## next step

5/13-15 user task:
- TFJV / TARGET frontier JV 経由 jra_races_full.csv 更新
- JV-Link 32-bit Python で 32 dataspec 実 fetch
- netkeiba 2026 catchup 手動実行 (auto-block 認可必要)

5/16 夜 rehearsal:
- V15 daily_predict 1 race test
- Strategy 8 sidecar 1 race test

5/17 (土) 本番:
- V15 戦略⑦ 案B改 単独運用 (現状不変)
- Strategy 8 sidecar shadow eval (Discord 通知のみ、 投資 0)

5/24+ V20/V21 path (改定):
- V15 retraining base + Phase 24/26 features → 真の V21 候補
- 期待 WF AUC 0.90+ (V15 + 新 features 上乗せ)
- 6/15 GO/no-go 判定
