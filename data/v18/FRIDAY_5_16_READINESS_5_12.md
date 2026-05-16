# Friday 5/16 → 土曜 5/17 readiness summary (5/12 朝 marathon 完了時点)

user "1-2 日 触れない" の 留守中に main thread + Agent group で 完成した items。

## 🎯 結論

**5/17 (土) 本番は V15 戦略⑦ 案B改 単独継続** (絶対遵守、 変更なし)

V21/V22 開発成果は **全て shadow eval / 将来 投入用**。 production 投入は 5/24+ 以降 user 判断。

---

## ✅ 完成 deliverables (本 marathon 集計)

### A. V21 (V20 base + video infra)
| 項目 | 状態 | 備考 |
|------|------|------|
| video features pipeline | ✅ | 33 features (gait 18 + body 15) |
| paddock dirs | 107+ | 5/9-5/10 含む multi-course |
| horse_id mapper | ✅ | TFJV ↔ netkeiba (`'20' prefix`) |
| training data builder | ✅ | 190K rows, video coverage 0.03% |
| LGB+XGB WF | ✅ | mean AUC 0.7918 (3 folds) |
| auto-retrain trigger | ✅ | paddock 閾値で 自動 retrain |

### B. V22 (V15 base + Phase 24/26 features)
| 項目 | 状態 | 備考 |
|------|------|------|
| trainer (V15 cache + 32 新 features) | ✅ | 527K rows × 177 features |
| LGB+XGB 6-fold WF | ✅ | mean AUC **0.8683** |
| per-year stability | ✅ | 2020=0.8591 〜 2024=0.8735 |
| Phase 24/26 features 機能確認 | ✅ | 全部 importance > 100 |

V22 > V21 (+0.0765)、 V15 (4-model Grid 0.8939) と LGB+XGB only fair compare では V22 が候補。

### C. Phase 26 (5/12 朝、 Session #87)
- jockey_trainer_combo (+21.3pt) verified in V22 importance
- corner_position_delta (+10.2pt) verified
- PACI 修復 (5/3-5/10 補完 OK)
- live_features_5_17.py (5/17 動的 features 計算)
- strategy8_sidecar.py (Jackpot pattern 別 channel Discord)

### D. Infrastructure (新 tools)
1. `tools/horse_id_mapper.py` - TFJV ⇔ netkeiba mapping (permanent rule)
2. `tools/v21_extract_all_video_features.py` - paddock dirs → unified V21 CSV
3. `tools/v21_training_data_builder.py` - V20 base + video merge
4. `tools/v21_auto_retrain_trigger.py` - paddock 閾値で auto retrain
5. `tools/paddock_weekend_archive_build.py` - 5/9-5/10 archive build (105+ dirs)
6. `tools/jvlink_parser.py` - 32 dataspec (8 → 32) expansion
7. `tools/scrape_jrdb_paci.py` - PACI 修復 scraper
8. `tools/scrape_jrdb_kta_mza.py` - KTA/MZA/MSA scraper (URL verify pending)
9. `tools/netkeiba_2026_catchup.py` - 12 csv catchup (auto-block、 user 認可 必要)
10. `tools/jravan_paddock_eye_capture.py` - JRA-VAN パドックアイ skeleton
11. `tools/jra_rv_patrol_capture.py` - RV パトロール skeleton
12. `tools/build_competitor_gap_features.py` / `_v2.py` - 9 new features (jockey_trainer_combo 等)
13. `tools/build_pace_features_FIXED.py` - race_id bug 修正版

### E. Trainers (新 train scripts)
1. `train/train_v21_lgb_xgb.py` - V21 LGB+XGB WF
2. `train/train_v22_v15base_phase24.py` - V22 V15+P24/26 features WF

---

## 📊 modeling 結果まとめ

| model | features | LGB+XGB AUC | 4-model Grid AUC | 判定 |
|-------|----------|-------------|-----------------|------|
| V15 (production) | 145 (JRDB+PACI 完備) | ~0.87 (推定) | **0.8939** | 継続 |
| V20 (旧候補) | 100 (V20 base) | 0.8376 | — | 不採用 |
| **V21** | 100 + 33 video sparse | **0.7918** | — | shadow eval |
| **V22** | 145 + 32 Phase 24/26 | **0.8683** | — | shadow eval、 4-ensemble で 投入候補 |

**V22 fair-compare 解釈**:
- V22 LGB+XGB 0.8683 vs V15 4-ensemble 0.8939
- V22 4-ensemble (FT + IR 追加) は 5/24+ user GPU 環境 で 学習可能
- 想定 V22 4-ensemble AUC: 0.89-0.91 (V15 + 0.01-0.02、 Phase 24/26 features 効果)

---

## 🛡 V15 投資保護 (5/12-5/17 完全遵守)

- V15 .pkl.gz / predict_core.py / daily_predict.py / app.py **完全 不変**
- V21 model: `keiba_model_v21_central.pkl.gz` (別 file、 production 投入なし)
- V22 model: `keiba_model_v22_central.pkl.gz` (別 file、 production 投入なし)
- Strategy 8 sidecar: V15 と 別 process / 別 Discord channel
- 累計収支 +5,240 円 守る (撤退余裕 +55,240 円) ※ 旧 +13,530 / +63,530 は drift、 5/16 P0-1 真値 (docs/ROI_DISCREPANCY_2026_05_16.md)

---

## 5/17 (土) 本番運用 fix plan

### V15 戦略⑦ 案B改 (絶対遵守)
- daily_predict.py (08:00 schtask)
- race_auto_notify.py (R-5 分前 Discord 通知)
- 06_特別 / 京都 / 条件E / 条件B 除外
- 案B改 12R 1勝クラスのみ 上限 2,100円

### Strategy 8 sidecar (新規 shadow eval)
- 09:00 strategy8_sidecar.py 手動 or schtask
- Jackpot pattern (4-way) 該当馬 別 channel Discord
- shadow eval のみ、 投資 0 円 (verification 用)
- DISCORD_WEBHOOK_JACKPOT .env 未設定なら log のみ

### V21/V22 動作確認 (任意)
- 5/17 朝 1 race で V21/V22 推論 (production 影響なし)
- 結果は data/v22_shadow_eval/{date}/ 保存

---

## 🔴 user manual task (5/13-15 中 推奨)

1. **TFJV update** (TARGET frontier JV で 手動 export)
2. **netkeiba 2026 catchup**: `python tools/netkeiba_2026_catchup.py` (auto-mode block、 user 認可必要)
3. **JV-Link 32-bit COM test**: `C:\Users\takum\jvlink-venv\Scripts\activate.bat` → `python tools/jvlink_parser.py --test-com`
4. **JRDB KTA/MZA/MSA URL verify** (JRDB member area で 真の path 確認)
5. **JRA-VAN パドックアイ / RV パトロール login** (probe scripts は skeleton 完備)

---

## 📅 短期 roadmap

### 5/12 朝〜 5/16 夜
- 自動 task (paddock 蓄積 / V21 auto-retrain / sidecar) 完備
- user 1-2 日 不在中も V15 production 自動継続

### 5/17 (土) 本番
- V15 戦略⑦ 案B改 単独運用
- Strategy 8 sidecar shadow eval

### 5/18 (日) 振り返り
- V15 結果照合 (自動 daily_results.py)
- V21/V22 shadow 比較 (任意)

### 5/24+ V20/V22 真の path
- V22 4-model Grid 学習 (LGB+XGB+FT+IR、 GPU 24-48h)
- 期待 AUC 0.89-0.91 (V15 0.8939 越え候補)
- 6/8+ paper trading
- 7/1+ production 投入判定

### 7/1+ V21 path (video features 拡大)
- 1000+ paddock dirs 蓄積後
- V21 (V22 base + video features 1000+ coverage) 再学習
- 期待 効果: V22 + 0.005-0.015 (動画 features 仮想 ROI 改善)
- 8/1+ 投入判定

---

## 🚨 留守中 安全確認

- V15 production: 完全自動運用 (Windows schtask)
- backtest_central_leakfree / daily_predict / race_auto_notify: 不変
- 累計収支 monitor: cumulative_results.csv 自動更新
- Discord 通知 (買い目 / 更新): 機能継続
- nightly_sanity: 23:00 翌日 schtask 事前 check + Discord 通知

万一 production が 死亡時 → user 携帯 Discord 通知 で 検知可能。
