# T1 features integrity audit (TYB merge bug 級事故 永久防止)

**実施日**: 2026-05-17
**目的**: V15 cache (`data/_v15_optuna_df_cache.pkl.gz`) の 145 features を完全 audit、
TYB merge bug (1 年以上 検出されず TYB 5 features が実質ゼロ寄与) と同型の事故を防ぐ
自動 monitor を構築する。

★ **V15 production model 完全不変**、 audit は read-only only。 修正・再学習なし。 ★

---

## 0. 結論

- **V15 model 真値性**: 健全 (RED_IMP_BUT_CONST = 0 件、 model に load されているが
  分散ゼロの features は存在しない)
- **V15 145 features audit 結果**:
  - RED_CONSTANT (unique <= 1): **8 件** — 全 既知 + LGB/XGB importance = 0
  - RED_LOW_UNIQUE (unique 2-10): 39 件 — 多くは categorical 正常
  - WARN_QUASI_CONSTANT (mcr > 95%): 12 件 — 大半は RED_CONSTANT と重複
  - WARN_HIGH_NULL (>50%): 0 件
- **TYB known suspects (sub-task 6 発見 5 件)**: V15 145 features に **含まれていない** ことを確認
  - `jrdb_paddock_idx`, `jrdb_odds_idx`, `jrdb_body_code`, `jrdb_demeanor_code`,
    `jrdb_live_composite_idx` → V15 で使われていないため model 害なし
  - (cache の 232 columns には残っているが unused)
- **monitor 自動化 ready**: `tools/features_integrity_monitor.py` 完成、
  daily 22:00 schtask 登録 script (`tools/register_features_integrity_schtask.bat`) ready
  ★ 実 schtasks /create は **5/18 user 判断後** に admin で実行 ★
- **regression test 拡張**: 23 → 33 (T1 で +10)、 全 PASS 想定

---

## 1. 145 features 全 audit table (priority 順)

判定 criteria:
- `RED_CONSTANT`: unique <= 1 (全行同値、 TYB 級事故候補)
- `RED_LOW_UNIQUE`: 2 <= unique <= 10 (low cardinality)
- `RED_IMP_BUT_CONST`: lgb_gain > 0 かつ unique <= 1 (★ critical: model 入力but分散なし ★)
- `WARN_HIGH_NULL`: null_rate > 50%
- `WARN_QUASI_CONSTANT`: most_common_rate > 95%

### RED_CONSTANT (8 件、全て既知)

| feature | unique | null | mcr | lgb_gain | xgb_gain | flags | judgment |
|---|---:|---:|---:|---:|---:|---|---|
| `is_nar` | 1 | 0.000 | 1.000 | 0.0 | 0.0 | RED_CONSTANT | ★ intentional ★ (JRA 専用 cache 0 固定) |
| `prev_odds_log` | 1 | 0.000 | 1.000 | 0.0 | 0.0 | RED_CONSTANT | log(16) ≈ 2.77 default fill、 LEAK 除去後残骸 |
| `prev_race_first3f` | 1 | 0.000 | 1.000 | 0.0 | 0.0 | RED_CONSTANT | 35.8s default、 取得 pipeline 死、 v15.2 で削除推奨 |
| `prev_race_last3f` | 1 | 0.000 | 1.000 | 0.0 | 0.0 | RED_CONSTANT | 36.5s default、 同上 |
| `prev_race_pace_diff` | 1 | 0.000 | 1.000 | 0.0 | 0.0 | RED_CONSTANT | 0.0 default、 同上 |
| `sire_shinba_top3r` | 1 | 0.000 | 1.000 | 0.0 | 0.0 | RED_CONSTANT | 0.22 default、 新馬戦評価死特徴量 |
| `pci` | 1 | 0.000 | 1.000 | 0.0 | 0.0 | RED_CONSTANT | 1.0195 default、 ペースチェンジ指数死特徴量 |
| `gaisha_rank` | 1 | 0.000 | 1.000 | 0.0 | 0.0 | RED_CONSTANT | 0 default、 4/26 audit 死特徴量確認済 |

→ 全 8 件で LGB/XGB gain = 0、 model に害なし。 ただし v15.2 学習時は 削除推奨。

### RED_LOW_UNIQUE (上位 22 件、 importance 順)

| feature | unique | null | mcr | lgb_gain | xgb_gain | judgment |
|---|---:|---:|---:|---:|---:|---|
| `paci_sogo_mark` | 6 | 0.000 | 0.642 | 18266.2 | 582.5 | ★ 健全 ★ (paci mark は 0-5 のため intentional low) |
| `surface_dist_enc` | 10 | 0.000 | 0.231 | 16267.4 | 49.6 | 健全 (categorical 2x5) |
| `paci_idm_mark` | 6 | 0.000 | 0.642 | 7326.0 | 58.7 | 健全 |
| `surface_enc` | 2 | 0.000 | 0.505 | 6970.1 | 53.6 | 健全 (芝/ダ) |
| `course_enc` | 10 | 0.000 | 0.161 | 6388.4 | 27.1 | 健全 (10 競馬場) |
| `training_intensity_enc` | 4 | 0.000 | 0.657 | 1670.0 | 22.6 | 健全 |
| `jrdb_running_style` | 5 | 0.000 | 0.321 | 1583.5 | 18.8 | 健全 |
| `jrdb_ze_furi_count` | 8 | 0.000 | 0.726 | 1438.9 | 14.2 | 健全 |
| `season` | 4 | 0.000 | 0.262 | 1302.7 | 12.2 | 健全 |
| `jrdb_tb_homestr_inner` | 6 | 0.000 | 0.565 | 1296.8 | 13.2 | 健全 |
| `sex_enc` | 3 | 0.000 | 0.547 | 1170.7 | 15.6 | 健全 |
| `jrdb_ranch_rank` | 6 | 0.000 | 0.319 | 1160.3 | 11.3 | 健全 |
| `jrdb_stable_rank` | 8 | 0.000 | 0.244 | 1121.3 | 12.0 | 健全 |
| `jrdb_dist_apt` | 6 | 0.000 | 0.320 | 853.5 | 12.2 | 健全 |
| `jrdb_training_arrow` | 5 | 0.000 | 0.769 | 745.0 | 18.5 | 健全 |
| `location_enc` | 4 | 0.000 | 0.515 | 693.8 | 18.7 | 健全 |
| `bracket` | 8 | 0.000 | 0.146 | 685.7 | 10.2 | 健全 |
| `dist_cat` | 5 | 0.000 | 0.408 | 646.5 | 33.0 | 健全 |
| `jrdb_heavy_apt` | 4 | 0.000 | 0.445 | 578.5 | 10.0 | 健全 |
| `course_renovated` | 2 | 0.000 | 0.987 | 70.5 | 5.5 | ⚠ quasi-const、 4/27 永久化適用済 |
| `age_group` | 6 | 0.000 | 0.417 | 0.0 | 194.5 | XGB 専用 (LGB 0)、 健全 |
| (他 17 件) | — | — | — | — | — | 全 categorical 正常 |

### WARN_QUASI_CONSTANT (RED_CONSTANT 除いた 4 件)

| feature | unique | mcr | lgb_gain | xgb_gain | judgment |
|---|---:|---:|---:|---:|---|
| `has_training` | 2 | 0.970 | 0.0 | 0.0 | 死特徴量 (97% 同値、 imp 0) |
| `jrdb_prev_interference` | 14 | 0.982 | 159.6 | 23.4 | ★ 健全 ★ (前走不利稀イベント 1.8% を捕捉) |
| `jrdb_prev_rise_code` | 6 | 0.976 | 34.7 | 0.0 | LGB が稀イベント捕捉 |
| `course_renovated` | 2 | 0.987 | 70.5 | 5.5 | 京都改修 1.3% events、 4/27 永久化適用済 |

---

## 2. red flag features 完全 list (8 + 0 = 8 件)

### 既知 RED_CONSTANT (8 件、 documented)

```python
KNOWN_RED_CONSTANT_FEATURES = {
    'is_nar',              # intentional (JRA only cache)
    'prev_odds_log',       # 2.77 default、 LEAK 除去後残骸
    'prev_race_first3f',   # 35.8s default、 prev race ラップ取得死
    'prev_race_last3f',    # 36.5s default、 同上
    'prev_race_pace_diff', # 0.0 default、 同上
    'sire_shinba_top3r',   # 0.22 default、 新馬戦評価死
    'pci',                 # 1.0195 default、 ペースチェンジ指数死
    'gaisha_rank',         # 0 default、 4/26 audit 死特徴量
}
```

→ 共通点: **全て LGB/XGB importance = 0** → V15 model に害なし。 v15.2 で削除推奨。

### 新規 RED_CONSTANT (0 件)

今回の audit で新たに発見された TYB 級事故: **なし**。

### TYB known suspects (sub-task 6 発見 5 件)

V15 145 features 内 **含まれていない** ことを確認。 cache 232 columns には残っているが
V15 model は使用しない。 V21 video pipeline / V22 distillation で再考。

| feature | V15 内? | cache 内? | 状態 |
|---|:---:|:---:|---|
| `jrdb_paddock_idx` | NO | YES (unique=1) | TYB merge 死 |
| `jrdb_odds_idx` | NO | YES (unique=1) | TYB merge 死 |
| `jrdb_body_code` | NO | YES (unique=1) | TYB merge 死 |
| `jrdb_demeanor_code` | NO | YES (unique=1) | TYB merge 死 |
| `jrdb_live_composite_idx` | NO | YES (unique=1) | TYB merge 死 |

---

## 3. 推奨 actions

### 直接 action (★ 今回 commit 範囲 ★)
1. **monitor 自動化**: `tools/features_integrity_monitor.py` 完成、 daily run で
   新規 red flag 発生時のみ Discord 警告。
2. **regression test 拡張**: `tests/T1_features_integrity_test.py` 追加 (10 tests)、
   既存 23 → 33 へ。
3. **schtask 登録 script ready**: `tools/register_features_integrity_schtask.bat`
   (★ 5/18 user 判断後に admin で実行 ★)

### ★ V15 modify 禁止 ★ — 以下は v15.2 学習時に適用
- 既知 8 件 RED_CONSTANT features を v15.2 features list から除外
- TYB 5 features は引き続き V15 系統で使用しない
- 学習 pipeline に build-time integrity check (unique <= 1 検出時 fail) 追加

### 5/18 user 判断後の手順
1. ADMIN PowerShell で `tools/register_features_integrity_schtask.bat` 実行
2. `schtasks /Query /TN "Keiba-FeaturesIntegrityCheck"` で確認
3. Discord (`DISCORD_WEBHOOK_UPDATES`) が設定済か確認

---

## 4. monitor 仕様

### `tools/features_integrity_monitor.py`

- 入力: `data/_v15_optuna_df_cache.pkl.gz` (V15 cache、 read-only)
       + `keiba_model_v15_central.pkl.gz` (V15 model、 read-only)
- 出力: `data/T1_features_audit_YYYY_MM_DD.json` (audit 結果)
- Discord 通知条件 (lazy notify):
  - **新規 RED_CONSTANT** 発生 (KNOWN list 外)
  - **RED_IMP_BUT_CONST** 発生 (importance > 0 + 分散ゼロ)
  - **MISSING** 発生 (V15 feature が df から消失)
- exit code: 常に 0 (read-only audit、 pipeline blocking しない)

### CLI
```bash
python tools/features_integrity_monitor.py             # 通常 run + Discord
python tools/features_integrity_monitor.py --check-only # JSON 保存なし
python tools/features_integrity_monitor.py --no-discord # Discord 抑制
```

### schtask 仕様

| 項目 | 値 |
|---|---|
| Task name | `Keiba-FeaturesIntegrityCheck` |
| Schedule | DAILY @22:00 |
| Command | `python tools/features_integrity_monitor.py` |
| Log | `logs/features_integrity_YYYYMMDD.log` |

---

## 5. V15 production 不変保証

| 項目 | 状態 |
|---|---|
| `keiba_model_v15_central.pkl.gz` | **完全不変** (read-only access のみ) |
| `keiba_model_v15_central_live.pkl.gz` | **完全不変** (touch なし) |
| `tools/predict_core.py` | **完全不変** |
| `tools/daily_predict.py` | **完全不変** |
| `tools/race_auto_notify.py` | **完全不変** |
| `app.py` | **完全不変** |
| `data/cumulative_results.csv` | **完全不変** |
| 既存 regression_test.py 23 tests | **logic 不変** (T1 は追加 only) |

✅ T1 sub-task は monitor + test + docs の **追加 only**。 既存資産に一切手を加えない。

---

## 6. 完了 metric

| 項目 | 状態 |
|---|---|
| V15 cache 145 features audit | ✅ 完了 |
| red flag features 検出 | ✅ 8 件 (全既知) |
| TYB 級事故新規発見 | 0 件 (健全) |
| `tools/features_integrity_monitor.py` | ✅ ready |
| `tools/register_features_integrity_schtask.bat` | ✅ ready (★ 5/18 user 判断 ★) |
| `tests/T1_features_integrity_test.py` (10 tests) | ✅ ready |
| `data/T1_features_audit_2026_05_17.json` | ✅ saved |
| 本 doc | ✅ saved |
| V15 production 不変 | ✅ 完全保証 |
