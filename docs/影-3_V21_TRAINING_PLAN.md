# 影-3: V21 Training Plan — TYB Feature Integration

**作成日**: 2026-05-22
**前提 docs**: `docs/TYB_RELEASE_TIMING_RE_AUDIT_2026_05_21.md` / `docs/TYB_PER_RACE_TIMING_AUDIT_2026_05_22.md`
**V15 baseline**: genuine WF LGB+XGB 6-fold mean = **0.8678** (V15-audit-2)
**V21 target**: genuine WF AUC ≥ **0.880** (V15 比 +0.012 以上)
**制約**: V15 production は完全不変。V21 は新規学習。

---

## 0. TL;DR

TYB (JRDB 直前累積データ) は全 26 fields が PRE_RACE content であることが確認済み。
過去の「LEAK 確定」「永久放棄」は配信タイミング問題 (標準 path = 17:00 JST) との混同による誤判定。
V21 では retrospective TYB データを学習データに merge し、tyokuzen path でのライブ予測を実現する。

| 確認済 | 内容 |
|--------|------|
| content leak | **0 件** — 全 26 fields PRE_RACE |
| per-race 更新 | **confirmed** (10/10 dates) |
| odds_idx LEAK 誤ラベル | odds_idx = PRE_RACE。corr_target +0.42 = 有用信号、post-race 混入でない |
| V21 retrospective merge | 1,340 files (TYB150104〜TYB260516) で 100% coverage 見込み |

---

## 1. Timeline

| 日付 | 作業 | 成果物 |
|------|------|--------|
| **6/1** | 影-3 shadow 観測 (per-race fetch 実確認) | `data/tyb_shadow/20260601/summary.csv` |
| **6/9-13** | JV-Link parser + TYB merge pipeline 実装 | `tools/v21/tyb_merge_pipeline.py` |
| **6/14-15** | V21 data spec 確定 (TYB fields 選定) | `docs/V21_DATA_SPEC.md` |
| **6/16-20** | V21 v1 学習 (LGB+XGB, 6-fold WF) | `models/keiba_model_v21_v1.pkl.gz` |
| **6/21-25** | V21 WF validation + ablation 結果確認 | `data/v21_wf_results.json` |
| **6/26-28** | Paper trading (shadow) | `data/race_notify_log_v2_summary/v21_paper/` |
| **6/29-30** | GO/NO-GO 最終判定 | 判定記録 doc |

---

## 2. TYB Feature Candidates for V21

### 2.1 フィールド一覧と評価

| Field | Source | 予測価値 (add-one 推定) | 懸念 |
|-------|--------|----------------------|------|
| `odds_idx` | JRDB 計算 index (-15 min) | **corr_target ~0.42 / 推定 +0.017** | V15 morning odds features (oz_base_pop_rank 等) との multicollinearity。ablation 後に判断 |
| `padock_idx` | パドック観察 (-30〜-40 min) | **corr_target ~0.35 / 推定 +0.015** | 主観的人間評価。外れ年の variance が大きい可能性 |
| `jockey_idx` | 騎手指数 (直前) | 推定 +0.010 | V15 に騎手系 features 既存。重複度確認が必要 |
| `info_idx` | 情報指数 (直前) | 推定 +0.008 | 情報源不明。黒箱 index |
| `tansho_odds` | 単勝オッズ (-15 min snapshot) | 推定 +0.008 | V15 の `odds_log` / `pop_rank` と重複する可能性。multicollinearity 確認要 |
| `fukusho_odds` | 複勝オッズ (-15 min) | 推定 +0.006 | tansho_odds と同根の懸念 |
| `padock_mark` | A/B/C/D グレード | 推定 +0.004 | ordinal encoding (A=4, B=3, C=2, D=1) が必要 |
| `ashimoto` | 歩様 (パドック -30 min 観察) | 推定 +0.002 | baseline 低い (~0.05)。個人差が大きい観察データ |
| `horse_weight` | 公式 -70 min 発表 | **negative add-one** | V15 Pattern B に既存。duplicate = negative |
| `weight_diff` | 馬体重増減 | 微小 (V15 内包) | 同上 |
| `sogo_idx` | idm+odds+padock 合成 | V15 内包の可能性 | 個別 fields が入れば redundant |
| `idm` | JRDB 総合指数 | V15 内包の可能性 | 確認要 |
| `kehai_code` | 気配コード | **negative add-one** | V15 audit で negative 確認済 |
| `bagu_change` | 馬具変更 | **negative add-one** | V15 audit で negative 確認済 |
| `batai_code` | 馬体コード | 微小 | body code、主観的 |
| `baba_code` | 馬場状態 | V15 内包 | V15 に condition_enc 既存 |
| `weather_code` | 天候 | V15 内包 | V15 に weather_enc 既存 |

### 2.2 優先候補 (ablation 実施対象)

ablation の優先順位は add-one 推定値の高い順。ただし multicollinearity 懸念のあるものは除外後に再テスト。

**Tier 1 (必ずテスト)**:
- `odds_idx` — corr_target ~0.42 は最高。multicollinearity を ablation で検証
- `padock_idx` — 直接観察値。V15 に同等 feature なし
- `jockey_idx` — V15 騎手系と重複度チェック後

**Tier 2 (Tier 1 次第)**:
- `info_idx`, `tansho_odds`, `fukusho_odds`, `padock_mark`

**Tier 3 (最後に確認)**:
- `ashimoto`, `sogo_idx`

**除外 (ablation 不要)**:
- `horse_weight`, `weight_diff`, `baba_code`, `weather_code` — V15 に既存
- `kehai_code`, `bagu_change` — negative add-one 確認済

---

## 3. Ablation Test Design

### 3.1 設計方針

**Baseline**: V15 genuine WF 6-fold LGB+XGB AUC = **0.8678**

全 ablation は V15 の 6-fold walk-forward (2020-2025) 上で実施。fold 構成は V15 と同一にして比較可能にする。

### 3.2 ステップ

**Step 1: Add-one test (各 TYB field を単独で追加)**

```python
for field in TYB_TIER1_FIELDS + TYB_TIER2_FIELDS:
    features_v21 = V15_FEATURES_145 + [field]
    auc = run_wf_6fold_lgb_xgb(features_v21, data_merged)
    delta = auc - 0.8678
    print(f"{field}: delta = {delta:+.4f}")
```

判断基準: `delta ≥ +0.002` → 採用候補 / `delta < 0` → 即時除外

**Step 2: multicollinearity 確認 (odds_idx 等)**

V15 には morning odds-based features (`oz_base_pop_rank`, `prev_odds_log`, `odds_change_rate`, `pop_rank_change`, `odds_sharp_drop`) が含まれる。

```python
# V15 morning odds features を除外した条件で odds_idx の add-one delta を再測定
features_no_morning_odds = V15_FEATURES_145 - MORNING_ODDS_FEATURES + ["odds_idx"]
auc_clean = run_wf_6fold_lgb_xgb(features_no_morning_odds, data_merged)
delta_clean = auc_clean - BASELINE_NO_MORNING_ODDS
```

- `delta_clean` が大きければ: odds_idx は genuine な追加信号。morning odds との trade-off を検討。
- `delta_clean` が小さければ: V15 morning odds が既に odds_idx の情報を捕捉 → 追加不要

**Step 3: Combo test (上位 3 field の組み合わせ)**

Step 1/2 で delta ≥ +0.002 かつ multicollinear でないと判定された field を 2-3 個組み合わせてテスト。

```python
top_fields = [f for f in TYB_FIELDS if delta[f] >= 0.002 and not multicollinear[f]]
features_combo = V15_FEATURES_145 + top_fields[:3]
auc_combo = run_wf_6fold_lgb_xgb(features_combo, data_merged)
```

**Step 4: V21 final feature set**

```
V21_FEATURES = V15_FEATURES_145 + TYB_NET_POSITIVE_FIELDS
```

`TYB_NET_POSITIVE_FIELDS` = Step 1-3 で採用判定した field のみ (追加数は 0〜5 件程度を想定)。

### 3.3 正直な warning

**同一 data で feature 選択と評価を行うことの限界**:
- add-one delta が Step 1 で +0.002 でも、hold-out (未来 2026 データ) では消える可能性がある
- **必須**: 6/26-28 paper trading で out-of-sample 確認を経てから GO/NO-GO を出す
- ablation 結果だけで GO を出さない

---

## 4. V21 Training Data Merge

### 4.1 TYB retrospective data

```
data/jrdb/extracted/Tyb/TYB{yymmdd}.txt
```

- ファイル数: `TYB150104.txt` 〜 `TYB260516.txt` で約 1,340 files (確認済、`TYB_RELEASE_TIMING_RE_AUDIT §2.1`)
- 年度カバー: 2015-2026 (JRA 開催日分のみ)
- レコード長: 128 bytes / horse

TYB ファイルカバレッジ確認コマンド:

```bash
# Windows PowerShell
(Get-ChildItem "data\jrdb\extracted\Tyb\*.txt" | Measure-Object).Count
# または
ls data/jrdb/extracted/Tyb/*.txt | wc -l
```

### 4.2 Merge key

```python
# TYB merge key = race_id (date + basho + race_num) + umaban (horse number)
# TYB: basho_code (01-10) + year (2桁) + kai (2桁) + nichi (2桁) + race_num (01-12) + umaban (01-18)
# jra_races_full.csv の race_id は YYYYMMDDRR 形式

def build_merge_key(tyb_record):
    # basho_code + year + kai + nichi → date変換
    # race_num, umaban はそのまま
    return f"{date}_{basho}_{race_num}_{umaban}"
```

詳細な変換ロジックは `tools/v21/tyb_merge_pipeline.py` に実装予定 (6/9-13)。

### 4.3 Coverage 見積もり

| データ | 行数 | 期間 |
|--------|------|------|
| `jra_races_full.csv` | ~781,161 | 2010-2025 |
| TYB txt files | ~1,340 files | 2015-2026 (開催日) |
| **期待 merge 率** | **~70-80%** | 2015以前は TYB なし → 2010-2014 はNaN fill |

2010-2014 の行 (約 30%) は TYB features = NaN となる。LGB/XGB は NaN をそのまま扱えるため問題なし。

### 4.4 Missing value 処理

```python
# TYB features の NaN 処理方針
tyb_features = ["odds_idx", "padock_idx", "jockey_idx", "tansho_odds", "fukusho_odds",
                 "padock_mark", "ashimoto"]

# LGB: categorical features → NaN は別ビン扱い (自動)
# XGB: NaN は内部で missing value branch として処理 (tree_method='hist' で対応)
# 明示的な fill は行わない (信号を混入させない)
```

---

## 5. V21 GO/NO-GO Decision Criteria

| 基準 | 値 | 根拠 |
|------|----|------|
| WF AUC (6-fold LGB+XGB) | **≥ 0.880** | V15 genuine 0.8678 から +0.012 以上 |
| LIVE retro winner_top1 | **≥ 30%** | V15 実運用 top-k recall 基準 |
| odds_time shift | **≤ 12x** | Session #38 NO-GO 判定基準と同一 |
| Paper trading ROI (2 週) | **≥ 110%** | V15 実運用 98.34% からの有意改善 |
| LEAK audit | **PASS** | TYB 全 fields PRE_RACE 確認済。新規 features も ablation 前に確認 |
| No new LEAK | **PASS** | V21 data spec 確定前に leak audit 実施 (V20_LEAK_FEATURES 除外を継承) |

**全基準 PASS → 7/1 V21 段階投入 (週末のみ、上限 5,000円/日)**
**1 基準でも FAIL → NO-GO。V15 継続 + 再設計。**

---

## 6. V21 Architecture

V15 の genuine production architecture (LGB+XGB 2-model) を踏襲しつつ、V15-audit-1 で判明した問題 (FT/IR の .pkl 未保存) を解決する。

| 項目 | V15 | V21 |
|------|-----|-----|
| Architecture | LGB+XGB (production) / FT+IR は WF 専用 | **LGB+XGB+FT+IR full ensemble、全モデル .pkl 保存** |
| Features | 145 (booster) | 145 + TYB net positive (予想 147-150) |
| Genuine WF AUC | 0.8678 (LGB+XGB) | **目標 ≥ 0.880** |
| Folds | 6-fold 2020-2025 | 6-fold 2020-2025 (同一) |
| SKB features | V15 に含まず (Session #38 NO-GO) | 引き続き除外 (`V20_LEAK_FEATURES` 継承) |
| sib_*_exp | V15 非採用 | 6/1-6/8 GO 判定後に追加検討 |
| TYB features | なし (truncate) | ablation PASS 分のみ追加 |

---

## 7. Production Fetch Design (V21 Live -15 min)

V21 が 7/1 投入された場合の per-race fetch flow:

```
race_auto_notify.py
  └─ for each race (08:45 loop):
       start_time = get_race_start_time(race_id)
       fetch_window = start_time - 20 min
       if now >= fetch_window:
           tyb = fetch_tyokuzen_tyb(date, jrdb_auth)  # tyokuzen path
           tyb_features = parse_tyb_for_race(tyb, race_num)
           features = merge_v21_features(base_features, tyb_features)
           pred = v21_model.predict(features)
           notify_discord(pred)
```

- fallback: tyokuzen fetch 失敗 → standard batch CSV (当日 17:00 以降は取得済み) にフォールバック → 翌日以降は影響なし、当日 live のみ精度低下

---

## 8. Honest 限界 / リスク

1. **同一 data での feature 選択 → hold-out での過楽観リスク**: ablation delta はバイアスあり。paper trading (6/26-28) で out-of-sample 確認必須。
2. **odds_idx の multicollinearity**: V15 morning odds features が odds_idx の情報を既に捕捉している場合、delta は +0 に収束。その場合 odds_idx 不採用。
3. **padock_idx の subjective variance**: 観察者 (JRDB スタッフ) の主観が入る。年によって variance が異なる可能性。2020-2022 等 COVID 期の特殊事情も要確認。
4. **tyokuzen path の安定性**: 5 週間観測で安定していたが、JRDB サーバー変更でパスが変わる可能性はゼロではない。6/1 shadow 観測で最終確認。
5. **JRDB 規約の明示的確認未完了**: kiyaku.html の TLS エラーで直接確認できていない。per-race fetch の商業利用許諾を確認できていない点は ongoing risk。

---

## 9. 参考 / 出典

| doc | 内容 |
|-----|------|
| `docs/TYB_RELEASE_TIMING_RE_AUDIT_2026_05_21.md` | TYB field schema / 誤 LEAK 訂正 / 予測価値推定 |
| `docs/TYB_PER_RACE_TIMING_AUDIT_2026_05_22.md` | per-race 更新 10/10 dates 確認 / R01-R12 odds_time 実測 |
| `docs/影-3_6_1_SHADOW_OBSERVATION_PLAN.md` | 6/1 観測計画 (本 doc の前提) |
| CLAUDE.md §Phase 3-4 roadmap | V20 timeline (V21 はその延長線) |
| V15-audit-1〜5 | V15 真値 (AUC 0.8678 / architecture LGB+XGB 2-model) |
| Session #38 | SKB POST-RACE LEAK 確定 / V20_LEAK_FEATURES |
