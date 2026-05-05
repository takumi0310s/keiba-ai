# JRA × NAR 統合方針 (Phase 2.5+ / 2026-05-05)

**前提**:
- JRA: V15 案B改 (AUC 0.8939, 累計 +14,140 円, 戦略⑦自動化済)
- NAR: NAR v4 (AUC 0.8145 reported / 0.8519 実測, 22 features Pattern B)

---

## 1. 並列運用 (Phase 2.5+ 推奨)

### 1.1 役割分担

| 平日 (月-金) | 主役 | 補助 |
|-------------|------|------|
| 競馬開催 | NAR (v4) | (JRA なし) |
| 投資 | NAR daily | - |

| 土日 | 主役 | 補助 |
|------|------|------|
| 競馬開催 | JRA + NAR (高知 等) | - |
| 投資 | **JRA V15 案B改 (継続)** | NAR (任意、極小) |

### 1.2 衝突回避設計

- pipeline: 既存 JRA 8 task + 新規 NAR 5 task = 13 task、時刻衝突なし
- model: V15 と NAR v4 は完全独立 (.pkl 別、features 共有 14 のみ)
- Discord: NAR 通知も同 webhook (#bets / #updates) 利用、prefix で区別 (例: `[NAR]`)
- 累計記録: `cumulative_results.csv` に source 列追加 (JRA / NAR)

### 1.3 投資配分 (累計上限 -50,000 円)

| 種別 | 1日上限 | 週上限 | 累計判断 |
|------|------:|------:|---------|
| JRA 案B改 | ~9,800 円 | ~19,600 円 (土日のみ) | 既存維持 |
| NAR (5/16-5/22) | 500 円 | 2,500 円 | -3,000 円で週停止 |
| NAR (5/23-5/29) | 1,400 円 | 7,000 円 | -3,000 円で週停止 |
| NAR (6 月以降) | 2,100 円 | 10,500 円 | -5,000 円で月停止 |

**累計 -50,000 円 ライン**: JRA + NAR 合算。到達 → 全停止 + user 判断。
現状 +14,140 円なので耐 drawdown -64,140 円 (約 1 年分の 1日 -200 円ペース)。

---

## 2. Phase 3 統合モデル v20 構想 (長期)

### 2.1 動機

- V15 と NAR v4 は features 共有 14 のみ → 学習データ統合できれば sample size 大幅増
- 中央 + 地方の race 数: 中央 ~3,500/年 × 6 年 = 21,000 race、 地方 ~5,000/年 × 6 年 = 30,000 race → 合計 51,000 race
- is_nar フラグで model が自動切り替え学習可能

### 2.2 実装案 (構想のみ)

```
v20 model 設計:
  features: 22 (NAR v4 と同じ Pattern B base)
  + 拡張 features (中央のみ計算、NAR は 0 fill):
    - sire_enc, bms_enc (血統)
    - jockey_wr_calc, jockey_course_wr (中央騎手系)
    - prev_finish, prev_last3f (前走履歴)
    + ~30 追加 → 計 ~52 features
  
  学習データ:
    - 中央: jra_races_full.csv (2020-2025)
    - 地方: chihou_races_2020_2025.csv (要生成、または nar_all_races.csv 拡張)
    - is_nar=1 → NAR-only features 0 fill、jra_only feats 0 fill
  
  目的変数: finish==1 (binary)
  ensemble: LGB + XGB + (FT-Transformer 任意)
  
  目標 AUC:
    - JRA subset: 0.85+ (V15 0.8939 比 -0.04 低下許容)
    - NAR subset: 0.85+ (v4 0.8145 比 +0.04 改善)
```

### 2.3 工数 / 期待 ROI

| step | 工数 | 期待効果 |
|------|------|---------|
| chihou_races_2020_2025.csv 生成 | 4h | base data |
| features 統合設計 + encode | 8h | 統合学習可能 |
| v20 学習 (Optuna 100 iter) | 4h | model |
| backtest (JRA + NAR 別) | 4h | AUC + ROI |
| 本番統合 (predict_v20.py + UI) | 8h | 単一 pipeline |
| **合計** | **~28h** | **Phase 3 (6 月後半)** |

### 2.4 Phase 2.5+ では実施しない

- 上記は **Phase 3 (5/末 〜 6/末)** に切り出し
- 5/16-5/24 期間は **並列運用** で観察、統合は実績累積後に判断

---

## 3. リスク管理 (詳細)

### 3.1 累計損失閾値

| 累計 | アクション |
|------|-----------|
| +5,000 円 〜 -5,000 円 | 通常運用 |
| -5,000 〜 -15,000 円 | **NAR 1日上限 50% 削減** (例 1,400 → 700 円) |
| -15,000 〜 -30,000 円 | **NAR 完全停止**、JRA 案B改 のみ |
| -30,000 〜 -50,000 円 | **JRA 1日上限 50% 削減** + NAR 停止 |
| -50,000 円 | **全停止**、user 判断 |

### 3.2 1日損失閾値 (即時停止)

- 当日 -1,000 円 (NAR 単独試行 中) → 残り race 停止
- 当日 -3,000 円 (JRA 通常) → 残り race 停止、原因確認
- 当日 -5,000 円 (合算) → 翌日も停止、原因究明

### 3.3 model 健全性 monitor

- 週次 AUC validation (新データ 100 race+)
- AUC < 0.75 (NAR) / < 0.83 (V15) → モデル再学習 + 投資停止

### 3.4 オッズ取得失敗時

| 失敗内容 | 対応 |
|---------|------|
| 1 race odds_log 取得不可 | 該当 race 見送り (predict skip) |
| 1日全体で odds 取得 80% 以上失敗 | 当日 NAR 停止 |
| netkeiba.nar 全失敗 | 楽天競馬 fallback (実装は別 task) |

---

## 4. 5/16-5/24 試行プラン (詳細)

### 4.1 5/12-5/15 (火-金): paper trading

| 日付 | アクション | 投資 | 期待 |
|------|-----------|------|------|
| 5/12 (火) | NAR 推論+Discord 通知のみ、investment 0 | 0 | pipeline 動作確認 |
| 5/13 (水) | 同上 | 0 | 通知精度 確認 |
| 5/14 (木) | 同上 | 0 | 取消対応 確認 |
| 5/15 (金) | 同上 + 5/16 投入判断 | 0 | go/no-go |

**判断基準** (5/15 夜):
- 5/12-5/15 paper retro で条件 A/D の prediction 精度 (winner_top1 rate ≥ 30%) → go
- pipeline 1日 1 失敗以下 → go
- どちらか不成立 → 5/16 NAR は no-go、V15 単独継続

### 4.2 5/16-5/17 (土日): JRA 単独 + NAR 試行

| 日付 | JRA | NAR |
|------|-----|-----|
| 5/16 (土) | V15 案B改 (継続、~9,800 円) | 試行 500 円/日 (条件 A/D に絞る) |
| 5/17 (日) | 同上 | 5/16 結果次第 (好調なら続行) |

**stop**: 5/16 NAR -1,000 円 → 5/17 NAR は no-go。

### 4.3 5/18-5/22 (月-金): NAR 専業

| 日付 | NAR |
|------|-----|
| 5/18 (月) | 大井 etc. 500 円/日 |
| 5/19 (火) | 名古屋 + 船橋 等 |
| 5/20 (水) | 同上 |
| 5/21 (木) | 同上 |
| 5/22 (金) | 同上 + 週判断 |

**週累計判断** (5/22 夜):
- 週累計 +500 円以上 → 5/23-5/29 は 1,400 円/日 へ ramp
- 週累計 ±0 〜 -1,500 円 → 500 円/日 で継続観察
- 週累計 -1,500 円超 → NAR 停止、原因究明

### 4.4 5/23-5/24 (土日): JRA + NAR 並列

| 日付 | JRA | NAR |
|------|-----|-----|
| 5/23 (土) | V15 案B改 (継続) | 試行継続 (前週末判断による) |
| 5/24 (日) | 同上 | 同上 |

### 4.5 5/25 (月) Phase 3 移行判断

| 累計 (5/12-5/24) | アクション |
|-----------------|-----------|
| NAR ROI > 100% かつ累計 +1,000 円以上 | **5/26 から NAR 1,400 円/日 ramp**、Phase 3 v20 設計開始 |
| NAR ROI 80-100% | 500 円/日 で継続、6 月再判断 |
| NAR ROI < 80% かつ累計 -3,000 円以上 | **NAR 完全停止**、Phase 3 v20 で根本対応 |

---

## 5. 結論

| 短期 (5/16-5/24) | 中期 (6 月) | 長期 (Phase 3) |
|-----------------|-----------|----------------|
| **並列運用 + 段階 ramp** | NAR ROI に応じて投資配分変動 | **v20 統合モデル** 学習 + 単一 pipeline 化 |
| 累計 -50,000 円 死守 | JRA + NAR 累計 monitor | features 拡張 (52+) |
| 5/12 paper → 5/16 go/no-go | week 単位 ramp | predict_v20.py で完全置換 |

**重要原則**: V15 案B改 (累計 +14,140 円) を絶対に毀損しない。NAR は補助、リスク isolate。

---

## 6. 関連 doc

- `data/v18/nar_v4_current_state.md` — NAR v4 model + データ現状
- `data/v18/nar_pipeline_design.md` — 自動化 pipeline 設計
- `data/v18/nar_v4_backtest_5_5.md` — backtest AUC 再現
- `data/v18/nar_schtasks_user_guide.md` — schtasks 登録手順
- `data/v18/v18_v19_integration_plan_5_4_pm.md` — JRA v18/v19 5/16 投入プラン (本 plan と並列)
