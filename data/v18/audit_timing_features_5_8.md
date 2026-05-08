# AUDIT-1 G: 取得 timing 別 features audit (5/8)

**作成**: 2026-05-08 (AUDIT-1 G 領域)
**前提**: V15 は Pattern A (前日) + Pattern B (直前情報込み) の 2 段階 model
**位置付け**: 二段階 / 三段階予測 system 設計の base

---

## 1. Tier 定義

| Tier | timing | 取得可能 source |
|------|--------|----------------|
| 0 | いつでも (過去 data) | TFJV 全 historical / netkeiba 全 db / JRDB 過去 |
| 1 | 朝 6:00-8:00 (前日まで data) | jrdb_kyi 朝 / netkeiba master 朝 / TFJV 前日 ES |
| 2 | レース 70 分前 (当日体重発表) | netkeiba 出馬表 当日 / horse_weight |
| 3 | レース 30 分前 (パドック) | パドック (静止画 / 動画) / 当日 odds |
| 4 | レース 5 分前 (確定オッズ) | 確定 odds / 馬場 |
| 5 | レース後 | 結果 / SED / HR |

---

## 2. Tier 別 V15 features 利用状況

### Tier 0 (いつでも) - V15 fully 使用

| feature | source |
|---------|--------|
| 全 expanding window features (jockey_wr / sire_*_wr 等) | TFJV historical |
| 全 horse_career_* | TFJV historical |
| 全 sib_*_exp | netkeiba_siblings 過去計算 |
| sire_enc / bms_enc / location_enc | TFJV |
| Tier 0 全部: V15 で 約 60 件 利用 |  |

### Tier 1 (朝、 前日まで) - V15 fully 使用

| feature | source |
|---------|--------|
| jrdb_kyi 22 features (前日 17:00 公開) | JRDB KYI 朝 |
| paci 11 features (Tier A 7 + Tier B 4) | JRDB PACI |
| sed_features 8 (前走) | JRDB SED |
| netkeiba speed_index 4 件 | netkeiba 朝 |
| netkeiba training_times 5 件 | netkeiba 朝 |
| Tier 1 全部: V15 で 約 50 件 利用 |  |

### Tier 2 (70 分前、 当日体重発表) - **V15 ほぼ 未使用**

| feature 候補 | source | V15 組込 |
|------------|--------|--------|
| horse_weight (当日体重) | netkeiba shutuba | ⚠️ Pattern B のみ |
| weight_change (馬体重変化 前走比) | netkeiba shutuba | ⚠️ Pattern B のみ |
| weight_cat / weight_cat_dist (派生) | 派生 | ⚠️ Pattern B のみ |
| condition_enc (馬場状態 朝) | JRA 公式 | ⚠️ Pattern B のみ |
| moisture_rate / cushion_value (馬場詳細) | JRA 公式 | ⚠️ Pattern B のみ |
| weather_enc (天候) | 気象庁 API | ⚠️ Pattern B のみ |

→ Tier 2 は **Pattern B (10 features) のみ**、 Pattern A (学習評価) には 未使用 (リーク扱い)

### Tier 3 (30 分前、 パドック) - **V15 完全 未使用** ★

| feature 候補 | source | V15 組込 |
|------------|--------|--------|
| パドック static 体格 score | 画像解析 | ❌ |
| パドック 動画 歩様 | 動画解析 | ❌ |
| 当日 odds 中間 | netkeiba | ❌ |
| 厩舎コメント 直前 | scrape | ❌ |
| jrdb_tyb 5 件 (paddock_idx / odds_idx 等) | JRDB TYB | ⚠️ Pattern B のみ |
| jrdb_tyb bagu_change / ashimoto / cancel_flag | JRDB TYB | ❌ |
| jrdb_jo soten/yoso_odds (基準オッズ) | JRDB JO | ❌ |

### Tier 4 (5 分前、 確定オッズ前) - **V15 部分使用**

| feature 候補 | source | V15 組込 |
|------------|--------|--------|
| 確定オッズ (odds_log) | netkeiba | ❌ POST-RACE LEAK 確定で 完全除外 |
| 人気 (pop_rank) | netkeiba | ⚠️ Pattern B のみ |
| 馬連 / 三連複 オッズ | netkeiba | ❌ |
| 取消馬 反映 | scrape | ⚠️ tools/predict_one_race.py で 手動 |

---

## 3. 二段階 / 三段階予測 system 設計 base

### 3.1 現状: 2 段階 (Pattern A 学習 + Pattern B 実運用)

- **Pattern A (リークフリー)**: Tier 0 + Tier 1 のみ → 学習 / 評価
- **Pattern B (実運用)**: Tier 0 + Tier 1 + Tier 2 (一部) → 当日予測

### 3.2 提案: 3 段階 / 4 段階予測 system

#### 3 段階予測

| 段階 | timing | 用途 | 入力 | 出力 |
|------|--------|------|------|------|
| 1 | 朝 8:00 (DailyPredict) | 投票 計画 | Tier 0 + 1 | スコア (主軸) |
| 2 | 70 分前 | 体重 補正 | + Tier 2 | スコア 微調整 (馬体重 急変 検知) |
| 3 | 5 分前 | 最終 | + Tier 4 (オッズ) | EV 計算 + 投資 確定 |

→ tools/morning_weight_check.py が 段階 2 候補 (既実装)
→ 段階 3 (オッズ反映) は **未実装** (Sprint 4 候補?)

#### 4 段階予測 (Phase 4 想定)

| 段階 | timing | 用途 |
|------|--------|------|
| 1 | 朝 8:00 | 主軸 |
| 2 | 70 分前 | 体重 補正 |
| 3 | 30 分前 | パドック画像 補正 (Phase 4) |
| 4 | 5 分前 | オッズ + EV 確定 |

---

## 4. Tier 別 V15 未使用 features list

### Tier 1 未使用 (即実装可)

- master_index 5 件 ★★★
- ai_opinion pace ★★
- track_bias ★★
- ai_position 位置取り ★
- jrdb_jo cid/ls/em/gaisha_bb/breeder_bb 9 件 ★★
- jrdb_kka 12 group × 4 = 48 件 ★★
- jrdb_cha oikiri 9 件 ★★
- jrdb_srb bias 6 件 ★★★
- 派生 (jockey_horse_recent_3 等) 5-8 件 ★

### Tier 2 未使用

- weight_change_trend (3 走 馬体重 trend) - 派生
- 馬体重 absolute (今走) - Pattern B 既組込

### Tier 3 未使用

- jrdb_tyb 5 残 - Pattern B 限定
- パドック static 体格 score - Phase 4
- パドック動画 歩様 - Phase 4
- 厩舎コメント 直前 - 既存 stable_comment_score 拡張

### Tier 4 未使用 / リーク risk

- 確定オッズ - リーク (使わない)
- 確定 配当 expected_value - リーク
- 馬連 odds - リーク
- 三連複 odds - リーク
- 三連単 odds - リーク

→ Tier 4 features は **EV 計算 (post-prediction)** で 使用、 model 入力 では 使わない

---

## 5. 取得 timing 整理

| Tier | features 候補 数 | V15 利用数 | 未使用数 |
|------|---------------|----------|--------|
| 0 (historical) | ~80 | ~60 | ~20 |
| 1 (朝、 前日まで) | ~80 | ~50 | ~30 |
| 2 (70 分前) | ~10 | 0 (Pattern A) / 6 (Pattern B) | ~4 |
| 3 (30 分前) | ~10 | 0 / 5 (Pattern B 限定) | ~5 |
| 4 (5 分前) | ~5 | 0 (リーク扱い) | (使用しない) |

**Tier 1 で 約 30 件 未使用** が 最大の改善 機会 (Sprint 4 / V20)

---

## 6. 推奨 system 設計

### 6.1 短期 (Sprint 4): Tier 1 features 拡張

- master_index 5 件 (即) → Tier 1 で 朝 取得済 + 学習 評価
- ai_opinion pace + track_bias 4 件 (即)
- jrdb_srb bias 6 件
- 期待: V15 0.886 → 0.894 (+0.008)

### 6.2 中期 (V20、 6/8): Tier 1 features 大量追加 + TFJV 90 年

- KKA / CHA / JO / 残 KYI / netkeiba master 全 + TFJV BR/BS/BN
- 期待: 0.894 → 0.896-0.900 (+0.002-0.006)

### 6.3 長期 (V21、 9/1): Tier 3 動画 features

- パドック画像 + 動画 歩様 features 5-7 件
- 期待: 0.900 → 0.905-0.910 (+0.005-0.010)

### 6.4 並行: 段階 3 (オッズ EV 反映 5 分前)

- model 入力 ではなく、 buy 判定 で 使用 (現状 race_auto_notify で 一部実装)
- 拡張: 5 分前 オッズ で 三連複 EV 再計算

---

## 7. 5/9 V15 投資保護

✅ Tier 1 / Tier 2 取得 timing 不変
✅ daily_predict 朝 8:00 / morning_weight_check 70 分前 のままで 動作
✅ 段階 3 (オッズ EV 反映) は 5/9 では race_auto_notify 既実装版を そのまま使用

---

## 8. 結論

✅ Tier 0-4 別 V15 features 利用状況 整理
✅ Tier 1 (朝) で **約 30 件 未使用** が 最大の改善 機会
✅ Tier 3 (パドック) は Phase 4 候補
✅ 二段階 → 三段階 (体重 補正 + オッズ EV 再計算) 設計 base 完成
