# FINAL QA Audit (5/12 朝、 5/16 までの 5/17 準備 verify)

## 📊 既存 data inventory (rows / 鮮度)

### JRDB Advance (¥2,000/月、 取得済 34 type、 6.5M rows)
| type | rows | size | 状態 |
|------|------|------|------|
| kka_v2 | 549K | 191MB | ✅ 最新 (1d) |
| paci | 549K | 140MB | ✅ |
| tyb | 548K | 56MB | ⚠ 8.8d 古い |
| sed | 548K | 116MB | ⚠ 8.8d 古い |
| skb | 547K | 154MB | ⚠ 8.8d 古い (POST-RACE LEAK) |
| ze | 537K | 114MB | ⚠ 古い (前走) |
| zk | 530K | 152MB | ⚠ 古い (前走 直近) |
| kta | 299K | 63MB | 古い |
| kyi | 292K | 92MB | ✅ 2.0d (重要情報) |
| jo | 302K | 26MB | 古い (騎手 ?) |
| cha | 302K | 27MB | 古い (調教情報) |
| ukc | 37K | 8MB | 古い (馬基本) |
| 他 20+ type | - | - | - |

**結論**: 34 type 全部 取得済、 多くは V15 学習で 未使用の可能性 (V15 124 features は KYI / SED / TYB 中心)。

### netkeiba マスターコース 取得済 csv (15 種類)
| csv | total | 2026 取得 | 状態 |
|------|------|---------|------|
| netkeiba_race_review | 277K | **0** | 🚨 1.5 ヶ月停止 |
| netkeiba_speed_index | 271K | 1,008 | ✅ 動作中 |
| netkeiba_stable_comments | 130K | 1,441 | ✅ 動作中 |
| netkeiba_training_times | 301K | **0** | ⚠ 古い |
| netkeiba_training_eval | 302K | 0 | ⚠ |
| netkeiba_master_index | 140K | **0** | 🚨 2026 取得停止 |
| netkeiba_ai_position | 68K | 0 | ⚠ |
| netkeiba_ai_opinion | 5K | 0 | ⚠ |
| netkeiba_ana_best | 42K | 0 | ⚠ |
| netkeiba_track_bias | 27K | 0 | 🚨 2026 全部 取れず |
| netkeiba_race_lap | 26K | 0 | 🚨 同上 |
| netkeiba_track_index | 21K | 0 | ⚠ |
| netkeiba_upset_level | 37K | 144 | ⚠ 5/3 頃まで |
| netkeiba_shinba_eval | 8K | 0 | ⚠ |

**結論**: 多くの netkeiba scrape が **2026 完全 未取得** or 取得停止。 V15 daily_predict は LIVE で 動的 取得してるが、 蓄積 csv が不完全 → features 蓄積 漏れ。

### TFJV / TARGET (個人 license)
- jra_races_full.csv: 532K rows、 7.2 d 古い、 2026/5/4 頃まで → **5/11 races 含まれず**
- jra_payouts.csv: 12K rows、 4/6 から 停止 (公式 DB ページ構造変更)

### JRA レーシングビュアー (¥550/月)
- prc.jp Web access 確認済 (login 必要)
- 動画 source: 全レース / 調教 / GI / ダートグレード / マルチカメラ
- パドック映像: **締切 15 分前 公開** = LIVE 予測 R-5 分前 に間に合う可能性

## ✅ 取得テスト 既実施 (5/11 marathon)

| test | result |
|------|--------|
| paddock 動画 capture | ✅ 26 frame 鮮明 (ウインイザナミ) |
| race 動画 capture | ✅ 18 frame 鮮明 (馬群 gallop) |
| YOLOv8 馬 bbox | ✅ paddock 100% / race 94% |
| gait features | ✅ 20 features 抽出 |
| body condition | ✅ score 0.717 |
| oikiri probe | 🟡 video element 未表示 (popup modal click 必要) |
| JV-Link COM | 🔴 64-bit Python 不可、 32-bit 必須 (user 5/12) |
| JRA RV web login | 🔴 未 test (user 5/12-15) |
| JRDB API | 🟡 自動 schtask 動作中 (DailyJrdbKyi 06:00) |

## 🚨 5/16 までに 必要 action (critical)

### user task (5/12-5/15)

#### 5/12 (月)
```bash
# 1. TFJV → jra_races_full.csv 更新 (admin、 TARGET 起動 後 extract)
python tools/extract_jvdata.py
# 結果: 5/11 races まで含まれた jra_races_full.csv

# 2. schtask 登録 (admin PowerShell)
python tools/register_all_phase24_schtasks.py

# 3. JV-Link 32-bit 動作確認
C:\Users\takum\jvlink-venv\Scripts\activate.bat
pip install pywin32
python tools/jvlink_parser.py --test-com
python tools/jvlink_movie_wrapper.py --probe --race-id 202603010112 --horse-id 2022106229

# 4. JRA RV web login + probe
python tools/jra_racing_viewer_capture.py --probe
# .env に JRA_RV_LOGIN_ID / JRA_RV_PASSWORD 追加

# 5. JRA 払戻 復活 試験 (Phase 22 Agent C)
python tools/scrape_jra_payouts_v2.py --dry-run
```

#### 5/13-14 (火-水) - 重要な不足 data 補完
```bash
# netkeiba 2026 不足 csv 取得 (★★★ critical)
python tools/scrape_master_index.py --year 2026
python tools/scrape_race_review.py --year 2026
python tools/scrape_master_course.py --year 2026
python tools/scrape_ai_opinion.py --year 2026  # 既存あれば

# 厩舎コメント / 専門家 印 scrape
python tools/bulk_scrape_stable_comments_v2.py --year-from 2024 --year-to 2026
python tools/bulk_scrape_expert_marks.py --year-from 2024 --year-to 2026

# JRDB 古い type を最新化
python tools/daily_jrdb_kyi.py  # 既存 schtask、 強制実行
```

#### 5/14-15 (水-木)
```bash
# features 全 rebuild (5/11 races 反映)
python tools/rebuild_all_features.py
# 出力: 8 features csv 全更新

# paddock 過去 archive build (V21 用 蓄積)
python tools/paddock_weekend_archive_build.py 20260504 20260510 --top-n 3
# 結果: 約 216 動画
```

#### 5/16 (金) 夜 - 5/17 準備 final
```bash
# 1. 全 chain rehearsal
python tools/rehearsal_5_17.py

# 2. 5/17 用 shadow runner verify
python tools/strategy8_shadow_runner.py 20260517

# 3. health check
python tools/check_video_sources.py
python tools/morning_briefing_5_17.py
```

#### 5/17 (土) 開催
- 06:30 morning_briefing 自動 (schtask)
- 08:00 V15 daily_predict 自動
- 08:55 YouTube LIVE 録画 自動
- 09:00 朝 user 手動 step:
  ```bash
  python tools/strategy8_shadow_runner.py 20260517
  # Jackpot 該当馬 確認 → 手動 単勝 1500 円 追加 (任意)
  ```

## 🟡 取得 漏れ items (5/11 marathon で 確認した gaps)

### 1. paddock 動画 自動取得 pipeline (V21 学習 data 蓄積)
- 5/4-5/10 archive まだ取得してない
- 5/12-5/14 で paddock_weekend_archive_build.py 実行 必要

### 2. JRA-VAN MovieType API (調教動画 / レース動画 公式)
- jvlink_movie_wrapper.py skeleton 実装済、 32-bit Python での 実 test 未

### 3. JRA レーシングビュアー Web (公式動画 SD)
- prc.jp top access 確認、 login flow 未実装
- .env に JRA_RV_LOGIN_ID 追加 → DOM 構造 確認 後 capture 実装

### 4. netkeiba 専門家予想 / AI 印 (14 件のみ)
- bulk_scrape_expert_marks.py 実装済、 実 scrape 未実行

### 5. netkeiba 短評 (43 日古い)
- scrape_race_review.py 動作再開 必要

### 6. JRA 公式 払戻 (4/6 停止)
- scrape_jra_payouts_v2.py で 復旧 path 構築済、 実 試験 必要

### 7. アメダス 1 分粒度
- scrape_amedas_1min.py skeleton 実装済、 実 API 動作確認 未

### 8. 30 年 backtest data
- TFJV 1995-2024 collector skeleton (Phase 22 Agent A)
- 容量 135 GB、 段階取得必要

## 📋 他社 AI 比較 (Agent 1 完了)

### 真に不足してる features (15 件、 priority 順)

#### 高 priority (即実装可能、 既存 data 流用、 1-2 週間)
1. **prev_race_disadvantage_score** — race_review.csv 277K rows 既存、 NLP/scoring のみ
2. **corner_position_delta** — prev_pass1 - prev_pass4 の単純差分 (既存 data 派生)
3. **jockey_trainer_combo_wr** — jockey × trainer pair expanding wr (既存 cross)
4. **bagu_change_flag** — 馬具変更 (ブリンカー初装着/解除) — JRDB 既存

#### 中 priority (V20 学習 5/25-6/13 で 同梱)
5. **start_index / middle_index** — 序盤 / 中盤 lap 別 speed (現状 last3f のみ)
6. **bms_shinba_top3r** — 母父 新馬戦 複勝率 (sire_shinba と対称)
7-9. **kireaji / suriko / pace_adapt career_avg** — career expanding 切れ味/末脚/ペース順応
10. **LPI / speed_gradient_E_M_L** — ラップ偏差指数 / 序盤中盤終盤 加速差
11. **PCI3 / RPCI** — 別 pace index (現 pci のみ)

#### 低 priority (V21+ 検討)
12. **h2h_winrate** — 同race 馬同士の past 対戦勝率 matrix (IntraRace Attn と重複可能)
13. **chakusa_score** — 期待 着差 vs 実 着差
14. **condition_specialist sub-model** — SPAIA 流 18 model 分割 (我々 6 条件分類で代替可)
15. **prev_pop_relative** — 人気 相対 (我々 popularity あり)

### 競合 AI 別 unique features (我々 未実装)

| 競合 | unique feature |
|-----|---------------|
| SPAIA | 18 model 条件別 specialist ensemble |
| netkeiba AI Master | start/middle 区間別 speed index、 不利フラグ score |
| UMAJIN.net | jockey × trainer combo / bms 新馬 |
| 学術 (arXiv) | head-to-head matrix |
| JRDB IDM 解説 | 累積平均 (切れ味 / 末脚 / ペース順応 expanding) |
| loots LPI | 速度勾配 / ラップ偏差 |

### 重要 finding (Agent 1 所見)

**Phase 24 新 features (class_down / hot_streak / Jackpot pattern / sire × class_down) は競合公開情報に 一切登場せず** = 我々の独自優位性。

競合 中核 = 伝統的 speed index + pedigree + lap analysis
我々 = hot_streak / class_down / Jackpot pattern + 動画 AI (paddock+race) = **先行領域**

真に効きそうな不足 (低 cost で 大 win):
- #1 prev_race_disadvantage_score (race_review 既存活用)
- #2 corner_position_delta (派生のみ)
- #4 jockey_trainer_combo_wr (cross expanding)
- #5 bagu_change_flag (JRDB 既存活用)

## 📋 加入 source 取得可能 data inventory (Agent 結果 待ち)

Agent 2 (4 source inventory) 完了次第 ここに 追記。

## 結論 (現状)

**5/17 までに 確実 必要 action**:
1. ✅ schtask 登録 (5/12 admin、 15 min)
2. ✅ TFJV → jra_races_full.csv update (5/12 user、 30 min)
3. ✅ JV-Link 32-bit 動作確認 (5/12 user、 30 min)
4. ✅ 全 features rebuild (5/12-13、 1h)
5. ✅ JRA RV login + probe (5/12-15、 30 min)
6. 🟡 netkeiba 2026 不足 csv 補完 (5/13-14、 1-2h)
7. 🟡 paddock 過去 archive build (5/13-14、 1-2h、 V21 用)
8. 🟡 専門家予想 / 厩舎コメント 実 scrape (5/14、 2h)

**5/17 開催 状態 (予測)**:
- V15 案 B 改 + 戦略⑦ 単独継続 = ✅ 確実
- Jackpot pattern shadow log = ✅ features rebuild 後 動作確実
- Jackpot 手動 単勝 追加 = 🟡 user 判断 (試験運用)

**真の features / data 取り漏れは Agent 結果 で判明**、 5/12 中に追記。
