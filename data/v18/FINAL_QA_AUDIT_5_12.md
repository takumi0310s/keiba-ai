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

## 📋 加入 source 完全 inventory (Agent 2 完了)

### TOP 10 不足 items (V20-V22 inject 候補、 priority 順)

| # | item | source | priority |
|---|------|--------|---------|
| 1 | **JV-Link DIFF マスター一括** | JRA-VAN | ★★★ V20 学習 base 強化 |
| 2 | **JV-Link 0B41/0B42 オッズ時系列 (1年)** | JRA-VAN | ★★★ V20 オッズ shift |
| 3 | **JV-Link MING 公式 DM 予想** | JRA-VAN | ★★ ensemble 第 5 model |
| 4 | **JV-Link 0B20 出走取消 速報** | JRA-VAN | ★★★ 朝 fire-check critical |
| 5 | **RV パトロール ビデオ** | JRA RV | ★★★ 不利検知 V22 RL |
| 6 | **RV パドックアイ AI 分割 + 歩様** | JRA RV | ★★★ Phase 4 比較 baseline |
| 7 | **netkeiba タイム指数マスター + start/middle/上がり 個別** | netkeiba master | ★★ V20 LGB +5 features |
| 8 | **netkeiba 期待値シミュレーション + AI 印** | netkeiba master | ★★ cross-check |
| 9 | **JRDB KTA / MZA / MSA** | JRDB | ★ 出走確定 + 抹消検知 |
| 10 | **TFJV JG/EX/TM/UM_DATA** | TFJV | ★ 取消 + base time + ID 補強 |

### 各 source 別 詳細

#### 1. netkeiba マスターコース (¥4,980/月、 25/25 主要 取得済)
**不足 7 items**:
- タイム指数マスター 上位モデル (重量補正)
- スタート/追走/上がり 区間別 指数
- 期待値シミュレーション
- AI 印 (◎○▲△ 直接値、 expert_marks には部分のみ)
- 走行解析 score
- UMAI ビルダー 出力 (cross-check 用)
- 9 AI parameters の生値 (気性 / 毛色 / 馬名文字数 / 画数 / 誕生月 / blinker)

#### 2. JRDB Advance (¥2,000/月、 15/18 主要 取得済)
**不足 3 items**:
- KTA (登録馬、 火曜先行) ← 出走確定 検知に 重要
- MZA/MSA (抹消馬) ← keep 馬 判定
- CZA / KZA 全件マスター (差分のみで base なし → 累積 error source)
- PACI 4/4 停止 (既知 bug、 要修復)

#### 3. JRA-VAN DataLab JV-Link (¥2,090/月、 8/30+ 実装)
**不足 15+ dataspec**:
- **DIFF** (馬/騎手/調教師/血統/累計 master 一括) ★★★
- **MING** (DataMining 公式予想)
- **SLOP** (坂路調教 公式)
- HOSE/HOYU (市場価格 / 馬名由来)
- COMM/YSCH/TOKU (コース / 開催 / 特別登録)
- 速報系 **全部**: 0B11 (馬体重) / 0B14 (馬場) / 0B15 (確定) / **0B20 (取消)** / 0B30-36 (オッズ各式) / **0B41/0B42 (時系列)**
- MV (動画 API)

#### 4. JRA レーシングビュアー (¥550/月、 5/15 取得済)
**不足 5+ 重要 items**:
- **パトロール ビデオ** (不利検知、 ゴール後 40分公開) ★★★
- **マルチカメラビュー** (重賞 のみ、 3 画面同時、 位置取り解析)
- **パドックアイ AI 分割動画 + 歩様解析** (発走 20 分前) ← Phase 4 V22 比較 critical
- GI ホースライブラリー (1984+、 V22 RL pretrain data 巨大)
- 重賞レビュー / ダートグレード / GI 特集

#### 5. TARGET frontier JV (個人 license、 4/24 抽出済)
**未抽出 7 datatype**:
- **JG_DATA** (競走除外 / 騎乗変更) ← 0B20 と被るが 蓄積
- **EX_DATA** (拡張データ)
- **TM_DATA** (タイム関連、 ベースタイム)
- **UM_DATA** (馬個体 master、 ID 紐付け強化)
- **DE_DATA** (出馬表 蓄積)
- **W5_DATA** (WIN5)
- **TXT/target_sakaro.csv** (公式坂路、 現状 netkeiba 経由)

### 競合 vs 我々 vs 不足 統合 summary

**我々の優位**:
- Phase 24 新 features (class_down / hot_streak / Jackpot pattern / sire boost) = 競合公開情報になし
- 動画 AI 自前構築 (paddock + race + body condition + gait) = 先行領域
- 4-ensemble + IntraRace Attention = 競合の中で上位
- 戦略⑦ 条件分類 + 黄金 pattern = unique

**不足で 真に効くもの (Phase 24 marathon 中 verify):**
- jockey_trainer_combo_wr +21.3pt (実装済)
- corner_position_delta +10.2pt (実装済)
- prev_race_disadvantage_score (NLP scoring 実装済)

**5/24+ V20 投入 で 追加すべき**:
- start/middle/上がり 個別 指数 (netkeiba)
- jockey_trainer combo, corner_delta は 既実装
- bagu_change (JRDB) - 詳細解析 必要
- JV-Link DIFF + 0B20 + 0B41 (V20 base 強化)

**5/24+ 〜 V21 で**:
- パトロールビデオ AI 解析
- パドックアイ AI 歩様 cross-check

**V22+ RL で**:
- GI ホースライブラリー 1984+ pretrain

### JV-Link DataSpec 既実装 vs 不足 (我々 中間検証)

**我々 実装済 (jvlink_parser.py 8 dataspec)**:
- RACE / SE / HR / UM / BLOD / WOOD / TCOV / O1

**JV-Link 公式 30+ dataspec のうち 我々 不足**:
| dataspec | 内容 | priority |
|----------|------|---------|
| **O2** | 馬連 オッズ (LIVE) | ★★★ Pari-mutuel optimizer 必要 |
| **O3** | 馬単 オッズ | ★★ |
| **O4** | Wide オッズ (LIVE) | ★★★ (Wide ROI 178% 確認済) |
| **O5** | 三連複 オッズ | ★★★ trio bet base |
| **O6** | 三連単 オッズ | ★★ |
| **WH** | 馬体重 LIVE | ★★★ Phase 21A morning_weight_check と統合 |
| **WE** | 天気 / 馬場 LIVE | ★★★ |
| **SC** | 出走馬 確定 (LIVE) | ★★ |
| **MVID** | Movie type API | ★★ |
| **MING** | DataMining 公式 予想 | ★ |
| **DM** | ダート / 芝 mining | ★ |
| **TOKU** | 特別レース | ★ |
| **CHAR** | 字幕 | - |
| **RC** | レコード | - |
| **JC** | 騎手 変更 (LIVE) | ★★ jockey_change LIVE |

→ jvlink_parser.py に **15+ dataspec 追加** で full coverage、 特に O2-6 / WH / WE / SC は 5/24+ V20 投入時に critical。

Agent 2 完了次第 詳細 inventory 追記。

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
