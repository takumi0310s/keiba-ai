# Phase 3 全 4 source 役割分担設計 (Session #39 D)

**作成**: 2026-05-07 (Session #39 D)
**目的**: 4 source (netkeiba / JRDB / NAR scraping / JV-Link) の重複排除 + 役割分担を確定し、 V20 学習 data 戦略を構築

---

## 1. 現在の data sources

| # | source | 種別 | 月額 | 信頼性 | 取得 channel |
|---|--------|------|------|--------|------------|
| 1 | netkeiba | scraping | 4,500円 (Premium) | 高 (要 cookie) | `requests` + Premium login |
| 2 | JRDB Advance | 加入 | 約 2,000円 | 中-高 (時々停止) | FTP-like download |
| 3 | NAR (chihou) scraping | scraping | 0 | 中 (要監視) | `requests` |
| 4 | JV-Link (5/24+) | 公式 | 2,090円 | **最高** (公式) | COM API |

合計 月額: 約 8,590円 (5/24+)、 V20 ROI 140% 想定で月利 +5万〜10万円見込で十分カバー

---

## 2. 各 source の強み + 弱み

### 2.1 netkeiba (Premium)

**強み**:
- speed_index (1986+) — 最強の自前指数、 V18/V19 で高 importance
- training_eval (調教評価) — A/B/C/D ランク
- パドック評価
- newspaper_ai_thisweek — netkeiba AI による週末予想
- race_lap (ハロンタイム)、 track_bias (トラック偏り)、 upset_level
- horse 詳細ページ (血統、 兄弟、 厩舎コメント)

**弱み**:
- BAN リスク (cookie 失効、 IP block)
- HTML 構造変更で scraper 故障リスク
- レスポンス遅延 (混雑時 30s+)

**用途**:
- ✅ speed_index / training_eval は他 source 代替不可 → **継続使用**
- ✅ パドック / 新馬評価 / track_bias → 補助データ
- ⚠ jra_payouts (現在 4/6 停止中) → JV-Link 切替 (5/24+)

### 2.2 JRDB Advance

**強み**:
- 26 種データ ファイル (BAC/CYB/SED/SRB/UKC etc.)
- `paci.csv`: 騎手期待値、 人気 idx、 展開予想 (V18/V19 importance #1〜#3)
- `kyi.csv`: PRE_RACE features (75.9% 結合率)
- `tyb.csv`: LIVE features (当日朝発表)
- `sed.csv`: PREV race detail (要修復、 0% 結合率)

**弱み**:
- 4/4 から `paci.csv` 取得停止 (取得経路不明、 V18/V19 winner_top1 -13.3pt の主因 #1)
- SED csv 破損 (PREV データ取れず)
- 各 file の更新遅延 (2026年分が一部未取得)

**用途**:
- ⚠ paci_*: V18/V19 復活には必須、 5/24+ JV-Link オッズ自前 paci 計算で代替検討
- ✅ kyi.csv: V15 で重要、継続
- ⚠ sed.csv: 修復 (V18/V19 復活前提) or 廃止判断、 5/24+ で決定
- ✅ skb.csv: **V20 では完全除外** (Session #38 LEAK 確定)

### 2.3 NAR (chihou) scraping

**強み**:
- 地方競馬全データ (JRA-VAN は JRA のみ)
- NAR v4 学習 data 元 (約 5 万 horse)
- 大井/船橋/川崎/浦和/名古屋/兵庫等

**弱み**:
- BAN リスク (netkeiba と同じ)
- 中央競馬と format 異なる (race_id / horse_id 区別必要)
- 地方所属馬の中央移籍時の連結 (`location_enc=2`)

**用途**:
- ✅ V20 NAR subset 学習 data → **継続使用** (代替なし)
- ✅ 地方→中央 transfer 馬の血統情報

### 2.4 JV-Link (5/24+ 加入)

**強み**:
- **公式 = 100% 正確**、 BAN リスク 0
- 全 JRA レース 1986+ 提供 (約 30 万レース)
- 払戻データ (HR) → jra_payouts.csv 復活
- リアルタイム オッズ (10 分毎) → 自前 paci 算出可能
- 馬体重 (WF) → condition encoding 公式値
- 調教 (TCOV/WOOD) → netkeiba と相補

**弱み**:
- Windows COM 依存 (Linux/Mac で動かない)
- 固定長 record format → 自前 parser 必要
- 加入手続き + DLL インストール工数 (1h)

**用途**:
- ✅ 過去レース結果 + 払戻 → V20 学習 data 主軸
- ✅ オッズ速報 → 自前 paci 復活、 jrdb_paci 代替
- ✅ 馬体重 → Pattern B 当日情報
- ✅ 調教 → netkeiba と統合 (公式 > netkeiba 優先)

---

## 3. 重複データの優先順位

### 3.1 過去レース結果

| データ | 旧 主軸 | 新 主軸 (5/24+) | 補助 |
|--------|--------|----------------|------|
| race 詳細 (date/距離/馬場/race_name) | netkeiba | **JV-Link** | netkeiba (race_lap など) |
| 着順/タイム/上がり | netkeiba | **JV-Link** | JRDB sed (修復後) |
| 払戻金 | scrape_jra_payouts (停止中) | **JV-Link HR** | netkeiba |

→ V20 では `jra_races_full.csv` を **JV-Link → parser → 新 master** に置き換え。

### 3.2 血統

| データ | 旧 主軸 | 新 主軸 | 補助 |
|--------|--------|--------|------|
| 父/母/母父 | netkeiba blood_full | **JV-Link BLOD** | netkeiba (補完) |
| 兄弟成績 | netkeiba_siblings (リーク) | **sib_expanding (Session #39 A)** | — |

### 3.3 騎手・調教師成績

統合戦略: **全 source merge → expanding window で集計**
- netkeiba: jockey_horse 履歴
- JV-Link SNPN: 公式騎手・調教師
- 学習 features (jockey_wr_calc, jockey_horse_top3r 等) は date < race_date で expanding

### 3.4 調教

| データ | 旧 主軸 | 新 主軸 | 補助 |
|--------|--------|--------|------|
| 木馬場 4F best | netkeiba | **netkeiba** (継続) | JV-Link WOOD (補完) |
| 坂路 best | netkeiba | **netkeiba** (継続) | JV-Link TCOV (補完) |
| 調教評価 (A/B/C/D) | netkeiba | **netkeiba** (代替なし) | — |

### 3.5 オッズ

| データ | 旧 主軸 | 新 主軸 (5/24+) | 補助 |
|--------|--------|----------------|------|
| 単勝/複勝 | netkeiba | **JV-Link O1** | netkeiba |
| 馬連/三連複 | netkeiba | **JV-Link O2/O5** | netkeiba |
| 基準オッズ (前日朝) | jrdb_kyi | **JV-Link O1 早朝取得** | jrdb_kyi |
| paci (騎手期待値) | jrdb_paci (停止) | **自前 paci (JV-Link オッズから算出)** | — |

### 3.6 払戻

旧: scrape_jra_payouts.py (4/6 停止中)
新: **JV-Link HR** で完全代替 (即時、 確実)

### 3.7 馬場 (track condition)

| データ | 旧 主軸 | 新 主軸 | 補助 |
|--------|--------|--------|------|
| クッション値 | scrape_jra_track | scrape_jra_track (継続) | — |
| 含水率 | 同上 | 同上 | — |
| 良/稍重/重/不良 | netkeiba + JV-Link | **JV-Link** | netkeiba |

---

## 4. 切替戦略 (5/24+ Phase 3)

### 4.1 段階的切替

**Phase 3 前半 (5/24-6/8)**:
- JRA-VAN 加入 + DLL インストール (5/24)
- JV-Link RACE / HR / O1 / WOOD parser 実装 (5/25-6/1)
- 過去 1 年 (5/24/2025 〜 5/23/2026) の bulk fetch + jra_races_full.csv 整合チェック
- 既存 source は **並行運用** (V15/V18/V19 は旧 master を使い続ける)

**Phase 3 後半 (6/9-6/30)**:
- V20 学習 data spec 確定 (JV-Link 主軸)
- 旧 jra_races_full.csv → JV-Link parser 出力で再構築
- V20 学習 + WF 検証 + LIVE retro

**Phase 3 投入後 (7/1+)**:
- V20 production deploy (JV-Link 主軸)
- 旧 master CSV は archive へ
- netkeiba scraping は補助 (training_eval / paddock 等)

### 4.2 V20 学習 data 戦略

```
V20 学習 data:
├── JRA subset (~50 万 horse)
│    ├── 主 source: JV-Link RACE / HR / O1 / TCOV / WOOD / BLOD
│    └── 補助: netkeiba (speed_index / training_eval / paddock) + JRDB (kyi)
└── NAR subset (~5 万 horse)
     └── 主 source: NAR scraping (代替なし)

共通 features 80%:
- 距離 / 馬場 / 性別 / 年齢 / 騎手 / 調教師 / 血統 / 過去成績 / 当日情報

具体 source:
- 距離 / 馬場 / 性別 / 年齢: JV-Link (JRA) + NAR scraping (NAR)
- 騎手成績: JV-Link SNPN + NAR (cross-source merge + expanding)
- 血統: JV-Link BLOD + NAR (cross-source merge)
- 過去成績: JV-Link RACE (JRA) + NAR (NAR)
- 当日情報: JV-Link O1/WF + NAR (NAR)

JRA-only features:
- speed_index (netkeiba)
- training_eval (netkeiba)
- jrdb_kyi PRE_RACE features

NAR-only features:
- 地方場特性 (大井/船橋/川崎 etc. 別 fr_wr)
```

### 4.3 sample weight 設計

学習時に sample weight で JRA / NAR の比率を調整:
- JRA 70% / NAR 30%
- AUC は subset 別に計算 (JRA AUC ≥ 0.88, NAR AUC ≥ 0.83)

---

## 5. リスク + 対策

| リスク | 対策 |
|--------|------|
| JV-Link 取得失敗 (DLL/COM trouble) | 旧 source (netkeiba/JRDB) を fallback として保持、 7/1 まで V15 production 継続 |
| 公式 data と既存 master の不整合 | 5/27 に 1 か月分整合チェック、 不整合があれば原因調査 |
| BAN リスク (netkeiba/NAR) | JV-Link 主軸化で netkeiba 依存度 50% 以下に下げる |
| データ取得中断 (paci / sed / payouts) | JV-Link 多数 datatype で代替経路確保、 監視 daily |
| Phase 3 後半 (6/9-30) で間に合わない | 6/30 時点 GO/no-go 判定、 NG なら V20 投入を 7/15+ に延期 |

---

## 6. 結論

✅ 4 source 役割分担確定:
- **JV-Link** (公式) = 過去 race / 払戻 / オッズ / 調教 / 血統 の主軸
- **netkeiba** (Premium) = speed_index / training_eval / paddock の補助
- **JRDB** = kyi (PRE_RACE) 継続、 paci 復活経路検討
- **NAR scraping** = 地方競馬全般 (代替なし)

✅ V20 学習 data 戦略確定:
- 共通 features 80% (基本情報)
- JRA-only / NAR-only の specific features 分離
- sample weight JRA 70% / NAR 30%

✅ 切替 schedule:
- 5/24-6/8: JV-Link 並行運用、 parser 実装
- 6/9-6/30: V20 学習 + WF 検証
- 7/1+: V20 production deploy

V15 動作不変: 本 plan は予約 doc、 production 完全不変。

---

**Session #39 D 完了**
