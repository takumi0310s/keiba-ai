# P0-4 TYB live fetch 最終判定 (Sub-task 11)

**作成日**: 2026-05-16 evening JST
**作成 source**: 親 agent 指示 Sub-task 11 (P0-4 着手判断、 read-only judgement)
**作業 mode**: docs のみ。 production code / model / commit/push なし。
**前提 docs**:
- `docs/P0_4_TYB_LIVE_FETCH_DESIGN_2026_05_16.md` (Sub-task 7 設計)
- `docs/TYB_MERGE_BUG_AUDIT_2026_05_16.md` (Sub-task 6 audit)

---

## 0. 結論 (★ TL;DR ★)

| 項目 | verdict |
|------|---------|
| **★ P0-4 (TYB -15 min live fetch) ★** | **★ 永久放棄 (PERMANENT ABANDONMENT) ★** |
| **理由** | TYB content 5 features は **採用候補 0** (LEAK 3 / 信号ゼロ 2) + V15 truncate で merge bug 直しても効果 0 |
| **工数節約** | Sub-task 7 想定 2-3 日 → **0 日** (代わりに JV-Link production fetch path へ注力) |
| **代替 path 候補 (5/24+ 検討)** | JV-Link O1 連続 odds (推奨優先) / JV-Link TCOV (馬場 live) / netkeiba SP テキスト数値化 / JRDB CYB merge bug fix verify |
| **V15 production への影響** | **0** (現状維持、 既存 race_auto_notify の TYB silent failure は behavior 変えず) |
| **honest 注記** | 「永久放棄」 は本 sub-task 11 内容ベースの結論、 user の最終判断は 別 sentence |

---

## 1. 判定根拠 (Sub-task 6 + 設計 doc 統合)

### 1-1. V15 model truncate (Sub-task 6 §5-1 / §8)

- V15 Pattern A/B model `num_feature() = 145`、 Pattern B saved features list 長 = 150 (= 145 + 5 TYB live)
- `tools/predict_core.py:2160-2163` で **明示的 slice**: `X = X_full[:, :n_lgb_features]` (= `:, :145`)
- TYB 5 columns は X 末尾 position → ★ 必ず truncate ★
- ★ V15 を そのまま使う限り、 P0-4 fetch 経路を確立しても 取り込む value は 0 ★

### 1-2. TYB 5 features の LEAK + 信号ゼロ確定 (Sub-task 6 §4-2 / §5-3)

| feature | corr_target | 判定 |
|---------|-------------|------|
| `jrdb_odds_idx` | **+0.4214** | ★ LEAK 確定 ★ (`popularity` corr -0.4242 と ほぼ同等 = odds-based) |
| `jrdb_paddock_idx` | **+0.3539** | ⚠ delivery 17:00 JST = post-race confirmed (P0-3 監査) |
| `jrdb_live_composite_idx` | **+0.2573** | ⚠ odds+padock 合成 = 部分 LEAK |
| `jrdb_body_code` | +0.0121 | ✅ safe、 ただし 信号 極小 |
| `jrdb_demeanor_code` | +0.0041 | ✅ safe、 ただし 信号 極小 |

★ 5 features 中 採用候補 = 0 ★:
- LEAK 3 件 (odds_idx / paddock_idx / live_composite) → V15/V21 retrain で採用すれば致命的 (V15 odds_log と同等の事故)
- safe 2 件 (body_code / demeanor_code) → corr ≤ 0.012 で +AUC delta は ≈ 0

### 1-3. P0-4 fetch 経路の value 評価

P0-4 設計 doc (`P0_4_TYB_LIVE_FETCH_DESIGN_2026_05_16.md`) は:
- JRDB tyokuzen path (`/member/{YYYYMMDD}/tyokuzen/TYB{yymmdd}.lzh`) の 16:16 JST live 経路を**技術的には**発見
- 5/2-5/16 の 5 週間 Last-Modified pattern で publication timing を validate
- nowracedata JSON polling で race -15 min trigger 検出可能 と confirm

**しかし** Sub-task 6 で:
- ★ content (5 features) は 採用不可、 全部 LEAK or signal 0 ★
- ★ V15 で truncate されるため retrain なし では 効果 0 ★
- ★ V15 retrain で safe features 採用しても +AUC ≈ 0 ★
- ★ V15 retrain で LEAK features 採用したら 大破綻 ★

★ つまり P0-4 fetch 経路は **技術的には成立するが、 取り込む content に value がない** ★

---

## 2. 代替 path 候補 比較

| source | timing | leak risk | 工数 | 期待 ROI delta | 推奨度 |
|--------|--------|-----------|------|----------------|--------|
| **TYB (本 P0-4)** | -15 min | high | 2-3 日 | **0** | ❌ 永久放棄 |
| **netkeiba 直前 scrape** | -15 min | high (IP ban 経験) | 1 週 | est. +1-2pt (★ assumption ★) | △ |
| **★ JV-Link O1 連続 odds ★** | 連続更新 | 低 (JRA-VAN 公式) | 2 週 | est. +2-3pt (★ assumption ★) | ★★ 最優先 |
| **JV-Link TCOV** (馬場 live) | -60 min | 低 (JRA-VAN 公式) | 1 週 | est. +1pt (★ assumption ★) | ★ |
| **JRDB CYB merge fix verify** | 数日前 | unknown | 1 週 + V15 retrain 前提 | unknown (要 verify) | ⚠ (verify only、 採用判断は 別 sub-task) |
| **netkeiba SP テキスト数値化高度化** | 1-3 日前 | 低 (既存 scrape path) | 1 週 | est. +1-2pt (★ assumption ★) | ★ |

### 2-1. JV-Link O1 連続 odds (推奨優先)

- JRA-VAN DataLab JV-Link `O1` dataspec
- -30/-15/-10/-5 min 4 段階 snapshot で odds 変動を 数値化
- Sub-task 6 §6 case D で `jrdb_odds_idx` は LEAK 確定 (popularity 同等) だが、 **時系列 odds 変動 (例: -30→-5 min の rank shift)** は V15 で 未使用 + delivery 安全
- V21/V22 学習 candidate feature

### 2-2. JV-Link TCOV (馬場 リアルタイム)

- JRA 公式 馬場情報 (cushion / moisture) は既存 (`cushion_value` / `moisture_rate` in V15)
- TCOV は **JRA-VAN 公式の live snapshot** 経路 (現在は scrape_jra_track.py で HTML scrape)
- 信頼性 ↑ (HTML 仕様変更 risk 回避)、 ただし **新規 AUC delta は ≈ 0** (同一 source 由来)

### 2-3. JRDB CYB merge fix verify (★ verify only ★)

- Sub-task 5-4 で「constant default」 発見 (548K rows、 unique=1 columns 多数)
- TYB と同様 「merge bug 確定 → fix しても V15 で truncate」 の可能性 高い
- ただし CYB は ★ 数日前 publish ★ で delivery 安全 (post-race leak なし)
- 5/24+ で merge audit + corr_target check + safe判定 のみ verify、 採用は 別 sub-task で判断

### 2-4. netkeiba SP テキスト数値化高度化

- 既存 `data/netkeiba_stable_comments.csv` / `data/netkeiba_race_review.csv` の text → score 変換 path
- 既存 LangChain / GPT base スコアリングを高度化 (例: BERT japanese embedding)
- delivery 1-3 日前 = post-race leak なし
- 既存 cumulative_results.csv の SP テキスト feature は単純な keyword count レベル

---

## 3. 5/24+ 検討候補 格上げ

★ 推奨優先順位 (5/24+ Phase 3 期間で別 sub-task として 着手) ★:

1. **JV-Link O1 連続 odds** (4 段階 snapshot、 時系列 odds shift feature)
2. **JV-Link TCOV** (馬場 リアルタイム、 既存 cushion/moisture の 代替経路)
3. **netkeiba SP テキスト数値化高度化** (BERT embedding 等)
4. **JRDB CYB merge bug fix verify only** (採用判断は 別 sub-task)

### 3-1. V15 retrain 前提 (重要)

- ★ 1-4 全候補は V15 retrain (= V21/V22) 前提 ★
- V15 production は **完全不変** 維持 (CLAUDE.md §「現行モデルのベースライン」 厳守)
- 5/24-6/30 (Phase 3) で V20 学習 path 整備 中、 候補 1-4 は V20+ の 追加 features 候補
- ★ 6 月以降 V20 学習開始時 ★ に 4 候補 individually ablation test、 +AUC delta ≥ +0.001 で OK のみ 採用

---

## 4. 工数節約 + 注力先

| 項目 | before (P0-4 着手) | after (永久放棄) |
|------|---------------------|------------------|
| Sub-task 7 設計 → 実装 工数 | 2-3 日 | **0 日** |
| schtask 登録 + shadow eval 工数 | 1 週間 | **0 日** |
| 規約 risk (JRDB scrape 60 秒 polling) | ⚠ 低-中 | **0** |
| V15 production 影響 risk | 0 (元から) | **0** |

★ 節約された 2-3 日 + 1 週間 を JV-Link production fetch path (Phase 3、 5/24+) に注力 ★

---

## 5. 実施 action items (即)

| # | task | 担当 | 期限 |
|---|------|------|------|
| 1 | 本 verdict doc commit (親集中) | — | 5/16 evening |
| 2 | `docs/P0_4_TYB_LIVE_FETCH_DESIGN_2026_05_16.md` §9 末尾 update | sub-task 11 (本) | 5/16 evening |
| 3 | `tools/v21/jrdb_tyb_live_fetch.py` 改修 着手なし、 既存 fetch 経路 frozen | — | — |
| 4 | `tools/register_tyb_live_fetch_schtasks.ps1` 新規作成なし | — | — |
| 5 | 5/24+ Phase 3 で JV-Link O1 / TCOV 経路 prioritize | 別 sub-task | 5/24+ |
| 6 | JRDB CYB merge bug fix verify を 5/24+ に schedule | 別 sub-task | 5/24+ |

---

## 6. honest 限界 (fabrication 防止)

1. ★ 「永久放棄」 は ★ 本 sub-task 11 の content 判断 (Sub-task 6 + P0-4 設計 doc を再読した上での verdict)、 user の最終判断 sentence は別途必要 ★
2. ★ 代替 path 候補の「期待 ROI delta」 は assumption (本物の WF AUC delta は未測定、 V15 retrain後の ablation で確定) ★
3. JRDB CYB merge bug の verify 結果は 未測定、 5/24+ 別 sub-task で audit 必須
4. JV-Link O1 / TCOV の live fetch 工数は 32-bit Python venv 構築 + DLL access の依存があり、 5/24+ JV-Link 加入 (既存) で初期動作確認のみ済 (Sub-task 39B / sub-task 80 等)、 production fetch は未動作
5. ★ P0-3 (delivery timing) 監査 で `jrdb_paddock_idx` の 17:00 JST 配信は confirmed、 本 verdict はこの delivery timing 制約 を依存している ★ — Sub-task 5-4 が誤っていた場合は 本 verdict も再評価必要

---

## 7. 参考 / 出典

- `docs/P0_4_TYB_LIVE_FETCH_DESIGN_2026_05_16.md` (Sub-task 7、 commit 5f877758)
- `docs/TYB_MERGE_BUG_AUDIT_2026_05_16.md` (Sub-task 6)
- `docs/TYB_LEAK_AUDIT_2026_05_16.md` (Sub-task 5-4、 P0-3 delivery timing audit)
- `tools/v21/_tyb_merge_audit.py` (read-only audit script)
- `data/v21/tyb_merge_audit.json` (生 metrics)
- `tools/predict_core.py:2160-2163` (truncate 明示 source)
- CLAUDE.md §「現行モデルのベースライン」 (V15 production 不変 ルール)

---

## 8. caveman summary

- TYB content useless. all 5 features dead.
- LEAK 3, signal 0 = 2. retrain死。 truncate不変 効果0。
- P0-4 fetch路 技術OK、 取込 value 0。
- 永久放棄。 工数 2-3 日節約。
- 5/24+ JV-Link O1 / TCOV / netkeiba SP / CYB verify 注力。
- V15 不変。 commit/push 親。
