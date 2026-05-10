# Phase 18 A: netkeiba マスター DOM 検証 plan

**作成**: 2026-05-10 (Session #91 Phase 18 A、 ★ Opus 4.7 ★)
**前提**: Phase 13 (commit f4d813bf) で 4 系統 25 features stub + selector hypothesis 実装済
**目的**: 1 R で実 DOM を取得し、 stub の selector hypothesis を真値化する harness を整備

---

## 1. tools/netkeiba_master_dom_probe.py (新規)

### 1.1 用途

1 R 指定で 4 系統 (AI 展開 / AI 波乱度 / 個別ラップ / トラックバイアス) の生 HTML
を gzip 保存し、 ユーザーが手動で BeautifulSoup REPL や DevTools で selector
真値化する。

### 1.2 仕様

- 既存 `tools/netkeiba_master_scraper.py` の Cookie / rate limit / kill switch を継承
- 1 R = 4 request × 3 sec = 12 sec で完結 (大量 fetch しない)
- 出力: `data/v18/dom_probe/{race_id}/{ai_tenkai|ai_haran|lap|track_bias}.html.gz`
- 各 page で BeautifulSoup quick summary (title, table 数, sample div classes 50 件) を JSON 化

### 1.3 使い方

```bash
# 1 R に対し 4 系統 を probe
python tools/netkeiba_master_dom_probe.py --race 202608030611 --umaban 5

# 結果確認
ls data/v18/dom_probe/202608030611/
cat data/v18/dom_probe/202608030611/probe_summary.json
```

### 1.4 selector 真値化 手順

1. ユーザーが PC ブラウザ (PC 版 提供開始時) または スマホ → Cookie で
   compatibility.html / upset.html / lap.html / track_bias.html を 開く
2. DevTools で「AI 展開予測」「波乱度メーター」「個別ラップ table」「トラックバイアス図」
   の DOM 構造を確認
3. 上記 probe で保存した HTML と一致確認
4. 真の selector を `tools/netkeiba_master_scraper.py` の各 parser に反映:
   - `_parse_ai_tenkai`: `.RaceData_PacePred` / `.score` / `.pred_pos` 等
   - `_parse_ai_haran`: `.haran_score` / `.haran_meter.lvN`
   - `_parse_lap`: `.first3f` / `.last3f` / `.umaban`
   - `_parse_track_bias`: `.bias_severity` + 「内有利」「外有利」テキスト

---

## 2. PC 版未提供問題 (Phase 13 既知)

| 機能 | スマホ web | iOS アプリ | PC 版 |
|------|-----------|-----------|-------|
| AI 展開予測 | ✅ | ✅ | ❌ (近日対応) |
| AI 波乱度 | ✅ | ✅ | ❌ |
| 個別ラップ | ✅ | ✅ | ❌ |
| トラックバイアス | ✅ | ✅ | ❌ |

→ 2026-05-10 時点 PC ブラウザ では未対応。 sp.netkeiba.com サブドメイン
(`race.sp.netkeiba.com/...`) は スマホ向けレイアウト。 既存 cookie で
PC からも HTTP fetch 可能だが、 真値化は スマホ DevTools か Android emulator
推奨。

### 2.1 Android emulator option (将来検討)

| 工程 | 内容 | 工数 |
|------|------|------|
| Android Studio install | AVD で Pixel 6 / API 34 emulator | 30 分 |
| netkeiba app install | Google Play 経由、 既存アカウントで login | 10 分 |
| mitmproxy 経由 通信 capture | DOM 直接取れないが API endpoint 解析 | 1-2 時間 |
| HTTP API map → scraper 反映 | request 形式 (header, query) を再現 | 1-2 時間 |

→ Phase 18 では着手せず、 PC 版提供 or 近日対応待ち が現実的

---

## 3. selector hypothesis (Phase 13 stub から継承)

### 3.1 AI 展開予測 (compatibility.html)

| feature | 推定 selector | 確信度 |
|---------|--------------|-------|
| master_pace_pred | `.RaceData_PacePred` / `.pace_pred` (text: スロー/ミドル/ハイ) | 中 |
| master_pred_finish_time | `.RaceData_FinishTime` (`m:ss.f` pattern) | 中 |
| master_horse_aitenkai_score | tr (umaban 一致) > `.score` / `.ai_score` | 中 |
| master_horse_pred_pos | tr > `.pred_pos` / `.pass_pos` | 低 |

### 3.2 AI 波乱度 (upset.html、 URL 推定)

| feature | 推定 selector | 確信度 |
|---------|--------------|-------|
| master_haran_score | `.haran_score` / `.upset_score` | 中 |
| master_haran_meter | `.haran_meter.lv1`-`.lv5` | 低 (推定) |
| master_top_pop_trust | (未推定、 真値化待ち) | — |

### 3.3 個別ラップ (lap.html、 URL 推定)

| feature | 推定 selector | 確信度 |
|---------|--------------|-------|
| master_horse_lap_avg_first3f | tr (umaban) > `.first3f` / `.lap_first` | 中 |
| master_horse_lap_avg_last3f | 同 > `.last3f` / `.lap_last` | 中 |
| master_horse_lap_consistency | parser 計算 (std) | 高 |
| 終速指標 / 加減速 phase | DOM 直接取得不可、 lap 配列から計算 | 低 |

### 3.4 トラックバイアス (track_bias.html、 URL 推定)

| feature | 推定 selector | 確信度 |
|---------|--------------|-------|
| master_track_inner_outer_bias | テキスト「内有利」/「外有利」 | 高 |
| master_track_front_back_bias | テキスト「逃げ有利」/「差し有利」 | 高 |
| master_track_today_severity | `.bias_severity` / `.track_severity` | 中 |

---

## 4. 5/11+ 真値化 schedule

| 日付 | 内容 | 工数 |
|------|------|------|
| 5/11 | DOM probe を 1 R で実行 + ユーザーが selector 真値化 | 30 min |
| 5/12 | scraper parser に真値 selector 反映 (commit) | 30 min |
| 5/13 | 5/11 の 35 R で fetch 試行 (kill switch ON で start)、 25 features 取得率確認 | 1 h |
| 5/14-5/16 | parser 精度改善 + V18 candidate 投入準備 | 2-4 h |
| 5/17+ | 当日 R 自動 fetch enable、 paper trade で V15 と並行評価 | 継続 |

→ 過去 backfill は 5/17+ 段階的に (Phase 18 B)

---

## 5. V15 投資保護 (絶対遵守)

✅ DOM probe は 1 R / 12 sec、 BAN risk 最小
✅ predict_core / daily_predict / app.py 不変
✅ 出力先 data/v18/dom_probe/ のみ、 V15 prediction に影響なし
✅ 既存 Cookie ベース、 scrape_premium と同じ guard

---

## 6. 結論

✅ tools/netkeiba_master_dom_probe.py 新規 (1 R 4 page 保存 + summary)
✅ 1 R = 12 sec で完了、 BAN risk 最小
⚠ PC 版未提供、 真値化は スマホ DevTools 主体
⚠ 真値化作業は ユーザー手動 + Claude 反映の協働
✅ V15 投資保護完全

---

**Phase 18 A 完了** (Opus 4.7)
