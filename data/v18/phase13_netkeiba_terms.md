# Phase 13 — netkeiba マスターコース 規約 + 機能確認

**date**: 2026-05-10
**operator**: れんはす + Claude Opus 4.7
**caveat**: ★ ToS risk 全 受け入れ で実施 ★ (user 明示確認済)

---

## 1. 加入状況

| 項目 | 値 |
|------|----|
| プラン | netkeiba マスターコース |
| 月額 | ¥4,980/月 (年額 ¥39,800、 約 ¥3,316/月相当) |
| 加入時期 | 2026-05-10 (本日) |
| 利用 device | スマホ web + iOS アプリ (PC 版近日対応予定) |
| 利用目的 | 個人利用 (keiba-ai は商用 service ではない) |

---

## 2. 機能 audit (master_introduction page 確認)

### 確認済 4 系統

| 系統 | netkeiba 表記 | scraping 対象? |
|------|----------------|----------------|
| **AI 展開予測** | 「My 馬柱 data と連動した AI 展開予測」 | ✅ Phase B |
| **AI 波乱度予測** | 「My 馬柱 data と連動した AI 波乱度予測」 | ✅ Phase C |
| **個別ラップ** | 「実況から 30 秒遅延、 JRA 公開 realtime ラップ」 | ✅ Phase C |
| **トラックバイアス** | 「馬場バイアス」 (独立機能) | ✅ Phase C |

### 副次機能 (今回 scope 外)

- AI レース分析 (テキスト見解、 NLP 必要)
- スマート POG (POG 候補抽出、 利用予定なし)
- My 馬柱 (個人 customize、 scraping 対象外)
- 動画 (Phase 4 で別途検討)

---

## 3. URL pattern (確認済)

| 用途 | URL pattern |
|------|-------------|
| マスター 紹介 | https://regist.sp.netkeiba.com/?pid=master_introduction |
| AI 予測 hub | https://race.sp.netkeiba.com/?pid=AI |
| AI 展開予測 (1 R) | https://race.sp.netkeiba.com/race/compatibility.html?kaisai_date=YYYYMMDD&race_id=RRRRRRRRRRRR |
| 個別ラップ | (要 PoC、 推定 race/lap.html?race_id=...) |
| トラックバイアス | (要 PoC、 推定 race/track_bias.html?kaisai_id=...) |

認証方式: Cookie ベース (.env の `NETKEIBA_COOKIE` 流用)、 `tools/refresh_cookie.py` 自動更新済 system 利用。

---

## 4. 規約 read (PoC 内 best-effort)

### read attempt 結果

| URL | result |
|-----|--------|
| https://www.netkeiba.com/?pid=info_detail&id=1303 | ❌ Cannot find block (page error) |
| https://regist.netkeiba.com/?pid=user_terms | ❌ blank content |

→ web fetch 経路では規約 raw text 取得不能。

### 代替 sourceing

netkeiba 利用規約 typical 条項 (一般 web service の自動 access 関連):
- 第 X 条 「自動取得 prohibition」 → bot / scraper / 自動化された access の禁止
- 第 X 条 「data 再配布 prohibition」 → 取得 data の第三者提供禁止
- 第 X 条 「商用利用 制限」 → 個人利用範囲超え 料金課金禁止
- 第 X 条 「著作権」 → コンテンツ著作権 netkeiba 帰属

→ ★ user は ToS risk を理解した上で個人利用 (商用なし、 再配布なし、 keiba-ai 個人運用) として実施判断 ★

---

## 5. compliance 設計 (技術側 mitigation)

ToS 全免許ではないが、 risk 最小化のため:

| 項目 | 実装 |
|------|------|
| **rate limit** | 3 sec interval (`time.sleep(3.0)` 必須) |
| **User-Agent** | brand 偽装なし、 自前 ID `keiba-ai/Phase13/1.0 (personal-use)` |
| **再配布禁止** | data/netkeiba_master/ 直下のみ保存、 git ignore (再配布ゼロ) |
| **fetch 頻度** | 朝 7:00 daily 1 回 + 直前 30 min realtime 1 回、 計 2 回/日 |
| **scope 限定** | 当日開催 R のみ、 過去 backfill しない |
| **fail-safe** | 規約改訂時 即停止 + 全 data 削除 可能 設計 (single dir 集約) |
| **自動化 dependency** | netkeiba master 取得失敗時も V15 + V18 が default fill で動作継続 |

---

## 6. Phase B-D 実装 方針

| Phase | 内容 | output |
|-------|------|--------|
| B | AI 展開予測 PoC | `tools/netkeiba_master_scraper.py` (skeleton + parser) |
| C | 波乱度 + ラップ + バイアス PoC | 同 file 拡張 |
| D | V20 統合 plan | `tools/predict_core_v18.py` Phase 13 features 追加 |

★ 即時実行 scraping は Phase 13 では行わない ★ — 構造解析 + skeleton 実装まで。 実 fetch は user が `tools/netkeiba_master_scraper.py --enable` 明示実行時のみ。

---

## 7. V15 投資保護 (絶対遵守)

| file | 状態 |
|------|------|
| `predict_core.py` | ★完全不変★ |
| `daily_predict.py` | ★完全不変★ |
| `app.py` | ★完全不変★ |
| `keiba_model_v15_central.pkl.gz` | ★完全不変★ |
| 累計収支 +¥14,140 維持 | ★絶対★ |

Phase 13 の影響範囲: `tools/netkeiba_master_scraper.py` (新規) + `tools/predict_core_v18.py` (Phase 13 features 追加) + `data/v18/phase13_*.md` のみ。

---

## 8. exit 条件 (規約改訂 / risk 顕在時)

| trigger | 対応 |
|---------|------|
| netkeiba から問い合わせ | 即 scraper 全停止 + data 削除 |
| ToS 改訂で個人利用 自動 access 明示禁止 | 即停止 + Discord アラート |
| Cookie BAN / アカウント警告 | 即停止 + 手動 review |

実装: `tools/netkeiba_master_scraper.py` 内 `KILL_SWITCH_FILE = data/netkeiba_master/.disabled` を check、 存在時 全 fetch skip。
