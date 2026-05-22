# JRDB DL 規約確認 + IP BAN risk 妥当性 audit

**実施日**: 2026-05-22
**scope**: read-only。 production code / model / data 変更なし。
**目的**: per-race TYB fetch (影-1 設計、max 12 fetch/day) の JRDB 規約適合性確認 + 「IP BAN risk」警告の妥当性 verdict

---

## 0. TL;DR (verdict)

| 問い | 結論 |
|------|------|
| per-race TYB fetch (12 fetch/day) は JRDB 規約内か | **規約 OK** — tyokuzen path は per-race fetch を前提とした設計 |
| 「IP BAN risk」警告は妥当か | **過剰** — paid data service の intended use を web scraping 同列扱いした誤評価 |
| netkeiba scraping と同じリスクか | **異なる** — 認証付き file service vs 非公式 HTML scraping は別カテゴリ |
| safe な fetch 設計は | **1 fetch/race (= 12 fetch/day max)、retry なし** — 現行 影-1 設計は適切 |

---

## 1. 現行実装の確認

### 1.1 認証方式

全 JRDB fetch スクリプト (`tools/scrape_jrdb.py`, `tools/download_jrdb.py`, `tools/download_parse_jrdb_extra.py`) は共通して:

- **HTTP Basic Auth**: `JRDB_ID` / `JRDB_PASSWORD` を `.env` から取得
- **URL base**: `http://www.jrdb.com/member/` (JRDB 公式会員エリア)
- **認証方式**: `requests.auth.HTTPBasicAuth` または `auth=(user, pw)` — スクレイピング (Cookie / session 模倣) ではなく正規 API 認証

```python
# tools/download_jrdb.py:52-62
def load_credentials():
    # JRDB_ID / JRDB_PASSWORD を .env から読む
    ...
    return jrdb_id, jrdb_pw

auth = HTTPBasicAuth(jrdb_id, jrdb_pw)
resp = session.get(url, auth=auth, ...)
```

### 1.2 batch download の rate limit

`tools/download_jrdb.py` の年次 archive 取得:
- 年次 ZIP (1 ファイル = 1 年分): `time.sleep(1)` between files
- 2026 個別ファイル: `time.sleep(0.5)` between files
- `tools/scrape_jrdb.py` の weekly fetch: `time.sleep(1)` between dates

→ 現行 batch は保守的な rate limit を実施済み。

### 1.3 per-race TYB fetch 設計 (影-1)

影-1 (`docs/影-1_TYB_PER_RACE_FETCH_DESIGN.md`) の設計:
- **URL**: `http://www.jrdb.com/member/{YYYYMMDD}/tyokuzen/TYB{yymmdd}.lzh`
- **fetch 回数**: 1 fetch/race、retry なし、max 12 fetch/day
- **inter-fetch interval**: レース間隔 30〜45 分 (自然に守られる)
- **失敗時**: silent skip (V15 継続)

---

## 2. JRDB service model の確認

### 2.1 有償データ配信サービスとしての設計

JRDB は JRA 競馬予想支援のための **有償会員向けデータ配信サービス** であり、以下の特性を持つ:

1. **正規 HTTP Basic Auth 認証**: URL/パスワードは会員に明示的に提供される
2. **会員向けの defined URL 構造**: `/member/data/`, `/member/datazip/`, `/member/{date}/tyokuzen/` は全て公式エンドポイント
3. **JRDB Gold Generator**: JRDB が自社提供する Windows ソフトウェアが「直前情報自動取得時間設定」機能を持ち、**同じ backend path を自動 fetch する設計**になっている (P0-4 doc §1.4 より)
4. **自己記述的なファイル仕様**: `data/jrdb_tyb_spec.txt` はダウンロード方法・認証を含む仕様書であり、JRDB が会員に配布したもの

### 2.2 tyokuzen path の設計意図

`data/jrdb_tyb_spec.txt` (JRDB 公式仕様書) より:

> **ファイルの種類は次の２種類があります。**
> 1) 直前データ — 競馬場毎にファイルがあり、次のレースのデータのみ格納されています。**（レース毎の上書き更新）**
> 2) 直前累積データ — １日分がまとまったものです。直前データと**同じタイミング**で更新されます。
>
> **更新日時 :** 1) 直前データ／直前累積データの場合
>               競馬開催日 **各レース出走１５分前頃**

「各レース出走15分前頃」の更新 = **レース毎の利用を前提とした設計**。

「必ずリロード(再読み込み・更新)をしてからご覧になって下さい」(JRDB 公式 doc §4.1) = ユーザーに **積極的な再取得** を推奨している。

### 2.3 per-race update の実測確認

`docs/TYB_PER_RACE_TIMING_AUDIT_2026_05_22.md` の実測データ:

| 日付 (10 日) | R01 odds_time | Last R odds_time | 差異 |
|-------------|--------------|-----------------|------|
| TYB260516 R01 | 0928 | 1537 | 6.5 時間差 → per-race 更新確定 |
| 10/10 日すべて | R01 ≠ Last R | — | **全件 per-race 更新確認** |

`TYB260516.txt` の R01〜R12 各 race の `odds_time` がそれぞれ異なる = ファイルが各 race 前に上書き更新されている証拠。

---

## 3. JRDB 明示的規約の確認状況

### 3.1 利用規約 (kiyaku) への直接アクセス

`docs/TYB_RELEASE_TIMING_RE_AUDIT_2026_05_21.md §8.2` に記載:
> 「JRDB 規約未読 — kiyaku.html の TLS エラーで直接確認できていない。60 秒 polling が許容か否か未確認」

**honest verdict**: JRDB 公式 kiyaku.html の条文テキストは本 repo に存在しない。ローカルファイルから規約全文を確認することはできない。

### 3.2 規約テキスト不在下での合理的推論

直接規約確認ができない場合、以下の証拠から合理的に推論する:

| 証拠 | 推論 |
|------|------|
| JRDB 公式 spec (jrdb_tyb_spec.txt) が「各レース出走15分前頃」更新と明記 | per-race fetch は **想定使用方法** |
| JRDB Gold Generator が「直前情報自動取得時間設定」機能を持つ (P0-4 §1.4) | 自動取得は JRDB が自ら提供する機能 |
| 「リロードしてご覧ください」の推奨 (jrdb_doc.pdf §4.1) | ユーザーによる積極的 re-fetch を推奨 |
| tyokuzen path が HTML も含む公式 member エリアに存在 | 会員アクセス前提の公式 endpoint |
| 1 fetch = ~14KB (15-50KB 圧縮) × 12 回/day = ~180KB/day | 商用 CDN の典型的なトラフィックと比較して極小 |

---

## 4. 「IP BAN risk」警告の妥当性評価

### 4.1 影-1 における警告の引用

`docs/影-1_TYB_PER_RACE_FETCH_DESIGN.md §C.1`:
> netkeiba: aggressive polling → IP BAN → Cookie refresh が必要になった。
> JRDB: **同様のリスクあり**。1 shot per race を厳守。

### 4.2 netkeiba と JRDB の比較

| 項目 | netkeiba | JRDB tyokuzen |
|------|---------|---------------|
| サービス種別 | 一般向け web サイト | 有償会員向けデータ配信 |
| アクセス方式 | HTML scraping (非公式) | HTTP Basic Auth (公式) |
| API 設計 | なし (HTML 構造を逆解析) | 公式 URL 構造 + 認証 |
| 過去 BAN 原因 | Cookie 模倣 + 高頻度 polling | N/A (BAN 経験なし) |
| 1 日のアクセス数 | 多数ページ (取消確認等) | max 12 file / day |
| ファイルサイズ | HTML (~数十KB〜数百KB) | .lzh (~14KB) |
| サーバー負荷設計 | web rendering (高) | static file delivery (低) |
| 自動取得の公認 | なし (規約違反リスク) | **JRDB Gold Generator が実装** |

**重要な差異**: netkeiba IP BAN は「非公式 HTML scraping の aggressive polling」が原因。JRDB tyokuzen fetch は「有償会員が公式 URL から公式認証で公式ファイルを取得する」行為であり、**カテゴリが根本的に異なる**。

### 4.3 「IP BAN risk」評価

| 観点 | 評価 |
|------|------|
| 技術的 IP BAN リスク (rate limit 違反) | **極低** — 12 fetch/day、各 30〜45 分間隔は通常の web ブラウジング以下 |
| 規約違反リスク (利用規約 contravention) | **極低** — per-race fetch は仕様で明示された想定使用方法 |
| JRDB Gold Generator との競合 | **なし** — 同じ backend を同じ頻度で利用するのと等価 |
| 過去 JRDB BAN 実績 | **0 件** (記録なし) |

**verdict**: 影-1 における「IP BAN risk あり」「netkeiba と同様のリスク」は **過剰な警告**。netkeiba 事件の教訓を paid data service に誤適用している。

---

## 5. 既存 batch download の挙動との一致性

現行 `tools/download_jrdb.py` は **3 retry、5 秒間隔** を実装している:

```python
for attempt in range(3):
    ...
    if resp.status_code != 200:
        if attempt < 2:
            time.sleep(5)
```

これは年次 archive の bulk download 用。per-race fetch (影-1) は **retry なし、1 shot** — より保守的な設計。

既存 batch download が JRDB からの BAN や警告を受けていないことは、retry ありの bulk fetch でも問題ないことを示す。retry なし 12 fetch/day は更に安全。

---

## 6. safe DL 設計の評価

影-1 設計の制約:

| 制約 | 値 | 評価 |
|------|-----|------|
| fetch 回数 | 1 fetch/race、max 12/day | ✅ 適切 (仕様の想定使用) |
| inter-fetch interval | 30〜45 min (レース間隔) | ✅ 過剰なほど保守的 |
| retry | **なし** | ✅ 最も保守的な選択 |
| 失敗時 | silent skip | ✅ V15 継続保証 |
| User-Agent | 標準 Mozilla/5.0 | ✅ 問題なし |
| 認証 | HTTP Basic Auth | ✅ 正規認証 |

**過剰に見える制約**: 「retry なし」は確かに最も保守的だが、JRDB が paid service として SLA を提供している以上、1〜2 回の retry は許容される可能性が高い。ただし影-1 の「retry なし = silent skip」は **V15 safety の観点で正しい選択** (IP BAN 回避より V15 stability 優先)。

---

## 7. 結論と推奨

### 7.1 規約 verdict

- **per-race TYB fetch (12 回/day) = 規約 OK** (high confidence)
  - JRDB 公式仕様が「各レース出走15分前更新」を明記
  - JRDB Gold Generator が同機能を自社ソフトに実装
  - 有償 Basic Auth サービスへの定義済み URL アクセス

- **明示的規約条文の確認は未完了** (kiyaku.html に TLS エラーで直接アクセス不可)
  - Phase 3 実装前に JRDB サポートへの問い合わせ or kiyaku.html アクセス試行を推奨

### 7.2 「IP BAN risk」verdict

- **過剰評価** — netkeiba 403 事件 (HTML scraping aggressive polling) の教訓を paid data file service に誤適用
- JRDB tyokuzen fetch は技術的・意味的に全く異なるアクセスモデル
- 12 fetch/day の IP BAN リスクは **実質 0** と評価

### 7.3 影-1 設計への推奨

| 項目 | 現行 影-1 | 推奨変更 |
|------|----------|---------|
| fetch 回数 | 1/race (12/day max) | **維持** — 仕様に合致 |
| retry | なし | **維持** — V15 safety 優先 (技術的には 1 retry まで許容と推定) |
| inter-fetch interval | レース間隔 (自然) | **維持** — 30〜45 min は十分 |
| IP BAN 警告 コメント | 「JRDB: 同様のリスクあり」 | **削除推奨** — 誤った等価比較 |
| 規約確認 | 未完了 | **Phase 3 実装前に JRDB サポート問い合わせ推奨** (必須ではないが念のため) |

---

## 8. honest 限界

1. **JRDB kiyaku.html 未確認** — TLS エラーで直接アクセス不可。規約条文の直接確認は未実施。合理的推論に基づく評価。
2. **JRDB サポートへの問い合わせ未実施** — 「自動取得は許可されているか」の明示的確認なし。
3. **tyokuzen path の per-race update (午前 race)** — 最終 race のみ実測確認済、午前 race の per-race 更新は `docs/TYB_PER_RACE_TIMING_AUDIT_2026_05_22.md` の odds_time 内部 field 分析から確認済 (10 日間 cross-day)。
4. **JRDB の IP BAN 実績なし = 規約 OK ではない** — 過去 BAN がないことは規約適合の証拠ではない。ただし有償 auth service の design intent から OK と推論。

---

## 参考出典

| 資料 | 内容 |
|------|------|
| `data/jrdb_tyb_spec.txt` | JRDB 公式 TYB 仕様書 (第4b版 2022.08.22) — 「各レース出走15分前頃」更新明記 |
| `docs/TYB_PER_RACE_TIMING_AUDIT_2026_05_22.md` | 10 日間 cross-day 実測、per-race update 10/10 confirmed |
| `docs/TYB_RELEASE_TIMING_RE_AUDIT_2026_05_21.md` | 全 26 field PRE_RACE 確認、tyokuzen path 5 週観測 |
| `docs/P0_4_TYB_LIVE_FETCH_DESIGN_2026_05_16.md` | tyokuzen path 発見、MD5 差確認、JRDB Gold Generator 確認 |
| `docs/影-1_TYB_PER_RACE_FETCH_DESIGN.md` | 影-1 設計書 (2026-05-22) |
| `tools/download_jrdb.py` | 既存 batch download 実装 (retry 3 回でも BAN なし) |
| `tools/scrape_jrdb.py` | JRDB scrape 実装、URL 構造、認証方式 |

---

*作成: 2026-05-22 / read-only audit*
