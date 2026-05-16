# 加入 source 規約 review (5/15、 我々の利用 適合性 audit)

実行: 2026-05-15、 Opus 4.7
目的: 各 加入 source の 規約 review + 我々の利用 法的 / 倫理的 リスク 識別

## ★ 重要 注意 ★

本 doc は **公開情報 + 一般的 規約 知識** に基づく 整理。 **法務 専門家 の最終確認 推奨**。 
正式 規約 は 各 source 公式 サイト で 取得 + review してください。

## 1. netkeiba マスターコース (月額 4,500 円)

### 一般的 規約 (公開情報、 2024 年時点)

| 項目 | 内容 |
|------|------|
| 私的利用 | OK (会員 個人 利用) |
| 商用利用 | NG (再販 / 配布 禁止) |
| 自動化スクレイピング | グレー (公式 否定的 だが API 公開なし) |
| 大量 download | NG (server 負荷、 規約違反 リスク) |
| 解析 / AI 学習 | グレー (私的利用範囲なら OK 解釈) |
| 二次配布 | NG (data 含む、 重要) |

### 我々の利用 vs 規約

| 我々の動作 | 規約 適合度 | リスク |
|----------|----------|------|
| netkeiba_master_index.csv (139K rows) 保有 | 私的利用範囲、 OK | 低 |
| 過去 race scrape 大量 | グレー、 rate-limit 必要 | **中-高** |
| 厩舎コメント / レビュー 集積 (407K rows) | 同上 | **中** |
| AI 予想 data 利用 (本日 統合) | 私的解析 OK | 低 |
| GitHub repo に csv commit | **再配布 NG 抵触 risk** | **★ 高 ★** |
| paddock 動画 大量 download | NG リスク 大 | **★ 高 ★** |
| 専門家印 200K bulk scrape (Tier 1-2) | rate limit 必須、 数日 分散 | **中** |

### 推奨 対応

1. **.gitignore に csv 全て** → repo に保存しない (現在 一部 commit されている、 ★ 改善 必要 ★)
2. **scrape rate limit**: 1-2 秒 / request、 並列禁止
3. **bulk scrape 数日 分散**: 1 日 1-2K races 上限
4. **再配布 NG 徹底**: GitHub public repo に csv 出さない、 private branch にも 注意
5. **個人利用 明示**: AI prj README に 「個人 利用、 再配布 禁止」

## 2. JRDB Advance (月額 約 2,000 円)

### 一般的 規約

| 項目 | 内容 |
|------|------|
| 私的利用 | OK |
| 商用利用 | 別 契約 必要 |
| データ加工 | OK (個人) |
| 再配布 | NG |
| download | 規約範囲内 OK (公式 LZH ダウンロード) |

### 我々の利用 vs 規約

| 動作 | 適合 |
|------|------|
| JRDB LZH dl + parse | ✅ 規約 範囲内 |
| jrdb_*.csv 大量保存 | ✅ 私的利用 |
| AI 学習 | ✅ 個人 |
| GitHub commit | **★ 再配布 NG 抵触 risk ★** |

### 推奨

- jrdb_*.csv も .gitignore (現状 含むものあり、 確認要)
- jrdb_*_spec.txt は 我々の memo、 commit OK

## 3. JRA-VAN DataLab (月額 2,090 円)

### 一般的 規約

| 項目 | 内容 |
|------|------|
| 私的利用 | OK |
| 商用利用 | NG (別 BtoB 契約 必要) |
| データ加工 | OK |
| 再配布 | **厳格 NG** (規約 違反 大) |
| RT 取得 | ID/PW 個人認証 必要 |

### 我々の利用 vs 規約

| 動作 | 適合 |
|------|------|
| JV-Link COM 経由 取得 | ✅ 個人認証 |
| TFJV binary parse | ✅ 個人 |
| data/jvlink/*.json 保存 | ✅ 私的 |
| GitHub commit | **★ 再配布 NG 抵触 risk ★** |
| **AI 自律 fetch** (5/24+) | ✅ 個人利用範囲 |

### 推奨

- JV-Link 取得 data も .gitignore 厳格
- 加工結果 (features csv) も 配布 注意

## 4. JRA レーシングビュワー (月額 約 1,000 円)

### 一般的 規約

| 項目 | 内容 |
|------|------|
| 動画 視聴 | 会員 個人 |
| **動画 download** | **★ 厳格 NG ★** (DRM / 利用規約) |
| frame 抽出 / 解析 | 私的 grey zone |
| 再配布 | NG |

### 我々の利用 vs 規約

| 動作 | 適合 |
|------|------|
| paddock frame 抽出 (現 237 dir) | **★ DRM 解除 NG risk、 公式 NG ★** |
| 動画 download (Phase 4 1000+ 蓄積 plan) | **★ 規約違反 高 risk ★** |
| YOLOv8 推論 | (動画 取得 自体 NG なら 抵触) |

### 推奨

- **★ 動画 大量 download / frame 抽出 は 規約違反 risk 大 ★**
- 別 source (JRA 公式 YouTube embed 等) で 代替 検討
- Phase 4 動画 features 計画 は 規約 厳格 確認 必須

## 5. TARGET TFJV (C:\TFJV、 無料 binary)

### 一般的 規約

| 項目 | 内容 |
|------|------|
| binary 取得 | OK (TARGET 経由) |
| AI 学習 | OK (個人) |
| 再配布 | NG |

### 我々の利用 vs 規約

| 動作 | 適合 |
|------|------|
| binary 直 parse | ✅ |
| RA/SE/HR features 抽出 | ✅ |
| Phase 13 parser 拡張 | ✅ |

## 6. 無料 / 公的 source

| source | 規約 | 我々の利用 |
|--------|------|----------|
| 気象庁 API | open / 商用 OK | ✅ |
| jpholiday lib | MIT license | ✅ |
| 国立天文台 公式 公式 | open | ✅ |
| 国土地理院 tile API | 利用規約 範囲内 | ✅ (hardcoded のみ) |
| JRA 公式 (馬場情報 / 配当) | 公式 公開、 個人利用 OK | ✅ |

## ★ 我々の 違反 / 高 risk 動作 サマリ ★

| risk level | 動作 | 対応 必要 |
|---------|------|--------|
| **★ 高 ★** | レーシングビュワー 動画 frame 抽出 / download | **Phase 4 着手前 規約 厳格 確認**、 別 source 検討 |
| **高** | csv (各 source data) を GitHub commit | **.gitignore 徹底 + 既 commit 確認** |
| 中-高 | 大量 scrape (200K rows、 専門家印 bulk 等) | rate limit + 分散 必須 |
| 中 | netkeiba paddock 静止画 取得 | 規約範囲内 だが 量 注意 |
| 中 | JV-Link 大量 RT fetch | 個人 認証範囲 OK だが server 負荷 注意 |

## ★ 即対応 推奨 ★

### A. .gitignore 強化 (即実行可能、 AI 自律 OK)

現状 .gitignore に欠落している可能性のある csv を audit + 追加:

```bash
# data/ 全 csv (config 除く)
data/jra_*.csv
data/jrdb_*.csv
data/netkeiba_*.csv
data/features_*.csv  # 加工後 features も safety 重視
data/v*.csv
data/*.csv.bak*

# 動画 / 画像
data/paddock_archive/
data/paddock_netkeiba/
data/race_video_frames/

# 大 model
*.pkl.gz
keiba_model_*.pkl*
models/v*/
```

### B. 既 commit 済 csv 確認 (要 user 判断)

git history に commit されている csv の audit:
```bash
git log --all --diff-filter=A --name-only -- 'data/*.csv' | grep -v test | sort -u
```

→ 必要 なら BFG / filter-repo で history rewrite (destructive、 user 認可必要)

### C. レーシングビュワー 動画 計画 再検討

Phase 4 (7-8月) plan で **動画 1000+ 蓄積** + **frame 抽出 + YOLOv8 推論** が:
- 規約 NG リスク 大 (DRM 解除 / 大量 download 禁止)
- **法務 確認 必須 → 必要なら Phase 4 plan 変更**
  - 代替 1: JRA 公式 YouTube embed (metadata のみ)
  - 代替 2: 規約範囲内 ストリーミング 再生 + 即時 推論 (保存なし)
  - 代替 3: Phase 4 中止

### D. scraping rate limit 標準化

全 scrape script で:
```python
import time
time.sleep(1.5)  # 各 request 後
```

→ 既存 tools/bulk_scrape_*.py に 標準 適用

## ★ 結論 ★

### 即対応 可能 (AI 自律)

1. ✅ .gitignore audit + 強化 (本日 commit)
2. ✅ scraping rate limit 強化 (script 更新)
3. ✅ 本 review doc commit

### user 判断 必要

1. **既 commit 済 csv の history rewrite** (destructive、 user 認可)
2. **レーシングビュワー Phase 4 plan 法務 確認**
3. **netkeiba bulk scrape 着手 適合性 確認** (専門家印 / paddock)
4. **GitHub repo public/private 判断** (csv 含む history)

### 法務 専門家 相談 推奨

- レーシングビュワー 動画 利用 (Phase 4)
- 大量 scrape (200K+ rows、 自動化)
- 商業化 path (将来)
- 個人情報 (馬主名 = 個人情報 抵触 risk)

## V15 投資保護 (規約 視点)

- V15 自体 は 私的 利用、 規約適合
- 累計 +5,240 円 (個人 投資収益、 規約 problem なし) ※ 旧 +13,530 円 は drift、 5/16 P0-1 真値 (docs/ROI_DISCREPANCY_2026_05_16.md)
- 但し data source 取得方法 が 規約適合 必須

## まとめ: priority action

1. **★ 即実行 ★** `.gitignore` audit + 強化 (AI 自律)
2. **★ 即実行 ★** scrape rate limit 強化
3. **★ user 帰宅後 ★** git history csv 削除 判断 (destructive)
4. **★ user 帰宅後 ★** レーシングビュワー Phase 4 規約 確認
5. **★ 中長期 ★** 法務 専門家 相談 (商業化 / 大量 scrape / 動画)
