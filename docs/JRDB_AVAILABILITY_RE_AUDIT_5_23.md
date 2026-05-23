# JRDB データ提供可否 再監査 (2026-05-23)

**監査目的**: 前回監査の「取得不可」判定を実際のローカルデータ・スクリプトで再検証する。
**監査者**: Claude Code read-only audit (V15 production 完全不変)
**前提**: ユーザー (JRDB 会員) が「新潟も JRDB に出ている」と主張 → 前回監査を疑う。

---

## 監査結果一覧

| データ | 前回判定 | 今回判定 | 真の状態 |
|--------|---------|---------|---------|
| KTA 新潟 | 「新潟=0件」 | **USER_CORRECT** | 40,288行あり。前回監査が誤り |
| KKA 遅延 | 「1-2週遅延」 | **FETCH_BUG** 前回誤り → **WORKING** | 毎日 6:01 取得済み、遅延ゼロ |
| PACI | 「scrape_jrdb_paci.py 未スケジュール」 | **NOT_SCHEDULED** (一部正) | スクリプトは機能するが自動化なし |
| OZ | 「3/29で停止」 | **NOT_SCHEDULED** (一部正) | CSV は 3/31 以降再生成されていない |

---

## STEP 2: KTA 新潟 再検証 (最重要)

### 前回監査の主張
> 「新潟 (venue code 04) KTA data = 0 records」

### 今回の実測値

```
jrdb_kta.csv 総行数: 298,551
新潟 (race_id like "??????04*") 行数: 40,288
最新新潟 race_id: 202509040912 (2025年9月)
```

2025年 KTA 新潟確認済みファイル (抜粋):
- KTA250503.txt, KTA250504.txt (5月開催)
- KTA250726.txt ～ KTA250831.txt (夏季新潟開催)
- KTA251018.txt, KTA251019.txt (秋季開催)

**結論: 前回監査の「新潟=0件」は完全に誤り。JRDB は新潟データを提供しており、ローカル CSV に 40,288 行存在する。**

### KTA 2026 の状況

KTA raw 最新: `KTA260405.lzh` (2026-03-31 ダウンロード)
KTA extracted 最新: `KTA260405.txt`

`daily_jrdb_kyi.bat` は毎日 6:00 に `download_parse_jrdb_batch2.py --types kta cha` を実行。
2026-05-23 06:00 ログ: **「Downloaded 0 new KTA files」**

**原因**: JRDB の datazip/Kta/index.html に KTA260523.zip / KTA260524.zip がまだ掲載されていなかった (金曜 6:00 時点)。KTA は通常木〜金曜に JRDB から公開される。夕方再試行すれば取得できる可能性がある。

**判定: TIMING** (fetchスクリプトは正常動作。JRDB側の公開タイミングの問題)

---

## STEP 3: KKA 遅延 再検証

### 前回監査の主張
> 「JRDB は 1-2 週遅延で配信」

### 今回の実測値

```
jrdb_kka.csv 最終更新: 2026/05/23 06:01
jrdb_kka.csv 総行数: 546,925
2026年分: 16,301 行
```

KKA 年別カバレッジ (05/23 06:01 ログより):
```
2015: 49,992 / 2020: 48,282 / 2021: 47,821 / 2022: 47,220
2023: 47,672 / 2024: 47,181 / 2025: 46,889 / 2026: 16,301
```

**仕組み**: `daily_jrdb_kyi.bat` が `download_parse_jrdb_extra.py --types kka` を実行。
KKA は yearly zip (KKA_2020.zip 等) + 2026年分は個別 zip で毎日取得。
今日のログ: 「KKA: already extracted (1229 files) → 546,925 rows → jrdb_kka.csv saved」

Health check (07:30): `OK KKA: ok (48,128,510B mtime 05/23 06:01)`

**結論: 前回監査の「1-2週遅延」は誤り。KKA は毎日 6:01 に更新されており遅延ゼロ。**

**判定: USER_CORRECT / FETCH_BUG** (前回監査側の調査誤り)

---

## STEP 4: PACI 再検証

### 前回監査の主張
> 「scrape_jrdb_paci.py 未スケジュール、chronic bug」

### 今回の実測値

**scrape_jrdb_paci.py**: 完全に動作するスクリプト (stub ではない)
- JRDB の `datazip/Paci/` から PACI*.zip をダウンロード
- `parse_jrdb.py` を呼び出して `jrdb_paci.csv` を再生成

**重要な発見: PACI = 週次全ファイル束**
PACI260510.zip の中身:
```
BAC260510.txt, CHA260510.txt, CYB260510.txt, JOA260510.txt
KAB260510.txt, KKA260510.txt, KYI260510.txt
OT260510.txt, OU260510.txt, OW260510.txt, OZ260510.txt
UKC260510.txt, ZED260510.txt, ZKB260510.txt
```
PACI は「競走馬本データ (ZKB)」単体ではなく、その週の全 JRDB ファイルのバンドル。

**ダウンロード状況**:
```
PACI260510.zip: 2026/05/12 06:09 (最新)
PACI260509.zip: 2026/05/09 12:08
PACI260503.zip: 2026/05/02 12:13
```

**スケジュール状況**: `scrape_jrdb_paci.py` はいかなる schtasks にも登録されていない。5/9 と 5/12 のダウンロードは手動実行によるもの。

**CSV 状況**:
```
jrdb_paci.csv 最終更新: 2026/05/12 06:11
jrdb_paci.csv 総行数: 549,604
2026年分: 17,985 行 / 2026年5月分: 2,357 行
```
5/16〜17、5/23〜24 週末分は未取得。

**判定: NOT_SCHEDULED** (スクリプトは機能するが自動化されていない)

**Fix**: `daily_jrdb_kyi.bat` または weekend bat に以下を追加:
```bat
python tools\scrape_jrdb_paci.py >> %LOGFILE% 2>&1
```

---

## STEP 5: OZ 再検証

### 前回監査の主張
> 「OZ 3/29 で停止」

### 今回の実測値

**OZ raw (datazip/Oz/)**: 最新 `OZ260329.zip` (2026-03-31)
**OZ extracted/Oz/**: 最新 `OZ260503.txt` (2026-05-02 ダウンロード)
**OZ extracted/Paci/**: 最新 `OZ260510.txt` (2026-05-12、PACI バンドル経由)

**jrdb_oz.csv 最終更新**: 2026/03/31 12:04 (3ヶ月前)
**jrdb_oz.csv 総行数**: 21,591 行

**問題の構造**:
1. `daily_jrdb_kyi.bat` は `batch2 --types kta cha` のみ実行 → OZ は非対象
2. OZ の個別 zip は 5/1〜5/2 に `extracted/Oz/` へダウンロード済み (OZ260502, OZ260503)
3. PACI バンドル経由で `extracted/Paci/` に OZ260426〜OZ260510 が存在
4. しかし `batch2 OZ parser` は `extracted/Oz/` のみ参照 → `extracted/Paci/` の OZ は未参照
5. `jrdb_oz.csv` は 3/31 以降再生成されていない

**実際の OZ データ可用性**:
- extracted/Oz/ に OZ260503.txt まで存在 → batch2 --types oz を実行すれば 5/3 データまで CSV に反映可能
- extracted/Paci/ に OZ260510.txt まで存在 → batch2 OZ parser を Paci/ も参照するよう拡張すれば 5/10 まで反映可能

**判定: NOT_SCHEDULED** (データは存在するが OZ が daily fetch に含まれていない)

**Fix オプション**:
- Option A: `daily_jrdb_kyi.bat` に `batch2 --types oz ow ot` を追加
- Option B: batch2 OZ parser を `extracted/Paci/OZ*.txt` も参照するよう修正
- 推奨: Option A (最小変更)

---

## STEP 6: 日次スケジュール全体像

| 時間 | タスク | 取得データ |
|------|--------|-----------|
| 06:00 (毎日) | `daily_jrdb_kyi.bat` | KYI, SED, TYB, CYB, JOA, KAB, KTA, CHA, KKA, JO |
| 07:30 (土日) | `jrdb_health_check.bat` | Health check: KYI, KAB, KTA, CHA, KKA, JO |
| 09:00 (土日) | `jrdb_retry_am9.bat` | TYB, SED, KYI, KAB retry |
| 手動のみ | `scrape_jrdb_paci.py` | PACI バンドル (OZ, KKA, KYI 等含む) |
| 手動のみ | `batch2 --types oz ow ot ov` | OZ/OW/OT/OV (オッズ系) |

---

## STEP 7: 各ソースの最終判定

| データ | 判定 | 理由 |
|--------|------|------|
| **KTA 新潟** | **USER_CORRECT** | 新潟データ 40,288行存在。前回監査が誤調査 |
| **KKA 遅延** | **FETCH_BUG → WORKING** | 前回監査誤り。毎日 06:01 取得、遅延ゼロ |
| **PACI** | **NOT_SCHEDULED** | スクリプト正常。スケジュール未登録 |
| **OZ** | **NOT_SCHEDULED** | データは extracted/ に存在。daily bat 非対象 |

---

## 前回監査の精度評価

| 項目 | 前回監査 | 正確性 |
|------|---------|--------|
| KTA 新潟 | 「0件」と誤判定 | **誤り** (40,288行存在) |
| KKA 遅延 | 「1-2週遅延」と誤判定 | **誤り** (毎日更新) |
| PACI 未スケジュール | 「未スケジュール」 | **正** (ただし「chronic bug」は不正確: スクリプトは正常) |
| OZ 3/29停止 | 「3/29停止」 | **部分的に正** (CSV は止まっているが extracted には 5/10 まで存在) |

**総評**: 前回監査は 4 項目中 2 項目で重大な誤りがあった。ユーザーの「新潟も JRDB に出ている」という主張は完全に正しい。前回監査は CSV の空行数ではなく抽出ロジックのバグに惑わされた可能性が高い。

---

## 推奨アクション (優先度順)

1. **[高] OZ の daily 取得追加**: `daily_jrdb_kyi.bat` に `batch2 --types oz ow ot` を追加 (3行)
2. **[高] PACI の weekend 取得自動化**: `jrdb_retry_am9.bat` または weekend bat に `scrape_jrdb_paci.py` を追加
3. **[中] KTA 5/23-24 再試行**: 金曜夕方に `batch2 --types kta` を再実行して新規 KTA zip を取得
4. **[低] batch2 OZ parser を Paci/ も参照するよう拡張**: PACI バンドルの OZ を自動的に Oz/ と統合

---

*作成: 2026-05-23 Claude Code read-only audit*
