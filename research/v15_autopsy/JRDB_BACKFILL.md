# C. JRDB バックフィル準備（調査・報告のみ / ダウンロードは承認後）

JRDB・JRA-VAN 再契約済み。6/27-8/9 の欠損を JRDB バックナンバーで埋められるか調査。

## 1. 欠損の実測（data/jrdb_raw/<type>/ 最新ファイル）
| type | 最新ファイル | 欠損開始 |
|------|-------------|---------|
| KYI（競走馬） | KYI260621 | **6/22〜** |
| CYB（調教） | CYB260621 | 6/22〜 |
| JOA | JOA260621 | 6/22〜 |
| KAB | KAB260627 | 6/28〜 |
| SED（成績） | SED260607 | **6/8〜** |
| TYB（直前） | TYB260418 | **4/19〜（最も古い）** |
| KTA / MSA / MZA / ZED | 0件（空） | 全欠損 |

→ **feat_dump の JRDB死(6/27で40/40定数)と整合**。主要データは 6/21 で停止（SEDは6/7、TYBは4/18）。

## 2. バックフィル可否: ★可能（gapは当方のfetch停止）★
- `tools/download_jrdb.py`: `http://www.jrdb.com/member/datazip/{Type}/index.html` を member Basic認証で開き、**年アーカイブ `{TYPE}26*.zip`** を取得→lzh展開。
- JRDBは日次公開を継続しており、**年アーカイブに 6/22-8/9 の日次分が含まれる**（欠損は当方が取りに行かなかったため）。→ 年アーカイブ再取得で埋まる見込み。

## 3. 必要ファイル種（要求 KYI/KTA/KKA/SR/SRB/SKB/TYB/CYB/OZ の対応）
| 要求 | JRDB種 | download_jrdb.py YEARLY_TYPES | 備考 |
|------|--------|------------------------------|------|
| KYI | Kyi | ✓ | 標準経路 |
| KKA | Kka | ✓ | 標準経路 |
| SKB | Skb | ✓ | **SR/SRB は Skb の解析サブ要素**（parse_jrdb系で分離） |
| SR / SRB | (Skb内) | ✓ | 同上 |
| TYB | Tyb | ✓ | 標準経路（最も古い4/18から要回収） |
| CYB | Cyb | ✓ | 標準経路 |
| KTA | Kta | ✗（別経路） | `tools/jrdb_daily_fix_fetch.py`（6/12新設・日付直フェッチ）で取得 |
| OZ | Oz/Ozi | ✗（要確認） | オッズ系。`data/jrdb_oz.csv` 存在＝過去は取得実績あり。index.html で Oz 系アーカイブ有無を確認要 |

- `download_jrdb.py` の YEARLY_TYPES = `[Sed, Tyb, Kyi, Bac, Hjc, Cyb, Skb, Kab, Ukc, Kka]` が6種(KYI/KKA/SKB/TYB/CYB/SED)を標準カバー。
- **KTA は別経路**（jrdb_daily_fix_fetch.py）、**OZ は要 index 確認**。

## 4. 推奨手順（承認後に実行）
1. `.env`/認証の JRDB credentials 有効性を確認（member Basic認証）。
2. **year-archive 取得**: `python tools/download_jrdb.py`（対象 Kyi/Cyb/Skb/Tyb/Kka/Sed の 2026アーカイブ）→ data/jrdb_raw/ に lzh 展開。
3. **KTA**: `python tools/jrdb_daily_fix_fetch.py`（6/22-8/9 の日付範囲）。
4. **OZ**: index.html で Oz アーカイブ確認 → 該当あれば取得。
5. parse: `tools/parse_jrdb*.py` / `build_jrdb_v2_csv.py` で CSV 再構築 → feat_dump 再生成で 640R Pattern A が健全化。
6. 検証: 再構築後、6/27-8/9 の feat_dump で JRDB定数率が 40/40→数個 に戻ることを確認。

## 5. 留意
- ダウンロード実行は**本報告後の承認が前提**（未実行）。jrdb.com への index アクセスも承認まで行っていない（本調査はローカル生データ+スクリプト解析のみ）。
- バックフィルで **640R Pattern A の健全化（先行実験の全12日化）** と **V15本番の劣化解消**の両方に効く。ただし autopsy の通り、ROI低下の一次要因はエッジ減衰でありデータ修復だけでは黒字化しない見込み。
- OZ（確定オッズ系）を特徴に使う場合は leak 注意（Pattern A は base/prev オッズのみ許容）。

---

## 6. 実行結果（2026-08-09 承認後・研究側で取得）
| 種別 | 経路 | 結果 |
|------|------|------|
| KYI/CYB/SKB/TYB/KKA/SED | download_jrdb.py --years 2026 | ✓ 6/22-8/9 完全取得（data/jrdb/extracted/、164DL/216展開） |
| KTA | jrdb_daily_fix_fetch.py --start 20260615 --end 20260809 | ✓ 取得（8/9まで） |
| OZ | Oz/index.html 日次zip | △ 6/13-6/21のみ既存・6/22+は取得失敗（命名/認証差、要follow-up）。★c-onlyはオッズ除外のため不要★ |

→ **c-only に必要な JRDB源は 6/22-8/9 まで取得完了**（data/jrdb/extracted）。

## 7. feat_dump 健全化の残工程（★production パイプライン・別途承認推奨★）
- jrdb_features.py は **構築済CSV**（data/jrdb_kyi.csv 等, nk_race_id+馬番）を読む。extracted→CSV再構築が必要。
- 再構築は複数スクリプト（parse_jrdb / build_jrdb_v2_csv / rebuild_jrdb_kyi …）が絡み、**生 jrdb_*.csv を上書き**＝production データ操作。手作業一括再構築は corruption リスクありと判断し**未実施**。
- 推奨: production の JRDB build を正規手順で走らせCSV再構築 → `daily_predict.py --date 20260627..20260809` で feat_dump 再生成（JRDB健全化）。これは運用系操作のため別途承認で。
- ★ただし item2 の答えは 186R 確定オッズ判定で既に **NO-GO** が明確（研究側 §5）。全640Rは方向確認の確認作業。★
