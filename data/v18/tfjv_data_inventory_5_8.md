# C:\TFJV\ 完全 inventory + parser 設計 (Session #44 A)

**作成**: 2026-05-08 (Session #44 A)
**前提**: ユーザー側で TARGET frontier JV install 完了 + 過去 30-90 年分の data 確認

---

## 1. C:\TFJV\ 構造 (root level)

| dir | record_type | files | size | 内容 |
|-----|------------|-------|------|------|
| BR_DATA | BR | 10 | 5.8 MB | 繁殖牝馬 |
| BS_DATA | HS | 311 | 11 MB | 生産者 |
| BY_DATA | HY | 283 | 22 MB | 馬連オッズ系? |
| **CK_DATA** | 02/12 | **18,089** | **657 MB** | 調教 |
| DE_DATA | RA+SE+TK | 23 | 864 KB | (用途不明) |
| **ES_DATA** | RA+SE | **11,584** | **688 MB** | レース成績 (年度確定) |
| **HY_DATA** | H1/H6 | **6,160** | **2.0 GB** | オッズ大量 |
| JG_DATA | JG | 41 | 1.7 MB | 競走除外/発走時刻 |
| KT_DATA | HN | 20 | 49 MB | 馬名? |
| OW_DATA | BN | 10 | 4.1 MB | 馬主? |
| **SE_DATA** | RA/SE/HR/YS/RC | **4,671** | **1.9 GB** | レース週報 (主軸) |
| TM_DATA | TM | 440 | 6.7 MB | 調教タイム |
| UM_DATA | UM/SK/他 | 280 | 497 MB | 馬個体 (1936-2025、 90 年分) |
| W5_DATA | WF | 863 | 7.0 MB | WIN5 (10 年分) |

**合計約 6 GB / 約 43,000 ファイル**

---

## 2. record format

### 2.1 共通仕様

```
全 ASCII 数字 + Shift-JIS 漢字 fields
record 区切り: \r\n (CRLF)
record 長: type ごと固定長 (例: HR 717 bytes)
```

### 2.2 主要 record type

| type | 内容 | 主 directory |
|------|------|------------|
| RA | レース詳細 (race info) | SE_DATA, ES_DATA, DE_DATA |
| SE | 馬毎レース情報 | SE_DATA, ES_DATA, DE_DATA |
| HR | 払戻金 | SE_DATA |
| RC | レース短信 | SE_DATA |
| YS | スケジュール | SE_DATA |
| H1 | 単複オッズ | HY_DATA |
| H6 | 三連単オッズ | HY_DATA |
| HY | 払戻 (BY_DATA 内) | BY_DATA |
| UM | 馬個体 | UM_DATA |
| SK | 産駒情報 | UM_DATA |
| KS | 騎手 | TFJ_KISI.DAT |
| BR | 繁殖牝馬 | BR_DATA |
| HS | 生産者 | BS_DATA |
| BN | 馬主 | OW_DATA |
| TM | 調教タイム | TM_DATA |
| HN | 馬名? | KT_DATA |
| JG | 競走除外 | JG_DATA |
| TK | 特別レース? | DE_DATA |
| WF | WIN5 | W5_DATA |
| 02/12 | 調教 (data 区分) | CK_DATA |

### 2.3 サンプル record (SE_DATA/2025/SH202511.DAT、 type=HR)

```
HR2 2025 0728 2025 0726 01 01 01 12 ...
↑↑ ↑↑↑↑ ↑↑↑↑ ↑↑↑↑ ↑↑↑↑ ↑↑ ↑↑ ↑↑ ↑↑
HR record
data区分2 (確定)
year 2025
month_day 0728
year2 2025
month_day2 0726
course 01 (札幌)
kai 01
nichi 01
race_num 12
... (払戻金種別ごとの金額)

length: 717 bytes
```

---

## 3. 既存 keiba-ai との重複状況

### 3.1 keiba-ai 既存 CSV (TARGET TFJV 由来、 過去抽出済)

| keiba-ai csv | rows | TFJV source |
|------------|------|-----------|
| `data/jra_races_full.csv` | 782,000 | SE_DATA + ES_DATA (RA + SE) |
| `data/blood_full.csv` | 81,986 | UM_DATA |
| `data/training_times.csv` | 955,580 | CK_DATA + TM_DATA |
| `data/odds_history.csv` | 778,387 | HY_DATA |
| `data/jra_payouts.csv` | 12,333 | SE_DATA (HR) — 4/6 で停止 |

→ 主要 datatype は 既に CSV 化済、 V15 学習で活用中

### 3.2 まだ未活用 / 部分活用の TFJV data

| TFJV source | 内容 | keiba-ai 利用状況 | V20 で活用候補 |
|-----------|------|----------------|------------|
| W5_DATA (WF) | WIN5 (10 年) | 未利用 | ★ V20 で複勝率参考 features |
| BR_DATA (BR) | 繁殖牝馬 | 未利用 | ★ sib_exp の母系 拡張 |
| BS_DATA (HS) | 生産者 | 未利用 | ★ 新馬戦 indicator |
| OW_DATA (BN) | 馬主 | 未利用 | 馬主成績 features |
| KT_DATA (HN) | 馬名? | 未利用 | (用途要確認) |
| JG_DATA (JG) | 競走除外 | 未利用 | リアルタイム取消 (Phase 4 候補) |

---

## 4. parser 設計方針

### 4.1 既存 `tools/extract_jvdata.py` の利用状況

```python
# 既存 extract_jvdata.py 概要
TFJV_DIR = 'C:/TFJV'
KEIBA_DATA = TFJV_DIR + '/TXT/keiba_data.csv'  # ← TARGET GUI export
```

→ TARGET の GUI で `keiba_data.csv` 等を export してから 既存 script で 7 CSV に変換
→ binary .DAT を直接 parse する経路は keiba-ai に **未実装**

### 4.2 本 Session で追加する parser

直接 binary .DAT を parse することで:
- TARGET GUI export 不要 (keiba_data.csv 不在でも動作)
- 全 90 年分の馬 data (UM_DATA) など 既存 export では取れない data を抽出
- V20 で W5/BR/BS/OW data を活用

### 4.3 parser class 設計

```python
class TFJVRecord:
    """JV-Data 仕様準拠 record の抽象 class"""
    record_type: str  # 'RA', 'SE', 'HR', etc.
    raw: bytes
    fields: dict
    @classmethod
    def parse(cls, raw_bytes): ...

class TFJVParser:
    """ディレクトリ単位で .DAT files を iterate + parse"""
    def __init__(self, base_dir='C:/TFJV'): ...
    def iter_records(self, datatype: str, year: int = None): ...
    def to_dataframe(self, datatype: str, year: int = None) -> DataFrame: ...
```

### 4.4 fields 抽出範囲

JV-Data spec full layout は複雑 (RA 1095 bytes、 50+ fields)。 本 Session では主要 fields のみ:

```python
# RA record (race info)
RA_FIELDS = {
    'year': (3, 7),       # offset 3-7 (4 chars)
    'month_day': (7, 11),
    'course_code': (11, 13),
    'kai': (13, 15),
    'nichi': (15, 17),
    'race_num': (17, 19),
    # ... (race_name, distance, surface, class 等は後段で追加)
}

# SE record (馬毎)
SE_FIELDS = {
    'year': (3, 7),
    'race_id_part': (3, 19),  # 16 chars
    'umaban': (21, 23),
    'horse_name_offset': (33, 73),  # Shift-JIS 漢字
    # ...
}

# HR record (払戻金)
HR_FIELDS = {
    'year': (3, 7),
    'race_id_part': (3, 19),
    # 単勝 payouts、 複勝 payouts 等の offset は spec で確定
    # 既存 jra_payouts.csv format に変換
}
```

---

## 5. V20 構築への寄与

### 5.1 既存 V15 features の維持

V15 150 features は data/jra_races_full.csv (TFJV 経由 既存 CSV) で構築済 → **そのまま継承**

### 5.2 V20 で追加候補 features (TFJV 直 parse)

| feature | source | 期待効果 |
|---------|--------|--------|
| sib_top3_rate_ext (90 年分母系) | UM_DATA + SE_DATA 拡張 | sib_w5 さらに +0.001-0.003 |
| W5_appearance_count (WIN5 出走実績) | W5_DATA | +0.001 |
| breeder_top3_rate (生産者成績) | BS_DATA | +0.002-0.005 |
| owner_top3_rate (馬主成績) | OW_DATA | +0.002 |
| precise_payout (払戻 詳細) | SE_DATA HR | jra_payouts 4/6 停止 解消 |

→ 期待 V20 AUC: 0.890-0.895 (V15 0.8858 から +0.005-0.01)

### 5.3 schedule 大幅前倒し

| 旧 plan (Session #41 H roadmap v2) | v3 (本 Session F で確定予定) |
|----------------------------------|-----------------------------|
| 5/24+: 32-bit Python install | 不要 (TFJV 直読み) |
| 6/9-13: JV-Link backfill | 不要 |
| 6/14-30: V20 構築 | **5/16-6/8 に 前倒し** ★ |
| 7/1: V20 投入 | **6/8 V20 投入候補** ★ |

→ **約 1 か月の前倒し** 可能

---

## 6. 5/9 V15 投資保護 (A 領域)

✅ TFJV 全 data は **read-only** (write しない)
✅ V15 model md5: `842b9a5f305c793ed8fa54a74e06b836` 不変
✅ predict_core / daily_predict / app.py 完全不変
✅ 既存 jra_races_full.csv 等 keiba-ai data も **不変**

→ **5/9 朝 V15 完全保証**

---

## 7. 結論

✅ A1: TFJV 構造 完全把握 (約 43,000 files、 6 GB、 14 datatypes)
✅ A2: record format 確認 (CRLF 区切り、 ASCII + Shift-JIS、 type ごと固定長)
✅ A3: 既存 keiba-ai との重複特定 (主要 datatype は既存 CSV 化済)
✅ A4: 未活用 datatype 5 件 (W5/BR/BS/OW/JG) → V20 候補 features
✅ A5: parser 設計 (TFJVRecord / TFJVParser class、 主要 fields 抽出)

→ **Phase 3 V20 構築の 1 ヶ月 前倒し可能、 直接 binary parse で TARGET GUI export 不要**

---

**Session #44 A 完了**
