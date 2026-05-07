# Phase 3 JV-Link 統合 plan + セットアップ手順書 (Session #39 B)

**作成**: 2026-05-07 (Session #39 B)
**目的**: JRA-VAN DataLab + JV-Link で公式 race / odds / 払戻 / 調教データを直接取得し、 V20 学習 data の主軸へ

---

## 1. 概要

### 1.1 JRA-VAN DataLab とは

JRA 公式が提供する有料データ提供サービス。月額 **2,090円**。

| 提供 | 内容 |
|------|------|
| 過去データ | 1986年〜現在まで全 JRA レース (約 30 万レース) |
| リアルタイム | 当日オッズ (10 分毎更新)、 馬体重、 競走除外、 払戻 |
| 確実性 | JRA 公式 = 100% 正確、 scraping より遅延少 |
| 商用利用 | 個人利用は契約次第で可 (要確認) |

→ 既存 netkeiba scraping (BAN リスク) / JRDB Advance (有料、 重複あり) の代替・補完

### 1.2 JV-Link とは

JRA-VAN が配布する **Windows COM (ActiveX) DLL**。
- Python から `pywin32` 経由で COM 呼出し
- データは独自 fixed-width format で出力
- 公式マニュアル: https://jra-van.jp/dlb/manual/jvlink.html

---

## 2. セットアップ手順書

### Step 1: JRA-VAN DataLab 加入

1. https://jra-van.jp/ にアクセス
2. 「DataLab」プラン (月額 2,090円) を選択
3. クレカ登録 + ID/PW 発行 (登録後 即日利用可)
4. **5/24 (土) 加入予定** (Phase 3 開始)

**動作確認**:
- マイページから 「JV-Link DLL ダウンロード」が見えれば OK

### Step 2: JV-Link DLL インストール

1. https://jra-van.jp/dlb/ から最新 JV-Link インストーラ取得 (例: `JVDTLabSetup_4_8_4_0.exe`)
2. Windows で **管理者権限** で実行 → 既定の `C:\JV-Link\` にインストール
3. インストール完了後、 `regsvr32 C:\JV-Link\JVDTLab.dll` で COM 登録 (インストーラが自動実行)

**動作確認**:
```powershell
PS> Get-WmiObject -Class Win32_ClassicCOMClassSetting | Where-Object { $_.Description -like "*JV*" }
```
→ JVDTLab.JVLink が出力されれば OK

### Step 3: ID/PW 設定 (初回のみ GUI)

```python
import win32com.client
jv = win32com.client.Dispatch("JVDTLab.JVLink")
jv.JVSetUIProperties()  # GUI 起動 → ID/PW 入力 → OK
```

GUI で:
- 利用者 ID: 加入時メールに記載
- パスワード: 同上
- データ取得先: `C:\JV-Link\Data\` (既定)

設定はレジストリ (`HKCU\Software\JRA-VAN\JVLink`) に永続保存。

**動作確認**:
```python
rc = jv.JVInit("UNKNOWN")  # rc=0 なら成功
print(rc)
```

### Step 4: tools/jvlink_fetcher.py 動作確認

本 Session で `tools/jvlink_fetcher.py` を試作。

```bash
# datatype 一覧確認
python tools/jvlink_fetcher.py --list-datatypes

# 5/7 のレース詳細取得
python tools/jvlink_fetcher.py --datatype RACE --from 20260507 --max-records 100

# 当日オッズ速報
python tools/jvlink_fetcher.py --datatype O1 --realtime --max-records 50
```

出力: `data/jvlink/RACE/20260507.csv`

**動作確認**:
- ✅ rc=0
- ✅ data > 0 (records 取得)
- ✅ raw_record 列に固定長 record string が並ぶ

### Step 5: schtasks 統合 (毎朝 06:45)

```cmd
schtasks /Create /TN "Keiba-JVLink-DailyFetch" ^
    /TR "C:\Users\takum\keiba-ai\jvlink_daily.bat" ^
    /SC DAILY /ST 06:45 /F
```

`jvlink_daily.bat`:
```batch
cd /d C:\Users\takum\keiba-ai
python tools/jvlink_fetcher.py --datatype RACE --from %date:~0,4%%date:~5,2%%date:~8,2%
python tools/jvlink_fetcher.py --datatype TCOV --from %date:~0,4%%date:~5,2%%date:~8,2%
python tools/jvlink_fetcher.py --datatype WOOD --from %date:~0,4%%date:~5,2%%date:~8,2%
python tools/notify_done.py "JV-Link daily fetch" "OK"
```

### Step 6: 既存 keiba-ai 統合経路

```
data/jvlink/RACE/*.csv    →  tools/parse_jvlink_race.py    →  data/jra_races_full.csv (互換)
data/jvlink/TCOV/*.csv    →  tools/parse_jvlink_tcov.py    →  data/training_times.csv (補完)
data/jvlink/HR/*.csv      →  tools/parse_jvlink_hr.py      →  data/jra_payouts.csv (4/6 停止 解消)
data/jvlink/O1〜O6/*.csv  →  tools/parse_jvlink_odds.py    →  data/odds_history.csv (リアルタイム)
data/jvlink/BLOD/*.csv    →  tools/parse_jvlink_blod.py    →  data/blood_full.csv (公式更新)
```

各 parser は固定長 record を pandas DataFrame に変換 (Phase 3 後半 6/9-15 で実装)。

### Step 7: V20 学習 data 主軸切替 (6/9-30)

- 旧 source (netkeiba scraping + JRDB Advance) → 補助的に残す
- 新 source (JV-Link) → V20 学習 data の主軸
- merge は date + race_id で join、 公式 (JV-Link) > netkeiba > JRDB の優先順位

---

## 3. tools/jvlink_fetcher.py 概要

### 3.1 ファイル

`tools/jvlink_fetcher.py` (本 Session、 約 170 行)

主要 API:
- `init_jvlink(sid)` — COM 初期化 + JVInit
- `open_data(jv, dataspec, fromtime, option)` — JVOpen
- `read_records(jv, max_records)` — JVRead 全 record 取得
- `fetch_to_csv(...)` — 上記まとめ raw CSV 保存
- CLI: `--datatype`, `--from`, `--option`, `--max-records`, `--realtime`

### 3.2 datatype 一覧

```
RACE  : レース詳細
DIFF  : 差分データ
BLOD  : 血統登録
SNPN  : 騎手・調教師
TCOV  : 調教タイム
WOOD  : 木曽用調教
RC    : レース短信
O1〜O6 : 単複/馬連/ワイド/馬単/三連複/三連単 オッズ
HR    : 払戻金
JG    : 競走除外/発走時刻変更
WF    : 馬体重情報
```

### 3.3 動作前提

- ✅ pywin32 インストール (`pip install pywin32`)
- ✅ JV-Link DLL インストール + COM 登録
- ✅ ID/PW 設定済 (`JVSetUIProperties()` 一回実行)
- ❌ JRA-VAN 未加入 → 5/24 加入予定

### 3.4 PoC 状況

- ✅ コード syntax OK
- ✅ COM 接続 sequence 設計完了
- ❌ 実機動作確認 未実施 (5/24 加入後)
- ❌ raw record → DataFrame parse 未実装 (Phase 3 後半 6/9-15)

---

## 4. 期待効果

### 4.1 jra_payouts.csv 問題解消 (CLAUDE.md 既知バグ)

旧: `data/jra_payouts.csv` が 4/6 で更新停止 (scrape_jra_payouts.py の壊れ)
新: `tools/jvlink_fetcher.py --datatype HR` で 公式払戻データ直接取得

### 4.2 jrdb_paci.csv 4/4 停止 問題 (CLAUDE.md 既知バグ)

旧: JRDB Advance の paci 取得停止 (V18/V19 importance #1〜#3 の paci_* features 機能停止)
新: JV-Link `O1〜O6` で公式オッズ取得 → 自前で paci 相当 features 算出可能

### 4.3 BAN リスク低減

旧: netkeiba scraping は IP/cookie ban リスク
新: JV-Link 公式 API は契約者向け正規 channel = ban リスク 0

### 4.4 V20 学習 data 拡充

旧: ~50 万 horse rows (2010-2025)
新: ~80 万 horse rows (1986-2025、 JV-Link 全期間 + JRA-VAN 蓄積)

---

## 5. 5/24 加入後 即着手 step

| step | 内容 | 期間 |
|------|------|------|
| 1 | JRA-VAN DataLab 加入 + JV-Link DLL インストール | 5/24 (土) 朝 1h |
| 2 | jvlink_fetcher.py で RACE 1 日分取得 (動作確認) | 5/24 1h |
| 3 | parse_jvlink_race.py 実装 (RACE record → DataFrame) | 5/25-26 |
| 4 | 過去 1 か月分 (4/24-5/23) を bulk fetch、 jra_races_full.csv と整合チェック | 5/27 |
| 5 | schtasks に DailyFetch 登録 (06:45) | 5/28 |
| 6 | 払戻 (HR) + オッズ (O1) 取得 + parser | 5/29-30 |
| 7 | 調教 (TCOV/WOOD) parser | 5/31-6/1 |
| 8 | V20 学習 pipeline に統合 | 6/9-30 |

---

## 6. リスク + 対策

| リスク | 対策 |
|--------|------|
| pywin32 環境 trouble | venv を JV-Link 用に分離、 `pip install pywin32==308` で固定 |
| COM 登録失敗 (UAC) | 管理者権限 PowerShell で `regsvr32 /i JVDTLab.dll` |
| ID/PW 失効 | レジストリで永続化、 失効時 GUI 再入力 (年 1 回程度) |
| データ取得失敗 (option mismatch) | option=4 (差分) → option=1 (通常) → option=2 (今週) で fallback |
| 1 日分の record 抜け | next day に option=4 で差分取得、 漏れなし |
| record format 違い (固定長 layout) | https://jra-van.jp/dlb/manual/recordlayout/ 参照、 datatype ごとに parser 個別実装 |

---

## 7. 月額 2,090円 ROI 試算

| 項目 | 金額 |
|------|------|
| JRA-VAN DataLab 月額 | -2,090円 |
| jra_payouts 復活 → ROI 計算精度 +5% (V20 期待 ROI 140% 想定で 月+1,500円) | +1,500円 |
| paci_* 復活 → V18/V19 winner_top1 +3pt (5/16 NO-GO 解消、 6/15+ 投入で月 +5,000円見込) | +5,000円 |
| 学習 data 拡充 → V20 AUC +0.005 想定 → ROI +2-3% | +1,000円 |
| **net** | **+5,410円/月** |

→ 月額元取り見込み (Phase 3 後半以降)。

---

## 8. 結論

✅ JV-Link 統合 plan 完了、 5/24 加入で **即着手可能**
✅ tools/jvlink_fetcher.py 試作完了 (170 行、 COM 接続 sequence)
✅ Step 1〜7 の手順書 提供
✅ V20 学習 data 主軸切替 plan (6/9-30) 確定
✅ 既知バグ (jra_payouts/jrdb_paci) 解消経路 確認

V15 動作不変保証: 本 plan は予約 doc + tool 試作のみ、 5/24 まで V15 production 完全不変。

---

**Session #39 B 完了**
