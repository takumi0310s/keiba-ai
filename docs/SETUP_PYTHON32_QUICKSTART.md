# 32-bit Python + JV-Link クイックスタート (Session #42 A)

**作成**: 2026-05-08 (Session #42 A、 ユーザー仕事中)
**前提**: Session #41 A で setup_python32.ps1 + jvlink_test_python32.py 試作済
**目的**: ユーザー実行手順を 1 ページに集約 (admin 操作 のみ minimum)

---

## 1. 実行 step (admin PowerShell、 約 15 分)

### Step 1: PowerShell admin 起動

スタートメニュー → PowerShell 右クリック → 「管理者として実行」

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
cd C:\Users\takum\keiba-ai
```

### Step 2: 32-bit Python install (約 5-10 分)

```powershell
.\tools\setup_python32.ps1
```

→ 自動的に:
1. Python 3.11.9 32-bit installer download
2. `C:\Python311-32bit\` に silent install
3. venv 作成: `C:\Users\takum\jvlink-venv\`
4. pywin32 / pandas / numpy install
5. pywin32 COM 登録
6. JV-Link 動作確認 (sid=UNKNOWN で JVInit 確認)

### Step 3: 動作確認 (約 1-2 分)

```powershell
& "C:\Users\takum\jvlink-venv\Scripts\python.exe" tools\jvlink_test_python32.py
```

期待 output:
```
[Step 0] Python arch: 32bit  OK
[Step 1] pywin32 import  OK
[Step 2] JVDTLab.JVLink COM Dispatch  OK
[Step 3] JVInit('UNKNOWN')  rc=0  OK
[Step 4] JVOpen('RACE', '20260503000000', option=4)  rc=0, files=29  OK
[Step 5] JVRead loop:
  [1] rc=2048, file=RA20260503001a.RC, len(buff)=174
      content: 'RA1 ...'
[Step 6] JVClose  rc=0  OK
```

### Step 4: 5/1-5/7 backfill 試行 (約 14 分)

```powershell
& "C:\Users\takum\jvlink-venv\Scripts\python.exe" tools\jvlink_backfill_5_1_5_7.py
```

→ data/jvlink/ 配下に raw + meta CSV 出力 (約 84 files、 5-15 MB)

---

## 2. trouble shoot

### admin 権限 不足

```
[ERROR] Admin 権限で実行してください。
```

→ PowerShell を 「管理者として実行」 で再起動

### 既 install 検出

```
[INFO] 32-bit Python 既に install 済 → install skip
```

→ 既存環境を流用、 venv のみ作成 (問題なし)

### COM Dispatch 失敗

```
[Step 2] JVDTLab.JVLink COM Dispatch
  [ERROR] Dispatch 失敗
```

→ JV-Link DLL 未登録疑い。 admin で:
```powershell
regsvr32 "C:\Windows\SysWow64\JVDTLAB\JVDTLab.dll"
```

### JVInit rc != 0

```
[Step 3] JVInit('UNKNOWN')
  rc=-301
  [WARN] rc != 0、 ID/PW 未設定の可能性
```

→ ID/PW 設定:
```python
import win32com.client
jv = win32com.client.Dispatch("JVDTLab.JVLink")
jv.JVSetUIProperties()  # GUI 起動 → ID/PW 入力
```

---

## 3. install 後の利用

### 3.1 daily fetch (推奨 schtasks 追加)

```cmd
schtasks /Create /TN "Keiba-JvlinkDailyFetch" ^
    /TR "powershell -File C:\Users\takum\keiba-ai\jvlink_daily.ps1" ^
    /SC DAILY /ST 06:45 /F
```

`jvlink_daily.ps1`:
```powershell
cd C:\Users\takum\keiba-ai
$today = Get-Date -Format "yyyyMMdd"
& "C:\Users\takum\jvlink-venv\Scripts\python.exe" tools\jvlink_fetcher_v2.py `
    --date $today --datatypes RACE,SE,HR --parse
& "C:\Users\takum\jvlink-venv\Scripts\python.exe" tools\notify_done.py `
    "JV-Link Daily Fetch" "$today RACE/SE/HR 完了"
```

### 3.2 6 年分 backfill (Phase 3 後半 6/9-13)

```powershell
# admin: schtasks に Nightly 登録
schtasks /Create /TN "Keiba-JvlinkBackfillNightly" `
    /TR "powershell -File C:\Users\takum\keiba-ai\jvlink_backfill_nightly.ps1" `
    /SC DAILY /ST 23:00 /F
```

`jvlink_backfill_nightly.ps1`:
```powershell
cd C:\Users\takum\keiba-ai
& "C:\Users\takum\jvlink-venv\Scripts\python.exe" tools\jvlink_full_backfill.py `
    --from 20200101 --to 20251231 --datatypes RACE,SE,HR,O1 `
    --resume --max-runs 200 --monthly
```

→ 毎晩 200 fetch (約 1.5h)、 6 年分を 30-40 日で完了見込

---

## 4. 5/9 V15 投資保護 (A 領域)

✅ 32-bit Python install は **別 path** (C:\Python311-32bit\)
✅ 既存 64-bit Python 3.14 (keiba-ai 本体) は完全維持
✅ predict_core / daily_predict / V15 model 不変
✅ schtasks 既存 task 不変
✅ install は 5/9 V15 投資には **不要**、 起床後の都合の良い時間で実行可

→ **5/9 朝 V15 完全保証**

---

## 5. install しない場合の影響

| 機能 | install 必須 | 代替 |
|------|------------|------|
| V15 daily_predict | **不要** | 既存 64-bit Python のまま動作 |
| netkeiba scraping | **不要** | 既存のまま |
| JRDB Advance | **不要** | 既存のまま |
| JV-Link 直接取得 | 必要 | 5/24+ Phase 3 で実行、 6/9-30 V20 構築用 |

→ 5/9 投資には install 不要、 5/24+ Phase 3 開始までに admin 実行で十分

---

## 6. 結論

✅ A1: setup_python32.ps1 (Session #41 A から維持)
✅ A2: jvlink_test_python32.py (Session #41 A から維持)
✅ A3: 本ファイル (1 ページ admin 手順書、 trouble shoot 含む)
✅ A4: 5/9 投資 完全独立 確認

→ **ユーザー手動 install で 約 15 分、 admin 操作のみ**
→ **5/24+ Phase 3 着手前に いつでも実行可能**

---

**Session #42 A 完了**
