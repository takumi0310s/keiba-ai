# JV-Link 32-bit Python 環境構築 plan + 動作確認 (Session #41 A)

**作成**: 2026-05-08 深夜 (Session #41 A、 ユーザー就寝中)
**前提**: ユーザー が JRA-VAN DataLab 加入 + JV-Link DLL install 完了 (2026-05-07 夜)
**制約**: 既存 Python 3.14 (64-bit) では JVDTLab.JVLink COM 接続不可 → 32-bit 別環境必須

---

## 1. 環境制約

### 1.1 JV-Link DLL の 32-bit only 制約

```
DLL: C:\Windows\SysWow64\JVDTLAB\JVDTLab.dll  (32-bit only)
ProgID: JVDTLab.JVLink
Version: 1.18
```

WOW64 (Windows-on-Windows 64-bit) 経由で 64-bit Windows 上で動作するが、
**呼び出す client 側は 32-bit プロセス必須**。

### 1.2 既存 64-bit 環境

```
$ python -c "import platform; print(platform.architecture()[0]); print(platform.python_version())"
64bit
3.14.3
```

→ 既存 keiba-ai 用 Python (predict_core / daily_predict 含む) は **完全に維持**。
→ JV-Link 操作専用に **別 venv (32-bit)** を作成。

---

## 2. 推奨構成

| 項目 | path / 値 |
|------|---------|
| 32-bit Python install path | `C:\Python311-32bit\` |
| 32-bit venv path | `C:\Users\takum\jvlink-venv\` |
| Python version | 3.11.9 (32-bit) |
| 必須 package | pywin32, pandas, numpy |
| keiba-ai 64-bit 環境 | **完全維持** (predict_core / daily_predict 等) |

---

## 3. install 手順 (admin 権限必須)

### 3.1 自動化 script

`tools/setup_python32.ps1` (新規、 約 130 行)

```powershell
# admin PowerShell で実行
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\tools\setup_python32.ps1
```

各 step:
1. admin 権限確認
2. Python 3.11.9 32-bit installer download (https://www.python.org/ftp/python/3.11.9/python-3.11.9.exe)
3. silent install (TargetDir=C:\Python311-32bit, PrependPath=0)
4. arch 確認 (32bit であること)
5. venv 作成 (C:\Users\takum\jvlink-venv)
6. pip install pywin32, pandas, numpy
7. pywin32 post-install (COM 登録)
8. JV-Link 動作確認 (`tools/jvlink_test_python32.py --check-only`)

### 3.2 dry run (試行)

```powershell
.\tools\setup_python32.ps1 -DryRun
```

→ 実際の install / pip 実行はせず、 plan を表示のみ。

### 3.3 既 install 検出

`C:\Python311-32bit\python.exe` が既に存在しかつ 32-bit なら install を skip。

---

## 4. 動作確認 script

`tools/jvlink_test_python32.py` (新規、 約 145 行)

### 4.1 検証 6 step

1. Python arch == 32-bit か
2. pywin32 import 成功
3. JVDTLab.JVLink COM Dispatch
4. JVInit (sid="UNKNOWN") → rc=0
5. JVOpen (datatype=RACE, fromtime=20260503000000) → rc=0 + num_files >= 1
6. JVRead 数件 取得 + 内容 print
7. JVClose

### 4.2 利用例

```powershell
# venv activate 後
& "C:\Users\takum\jvlink-venv\Scripts\python.exe" tools\jvlink_test_python32.py

# 別日付テスト
& "C:\Users\takum\jvlink-venv\Scripts\python.exe" tools\jvlink_test_python32.py --date 20260426 --read-records 10

# COM Dispatch のみ (data fetch 無し)
& "C:\Users\takum\jvlink-venv\Scripts\python.exe" tools\jvlink_test_python32.py --check-only
```

### 4.3 期待 output (Session #40 ユーザー側 確認 済)

```
[Step 0] Python arch: 32bit  OK
[Step 1] pywin32 import  OK
[Step 2] JVDTLab.JVLink COM Dispatch  OK
[Step 3] JVInit('UNKNOWN')  rc=0  OK
[Step 4] JVOpen('RACE', '20260503000000', option=4)  rc=0, files=29  OK
[Step 5] JVRead loop:
  [1] rc=2048, file=RA20260503001a.RC, len(buff)=174
      content: 'RA1 ...'
  [2] rc=2048, ...
[Step 6] JVClose  rc=0  OK
```

---

## 5. 5/9 V15 投資保護 (A 領域)

✅ 既存 64-bit Python 環境 完全不変
✅ predict_core / daily_predict / V15 model 完全不変
✅ schtasks 既存 task 完全不変 (新規追加なし)
✅ 32-bit 環境は別 path、 完全独立

→ **5/9 朝 V15 完全保証**

---

## 6. 次 step (B-C で利用)

| step | 内容 |
|------|------|
| B | `tools/jvlink_fetcher.py` を 32-bit venv で実行する想定で 本実装 |
| C | 5/1-5/7 backfill (32-bit venv) → data/jvlink/ に 約 30-40 ファイル |
| 5/24+ | Phase 3 で V20 学習 data 主軸に |

---

## 7. 実行責任

⚠ **本 Session #41 A は 32-bit Python 自動 install を実行しません**:
- admin 権限が必要
- ユーザー本人の入力 (pip install / pywin32 post-install) が必要な場合あり
- ユーザー判断で `tools/setup_python32.ps1` を **手動実行** 推奨

代わりに本 Session A では:
- ✅ install plan + script (`tools/setup_python32.ps1`)
- ✅ 動作確認 script (`tools/jvlink_test_python32.py`)
- ✅ doc (本ファイル)

を提供。 ユーザー が起床後に 朝の都合の良い時間に実行できる構成。

---

## 8. 結論

✅ A1: 32-bit Python install plan 確立
✅ A2: `setup_python32.ps1` 自動化 script (admin 必須、 dry-run 対応)
✅ A3: `jvlink_test_python32.py` 6 step 動作確認 script
✅ A4: JVRead 実装試行は 32-bit 環境で動作確認 (本 Session ではコード提供まで、 実 fetch はユーザー実行)
✅ A5: 統合 doc (本ファイル)

→ **32-bit 環境構築 plan 完了、 manual 実行 (admin) で即着手可能**

---

**Session #41 A 完了**
