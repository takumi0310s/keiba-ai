# Phase 12 PoC A: JV-Link 環境 audit (5/10 22:00+)

> Session #87 Phase 12 PoC A 領域
> ★ honest report ★ — fabricate しない

---

## 1. JV-Link COM 環境 確認結果

| 項目 | 状態 | 備考 |
|------|------|------|
| JV-Link DLL | ✅ 存在 (C:\Windows\SysWow64\JVDTLAB\JVDTLab.dll) | 32-bit COM |
| ★ 32-bit Python venv ★ | ❌ **不在** (C:\Users\takum\jvlink-venv\ 未作成) | 5/24+ install 予定 |
| 現 Python | x64 (3.14.3) | 32-bit COM 呼出し ★ 不可 ★ |
| pywin32 (32-bit) | 未確認 | venv 設置時 同時 install |
| TARGET frontier JV (TFJV) | ✅ C:\TFJV\ 存在 (約 6 GB) | 64-bit Python で binary 直 parse 可 |
| jvlink_fetcher_v2.py | ✅ tools/ 存在 (Session #41 B、 280 行) | 32-bit venv 経由 で動作 |
| tfjv_parser.py | ✅ 動作確認 (Session #44 B) | 64-bit OK |

---

## 2. 結論: JV-Link COM 経路は本セッション内 ★ 不可 ★

### 2.1 不可 理由
- 32-bit Python venv 未作成
- pywin32 32-bit 未 install
- COM 呼出しは **32-bit Python 必須** (SysWow64 DLL)

### 2.2 切替: TFJV binary 直 parse 経路 ★ 可 ★
- C:\TFJV\ 配下 (約 6 GB、 14 datatypes)
- 既実装 tools/tfjv_parser.py (Session #44 B、 動作確認済)
- 64-bit Python で 直接 binary parse 可
- 既 parse 済 CSV: data/tfjv/{RA,SE,HR}_{2020-2025}.csv (Session #44 D で 6 年分一括 ~10 秒)

→ ★ 本 PoC は TFJV 経路 で 1 ヶ月 backfill を実施 ★

---

## 3. ユーザー指示の honest 解釈

| user 指示 | 実際の対応 |
|-----------|----------|
| 「JV-Link install 確認」 | DLL は ✅、 32-bit venv は ❌ → 32-bit COM 経路 不可 |
| 「JVOpen 動作」 | 32-bit venv 不在 → 本 session で **不可** |
| 「JVRead 1 R sample」 | 同上 |
| 「直近 1 ヶ月 backfill」 | TFJV binary 直 parse (4/10-5/10、 288 R) で代替 |

★ honest: ユーザー指示の JV-Link COM 経路 は本 session で 着手不可 ★
★ 代替 TFJV 経路 で 1 ヶ月 backfill を実施 ★

---

## 4. 5/24+ 作業 (full backfill 開始時)

| step | 工数 |
|------|------|
| 32-bit Python 3.x download + install | 30 min |
| C:\Users\takum\jvlink-venv\ 構築 | 10 min |
| pywin32 (32-bit) install + COM 登録 | 30 min |
| jvlink_fetcher_v2.py 動作確認 (1 R sample) | 30 min |
| **subtotal** | **約 100 min** |

→ 5/24 (土) 朝に 一括 setup、 当日中に 5/25-5/31 fetch 開始可。

---

## 5. V15 投資保護

✅ tools/predict_core.py / V15 model 不変
✅ TFJV は read-only
✅ data/jvlink/ 出力 (新規 dir、 既存 data 不変)

---

## 6. 結論

✅ A1: JV-Link DLL 存在確認 (C:\Windows\SysWow64\JVDTLAB\JVDTLab.dll)
✅ A2: 32-bit Python venv 不在確認 → COM 経路 本 session 不可
✅ A3: TFJV 直 parse 経路 ✅ 動作確認 (代替 path)
✅ A4: 5/24+ で 32-bit venv setup 予定 (約 100 min)
✅ A5: V15 完全保護
