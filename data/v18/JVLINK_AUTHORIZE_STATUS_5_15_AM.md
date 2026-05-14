# JV-Link AI 自律実行 limit (5/15 AM、 user authorize 状態)

実行: 2026-05-15 AM、 Opus 4.7
user 指示: "JVOpen 実 data fetch 着手 authorize する"

## 現状

### ✅ 動作確認済 (AI 自律 OK)

- 32-bit Python 3.11.9 install (C:/Users/takum/python32/)
- pywin32 install
- win32com.client.Dispatch('JVDTLab.JVLink') → COMObject 取得
- **JVInit('UNKNOWN') → ret=0 (OK)**
- **JVOpen('RACE', '20260503000000', 4) → ret=0, 32 files, 30 to download** ★ 一度成功 ★
- JVClose() → 正常

### ❌ 続行 blocked (auto-mode classifier)

user 自然言語 で authorize したが、 Claude Code auto-mode classifier は:
- Production Reads / Credential Exploration / 外部 paid service 経由 と判定
- 32-bit Python 自律 install を **scope escalation** と判定
- **bash permission rule (settings.json) で 明示 allow 必要**

実 fetch (JVOpen 2 度目以降、 JVRead loop、 JVStatus loop) が:
```
Permission denied by Claude Code auto mode classifier.
Reason: ... Production Reads-style operation against external production infrastructure.
```

## user 解決 path (3 択)

### Option A: settings.local.json 編集 (推奨)

`C:/Users/takum/.claude/settings.local.json` (or プロジェクト .claude/settings.local.json) に追加:

```json
{
  "permissions": {
    "allow": [
      "Bash(C:/Users/takum/python32/python.exe:*)",
      "Bash(C:/Users/takum/python32/*.exe:*)"
    ]
  }
}
```

→ 32-bit Python 経由の JV-Link 操作 を 自動 allow。

### Option B: 各 fetch を user 個別 認可

user が各 JVOpen/JVRead call を `permission grant` ボタンで個別認可。
- 工数 大 (28 種 datatypes × 各 fetch)
- 留守中 不可

### Option C: AI スコープ外で user 手動

user が直接 32-bit Python で fetch script 実行 (留守復帰後)。
- AI が 一旦 data 取得後の処理 (parse / merge / 学習) 着手

## 提案

★ **Option A 推奨 + 即実行可能** ★

user は 既に "JVOpen 着手 authorize する" と verbal で OK 出している。 これを 永続化 する settings.local.json 編集 で:
- AI 自律 で 17 features 真値化 着手可能
- V20 真の構築 1-2 週間 前倒し path 開通
- user 帰宅 待ち time 損失 ゼロ

settings.local.json は 既存:
```json
// 現在 (recurring 中 確認):
{
  "permissions": {
    "allow": [ ... 既存 rules ],
    "ask": [ ... ],
    "deny": [ ... ]
  }
}
```

`allow` に 1 行 追加 で 解禁:
```
"Bash(C:/Users/takum/python32/python.exe:*)"
```

## 暫定 (AI 自律 続行可能 task)

JV-Link RT fetch 待たず でも AI 着手可能:
1. ✅ 28 種 datatypes record layout 仕様 整理 doc 作成
2. ✅ JV-Link wrapper module 設計 + skeleton 実装
3. ✅ 既存 TFJV binary 経由 で SE/HR/RA 追加 features 抽出 (Phase 13 続き)
4. ✅ proper calibrator rebuild (5/16+ data 蓄積後)

これらは AI 自律 で 並行可能。 JV-Link RT は user authorize 後 着手。

## V15 投資保護 完全 (本日も遵守)

V15 .pkl.gz / predict_core / daily_predict / app.py 完全不変。
32-bit Python は別 path、 V15 inference 干渉なし。

## まとめ

★ AI 単独 で **JV-Link COM 接続性 完全動作確認** ★
★ JVOpen 一度成功 (32 files / 30 download 確認) ★
★ 但し 後続 fetch loop は auto-mode で blocked、 user の settings.local.json 編集が次 unlock 鍵 ★

User 帰宅後 1 行 追加で:
- 17 features 残 10 件 真値化 (AI 自律 1-2 日)
- V20 真の構築 (AI 自律 3-5 日)
- 6/15+ V20 production 投入判定 (V15 比較 後)
