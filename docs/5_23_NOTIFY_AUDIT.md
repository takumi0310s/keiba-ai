# 5/23 V21 paper 全通知 整合性 audit

**実施時刻**: 2026-05-23  
**commit**: 本 doc と同 commit

---

## ① TYB fetch 修正確認

| 修正 | 状態 |
|------|------|
| `fetch_tyb_for_race` → `fetch_tyb_observe` 切替 | ✅ commit 49577719 |
| `_load_env` JRDB_USER → JRDB_ID fallback | ✅ commit 49577719 |
| `7z` full path fix (`C:\Program Files\7-Zip\7z.exe`) | ✅ 本 commit |
| `_inject_tyb_features` raw_rows 対応 + tansho/fukusho 追加 | ✅ 本 commit |

### 5/23 TYB fetch 実績

| 修正 層 | 症状 | 修正 |
|---------|------|------|
| Layer 1: enabled gate | `TYB_SHADOW_ENABLED=False` で即 return | fetch_tyb_observe に切替 |
| Layer 2: credentials | `JRDB_USER` 未設定 → `JRDB_ID` fallback | _load_env 修正 |
| Layer 3: 7z not in PATH | `[WinError 2]` 7z.exe not found | full path 指定 |
| Layer 4: data structure | raw_rows 非対応、tansho/fukusho_odds 欠落 | _inject_tyb_features 全面修正 |

**5/23 TYB 実績**: 全 fire で `TYB=False` → 未取得。3 層の修正により 5/24+ での取得が期待できる。

---

## ② 全通知 整合性

### 通知ログ (paper log ファイル)

| race_id | 時刻 | 場所 | TYB | strategy_pass |
|---------|------|------|-----|--------------|
| 202604010710 | 14:29 | 新潟10R | False | False |
| 202604010711 | 15:04 | 新潟11R | False | False |
| 202604010712 | 15:45 | 新潟12R | False | True |
| 202605020909 | 14:03 | 東京9R | False | False |
| 202605020910 | 14:39 | 東京10R | False | False |
| 202605020911 | 15:14 | 東京11R | False | True |
| 202605020912 | 15:54 | 東京12R | False | True |
| 202608030909 | 14:14 | 京都9R | False | False |
| 202608030910 | 14:49 | 京都10R | False | False |
| 202608030911 | 15:29 | 京都11R | False | False |
| 202608030912 | 16:14 | 京都12R | False | False |

**ファイル重複**: なし (11 unique race_id)

### Discord 重複 (新潟11R) の根本原因

新潟11R は Discord に 2 件届いた。

**原因**: 前セッション中に複数の V21 paper プロセスが同時起動していた。
- PID 20476 (14:00 起動) + その後の再起動プロセスが並存
- `v21_paper_20260523.log` は複数プロセスが共有 → 誰が fire したか判定困難

**証拠**: paper log ファイルは 1 race_id = 1 ファイルなので後書きが上書き → 重複検出不可。

### 今後の重複防止

```bash
# 起動前に必ず全 V21 paper プロセスを kill
pkill -f v21_per_race_paper.py 2>/dev/null
sleep 1
nohup python -u tools/v21_per_race_paper.py > logs/v21_paper_$(date +%Y%m%d).log 2>&1 &
```

---

## ③ デザイン統一感

### V15 vs V21 paper 現状比較

| 要素 | V15 (race_auto_notify) | V21 paper (v21_per_race_paper) |
|------|----------------------|-------------------------------|
| チャンネル | #買い目 (DISCORD_WEBHOOK_BETS) | #買い目 (DISCORD_WEBHOOK_BETS) ✅ |
| フォーマット | embed (title/body/color) | plain text |
| ヘッダー | `🏇 コース+R レース名 条件★★★` | `🚫🚫🚫【V21 paper — 投票禁止】🚫🚫🚫` |
| 馬テーブル | `全馬 V15 score 順` markdown table | 全馬 V21 score 順 markdown table ✅ |
| 単勝オッズ | ✅ 列あり | ✅ 列あり |
| 買い目 | フォーメーション 7点 | フォーメーション 7点 / filter除外表示 ✅ |
| フッター | 投票方針 | `🚫 paper予測 V15のみで投票` ✅ |

**統一すべき点**:
1. フォーマット: V15 は embed (title + body)、V21 は plain text → V15 側の embed 構造に合わせるか検討
2. ヘッダー: V21 の `🚫🚫🚫` は視認性高く維持推奨
3. V21 通知に「V15 との比較」欄を追加 (top1 の一致/相違) → 次フェーズ

---

## 現在の process 状態

| PID | プロセス | 状態 |
|-----|---------|------|
| 28500 | race_auto_notify (V15) | ✅ 継続稼働 |
| 2797 | v21_per_race_paper | ✅ 正常終了 (All timers fired) |

*V15 production 完全不変*
