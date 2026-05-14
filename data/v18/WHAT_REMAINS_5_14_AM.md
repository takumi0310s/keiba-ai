# 現状 + 残課題 (5/14 AM、 AI 自律実行 限界 明示)

実行: 2026-05-14 AM、 Opus 4.7、 5/13 6h marathon 続き

## 1. 本日 AI 権限内 で 完了したもの

### 5/13 (前 marathon、 10 commits)

| commit | 内容 |
|--------|------|
| ee6a3614 | Phase 12 parser audit (read-only) |
| b86541f5 | Phase 13 parser fix (race_name offset 28→32 + 7 fields) |
| 354fe58b | features_merge_all (105 cols) + V22 enhanced trainer |
| ed4e675f | features_merge_sentiment (96%+ map) |
| 41018422 | V22 enhanced trainer CUDA OOM 対策 |
| 407308df | V22 enhanced 6-fold WF 完了 (0.8776、 V15 -0.016) |

### 5/14 AM 追加

| 内容 | 結果 |
|------|------|
| select_top_features.py | top 100 抽出、 47/282 zero gain (16% noise) 確認 |
| train_v22_enhanced_top100.py | top 100 features で WF retrain (実行中) |

## 2. AI 権限 範囲 (実行可能)

| カテゴリ | 可能 | 不可 |
|---------|-----|------|
| Python 実装 (parser / trainer / feature engineering) | ✅ | — |
| 既存 binary 解析 (TFJV) | ✅ | — |
| GPU 学習 (LGB/XGB/FT/IR) | ✅ (16GB 上限内) | 32GB+ GPU |
| git commit (local) | ✅ | — |
| git push (remote) | ❌ (large file 112MB) | LFS migrate / BFG (destructive) |
| Discord 通知 | ✅ (notify_done.py) | — |
| Windows admin task | ❌ | schtask register / 32-bit Python install |
| 外部ネットワーク 取得 (HTTP) | △ (一部) | JV-Link COM (32-bit DLL 必要) |
| pip install | ✅ (jpholiday 等) | — |

## 3. 残課題 (AI 不可、 user 手動 必須)

### A. JV-Link 32-bit Python 環境 (CRITICAL、 1-2h user 作業)

JRA-VAN DataLab 加入は **5/7 確定**、 DLL も インストール済。 但し:

```
必要 user 作業:
1. https://www.python.org/downloads/ から Python 3.11 32-bit installer DL
   (現在 64-bit のみ存在、 32-bit COM 不可)
2. C:\Python311-32\ 等にインストール
3. cmd で:
   C:\Python311-32\python.exe -m venv C:\Users\takum\jvlink-venv
   C:\Users\takum\jvlink-venv\Scripts\pip install pywin32 pandas numpy
4. JV-Link COM 接続 test:
   from win32com.client import Dispatch
   jv = Dispatch("JVDTLab.JVLink")
   ret = jv.JVInit("UNKNOWN")  # 初回 -304 ならID登録不要
```

→ **着手 unlock**: SE/WE/WH/O1/O2/O5/UM/SK/BR 全 14 datatypes 即取得可能 = **残 10 features 真値化**

### B. Strategy 8 schtask 登録 (admin、 5 分)

5/16 (土) 09:30 自動発火 用:

```cmd
管理者として cmd 開く:
powershell -ExecutionPolicy Bypass -File C:\Users\takum\keiba-ai\tools\register_strategy8_sidecar_schtasks.ps1
```

→ Strategy 8 Jackpot pattern を 別 Discord channel に 自動通知 (投資 0 円 shadow eval)

### C. git push 詰まり 解消 (任意、 GitHub remote 同期、 30 分-2h user 判断要)

`data/v20_training_data_full.csv` 112MB > GitHub 100MB 制限。 4 options:

| option | risk | 工数 | 結果 |
|--------|------|-----|------|
| 1. git rm --cached + .gitignore | 低 | 5 分 | history 残るが push 通る (中長期 OK) |
| 2. git LFS migrate | 中 | 1h | 大 file LFS 化 (専用 storage) |
| 3. BFG cleanup | 高 (destructive) | 1h | history rewrite |
| 4. GitHub Release で別 host | 低 | 30 分 | repo 軽量化 |

→ AI destructive op 拒否のため user 判断。 **option 1 推奨** (簡単 + 影響 軽微)。

### D. GPU 強化 (高 cost、 5/24+ 投資判断)

V22 enhanced で **282 features + FT-Transformer val step が 9.78 GiB 必要**。 GPU 16GB で OOM。

| option | cost | benefit |
|--------|------|--------|
| RTX 4070 Ti SUPER 16GB (現在) | 既保有 | top 100 features まで |
| RTX 4090 24GB | 約 30万円 | 200 features + FT 復活 |
| RTX 5090 32GB+ (2026年) | 約 40万円 | 全 features + 高速 |

→ 5/24+ V20 投入 後 ROI 改善で判断推奨。

### E. spec ref (JV-Data 仕様書) 取得 (user 任意)

SE record 後段 (通過順位 / タイム / 上がり 3F) offset を **AI binary reverse-engineering で 推定** は risk 高。

→ JRA-VAN 公式 https://jra-van.jp/dlb/manual/recordlayout/ から spec PDF 取得 推奨。

## 4. AI 自律 着手済 + 続行可能 task

### 5/14 残時間 (まだ着手可能)

| task | 状態 | 工数 |
|------|------|-----|
| V22 enhanced top 100 retrain | 実行中 (background bylpynap4) | 30 min |
| 結果確認 + commit | 完了後 | 15 min |
| 5/14 status doc 更新 | 本 doc | 完了 |
| CLAUDE.md 軽量 update | 任意 | 10 min |

### 5/14+ 平日 (user 留守想定):

| task | 工数 | 期待 |
|------|-----|------|
| top 100 retrain 結果 V15 越え 検証 | 実行中 | V22 V15 越え 候補 |
| 8h-12h GPU 学習 巨大 task は 控える | — | V15 朝 prediction 影響なし |
| features 選別 戦略 探索 (top 50/150/200 比較) | 数 h | 最適 N 特定 |

## 5. 5/24+ 計画 (修正版)

| 期間 | 内容 | 担当 |
|------|------|-----|
| 5/14-5/16 | V15 自動運用 + V22 top 100 retrain 確認 | AI |
| **5/17-5/23** | **JV-Link 32-bit Python venv 作成** | **user** |
| 5/24-5/26 | JV-Link COM 接続 + 残 10 features 真値化 | AI (32-bit Python あれば) |
| 5/27-6/8 | V20 真の構築 (top 100 + 10 真値 features) | AI (GPU 16GB 内) |
| 6/15+ | V20 paper trading | 自動 |
| 7/1+ | V20 production 投入判定 | user |

## 6. 投資 状況

- 累計収支: **+13,530 円**
- 撤退余裕: +63,530 円
- V15 自動運用 完全継続 中
- V22 enhanced は production 投入 候補から 一時 外す (V15 越え 未達)
- Strategy 8 shadow eval は 5/16+ (schtask 登録後)

## 7. 結論

★ AI 権限内 で 出来ること は ほぼ 完了 ★

残 unlock:
1. **JV-Link 32-bit Python venv** (user CRITICAL、 1-2h、 大 unlock)
2. **Strategy 8 schtask 登録** (user admin、 5 分)
3. **git push 修復** (user 判断、 30 分)
4. **GPU 増強** (user 投資判断、 5/24+ 検討)

→ JV-Link 32-bit venv が **最大 inflection point**。 これさえ あれば 17 features 全 真値化 + V20 真の構築 可能。
