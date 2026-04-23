# jra_races_full.csv 3/27停止 根本原因調査

調査日時: 2026-04-23 14:25
ブランチ: fix/precision-boost-20260423
**修正は本タスクでは実施しない（記録のみ）**

---

## 結論

**根本原因**: `tools/extract_jvdata.py` が完全手動運用、かつ TARGET Frontier JV (`C:\TFJV`) が現在マシン上に存在しない。3/27 が最後の手動実行で、以降データ供給チェーンが完全停止。

---

## 1. 現状

### ファイル鮮度
| ファイル | 最終更新 | サイズ | 行数 |
|----------|----------|--------|------|
| data/jra_races_full.csv | **2026-03-27 00:40** | 178MB | 531,620行 |
| data/target_odds.csv (ソース1) | **2026-03-11 01:08** | 223MB | — |
| C:\TFJV\TXT\keiba_data.csv (ソース2) | **存在せず** | — | — |

### CSV内最終データ
- 最終レース日: **2025-12-28** (中山12R)
- 2026年データは一切含まれない

---

## 2. データ供給チェーン

```
TARGET Frontier JV (C:\TFJV)
    ├── TXT/keiba_data.csv          ← TARGET エクスポート (現在不在)
    ├── TXT/target_sakaro.csv       ← 坂路調教
    └── TXT/target_wood.csv         ← 木馬場調教
            ↓
    + data/target_odds.csv (3/11 停止)
            ↓
    [tools/extract_jvdata.py 手動実行]
            ↓
    data/jra_races_full.csv (3/27 停止)
            ↓
    [tools/precompute_lookups.py 手動実行]
            ↓
    data/feature_lookups.pkl (3/27 停止)
```

---

## 3. 調査結果

### 3.1 タスクスケジューラ
extract_jvdata 関連タスク: **未登録**

登録済み Keiba タスク (10本) — いずれも extract_jvdata を呼ばない:
- Keiba-AM3FireCheck / AM6FireCheck / AM8FireCheck
- Keiba-FridayWeekendScrape
- Keiba-MorningDigest
- Keiba-NightlySanity
- Keiba-PreFireCheck
- KeibaAI_DriftDetector
- Keiba-ScrapeProgress
- Keiba-WeeklyScrapeResume

### 3.2 ログ
`logs/` 配下に extract_jvdata の実行ログ **0件**。完全手動運用を裏付ける。

### 3.3 ソースディレクトリ
`C:\TFJV` ディレクトリ自体が存在しない (PowerShell `Test-Path` でも `find` でも確認)。
TARGET Frontier JV がアンインストール済み、または別パスへ移動された可能性。

### 3.4 git履歴
3/25-4/23 の期間で `data/jra_races_full.csv`, `data/target_odds.csv`, `tools/extract_jvdata.py` への commit は **0件**。スクリプト変更も CSV コミットもなし (.gitignore 対象なので csv は本来 commit されない)。

---

## 4. 影響範囲

### 4.1 学習データ
v15 モデルは既に学習済み (`keiba_model_v15_central_live.pkl.gz`) なので、本番予測には直接影響なし。

### 4.2 feature_lookups.pkl
`tools/precompute_lookups.py` がこの CSV を入力とするため、3/27 で同時に停止 (前回タスク1で確認)。
- sire/BMS/trainer/jockey の勝率テーブルが2025/12/28 までのデータで固定
- 2026/1-4 の最新成績 (新人騎手・新規種牡馬等) が反映されない
- 影響度: 中 (主要な統計は安定するが、新規参入要素はカバーされない)

### 4.3 予測時のJRDB特徴量
JRDB系 (`data/jrdb_kyi.csv` 等) は別チェーン (4/19 まで最新) で独立運用、本停止の影響は受けない。

### 4.4 テスト用 train_df_cache
`data/_v15_train_df_cache.pkl` (Apr 13 15:36) は古いCSVから生成済みでキャッシュ。学習再実行しなければ問題なし。

---

## 5. 修正方針 (来週以降)

### 5.1 必須対応
1. TARGET Frontier JV の再インストールまたはパス特定
2. `C:\TFJV\TXT\keiba_data.csv` の最新エクスポート手順確立
3. `data/target_odds.csv` の最新化フロー確立

### 5.2 自動化候補
- `tools/extract_jvdata.py` をタスクスケジューラに登録 (例: 毎週月曜 AM7:00)
- TARGET エクスポートが手動なので完全自動化は不可。半自動化が現実的。

### 5.3 短期回避策
- 既存 v15 モデルで継続運用 (学習データは2025末まで)
- feature_lookups も現状維持で、本番予測精度は v15 の学習時水準
- 新規騎手/種牡馬の影響は当面許容

---

## 6. 本番 (4/25) への影響

**影響なし**。v15 モデル + JRDB系データチェーンは独立して動作中。
本停止が問題化するのは、再学習やルックアップ更新を試みる時のみ。

---

## 7. アクションアイテム

- [ ] TARGET Frontier JV の再インストール (来週)
- [ ] `data/target_odds.csv` 更新フロー確認
- [ ] `tools/extract_jvdata.py` の半自動化検討
- [ ] CLAUDE.md に「TARGETが再導入されるまで feature_lookups は固定」と明記
