# extract_jvdata.py 自動化検討レポート

調査日時: 2026-04-23 18:50
ブランチ: feat/maxvalue-pack-20260423-pm
**判定: 保留 (技術的に完全自動化不可)**

---

## 結論

`tools/extract_jvdata.py` は TARGET Frontier JV (`C:\TFJV`) という商用ローカルソフトの
エクスポート CSV を入力とする。完全自動化には TARGET の GUI 操作 (export) が必須で、
プログラムからの自動化は不可能。

**今回は自動化スクリプト作成・タスク登録ともに見送り**。
代替案 (netkeiba スクレイピング合成) は形式不一致で学習パイプラインを壊すリスクがあり、
本番48時間前に投入すべきではない。

---

## 1. 依存関係

```
C:\TFJV\TXT\keiba_data.csv   (TARGET エクスポート、52 columns、cp932)
C:\TFJV\TXT\target_sakaro.csv (坂路調教)
C:\TFJV\TXT\target_wood.csv   (ウッドチップ調教)
data/target_odds.csv          (別チェーン、こちらも3/11停止)
```

`tools/extract_jvdata.py` の `KEIBA_DATA = 'C:/TFJV/TXT/keiba_data.csv'` をハードコード参照。
TARGET アンインストール時はパス全体が消失。

---

## 2. 代替案検討 (すべて却下)

### 案A: netkeiba 結果ページ → jra_races_full.csv 合成
- **却下**: netkeiba の result page は出力フォーマットが TARGET と異なる
  (52 列の順序・カテゴリエンコード・調教コード等)
- 既存 `train_v15_master.build_v15_dataframe()` が前提とするフォーマットと不一致
- 強引にマッピングしても欠落カラムが多数 (`prize`, `time_margin`, `pass1-4` 等)
- 結果: 学習・予測パイプラインが壊れる

### 案B: JRA-VAN webサービス (有料 API)
- **却下**: TARGET と同じ JRA-VAN ベースだが API 別途契約・実装が大規模
- 本タスクの時間制約 (60分) では実装不可

### 案C: 既存 jra_races_full.csv を維持、新規データなしで運用
- **採用**: v15モデルは既学習で固定。新規レースデータはJRDB系チェーン (jrdb_kyi.csv等) で別途供給
- feature_lookups は3/27時点で固定 → 新規騎手・種牡馬の統計が取れないが、影響は限定的

### 案D: TARGET の自動エクスポート (Win32 API/SendKeys 経由)
- **却下**: 不安定・保守困難・TARGET 利用規約違反の可能性

---

## 3. 短期推奨事項

1. **来週月曜以降**: TARGET Frontier JV の再インストールを手動で実施
   - インストール後、TARGET の機能で `keiba_data.csv` 等をエクスポート
   - `python tools/extract_jvdata.py` を手動実行
   - `python tools/precompute_lookups.py` で feature_lookups 再生成

2. **半自動化** (来週): TARGETエクスポートを毎週月曜朝に手動実施し、
   その後 extract_jvdata.py を Windows タスクスケジューラから自動起動するハイブリッド方式。
   完全自動化は不可、この方式が現実的。

---

## 4. 本番 (4/25) への影響

**影響なし**。v15 モデル + JRDB 系チェーン (4/19 まで最新) は独立稼働。
新規騎手・種牡馬の統計が反映されないだけで、メイン予測ロジックは健在。

---

## 5. アクション

- [ ] 来週月曜: TARGET Frontier JV 再インストール手順を docs/target_reinstall_guide.md に記録
- [ ] 半自動化スクリプト作成は来週以降
- [x] **本タスクは保留**、commit はこのレポートのみ

---

## 6. なぜ自動化スクリプト作成しなかったか

ユーザー指示「dry-run で1週間分取得成功 → 採用、タスク登録/失敗 → 原因記録、タスク登録せず」に従い:

- 代替ソースが技術的に存在しない (TARGET エクスポートに代わる API なし)
- 既存パイプラインを壊すリスクが高い (形式不一致)
- 本番48時間前の改修としてリスクが許容範囲超

→ **失敗扱いで原因記録のみ**
