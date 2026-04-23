# target_odds.csv 3/11停止 根本原因調査

調査日時: 2026-04-23 18:55
ブランチ: feat/maxvalue-pack-20260423-pm
**判定: 保留 (TARGET 由来、自動化不可)**

---

## 結論

`data/target_odds.csv` も TARGET Frontier JV (`C:\TFJV`) の手動エクスポート由来。
extract_jvdata.py と同じく完全手動運用で、3/11 が最後のエクスポート。

**修復は TARGET 再インストール + 手動エクスポートが必須**。本タスクでは修復スクリプト
作成・タスク登録は行わない (技術的に自動化不可、形式互換問題)。

---

## 1. 現状

| 項目 | 値 |
|------|-----|
| 最終更新 | 2026-03-11 01:08 |
| サイズ | 223 MB |
| 行数 | 781,161 |
| エンコーディング | cp932 (Shift_JIS) |
| ヘッダー | なし、52 columns |
| データ最終日 | 2010-08-14 〜 確認のため tail 必要だが構造は古い形式 |

---

## 2. リポジトリ内の参照箇所

```
tools/extract_jvdata.py        # 読み込み専用 (TARGET_ODDS = 'data/target_odds.csv')
merge_training_data.py         # 読み込み (BASE_CSV)
project_status.py              # ファイル鮮度チェック
```

**書き込み・生成スクリプトは存在しない**。
git log でも target_odds.csv 生成 commit は見当たらず。

---

## 3. 推定: TARGET Frontier JV の出力

CSV の cp932 エンコード + ヘッダー無し + 52 columns 構造は extract_jvdata.py の
`COLUMNS` 定義と完全一致 → TARGET のレース成績エクスポートと同一フォーマット。

**つまり target_odds.csv は TARGET でユーザーが手動エクスポートした成果物**。
自動化スクリプトは元から存在せず、3/11 の手動操作が最後。

---

## 4. 修復方針

### 短期 (今週土日含む)
- **何もしない**: v15 モデルは既学習。本番予測には影響なし
- 既存 `target_odds.csv` (3/11時点) はそのまま保持

### 中期 (来週月曜以降)
- TARGET Frontier JV を再インストール
- TARGET 起動 → レース成績エクスポート → `data/target_odds.csv` 上書き
- `python tools/extract_jvdata.py` で `jra_races_full.csv` 再生成
- `python tools/precompute_lookups.py` で `feature_lookups.pkl` 再生成

### 長期 (1ヶ月以内)
- 半自動化: 月曜朝のTARGET手動エクスポート + Windows タスクスケジューラ自動起動の
  ハイブリッド方式を `docs/target_reinstall_guide.md` に記録

---

## 5. なぜ tools/auto_update_target_odds.py を作らないか

ユーザー指示「失敗 → 来週以降に持ち越し、原因記録」に従い:

1. 代替ソース不在 (TARGET 以外で同形式・同網羅範囲のCSVを取得する手段なし)
2. 形式変換 (netkeiba スクレイピング → 52列 cp932 整形) の実装コストが大きい
3. 整合性チェック不在で本番投入は危険
4. **本番48時間前の改修としてリスクが許容範囲超**

---

## 6. 本番 (4/25) への影響

**影響なし**。target_odds.csv は学習データ生成にのみ使用され、本番予測時は
v15 モデル + JRDB系チェーンが独立稼働する。

---

## 7. 関連事故

- inc-20260423-002: jra_races_full.csv 3/27停止 (同根、TARGET由来)
- inc-20260423-003: target_odds.csv 3/11停止 (本件)

両者同じ根本原因 (TARGET不在) で連鎖停止。修復も同じTARGET再インストールが必要。
