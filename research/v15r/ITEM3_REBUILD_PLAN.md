# item3 feat_dump健全化 実行計画（承認済・backup→rebuild→verify→rollback）

## 対象
6/27-8/9 の feat_dump で JRDB 40/40 定数（死）→ 健全化（数個定数まで回復）。
JRDBデータは 6/22-8/9 取得済（data/jrdb/extracted、item1バックフィル）。

## 安全手順
### Step 0: バックアップ（必須・先に）
```
mkdir research/v15r/backup_pre_heal_<date>/jrdb_csv
cp data/jrdb_{kyi,sed,tyb,cyb,skb,srb,kta,kka,cha,jo}.csv → backup/jrdb_csv/
# feat_dump対象日もコピー
cp -r data/v15_feat_dump/{20260627,20260628,20260711,20260802,20260808,20260809} → backup/feat_dump/
```

### Step 1: JRDB CSV 再構築（extracted → CSV, フル再構築でスキーマ不変）
| CSV | 再構築コマンド | 方式 |
|-----|--------------|------|
| jrdb_tyb / skb / srb | `fable_rebuild_type_20260612.py --types tyb skb srb` | ★parse専用・安全★ |
| jrdb_sed / cyb / kta | `download_parse_jrdb_batch2.py --types sed cyb kta` | download skip(raw有)+parse |
| jrdb_kka | `download_parse_jrdb_extra.py --types kka` | 同上 |
| jrdb_kyi | `rebuild_jrdb_kyi.py` | lzh→CSV |
| jrdb_cha / jo | 変化小・据置検討 | — |

### Step 2: feat_dump 健全化（2通り）
- **A. 最小パッチ（推奨・netkeiba非依存）**: 既存 parquet の netkeiba由来特徴(調教/指数)は維持し、
  `jrdb_features.merge_jrdb_predict_features(horses, race_id)` で JRDB 38特徴のみ再計算→該当列を上書き。
  netkeiba再取得なし。研究側スクリプトで parquet を patch。
- **B. 全再生成**: `daily_predict.py --date 20260627..20260809`（netkeiba shutuba再取得込み）。
  ※netkeiba解約方針のため A を優先。

### Step 3: 検証（NGなら即ロールバック）
- 各対象日 feat_dump の JRDB定数特徴数が 40/40 → ≤10 に回復すること。
- Pattern A/本番スコアの range が 0.0-0.9 域に回復すること（8/2で確認）。
- サンプルレースで JRDB特徴値が妥当域（jrdb_idm≠50一定 等）。
- NG時: backup から jrdb_csv と feat_dump を復元。

### Step 4: 全640R item2 再判定（確定オッズ）
健全化後 `item2_final_odds_eval.py`（JRDB_HEALTH_MAX閾値で全日採用）→ NO-GO確認。

## リスク・注意
- 再構築は production の生 jrdb_*.csv を上書き。**必ず backup 後**。
- 稼働中CSV(paci等)を巻き込まない種別単独ドライバ(fable_rebuild)を優先。
- item2は186R確定オッズで既にNO-GO。全640Rは方向確認。健全化の主目的は**再開に向けたproductionデータ健全性回復**。
