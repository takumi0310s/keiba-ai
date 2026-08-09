# T1v2 監視 設計案（実装しない・設計のみ）

## 背景（現行T1の致命欠陥）
現行 `T1_features_audit` は **学習キャッシュ(2020-25静的)を毎日監査** → 全日 byte-identical、
ライブのJRDB死(6/27 40/40定数)を**構造的に検知不能**でアラート不発（死因解剖 §3）。

## 設計目標
1. **監査対象をライブ当日 feat_dump に変更**（学習キャッシュではなく `data/v15_feat_dump/<当日>/`）。
2. **ゾンビ検知**: 前日と md5 一致（byte-identical）で即アラート。
3. **特徴死のライブ検知**: 定数化・欠損急増を当日データで判定。

## 監査フロー（日次・予測後）
```
入力: data/v15_feat_dump/<today>/*.parquet （当日全レースの145特徴）
      data/v15_feat_dump/<prev_open_day>/  （前開催日）
出力: data/T1v2_audit/<today>.json + 異常時 Discord アラート
```

### チェック項目
| # | 項目 | 条件 | 深刻度 |
|---|------|------|--------|
| A | **ゾンビ検知** | 当日集約 md5 == 前開催日 md5 | ★CRITICAL（即停止検討）★ |
| B | **参照元検知** | 監査対象パスが feat_dump でなくキャッシュを指す（設定ガード） | CRITICAL |
| C | JRDB群 定数化 | jrdb_* 40特徴中 N個が unique≤2（当日） | N≥10:WARN / N≥25:CRITICAL |
| D | 特徴 欠損急増 | 任意特徴の null_rate が前日比 +0.3 超 | WARN |
| E | 本番スコア圧縮 | 当日 `スコア` の range < 0.3（健全日~0.9） | WARN（識別能力喪失の代理） |
| F | 特徴 分布ドリフト | 主要特徴の mean/std が学習分布から Nσ 逸脱 | INFO |

### ゾンビ検知の具体（Aの核）
```
today_hash  = md5( concat(sorted(当日parquetの145特徴値)) )
prev_hash   = 前開催日の同ハッシュ
if today_hash == prev_hash: ALERT("T1 ZOMBIE: feat_dump が前日と同一。生成停止の疑い")
```
- 集約は「レース跨ぎで特徴値を連結→md5」。開催構成が違えば必ずハッシュは変わるはずで、
  一致＝生成が動いていない（現行T1がまさにこの状態だった）。
- 併せて **各 T1v2_audit 出力ファイル自体の日次md5** も比較（出力側ゾンビの二重検知）。

### アラート連携
- CRITICAL: Discord #アップデート + **翌日の予測をブロック**（degraded明示 or 見送り）。
- WARN: Discord 通知のみ。
- 既存 `tools/anomaly_auto_detector.py` / `data_freshness_monitor` に統合（内容ベース停止検知は
  既に一部実装済＝6/11 Fable sweep）。T1v2 はその feat_dump 特化版。

## 実装時の配置（案・未実装）
- `tools/t1v2_feature_audit.py`（当日feat_dump監査）+ 予測後スケジュール。
- 出力 `data/T1v2_audit/<date>.json`。設定に「監査対象=live固定」ガード。
- 回帰テスト: 意図的に前日コピーを置いてゾンビ検知が発火するか。

## 現行T1との差分（要点）
| | 現行T1 | T1v2 |
|--|--------|------|
| 監査対象 | 学習キャッシュ(静的) | **当日ライブ feat_dump** |
| ゾンビ検知 | なし（自身がゾンビ） | **md5前日一致で即アラート** |
| JRDB死検知 | 不可(常に1/40) | **当日unique判定で可** |
| 予測ブロック | なし | CRITICAL時ブロック |

*注: 本書は設計のみ。実装・運用反映は別途承認後。*
