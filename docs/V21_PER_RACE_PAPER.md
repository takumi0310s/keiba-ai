# V21 Per-Race Paper Scheduler

`tools/v21_per_race_paper.py`

## 概要

V21候補モデルの**paper予測**を発走17分前（15-20分ウィンドウの中点）に自動実行するスケジューラ。  
**V15 production は完全不変**。V21はpaper/比較観測専用。

## 起動方法

```bash
# 当日分（今日のレース）
python tools/v21_per_race_paper.py

# 日付指定
python tools/v21_per_race_paper.py --date 20260524
```

## タイミング

| 設定 | 値 |
|------|-----|
| 発火タイミング | race_start - 17分 |
| ウィンドウ | 15-20分前の中点 |
| TYB fetch | 発火時に実行（TYB_SHADOW_ENABLED=True の場合のみ） |

## 動作フロー

```
起動
 └─ V21モデル読み込み (models/v21_candidate.pkl.gz)
 └─ レース一覧取得 (netkeiba → fallback: daily_predictions CSV)
 └─ 各レースに threading.Timer セット

Timer発火 (race_time - 17min)
 ├─ 1. TYB fetch (fetch_tyb_shadow, enabled時のみ)
 ├─ 2. V21予測 (predict_core + V21モデル)
 ├─ 3. 戦略フィルタ (戦略⑦案C + C4)
 ├─ 4. Discord送信 (strategy_pass=True のみ)
 └─ 5. paper log保存 (pass/fail 問わず常に記録)
```

例外は全てswallow — スケジューラはクラッシュしない。

## V15との独立性

| 項目 | 状態 |
|------|------|
| V15モデル読み込み | しない (predict_core.load_models() 未呼び出し) |
| V15買い目への干渉 | なし |
| race_auto_notify.py との共有状態 | なし |
| Discord channel | DISCORD_WEBHOOK_V21_PAPER (独立) |

## Discord メッセージ形式

```
【V21 paper — 投票しないでください】
東京11R テストレースS
   芝1600m 良 条件A (14頭)
V21候補モデル WF AUC 0.8696 / V15+TYB10 features

三連複フォーメーション 7点 (V21 paper)
1列目: 1
2列目: 2, 3
3列目: 2, 3, 4, 5, 6

軸: 1(ホースA) スコア0.8200
Top5: 1 → 2 → 3 → 4 → 5
TYB: TYB未取得 (V15同等スコア)

⚠ これはpaper予測です。実 cash 投票は V15 買い目のみ使用してください。
```

## Paper log

`data/v21_paper_log/{YYYYMMDD}/{race_id}.json`

```json
{
  "race_id": "202605240511",
  "timestamp": "2026-05-24T15:13:00",
  "v21_top5": [1, 2, 3, 4, 5],
  "v21_formation": [[1, 2, 3], ...],
  "v21_strategy_pass": true,
  "cond_key": "A",
  "tyb_injected": false
}
```

## 環境変数

| 変数 | 用途 |
|------|------|
| `DISCORD_WEBHOOK_V21_PAPER` | V21 paper専用Discord webhook (推奨) |
| `DISCORD_WEBHOOK_UPDATES` | fallback channel |
| `TYB_SHADOW_ENABLED` | `True` にするとTYB直前データ取得を有効化 |
| `JRDB_USER` / `JRDB_PASSWORD` | TYB fetch用JRDB認証情報 |

## TYB状態

| 状態 | 動作 |
|------|------|
| `TYB_SHADOW_ENABLED=False` (default) | TYB未取得、V15同等特徴量で予測 |
| `TYB_SHADOW_ENABLED=True` | JRDB直前パスからTYB取得、10特徴量注入 |
| JRDB認証失敗 | TYB skip、予測は続行 |

## Windowsタスクスケジューラ登録例

```bat
@echo off
cd /d C:\Users\takum\keiba-ai
set PYTHONIOENCODING=utf-8
C:\Users\takum\AppData\Local\Python\pythoncore-3.14-64\python.exe tools/v21_per_race_paper.py >> logs/v21_per_race_paper.log 2>&1
```

土日朝 8:45 に登録（race_auto_notify.bat と同時刻でも独立して動作）。

## モデル情報

| 項目 | 値 |
|------|-----|
| ファイル | `models/v21_candidate.pkl.gz` |
| WF AUC | 0.8696 (6-fold) |
| Features | V15 145 + TYB 10 = 155 |
| V15との差 | +TYB10特徴量 (padock_idx / sogo_idx / odds_idx / idm 等) |

## テスト

```bash
cd C:\Users\takum\keiba-ai
python -m pytest tests/test_v21_per_race_paper.py -v
```
