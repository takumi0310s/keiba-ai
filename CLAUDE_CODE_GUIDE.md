# KEIBA AI - Claude Code 移行ガイド

## リポジトリ構成
```
keiba-ai/
├── app.py                    # Streamlit メインアプリ
├── keiba_model_v8.pkl        # 現行モデル（AUC 0.8501）
├── jockey_wr.json            # 騎手勝率データ
├── requirements.txt          # 依存ライブラリ
├── train/
│   └── keiba_v8_train.ipynb  # 学習ノートブック
└── data/                     # ローカルデータ置き場
    ├── keiba_data.csv        # 中央データ（cp932）
    └── chihou_races_2020_2025.csv
```

---

## Claude Codeで実装するタスク一覧

### 優先度★★★（即効・効果大）

#### 1. netkeibaオッズリアルタイム取得
- 出馬表URLから単勝オッズを自動取得
- app.pyのget_horse_stats()に組み込み
- 特徴量: odds_log（単勝オッズのlog変換）
- 期待AUC: +0.01〜0.02

#### 2. 予測結果の自動記録
- 予測するたびにSQLiteに保存
- 馬名・AIスコア・着順を記録
- 的中率・回収率をダッシュボードに表示

#### 3. 買い目の期待値自動計算
- 三連複オッズ × 的中確率 = 期待値
- 期待値1.0以上の買い目だけハイライト

---

### 優先度★★（精度改善）

#### 4. v8再学習パイプライン自動化
- keiba_data.csvをローカルで直接読み込み
- python train/train_v8.py で即学習
- Colabいらず

#### 5. 調教タイムスクレイピング
- netkeibaの調教ページから直前1週間のタイムを取得
- 特徴量: chukyo_time, wood_time, poly_time
- 期待AUC: +0.01

#### 6. 過去5走への拡張（現在3走）
- train_v8.pyのlag特徴量を5走分に拡張
- prev4_finish, prev5_finish追加
- 期待AUC: +0.005

---

### 優先度★（アプリ改善）

#### 7. LINE通知
- レース1時間前に自動でLINE送信
- TOP3予想 + 買い目を通知

#### 8. 複数レース一括予測
- 開催日の全レースURLを自動収集
- 一括でスコアリング

#### 9. 週次回収率レポート
- 毎週月曜に先週の成績を自動集計
- 的中率・回収率・収支をグラフ化

---

### 優先度★（発展・将来）

#### 10. v9モデル（アンサンブル）
- LightGBM + XGBoost のアンサンブル
- 期待AUC: 0.87〜0.89

#### 11. 展開予測モデル分離
- 逃げ・先行・差し・追込ごとに別モデル
- ペース予測精度UP

#### 12. 馬場バイアス自動検出
- 当日の前レース結果から内外・前後バイアスを計算
- リアルタイムで補正

---

## 現在のモデル状況

| バージョン | AUC | 状態 |
|-----------|-----|------|
| v6.2 | 0.8769 | バックアップ |
| v8 | 0.8501 | **現行（GitHub）** |
| v8.1（次） | 0.87台期待 | オッズ追加後 |

---

## 既知の問題・メモ

- keiba_data.csv: cp932エンコード（euc-jpではない）
- col26はタイム×10（オッズではない）→ 学習に使わない
- 中京コースのrace_keyマッチングが甘い（要修正）
- 地方AUC 0.68 → 地方は参考程度

---

## app.py 主要関数メモ

| 関数 | 役割 |
|------|------|
| load_model() | v8→v6→v5の順でpklを検索 |
| get_horse_stats() | netkeibaから馬の過去成績取得 |
| parse_shutuba() | 出馬表スクレイピング |
| calc_pace_advantage() | ペース有利度計算 |
| render_horse_card() | 馬カード描画 |
| render_buy_section() | 買い目セクション描画 |

---

## Claude Code 最初のコマンド

```bash
# リポジトリクローン
git clone https://github.com/takumi0310s/keiba-ai .

# 依存インストール
pip install -r requirements.txt

# ローカル動作確認
streamlit run app.py

# Claude Code起動
claude
```

最初のお願い:
「netkeibaの出馬表URLから単勝オッズを取得してapp.pyのget_horse_stats()に組み込んで」
