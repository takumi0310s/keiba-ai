# KEIBA AI - 中央競馬予測システム

LightGBM + XGBoost アンサンブルモデルで中央競馬の複勝圏（3着以内）を予測し、条件別に最適な買い目を自動生成するシステム。

- **Streamlit**: https://keiba-ai-l2klehd4rfoupnj5g7rw8b.streamlit.app
- **モデル**: Pattern A（リークフリー67特徴量, AUC 0.8095）+ Pattern B（当日情報込み75特徴量）
- **実績**: WF 2020-2025, 20,579レース, 全条件ROI 100%超え

## クイックスタート

```bash
git clone https://github.com/takumi0310s/keiba-ai
cd keiba-ai

# Windows
setup.bat

# Mac/Linux
chmod +x setup.sh && ./setup.sh
```

## 手動セットアップ

```bash
# 1. 依存パッケージ
pip install -r requirements.txt

# 2. アプリ起動
streamlit run app.py
```

## 新PCへの移行手順

git cloneだけでは動作しません。以下の手動コピーが必要です。

### Step 1: リポジトリクローン

```bash
git clone https://github.com/takumi0310s/keiba-ai
cd keiba-ai
```

### Step 2: 大容量データファイルのコピー

以下のファイルは `.gitignore` で除外されているため、旧PCから手動コピーが必要です。

```
旧PC                          → 新PC
data/feature_lookups.pkl       (37MB, 特徴量エンコーディング)
data/jra_races_full.csv        (781,161行, 中央競馬全レース 2010-2025)
data/training_times.csv        (955,580行, 調教タイムデータ)
data/odds_history.csv          (778,387行, オッズ履歴)
data/blood_full.csv            (81,986行, 血統データ)
data/jra_payouts.csv           (27,541行, JRA公式配当データ)
```

USBメモリ、ネットワークドライブ、クラウドストレージ等でコピーしてください。

**一括コピーコマンド例（旧PC → USBドライブ E:）:**

```bat
:: 旧PCで実行
xcopy data\feature_lookups.pkl E:\keiba-backup\data\ /Y
xcopy data\jra_races_full.csv E:\keiba-backup\data\ /Y
xcopy data\training_times.csv E:\keiba-backup\data\ /Y
xcopy data\odds_history.csv E:\keiba-backup\data\ /Y
xcopy data\blood_full.csv E:\keiba-backup\data\ /Y
xcopy data\jra_payouts.csv E:\keiba-backup\data\ /Y
xcopy *.db E:\keiba-backup\ /Y
```

```bat
:: 新PCで実行
xcopy E:\keiba-backup\data\* data\ /Y
xcopy E:\keiba-backup\*.db . /Y
```

### Step 3: SQLite DBファイルのコピー

DBファイル（`*.db`）も `.gitignore` で除外されています。過去の予測・結果履歴を引き継ぐ場合は旧PCからコピーしてください。

```
keiba_predictions.db     (app.py メインDB: 予測履歴・結果・買い目)
keiba_race_results.db    (tools用 予備DB)
```

DBが不要な場合（新規スタート）は `setup.bat` が空のDBを自動作成します。

### Step 4: セットアップ実行

```bash
# Windows
setup.bat

# Mac/Linux
chmod +x setup.sh && ./setup.sh
```

セットアップスクリプトが以下を自動実行します:
- Python バージョン確認
- `pip install -r requirements.txt`
- ディレクトリ作成（data/, logs/）
- DB初期化（存在しない場合のみ）
- データファイル・モデルファイルの存在確認

### Step 5: タスクスケジューラ設定（任意）

自動運用を行う場合、Windowsタスクスケジューラに以下を登録:

| 時間 | タスク | バッチ |
|------|--------|--------|
| 毎朝8:00 | 当日全レース予測 | `daily_predict.bat` |
| 毎晩20:00 | 結果照合・ROI計算 | `daily_results.bat` |
| 毎週月曜9:00 | 週次レポート | `weekly_report.bat` |

**注意**: バッチファイル内のパスを新PCに合わせて編集してください。

## ファイル構成

```
keiba-ai/
├── app.py                      # Streamlitメインアプリ
├── setup.bat                   # Windowsセットアップ
├── setup.sh                    # Mac/Linuxセットアップ
├── requirements.txt            # Python依存パッケージ
├── CLAUDE.md                   # AI開発ガイド（詳細仕様）
│
├── keiba_model_v9_central_live.pkl  # Pattern B (実運用)
├── keiba_model_v9_central.pkl       # Pattern A (評価用)
│
├── data/                       # データ（大容量はgitignore）
│   ├── feature_lookups.pkl     # [要コピー] 特徴量エンコーディング
│   ├── jra_races_full.csv      # [要コピー] 全レースデータ
│   └── ...
│
├── train/                      # 学習スクリプト
├── tools/                      # 運用・検証ツール
├── tests/                      # テスト
└── results/                    # 分析レポート
```

## コマンド一覧

```bash
# アプリ起動
streamlit run app.py

# CLI予測
python predict_and_log.py "https://race.netkeiba.com/..."

# 結果照合
python check_results.py --summary

# 自動運用
python tools/daily_predict.py
python tools/daily_results.py
python tools/weekly_report.py

# モデル学習
python train/train_v93_leakfree.py      # Pattern A
python train/train_v93_pattern_b.py     # Pattern B

# バックテスト
python backtest_central_leakfree.py
python calc_actual_roi.py
python backtest_condition_roi.py
```

## 条件別ROI格付け (OOS 2023-2025)

| 条件 | 説明 | trio ROI | 格付け |
|------|------|---------|--------|
| A | 8-14頭/1600m+/良稍 | 210.9% | ★★★ |
| B | 8-14頭/1600m+/重不 | 209.9% | ★★★ |
| C | 15頭+/1600m+/良稍 | 279.4% | ★★★ |
| D | 1200-1400m | 145.9% | ★★★ |
| E | 7頭以下 | 118.8% (umaren) | ★★ |
| X | 15頭+/重不 | 260.7% | ★★★ |

詳細は `CLAUDE.md` を参照。
