# 別PC移行ガイド

## 1. 前提ソフトウェア

- **Python 3.10+** (miniconda推奨)
- **Git**
- **Node.js 18+** (Claude Code用)

## 2. リポジトリ取得

```bash
git clone https://github.com/takumi0310s/keiba-ai.git
cd keiba-ai
pip install -r requirements.txt
```

## 3. 移行パッケージ適用

旧PCで生成:
```bash
python tools/migrate_to_new_pc.py
# → migrate_package_YYYYMMDD.zip が生成される
```

新PCで展開:
```bash
# zipを keiba-ai/ ディレクトリにコピーして展開
# Windows: 右クリック → すべて展開
# または: python -c "import zipfile; zipfile.ZipFile('migrate_package_YYYYMMDD.zip').extractall('.')"
```

含まれるファイル:
- `.env` (Cookie, Discord Webhook URL)
- `*.db` (予測・結果DB)
- `data/feature_lookups.pkl` (特徴量キャッシュ)
- `data/*.csv` (学習データ, 配当データ, プレミアムデータ)

## 4. 動作確認

```bash
# Streamlit起動
python -m streamlit run app.py

# 事前チェック
python tools/pre_race_check.py
```

## 5. Claude Code

```bash
npm install -g @anthropic-ai/claude-code
cd keiba-ai
claude
```

## 6. タスクスケジューラ登録

管理者権限でコマンドプロンプトを開いて:
```
setup_all_tasks.bat
```

8タスクが一括登録される:
| 時間 | タスク |
|------|--------|
| 毎日 03:00 | プレミアムデータ事前取得 |
| 毎日 08:00 | 全レース予測 |
| 土日 09:30 | レース5分前自動予測 |
| 土日 18:00 | 結果照合 |
| 毎晩 20:00 | 結果照合（平日含む） |
| 月曜 08:00 | 週次レポート |

## 7. Discord設定

```bash
python tools/setup_discord.py
```

## 8. netkeiba Cookie更新

ブラウザでnetkeibaにログイン → F12 → Network → Cookie値をコピー → `.env` を編集:
```
NETKEIBA_COOKIE="コピーした値"
```

## 9. TARGET JV連携（任意）

TARGET Frontier JVをインストールし `C:\TFJV` にデータ同期後:
```bash
python tools/extract_jvdata.py
python tools/precompute_lookups.py
```
