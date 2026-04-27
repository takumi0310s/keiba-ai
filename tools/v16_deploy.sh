#!/bin/bash
# v16 採用時の本番投入スクリプト
# 学習完了後に実行: bash tools/v16_deploy.sh

set -e

echo "============================================================"
echo "  v16 デプロイ開始"
echo "============================================================"

cd /c/Users/takum/keiba-ai

# 1. v16 結果確認
echo ""
echo "[1] v16 学習結果の確認"
if [ -f "data/v16_wf_results.json" ]; then
    python -c "
import json
with open('data/v16_wf_results.json') as f:
    data = json.load(f)
print('結果ファイル:')
for k, v in data.items():
    if isinstance(v, (int, float, str, bool)):
        print(f'  {k}: {v}')
    elif isinstance(v, dict):
        print(f'  {k}: {len(v)} keys')
"
else
    echo "  [ERROR] 結果ファイルなし、学習未完了"
    exit 1
fi

# 2. v16 モデルファイル確認
echo ""
echo "[2] v16 モデルファイル確認"
if [ -f "keiba_model_v16_central_live.pkl.gz" ]; then
    ls -la keiba_model_v16_central_live.pkl.gz
else
    echo "  [WARN] keiba_model_v16_central_live.pkl.gz が見つかりません"
    echo "  → 別名 (keiba_model_v16_*.pkl.gz) を探します"
    ls -la keiba_model_v16_*.pkl.gz 2>/dev/null
fi

# 3. ユーザー確認
echo ""
echo "[3] 本番投入の確認"
echo "  以下を実行します:"
echo "  - keiba_model_v15_central_live.pkl.gz (現本番) を v16 に置換"
echo "  - 既存モデルは .bak_$(date +%Y%m%d) としてバックアップ"
echo ""
read -p "  続行しますか? [y/N]: " ans
if [ "$ans" != "y" ]; then
    echo "  キャンセル"
    exit 0
fi

# 4. バックアップ
echo ""
echo "[4] 既存モデルのバックアップ"
DATE=$(date +%Y%m%d)
cp keiba_model_v15_central_live.pkl.gz keiba_model_v15_central_live.pkl.gz.bak_${DATE}
cp keiba_model_v15_central.pkl.gz keiba_model_v15_central.pkl.gz.bak_${DATE}
echo "  [OK] バックアップ完了"

# 5. v16 を本番に
echo ""
echo "[5] v16 を本番モデルに"
cp keiba_model_v16_central_live.pkl.gz keiba_model_v15_central_live.pkl.gz
echo "  [OK] 本番モデル差替完了"

# 6. 動作確認
echo ""
echo "[6] 動作確認"
PYTHONIOENCODING=utf-8 python tools/predict_one_race.py 202605020211 2>&1 | head -20

# 7. regression_test
echo ""
echo "[7] regression_test 実行"
PYTHONIOENCODING=utf-8 python tests/regression_test.py 2>&1 | tail -20

# 8. git commit準備
echo ""
echo "[8] git commit 準備"
echo ""
echo "  以下のコマンドで commit & push してください:"
echo ""
echo "  git add keiba_model_v15_central_live.pkl.gz"
echo "  git commit -m \"deploy: v16 model adopted (AUC: X.XXXX)\""
echo "  git push origin main"

echo ""
echo "============================================================"
echo "  v16 デプロイ完了"
echo "============================================================"
