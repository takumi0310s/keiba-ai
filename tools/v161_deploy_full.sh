#!/bin/bash
# v16.1 完全デプロイスクリプト

set -e
cd /c/Users/takum/keiba-ai

if [ ! -f "data/v161_wf_results.json" ]; then
    echo "[ERROR] 結果ファイルなし"
    exit 1
fi

MEAN_AUC=$(python -c "import json; d=json.load(open('data/v161_wf_results.json')); print(f'{d[\"v161\"][\"mean_auc\"]:.6f}')")
ADOPTED=$(python -c "import json; d=json.load(open('data/v161_wf_results.json')); print(d.get('adopted', False))")

echo "v16.1 結果:"
echo "  mean_auc: $MEAN_AUC"
echo "  adopted: $ADOPTED"

if [ "$ADOPTED" != "True" ]; then
    echo "[STOP] 不採用、v15 継続"
    exit 0
fi

read -p "デプロイしますか? (y/N): " confirm
if [ "$confirm" != "y" ]; then exit 0; fi

# モデル探索
ls -la *.pkl* keiba_model* 2>/dev/null | head -10

if [ -f "keiba_model_v161.pkl.gz" ]; then
    cp keiba_model_v15_central_live.pkl.gz "keiba_model_v15_central_live.pkl.gz.bak_v161_$(date +%Y%m%d_%H%M)"
    cp keiba_model_v161.pkl.gz keiba_model_v15_central_live.pkl.gz
    echo "[OK] v16.1 デプロイ"
else
    echo "[WARN] v161 pkl 未生成"
    echo "  → run_v16_and_am8_wf の run_wf 関数は WF評価のみ"
    echo "  → 別途本番モデル学習が必要"
    exit 1
fi

# テスト
PYTHONIOENCODING=utf-8 python tests/regression_test.py 2>&1 | tail -10

# git push
git add data/v161_wf_results.json keiba_model_v15_central_live.pkl.gz
git commit -m "deploy: v16.1 with training_eval_rank (mean AUC: $MEAN_AUC)"
git push origin main
