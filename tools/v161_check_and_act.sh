#!/bin/bash
# v16.1 完了確認 + 即アクション

cd /c/Users/takum/keiba-ai

echo "============================================================"
echo "[V161-CHECK] $(date)"
echo "============================================================"

# 結果ファイル確認
if [ ! -f "data/v161_wf_results.json" ]; then
    echo "🟡 v16.1 まだ未完了"
    powershell -Command "Get-Process -Id 1504 -ErrorAction SilentlyContinue | Format-List CPU, Responding"
    exit 0
fi

echo "🎉 v16.1 完了!"
echo ""

# 結果サマリー
python << 'PYEOF'
import json
with open('data/v161_wf_results.json', encoding='utf-8') as f:
    d = json.load(f)

v161 = d.get('v161', {})
mean_auc = v161.get('mean_auc', 0)
baseline = d.get('baseline', 0.8856)
elapsed = v161.get('elapsed_min', 0)
adopted = d.get('adopted', False)

print('='*60)
print('  v16.1 結果')
print('='*60)
print(f'  mean AUC: {mean_auc:.6f}')
print(f'  baseline: {baseline}')
print(f'  diff: {(mean_auc - baseline) * 10000:+.0f}bp')
print(f'  elapsed: {elapsed:.1f}分')
print(f'  adopted: {adopted}')
print()
print('  Per Year:')
for y in v161.get('per_year', []):
    yr = y.get('year')
    grid = y.get('grid_auc', 0)
    gap = y.get('gap', 0)
    print(f'    {yr}: grid={grid:.4f}, gap={gap:.4f}')

# 判定
print()
print('='*60)
if adopted:
    print('  ✅ 採用可能 → デプロイ推奨')
    print('  → 次: bash tools/v161_deploy_full.sh')
else:
    print('  ❌ 不採用 → v15 継続')
    print('  → 5/2 GW初日 v15 + 戦略⑦ で挑む')
print('='*60)
PYEOF

# モデルファイル確認
echo ""
echo "=== モデルファイル ==="
ls -la *.pkl* 2>/dev/null | head -5

# Discord 通知準備
echo ""
echo "=== Discord 通知準備 ==="
ls tools/notify_done.py 2>/dev/null
echo ""
echo "完了。次のアクション:"
echo "  採用なら: bash tools/v161_deploy_full.sh"
echo "  不採用なら: git add data/v161_wf_results.json && git commit -m 'test: v16.1 not adopted' && git push"
