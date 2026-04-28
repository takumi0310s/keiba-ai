#!/bin/bash
# v16.1 学習進捗監視

LOGFILE=$(ls -t logs/v161_*.log 2>/dev/null | head -1)
echo "監視中: $LOGFILE"
echo ""

while true; do
    clear
    echo "=== v16.1 学習進捗 $(date) ==="
    echo "ログ: $LOGFILE"
    echo ""
    
    # プロセス確認
    if tasklist //FI "IMAGENAME eq python.exe" //FO CSV 2>/dev/null | grep -q python; then
        echo "🟢 プロセス継続中"
    else
        echo "🔴 プロセス終了 (完了 or クラッシュ)"
        echo ""
        echo "=== 末尾20行 ==="
        tail -20 "$LOGFILE"
        break
    fi
    
    # 結果ファイル確認
    if [ -f "data/v161_wf_results.json" ]; then
        echo "✅ 結果ファイル生成済み (完了!)"
        cat data/v161_wf_results.json | python -m json.tool | head -30
        break
    fi
    
    # 末尾10行
    echo ""
    echo "=== ログ末尾10行 ==="
    tail -10 "$LOGFILE"
    
    echo ""
    echo "[Ctrl+C で監視停止]"
    sleep 60
done
