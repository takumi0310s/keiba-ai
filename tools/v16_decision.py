#!/usr/bin/env python
"""v16 学習結果の解釈と本番投入判断

使い方: python tools/v16_decision.py
"""
import os
import json
import sys
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ADOPTION_CRITERIA = {
    'min_auc': 0.8856,        # baseline (v15)
    'all_year_min_auc': 0.85,  # 全年最低
    'max_year_gap': 0.05,      # 年度間差
}


def main():
    print("=" * 60)
    print(f"  v16 Adoption Decision")
    print(f"  時刻: {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 60)
    
    results_path = os.path.join(BASE_DIR, 'data', 'v16_wf_results.json')
    
    if not os.path.exists(results_path):
        print("\n[WAITING] v16 結果ファイルがまだ生成されていません")
        print("  対応:")
        print("  1. tail -f logs/v16_wf_*.log で進捗確認")
        print("  2. 完了後に再実行")
        return
    
    with open(results_path) as f:
        data = json.load(f)
    
    print(f"\n[1] 結果ファイル読み込み")
    print(f"  パス: {results_path}")
    print(f"  キー: {list(data.keys())}")
    
    # AUC の確認
    if 'v16_auc' in data or 'mean_auc' in data:
        auc = data.get('v16_auc', data.get('mean_auc', 0))
        print(f"\n[2] WF AUC")
        print(f"  v16: {auc:.4f}")
        print(f"  v15 baseline: 0.8856")
        print(f"  差分: {auc - 0.8856:+.4f}")
    
    # 採用判断
    print(f"\n[3] 採用判断")
    if 'adopted' in data:
        adopted = data['adopted']
        print(f"  result: {'[OK] 採用' if adopted else '[NG] 却下'}")
    
    # 年度別 AUC
    if 'year_aucs' in data or 'wf_results' in data:
        print(f"\n[4] 年度別 AUC")
        years = data.get('year_aucs', data.get('wf_results', {}))
        if isinstance(years, dict):
            for y, v in years.items():
                print(f"  {y}: {v}")
        elif isinstance(years, list):
            for entry in years:
                print(f"  {entry}")
    
    # 採用された場合の本番投入手順
    print(f"\n[5] 本番投入手順 (採用された場合)")
    print(f"""
  cd /c/Users/takum/keiba-ai
  
  # 1. 既存モデルバックアップ
  cp keiba_model_v15_central_live.pkl.gz keiba_model_v15_central_live.pkl.gz.bak_$(date +%Y%m%d)
  
  # 2. v16 をコピー
  cp keiba_model_v16_central_live.pkl.gz keiba_model_v15_central_live.pkl.gz
  
  # 3. 動作確認
  python tools/predict_one_race.py 202605020211
  
  # 4. regression_test
  python tests/regression_test.py
  
  # 5. git commit & push
  git add keiba_model_v15_central_live.pkl.gz
  git commit -m "deploy: v16 model adopted (AUC: X.XXXX)"
  git push origin main
""")


if __name__ == '__main__':
    main()
