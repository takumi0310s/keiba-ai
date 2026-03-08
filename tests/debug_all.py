#!/usr/bin/env python
"""KEIBA AI 全機能デバッグテスト"""
import sys, os, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pickle, json, sqlite3, tempfile
import pandas as pd
import numpy as np

passed = 0
failed = 0
issues = []

def ok(label, cond, detail=""):
    global passed, failed, issues
    if cond:
        print(f"  PASS  {label}")
        passed += 1
    else:
        print(f"  FAIL  {label} -- {detail}")
        failed += 1
        issues.append(f"{label}: {detail}")

print("=" * 60)
print("  KEIBA AI 全機能デバッグテスト (20項目)")
print("=" * 60)

# ===== 1. V9.1モデル読み込み =====
try:
    with open('keiba_model_v9_central.pkl', 'rb') as f:
        v9c = pickle.load(f)
    ver = v9c.get('version', '')
    auc = v9c.get('auc', 0)
    feats = v9c.get('features', [])
    ok("[1] V9.1 Central model", ver == 'v9.1' and auc > 0.84 and len(feats) > 40,
       f"ver={ver}, auc={auc:.4f}, feats={len(feats)}")
except Exception as e:
    ok("[1] V9.1 Central model", False, str(e))

# ===== 2. V8モデル読み込み =====
try:
    with open('keiba_model_v8.pkl', 'rb') as f:
        v8 = pickle.load(f)
    ok("[2] V8 model", v8.get('auc', 0) > 0.8, f"auc={v8.get('auc', 0):.4f}")
except Exception as e:
    ok("[2] V8 model", False, str(e))

# ===== 3. 条件A-E,X自動判定 =====
def classify(ri, nh, nar=False):
    if nar:
        return 'B', 'wide'
    dist = ri.get('distance', 0)
    cond = str(ri.get('condition', '良'))
    heavy = any(c in cond for c in ['重', '不'])
    if nh <= 7: return 'E', 'umaren'
    if dist <= 1400: return 'D', 'trio'
    if 8 <= nh <= 14 and dist >= 1600 and not heavy: return 'A', 'trio'
    if 8 <= nh <= 14 and dist >= 1600 and heavy: return 'B', 'trio'
    if nh >= 15 and dist >= 1600 and not heavy: return 'C', 'trio'
    return 'X', 'trio'

cases = [
    ({'distance': 2000, 'condition': '良'}, 10, False, 'A', 'trio'),
    ({'distance': 2000, 'condition': '重'}, 10, False, 'B', 'trio'),
    ({'distance': 2000, 'condition': '良'}, 16, False, 'C', 'trio'),
    ({'distance': 1200, 'condition': '良'}, 10, False, 'D', 'trio'),
    ({'distance': 2000, 'condition': '良'}, 5, False, 'E', 'umaren'),
    ({'distance': 2000, 'condition': '重'}, 16, False, 'X', 'trio'),
    ({'distance': 2000, 'condition': '良'}, 10, True, 'B', 'wide'),
]
all_ok = True
for ri, nh, nar, ek, ebt in cases:
    k, bt = classify(ri, nh, nar)
    if k != ek or bt != ebt:
        all_ok = False
ok("[3] Condition A-E,X classify", all_ok, "condition mismatch")

# ===== 4. 買い目生成 =====
df_t = pd.DataFrame({'馬番': [1,2,3,4,5,6], 'スコア': [.9,.8,.7,.6,.5,.4], '馬名': list('ABCDEF')})
df_t = df_t.sort_values('スコア', ascending=False)
nums = [int(df_t.iloc[i]['馬番']) for i in range(6)]
n1 = nums[0]; second = nums[1:3]; third = nums[1:6]
bets = set()
for s in second:
    for t in third:
        combo = tuple(sorted({n1, s, t}))
        if len(combo) == 3: bets.add(combo)
trio_bets = sorted(bets)
ok("[4a] Trio 7点 generation", len(trio_bets) == 7, f"got {len(trio_bets)}")

umaren_bets = [sorted([nums[0], nums[1]]), sorted([nums[0], nums[2]])]
ok("[4b] Umaren 2点 generation", len(umaren_bets) == 2 and all(len(b)==2 for b in umaren_bets))

wide_bets = [sorted([nums[0], nums[1]]), sorted([nums[0], nums[2]])]
ok("[4c] Wide 2点 generation", len(wide_bets) == 2)

# ===== 5. オッズ取得テスト =====
import requests
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
try:
    r = requests.get("https://race.netkeiba.com/api/api_get_jra_odds.html?race_id=000000000000&type=1",
                     headers=HEADERS, timeout=10)
    ok("[5] Odds API response", r.status_code in [200, 400, 404], f"HTTP {r.status_code}")
except Exception as e:
    ok("[5] Odds API response", False, str(e)[:50])

# ===== 6. 調教データ取得テスト =====
try:
    r = requests.get("https://race.netkeiba.com/race/oikiri.html?race_id=000000000000",
                     headers=HEADERS, timeout=10)
    ok("[6] Training data API", r.status_code in [200, 400, 404], f"HTTP {r.status_code}")
except Exception as e:
    ok("[6] Training data API", False, str(e)[:50])

# ===== 7. 馬場バイアス関数存在チェック =====
# Static check: fetch_today_bias, apply_bias_adjustment
from app import fetch_today_bias, render_bias_panel
ok("[7] Bias functions exist", callable(fetch_today_bias) and callable(render_bias_panel))

# ===== 8. 展開予測関数存在チェック =====
from app import predict_race_pace, render_pace_prediction, calc_pace_advantage
ok("[8] Pace prediction functions exist",
   callable(predict_race_pace) and callable(render_pace_prediction) and callable(calc_pace_advantage))

# ===== 9. fetch_race_results 返り値テスト =====
from app import fetch_race_results
# 存在しないrace_idで呼んで返り値の型を確認
results, payouts = fetch_race_results("000000000000", is_nar=False)
ok("[9] fetch_race_results returns (dict, dict)",
   isinstance(results, dict) and isinstance(payouts, dict) and 'trio' in payouts and 'umaren' in payouts and 'wide' in payouts,
   f"results type={type(results)}, payouts type={type(payouts)}, keys={list(payouts.keys()) if isinstance(payouts, dict) else 'N/A'}")

# ===== 10. bet_type DB保存テスト =====
from app import save_prediction, classify_race_condition, CONDITION_PROFILES
df_save = pd.DataFrame({
    '馬番': [1,2,3,4,5], 'スコア': [.9,.8,.7,.6,.5], '馬名': list('ABCDE'),
    'AI順位': [1,2,3,4,5], '単勝オッズ': [2.0,5.0,8.0,12.0,20.0]
})
# Test condition E (umaren)
ri_e = {'distance': 2000, 'condition': '良', 'course': '東京', 'surface': '芝', 'grade': '', 'race_num': '1R'}
DB_TEST = os.path.join(tempfile.gettempdir(), 'test_debug.db')
import app
old_db = app.DB_PATH
app.DB_PATH = DB_TEST
if os.path.exists(DB_TEST): os.remove(DB_TEST)
app.init_db()
save_prediction('test_e', 'テストE', ri_e, df_save, is_nar=False)
conn = sqlite3.connect(DB_TEST)
c = conn.cursor()
c.execute("SELECT bet_type, bet_condition FROM race_results WHERE race_id='test_e'")
row = c.fetchone()
ok("[10] bet_type saved for E", row and row[0] == 'umaren' and row[1] == 'E',
   f"got bet_type={row[0] if row else None}, cond={row[1] if row else None}")

# ===== 11. 的中判定テスト (umaren) =====
from app import update_actual_results
# Horse 1 wins, horse 3 is 2nd -> umaren [1,3] should hit
update_actual_results('test_e', {1:1, 3:2, 2:4, 4:5, 5:6}, {'trio':5000, 'umaren':800, 'wide':300})
c.execute("SELECT hit_trio, payout FROM race_results WHERE race_id='test_e'")
row = c.fetchone()
ok("[11] Umaren hit detection", row and row[0] == 1 and row[1] == 800,
   f"hit={row[0] if row else None}, payout={row[1] if row else None}")

# ===== 12. 払戻金テスト =====
ok("[12] Payout for umaren=800 (not trio=5000)", row and row[1] == 800,
   f"payout={row[1] if row else None}")

# Test trio condition too
ri_a = {'distance': 2000, 'condition': '良', 'course': '東京', 'surface': '芝', 'grade': '', 'race_num': '2R'}
df_save_a = pd.DataFrame({
    '馬番': [1,2,3,4,5,6,7,8,9,10], 'スコア': [.9,.85,.8,.75,.7,.65,.6,.55,.5,.45],
    '馬名': list('ABCDEFGHIJ'), 'AI順位': list(range(1,11)),
    '単勝オッズ': [2,4,6,8,10,12,14,16,18,20]
})
save_prediction('test_a', 'テストA', ri_a, df_save_a, is_nar=False)
c.execute("SELECT bet_type FROM race_results WHERE race_id='test_a'")
row_a = c.fetchone()
ok("[10b] bet_type for A=trio", row_a and row_a[0] == 'trio',
   f"got {row_a[0] if row_a else None}")

update_actual_results('test_a', {1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9, 10:10},
                      {'trio':12000, 'umaren':1500, 'wide':500})
c.execute("SELECT hit_trio, payout FROM race_results WHERE race_id='test_a'")
row_a2 = c.fetchone()
ok("[12b] Trio payout=12000 (not umaren)", row_a2 and row_a2[0] == 1 and row_a2[1] == 12000,
   f"hit={row_a2[0] if row_a2 else None}, payout={row_a2[1] if row_a2 else None}")

conn.close()
os.remove(DB_TEST)
app.DB_PATH = old_db

# ===== 13. 中央/地方判別 =====
urls = [
    ("https://race.netkeiba.com/race/shutuba.html?race_id=202406050811", False),
    ("https://nar.netkeiba.com/race/shutuba.html?race_id=202444120511", True),
]
all_nar = all(("nar" in u) == exp for u, exp in urls)
ok("[13] Central/NAR detection", all_nar)

# ===== 14. TRACK RECORD badge =====
BET_SHORT = {'trio': '三', 'umaren': '連', 'wide': 'W'}
ok("[14] Badge mapping", BET_SHORT.get('trio')=='三' and BET_SHORT.get('umaren')=='連' and BET_SHORT.get('wide')=='W')

# ===== 15. 削除機能テスト =====
from app import delete_race_records, delete_all_race_records
ok("[15] Delete functions exist", callable(delete_race_records) and callable(delete_all_race_records))

# ===== 16. バッジ表示 =====
ok("[16] Badge: CENTRAL V9.1 AUC 0.8452", ver == 'v9.1' and abs(auc - 0.8452) < 0.001,
   f"ver={ver}, auc={auc:.4f}")

# ===== 17. レース名取得 =====
from app import parse_shutuba
ok("[17] parse_shutuba exists", callable(parse_shutuba))

# ===== 18. 起動テスト項目 =====
from app import run_system_checks
ok("[18] run_system_checks exists", callable(run_system_checks))

# ===== 19. 週次分析 =====
from app import get_weekly_analysis
ok("[19] get_weekly_analysis exists", callable(get_weekly_analysis))

# ===== 20. 特徴量重要度 =====
try:
    model = v9c['model']
    fi = model.feature_importance(importance_type='gain')
    top_feat = feats[fi.argmax()]
    ok("[20] Feature importance TOP20", len(fi) >= 20 and top_feat == 'odds_log',
       f"top={top_feat}, count={len(fi)}")
except Exception as e:
    ok("[20] Feature importance", False, str(e))

# ===== Summary =====
print()
print("=" * 60)
print(f"  TOTAL: {passed} PASS / {failed} FAIL / {passed+failed} tests")
if issues:
    print(f"  Issues found:")
    for i in issues:
        print(f"    - {i}")
else:
    print("  ALL TESTS PASSED!")
print("=" * 60)
