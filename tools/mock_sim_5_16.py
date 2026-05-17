"""
比較-3: 5/16 全 R mock -15min simulation (read-only).

mock direct odds = 朝 odds * Normal(1.0, 0.05) seed=42
mock weight_diff = 0 (neutral)
calibrator_overlay 適用 → 朝 vs mock -15min ranking 比較
→ 仮想 trio 7点 ROI delta

★ V15 / predict_core 不変、 git op なし、 read-only ★
"""
from __future__ import annotations

import sys
import io
from pathlib import Path
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tools.calibrator_overlay import overlay_race_ranking  # noqa: E402

DATE = "20260516"
SEED = 42
NOISE_STD = 0.05
NOISE_MEAN = 1.0


def severity(morning_top3, mock_top3):
    """severity 判定 (discord_recalc_notify と同形式)."""
    m_set = set(morning_top3)
    k_set = set(mock_top3)
    if m_set == k_set and morning_top3 == mock_top3:
        return "none"
    # top1 changed?
    if morning_top3[0] != mock_top3[0]:
        return "critical"
    # top1 same but top2/3 set differs?
    if {morning_top3[1], morning_top3[2]} != {mock_top3[1], mock_top3[2]}:
        return "major"
    # same set, only top2/top3 swap
    return "minor"


def build_trio_formation(top1, top2, top3, all_ranking):
    """top1 軸 trio 7点 formation 構築 (predict_core.py と同 logic).

    1列目: top1 only
    2列目: top2, top3
    3列目: top2 〜 top6 (5頭)
    → C(2,1) * C(5,2) なんちゃら... predict_core.py 実装の通り 7点。
    実 daily_predictions trio_bets_str を再現するのは複雑なので、
    ここでは simplified: top1 - {top2, top3} - {top2..top6} の 7組み合わせ。
    実 result の trio_bets_str を base にして hit 判定する形が無難。
    """
    # all_ranking = [(horse_num, score)] sorted desc
    top6 = [h[0] for h in all_ranking[:6]]
    if len(top6) < 3:
        return []
    t1 = top6[0]
    # 2列目候補 = top6[1:3]
    second = top6[1:3]
    # 3列目候補 = top6[1:6] (top2-top6)
    third = top6[1:6]
    bets = set()
    for s in second:
        for t in third:
            if s == t:
                continue
            trio = tuple(sorted([t1, s, t]))
            if len(set(trio)) == 3:
                bets.add(trio)
    bets = sorted(bets)
    # 7点 cap
    return bets[:7]


def check_trio_hit(bets, actual_top3):
    """trio hit 判定 (actual_top3 = list/set of 3 horse_num)."""
    actual_set = tuple(sorted(actual_top3))
    for bet in bets:
        if tuple(sorted(bet)) == actual_set:
            return True
    return False


def parse_trio_result(s):
    """'1-6-9' → [1, 6, 9]."""
    if pd.isna(s) or not isinstance(s, str):
        return None
    try:
        return [int(x) for x in s.split('-')]
    except Exception:
        return None


def main():
    np.random.seed(SEED)

    full = pd.read_csv(REPO / "data" / "daily_predictions_full" / f"{DATE}.csv", encoding='utf-8-sig')
    res = pd.read_csv(REPO / "data" / "daily_results" / f"{DATE}.csv", encoding='utf-8-sig')

    # race_id を str 化
    full['race_id'] = full['race_id'].astype(str)
    res['race_id'] = res['race_id'].astype(str)

    print(f"# 比較-3: 5/16 全 R mock -15min simulation")
    print(f"#races (full): {full['race_id'].nunique()}, #races (results): {len(res)}")
    print(f"seed={SEED}, noise std={NOISE_STD}")
    print()

    rows = []
    morning_total_invest = 0
    morning_total_payout = 0
    mock_total_invest = 0
    mock_total_payout = 0

    sev_counter = {"none": 0, "minor": 0, "major": 0, "critical": 0}
    morning_hit_count = 0
    mock_hit_count = 0

    # 戦略⑦案 C: 京都 / 06_特別 (非G1, OPEN特別) / 条件E / 条件B 除外
    # 5/16 actual: condition は A/C/D のみ (B/E なし)
    # 京都除外
    # 「06_特別」 = race_num >= 6 で race_name に "特別" or "S" 含むけど G1/G3 ではない … 簡易判定難。
    # ここは strict に condition + course のみで判定。

    for race_id, group in full.groupby('race_id'):
        race_id_str = str(race_id)
        group = group.sort_values('rank_in_race')

        # 朝 ranking (V15_score desc)
        morning_ranking = group.sort_values('V15_score', ascending=False)
        morning_top6 = morning_ranking.head(6)[['horse_num', 'V15_score']].values.tolist()
        morning_top3 = [int(morning_top6[i][0]) for i in range(min(3, len(morning_top6)))]

        # mock -15min odds = morning odds * N(1.0, 0.05)
        odds_arr = group['odds'].astype(float).values
        noise = np.random.normal(NOISE_MEAN, NOISE_STD, size=len(odds_arr))
        mock_odds = odds_arr * noise
        mock_odds = np.clip(mock_odds, 1.01, 9999.0)

        # overlay 適用
        df_in = pd.DataFrame({
            'horse_num': group['horse_num'].astype(int).values,
            'v15_prob': group['V15_score'].astype(float).values,
            'odds_15min': mock_odds,
            'weight_diff': np.zeros(len(group)),
        })
        recalc = overlay_race_ranking(df_in)
        mock_ranking = recalc.sort_values('corrected_prob', ascending=False)
        mock_top6 = mock_ranking.head(6)[['horse_num', 'corrected_prob']].values.tolist()
        mock_top3 = [int(mock_top6[i][0]) for i in range(min(3, len(mock_top6)))]

        # severity
        sev = severity(morning_top3, mock_top3)
        sev_counter[sev] += 1

        # 実 result
        res_row = res[res['race_id'] == race_id_str]
        if len(res_row) == 0:
            # No result for this race → skip ROI
            continue
        res_row = res_row.iloc[0]
        actual_top3 = parse_trio_result(res_row.get('trio_result'))
        if actual_top3 is None:
            continue

        # 朝 trio formation
        morning_top6_pairs = [(int(h), float(s)) for h, s in morning_top6]
        mock_top6_pairs = [(int(h), float(s)) for h, s in mock_top6]
        morning_bets = build_trio_formation(*morning_top3[:3], morning_top6_pairs) if len(morning_top3) >= 3 else []
        mock_bets = build_trio_formation(*mock_top3[:3], mock_top6_pairs) if len(mock_top3) >= 3 else []

        morning_hit = check_trio_hit(morning_bets, actual_top3)
        mock_hit = check_trio_hit(mock_bets, actual_top3)

        # 配当 (実 trio_payout で全 bet 同等扱い)
        # 実 daily_results は 700円投資、 hit すれば配当 = trio_payout (1点 base)
        payout_unit = float(res_row.get('trio_payout', 0))  # 1点 hit時の配当
        # 投資 = 100円×7点 = 700円 (predict_core 標準)
        invest_per_race = 700

        morning_payout_amt = payout_unit if morning_hit else 0
        mock_payout_amt = payout_unit if mock_hit else 0

        morning_total_invest += invest_per_race
        morning_total_payout += morning_payout_amt
        mock_total_invest += invest_per_race
        mock_total_payout += mock_payout_amt

        morning_hit_count += int(morning_hit)
        mock_hit_count += int(mock_hit)

        course = res_row.get('course', '')
        race_num = int(res_row.get('race_num', 0))
        race_name = res_row.get('race_name', '')
        condition = res_row.get('condition', '')

        rows.append({
            'race_id': race_id_str,
            'course': course,
            'race_num': race_num,
            'race_name': race_name,
            'condition': condition,
            'morning_top3': '-'.join(str(x) for x in morning_top3),
            'mock_top3': '-'.join(str(x) for x in mock_top3),
            'severity': sev,
            'actual_top3': '-'.join(str(x) for x in actual_top3),
            'morning_hit': int(morning_hit),
            'mock_hit': int(mock_hit),
            'payout_unit': int(payout_unit),
            'morning_payout': int(morning_payout_amt),
            'mock_payout': int(mock_payout_amt),
            'invest': invest_per_race,
        })

    df = pd.DataFrame(rows)
    print(df.to_string())
    print()
    print("=" * 80)
    print("# 統計")
    print(f"  total races evaluated: {len(df)}")
    print(f"  severity distribution: {sev_counter}")
    print(f"  morning hits: {morning_hit_count} / {len(df)} = {morning_hit_count/len(df)*100:.1f}%")
    print(f"  mock -15min hits: {mock_hit_count} / {len(df)} = {mock_hit_count/len(df)*100:.1f}%")
    print(f"  delta hits: {mock_hit_count - morning_hit_count:+d}")
    print()
    print(f"  morning invest: {morning_total_invest:,} 円")
    print(f"  morning payout: {morning_total_payout:,} 円")
    print(f"  morning ROI: {(morning_total_payout / morning_total_invest * 100):.2f}%")
    print(f"  morning profit: {morning_total_payout - morning_total_invest:+,} 円")
    print()
    print(f"  mock -15min invest: {mock_total_invest:,} 円")
    print(f"  mock -15min payout: {mock_total_payout:,} 円")
    print(f"  mock -15min ROI: {(mock_total_payout / mock_total_invest * 100):.2f}%")
    print(f"  mock -15min profit: {mock_total_payout - mock_total_invest:+,} 円")
    print()
    delta_roi = (mock_total_payout / mock_total_invest * 100) - (morning_total_payout / morning_total_invest * 100)
    print(f"  delta ROI: {delta_roi:+.2f} pt")
    print()
    # 戦略⑦案 C 適用
    df_c = df[(df['course'] != '京都') & (df['condition'].isin(['A', 'C', 'D']))]
    # 06_特別 除外 (race_num >= 6 で race_name に 特別 含む) は OPEN/G の区別が必要 — 簡易: race_name に 「特別」 含む & race_num >= 6
    is_06_tokubetsu = df_c['race_name'].astype(str).str.contains('特別') & (df_c['race_num'] >= 6)
    # ただし G1/G3/重賞 (大賞典 etc) は除外しない (race_name に "大賞典", "S" 含む)
    is_grade = df_c['race_name'].astype(str).str.contains('大賞典|G[1-3]')
    df_c = df_c[~(is_06_tokubetsu & ~is_grade)]
    print(f"# 戦略⑦案 C 適用後 (京都除外 + 06_特別 除外):")
    print(f"  candidate races: {len(df_c)}")
    if len(df_c) > 0:
        c_morning_invest = df_c['invest'].sum()
        c_morning_payout = df_c['morning_payout'].sum()
        c_mock_payout = df_c['mock_payout'].sum()
        print(f"  morning ROI (案C): {(c_morning_payout / c_morning_invest * 100):.2f}% profit {c_morning_payout - c_morning_invest:+,} 円")
        print(f"  mock -15min ROI (案C): {(c_mock_payout / c_morning_invest * 100):.2f}% profit {c_mock_payout - c_morning_invest:+,} 円")
        print(f"  morning hits (案C): {df_c['morning_hit'].sum()} / {len(df_c)}")
        print(f"  mock hits (案C): {df_c['mock_hit'].sum()} / {len(df_c)}")
    print()
    # 重賞個別
    print("# 重賞個別")
    for race_name_q in ['新潟大賞典', '六社S', '上賀茂S']:
        sub = df[df['race_name'] == race_name_q]
        if len(sub) > 0:
            row = sub.iloc[0]
            print(f"  {race_name_q} ({row['course']}{row['race_num']}R):")
            print(f"    朝 V15 top3: {row['morning_top3']}")
            print(f"    mock -15min top3: {row['mock_top3']}")
            print(f"    severity: {row['severity']}")
            print(f"    実 1-3着: {row['actual_top3']}")
            print(f"    morning hit: {row['morning_hit']}, mock hit: {row['mock_hit']}")
            print(f"    payout (hit時): {row['payout_unit']:,} 円")
            print()

    # 出力 csv (詳細記録、 docs 構築用)
    out_csv = REPO / "data" / "v21" / "mock_sim_5_16_detail.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False, encoding='utf-8-sig')
    print(f"detail csv: {out_csv}")

    # JSON summary
    import json
    summary = {
        "date": DATE,
        "seed": SEED,
        "noise_std": NOISE_STD,
        "n_races_evaluated": len(df),
        "severity_distribution": sev_counter,
        "morning_hits": int(morning_hit_count),
        "mock_hits": int(mock_hit_count),
        "delta_hits": int(mock_hit_count - morning_hit_count),
        "morning_invest": int(morning_total_invest),
        "morning_payout": int(morning_total_payout),
        "morning_roi_pct": float(morning_total_payout / morning_total_invest * 100) if morning_total_invest else 0,
        "mock_invest": int(mock_total_invest),
        "mock_payout": int(mock_total_payout),
        "mock_roi_pct": float(mock_total_payout / mock_total_invest * 100) if mock_total_invest else 0,
        "delta_roi_pt": float(delta_roi),
        "strategy_c_candidates": int(len(df_c)),
        "strategy_c_morning_roi": float(df_c['morning_payout'].sum() / df_c['invest'].sum() * 100) if len(df_c) else 0,
        "strategy_c_mock_roi": float(df_c['mock_payout'].sum() / df_c['invest'].sum() * 100) if len(df_c) else 0,
    }
    out_json = REPO / "data" / "v21" / "mock_sim_5_16_summary.json"
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"summary json: {out_json}")


if __name__ == "__main__":
    main()
