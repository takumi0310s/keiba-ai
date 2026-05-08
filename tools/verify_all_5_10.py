"""Session #47 D: 5/10 朝 結果照合 framework.

5/9 全 R の予測 (predict_all_5_9.py が出力した json) と netkeiba 実結果を照合し
metrics + Discord 通知。

出力:
  data/v18/verification_5_10.md
  data/v18/verification_5_10.json

Usage:
  python tools/verify_all_5_10.py                 # 5/9 全 R 照合 (5/10 朝 想定)
  python tools/verify_all_5_10.py --dry-run       # 5/3 sample で動作確認
  python tools/verify_all_5_10.py --no-discord    # Discord 通知 skip

NEVER:
- daily_results.py 変更
- predict_core.py 変更
- 既存 result CSV 上書き

OK:
- daily_results.py の fetch_race_result() 流用 (read-only)
- check_results.py の 配当判定 流用
- Discord 新規 message
"""
import os
import sys
import json
import argparse
import time
from datetime import datetime
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, 'tools'))

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

V18_DIR = os.path.join(BASE_DIR, 'data', 'v18')
PREDICTIONS_JSON = os.path.join(V18_DIR, 'predictions_5_9_all.json')


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def fetch_result_safe(race_id):
    """race_id から実結果 (top3 着順 + 配当) を取得。

    Strategy:
      1. tools/daily_results.py の関数 import 試行
      2. tools/check_results.py の関数 import 試行
      3. 直接 netkeiba scrape (最後の手段)
    """
    # Strategy 1: daily_results
    try:
        from daily_results import fetch_race_result
        result = fetch_race_result(race_id)
        if result:
            return result
    except Exception:
        pass

    # Strategy 2: check_results
    try:
        from check_results import fetch_result
        result = fetch_result(race_id)
        if result:
            return result
    except Exception:
        pass

    # Strategy 3: 直接 (最低限)
    try:
        import requests
        from bs4 import BeautifulSoup
        url = f'https://race.netkeiba.com/race/result.html?race_id={race_id}'
        h = {'User-Agent': 'Mozilla/5.0'}
        r = requests.get(url, headers=h, timeout=10)
        soup = BeautifulSoup(r.text, 'html.parser')
        rows = soup.select('table.RaceTable01 tr')
        top3 = []
        for tr in rows[1:4]:
            cells = tr.find_all('td')
            if len(cells) >= 3:
                try:
                    finish = int(cells[0].get_text(strip=True))
                    umaban = int(cells[2].get_text(strip=True))
                    top3.append({'finish': finish, 'umaban': umaban})
                except Exception:
                    continue
        return {'top3': top3, 'race_id': race_id}
    except Exception as e:
        log(f"  fetch_result_safe failed for {race_id}: {e}")
        return None


def compute_metrics(predictions, results):
    """予測 vs 結果 の metrics 計算。"""
    n_total = 0
    n_top1_hit = 0
    n_top3_hit = 0
    n_in_top3_anywhere = 0  # 予測 top3 のいずれかが 3着以内
    by_grade = defaultdict(lambda: {
        'count': 0, 'top1_hit': 0, 'top3_hit': 0,
    })
    detail = []

    for p in predictions:
        rid = p.get('race_id')
        if not rid or p.get('error'):
            continue
        if rid not in results:
            continue
        result = results[rid]
        actual_top3 = result.get('top3', [])
        if not actual_top3:
            continue

        v15_top3 = p.get('v15_top3', [])
        if not v15_top3:
            continue

        # extract umaban from v15_top3 (format may vary)
        v15_umaban_top3 = []
        for it in v15_top3:
            if isinstance(it, dict):
                v15_umaban_top3.append(it.get('umaban') or it.get('horse_num') or it.get('num'))
            else:
                v15_umaban_top3.append(it)

        actual_top1 = actual_top3[0].get('umaban') if actual_top3 else None
        actual_umaban_set = {a.get('umaban') for a in actual_top3}

        n_total += 1
        grade = p.get('grade', 'unknown')
        by_grade[grade]['count'] += 1

        if v15_umaban_top3 and v15_umaban_top3[0] == actual_top1:
            n_top1_hit += 1
            by_grade[grade]['top1_hit'] += 1

        if v15_umaban_top3 and v15_umaban_top3[0] in actual_umaban_set:
            n_top3_hit += 1
            by_grade[grade]['top3_hit'] += 1

        if any(u in actual_umaban_set for u in v15_umaban_top3[:3]):
            n_in_top3_anywhere += 1

        detail.append({
            'race_id': rid,
            'grade': grade,
            'race_name': p.get('race_name', ''),
            'v15_top3_umaban': v15_umaban_top3[:3],
            'actual_top3_umaban': [a.get('umaban') for a in actual_top3],
            'top1_correct': v15_umaban_top3[0] == actual_top1 if v15_umaban_top3 else False,
            'fukusho_hit': v15_umaban_top3[0] in actual_umaban_set if v15_umaban_top3 else False,
        })

    summary = {
        'total': n_total,
        'top1_hit_rate': n_top1_hit / n_total if n_total else 0,
        'fukusho_hit_rate': n_top3_hit / n_total if n_total else 0,
        'top3_anywhere_rate': n_in_top3_anywhere / n_total if n_total else 0,
        'by_grade': {k: dict(v) for k, v in by_grade.items()},
        'detail': detail,
    }
    return summary


def discord_notify(summary, dry=False):
    """Discord 通知 (DISCORD_WEBHOOK_UPDATES → URL fallback)。"""
    if dry:
        log("Dry-run: skip Discord")
        return
    webhook = (os.environ.get('DISCORD_WEBHOOK_UPDATES')
               or os.environ.get('DISCORD_WEBHOOK_URL'))
    if not webhook:
        log("No Discord webhook env var set, skip")
        return
    try:
        import requests
        msg = (
            f"## Session #47 D: 5/9 全 R 結果照合\n"
            f"- 全 R: {summary['total']}\n"
            f"- top1 hit: {summary['top1_hit_rate']:.1%}\n"
            f"- 複勝 hit: {summary['fukusho_hit_rate']:.1%}\n"
            f"- top3 anywhere: {summary['top3_anywhere_rate']:.1%}\n\n"
            f"クラス別:\n"
        )
        for grade, m in summary['by_grade'].items():
            if m['count'] > 0:
                msg += f"- {grade}: N={m['count']}, top1={m['top1_hit']}/{m['count']}\n"
        r = requests.post(webhook, json={'content': msg[:1900]})
        log(f"Discord: HTTP {r.status_code}")
    except Exception as e:
        log(f"Discord notify failed: {e}")


def write_markdown(summary, predictions_count, date_str):
    """data/v18/verification_5_10.md を書く。"""
    out = os.path.join(V18_DIR, 'verification_5_10.md')
    with open(out, 'w', encoding='utf-8') as f:
        f.write(f"# Session #47 D: 5/{date_str[6:8]} 結果照合 verdict\n\n")
        f.write(f"date: {date_str}\n")
        f.write(f"predictions: {predictions_count} R\n")
        f.write(f"matched: {summary['total']} R\n\n")
        f.write(f"## 全体 metrics\n\n")
        f.write(f"| metric | value |\n|--------|------|\n")
        f.write(f"| top1 hit rate | {summary['top1_hit_rate']:.1%} |\n")
        f.write(f"| 複勝 hit rate | {summary['fukusho_hit_rate']:.1%} |\n")
        f.write(f"| top3 anywhere rate | {summary['top3_anywhere_rate']:.1%} |\n\n")
        f.write(f"## クラス別 metrics\n\n")
        f.write(f"| grade | N | top1 hit | top3 hit |\n|-------|---|----------|----------|\n")
        for grade, m in summary['by_grade'].items():
            if m['count'] > 0:
                f.write(f"| {grade} | {m['count']} | "
                        f"{m['top1_hit']}/{m['count']} ({m['top1_hit']/m['count']:.0%}) | "
                        f"{m['top3_hit']}/{m['count']} ({m['top3_hit']/m['count']:.0%}) |\n")
        f.write(f"\n## 詳細\n\n")
        f.write(f"| race | grade | v15 top3 | 実 top3 | top1 OK | 複勝 OK |\n")
        f.write(f"|------|-------|----------|---------|---------|--------|\n")
        for d in summary['detail'][:50]:
            f.write(f"| {d['race_id'][-4:]} | {d['grade']} | "
                    f"{d['v15_top3_umaban']} | {d['actual_top3_umaban']} | "
                    f"{'OK' if d['top1_correct'] else '-'} | "
                    f"{'OK' if d['fukusho_hit'] else '-'} |\n")
    log(f"Saved: {out}")


def run_verify(date_str=None, dry_run=False, no_discord=False):
    if dry_run:
        log("=== Dry-run mode (sample) ===")
    if date_str is None:
        date_str = '20260509'
    log(f"Verify {date_str} predictions")

    if not os.path.exists(PREDICTIONS_JSON):
        log(f"NG: predictions json not found: {PREDICTIONS_JSON}")
        log("→ run tools/predict_all_5_9.py first")
        return None

    with open(PREDICTIONS_JSON, 'r', encoding='utf-8') as f:
        pred_data = json.load(f)

    predictions = pred_data.get('predictions', [])
    log(f"Loaded {len(predictions)} predictions")

    # Fetch results
    results = {}
    if dry_run:
        log("Dry-run: skip actual fetch, simulate empty results")
    else:
        for i, p in enumerate(predictions):
            rid = p.get('race_id')
            if not rid:
                continue
            log(f"[{i + 1}/{len(predictions)}] result fetch {rid}")
            r = fetch_result_safe(rid)
            if r:
                results[rid] = r
            time.sleep(1)

    # Metrics
    summary = compute_metrics(predictions, results)
    log(f"Metrics: top1={summary['top1_hit_rate']:.1%}, "
        f"複勝={summary['fukusho_hit_rate']:.1%}, "
        f"matched={summary['total']}")

    # Save JSON
    out_json = os.path.join(V18_DIR, 'verification_5_10.json')
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    log(f"Saved: {out_json}")

    # Markdown
    write_markdown(summary, len(predictions), date_str)

    # Discord
    if not no_discord and not dry_run:
        discord_notify(summary)

    return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--date', default='20260509')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--no-discord', action='store_true')
    args = p.parse_args()

    os.makedirs(V18_DIR, exist_ok=True)
    run_verify(date_str=args.date, dry_run=args.dry_run, no_discord=args.no_discord)
