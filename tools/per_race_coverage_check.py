#!/usr/bin/env python3
"""per-race 通知カバレッジ 突合チェック (夜間・二重の検知)。

開催R数(JRDB KYI) vs per-race通知記録数(race_notify_log_v2/{date}/phase2/*.json) を突合し、
漏れがあれば #アップデート(UPDATES) に警告。 完走サマリ(race_auto_notify末尾)を見逃しても夜に気づける。

★本番無影響★: 読み取り + 警告通知のみ。 予測・投票・per-race本体に一切触れない。 例外は握り潰す。
使い方: python tools/per_race_coverage_check.py [--date YYYYMMDD]
       (DailyResults の後 ~20:45 に実行、 or daily_results 末尾から guarded 呼び出し)
"""
from __future__ import annotations
import os, sys, glob, argparse
if sys.platform == "win32":
    try: sys.stdout.reconfigure(encoding="utf-8")
    except Exception: pass
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, 'tools'))


def check_coverage(date_str, notify=True):
    """戻り値 dict {n_races, n_notified, missing, is_race_day}。漏れがあれば UPDATES 警告。"""
    res = {'date': date_str, 'n_races': 0, 'n_notified': 0, 'missing': 0, 'is_race_day': False, 'warned': False}
    try:
        # 開催R数 = JRDB 出走表(KYI)。 非開催日(KYIなし)は突合不要。
        from race_day_check import jrdb_race_day_info
        info = jrdb_race_day_info(date_str, BASE)
        res['is_race_day'] = info['is_race_day']; res['n_races'] = info['n_races']
        if not info['is_race_day'] or info['n_races'] == 0:
            print(f"[coverage] {date_str}: JRDB出走表なし=非開催日 → 突合不要")
            return res
        # 通知記録 = race_notify_log_v2/{date}/phase2/*.json (1レース1ファイル=通知済R)
        phase2 = glob.glob(os.path.join(BASE, 'data', 'race_notify_log_v2', date_str, 'phase2', '*.json'))
        n_notified = len(phase2)
        # fallback: race_notify_log/{date}.json
        if n_notified == 0:
            fb = os.path.join(BASE, 'data', 'race_notify_log', f'{date_str}.json')
            if os.path.exists(fb):
                import json
                try: n_notified = len({r.get('race_id') for r in json.load(open(fb, encoding='utf-8'))})
                except Exception: pass
        res['n_notified'] = n_notified
        res['missing'] = max(0, info['n_races'] - n_notified)
        print(f"[coverage] {date_str}: 開催 {info['n_races']}R / per-race通知記録 {n_notified}R / 漏れ {res['missing']}R")
        if res['missing'] > 0 and notify:
            try:
                from notify import send_discord
                send_discord(
                    f"⚠️ per-race通知 漏れ {res['missing']}件 ({date_str})",
                    f"開催 {info['n_races']}R のうち per-race通知記録は {n_notified}R のみ。"
                    f"{res['missing']}R 未通知の可能性(プロセス途中死/取得失敗/未発火)。\n"
                    f"※買い目が来なかったレースがある可能性。 race_auto_notify ログを確認してください。",
                    color="red", channel="updates")
                res['warned'] = True
                print(f"[coverage] UPDATES に漏れ警告を送信")
            except Exception as e:
                print(f"[coverage] 警告送信失敗(処理継続): {e}")
    except Exception as e:
        print(f"[coverage] 突合スキップ(例外・本番無影響): {e}")
    return res


def main():
    import time
    ap = argparse.ArgumentParser()
    ap.add_argument('--date', default=time.strftime('%Y%m%d'))
    ap.add_argument('--no-notify', action='store_true')
    a = ap.parse_args()
    check_coverage(a.date, notify=not a.no_notify)


if __name__ == '__main__':
    main()
