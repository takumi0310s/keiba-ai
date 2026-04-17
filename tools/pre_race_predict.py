#!/usr/bin/env python
"""前夜予測モード（金曜夜・土曜夜に翌日分を予測→Discord通知）

daily_predict.py と同じロジックで指定日分の予測を実行し、整形済み買い目を
🔮 前夜予測タグ付きで Discord #買い目 に送信する。

Usage:
    python tools/pre_race_predict.py                      # 翌日
    python tools/pre_race_predict.py --date 2026-04-18    # 指定日
    python tools/pre_race_predict.py --no-notify          # 予測のみ

注意:
    - 前夜時点ではオッズ・馬体重が未確定のため、参考値として扱う
    - 当日 AM8:00 の daily_predict.py が本予測を再実行する
    - SCRAPER-GUARD (金22時〜月6時) が稼働している間は netkeiba側スクレイプ
      が抑制されるため、pre_race_predict は SCRAPER_GUARD_DISABLE=1 を自動設定して実行
"""
from __future__ import annotations
import argparse
import os
import subprocess
import sys
from datetime import datetime, timedelta

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, 'tools'))


def normalize_date(s: str) -> str:
    """空 -> 翌日 / YYYY-MM-DD or YYYYMMDD -> YYYYMMDD"""
    if not s:
        s = (datetime.now() + timedelta(days=1)).strftime('%Y%m%d')
    return s.replace('-', '').replace('/', '')


def main():
    ap = argparse.ArgumentParser(description='前夜予測モード')
    ap.add_argument('--date', type=str, default='',
                    help='YYYY-MM-DD or YYYYMMDD (default: tomorrow)')
    ap.add_argument('--no-notify', action='store_true',
                    help='Discord通知をスキップし予測のみ実行')
    ap.add_argument('--channel', default='bets',
                    help='Discord channel (default: bets)')
    args = ap.parse_args()

    date_yyyymmdd = normalize_date(args.date)
    print(f"[pre_race_predict] 対象日: {date_yyyymmdd}")

    # daily_predict.py を呼び出す (予測＋CSV保存まで担当させる)
    env = os.environ.copy()
    # 週末ガード中でも強制実行 (前夜予測はレース情報のみ取得)
    env['SCRAPER_GUARD_DISABLE'] = '1'

    daily_predict_py = os.path.join(BASE_DIR, 'tools', 'daily_predict.py')
    cmd = [sys.executable, '-u', daily_predict_py, '--date', date_yyyymmdd]
    print(f"[pre_race_predict] run: {' '.join(cmd)}")
    rc = subprocess.call(cmd, cwd=BASE_DIR, env=env)
    print(f"[pre_race_predict] daily_predict exit={rc}")

    csv_path = os.path.join(BASE_DIR, 'data', 'daily_predictions',
                            f'{date_yyyymmdd}.csv')
    if not os.path.exists(csv_path):
        print(f"[pre_race_predict] CSV未生成: {csv_path}")
        try:
            from notify import send_discord
            send_discord('🔮 前夜予測エラー',
                         f"{date_yyyymmdd} CSV生成失敗 (exit={rc})",
                         color='red', channel='updates')
        except Exception:
            pass
        return rc or 1

    if args.no_notify:
        print('[pre_race_predict] --no-notify 指定により通知スキップ')
        return 0

    # daily_predict.py 内部でも morning モード通知が走るので、ここでは
    # 明示的に pre_race モードを上書き送信するため、先行通知と重複しないよう
    # pre_race 専用タグで送る。
    try:
        from notify_bets_formatted import notify_formatted
        sent = notify_formatted(date_yyyymmdd, mode='pre_race',
                                channel=args.channel)
        print(f"[pre_race_predict] 前夜予測通知送信: {sent} messages")
    except Exception as e:
        print(f"[pre_race_predict] 通知失敗: {e}")
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
