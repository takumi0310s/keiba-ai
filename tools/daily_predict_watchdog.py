"""daily_predict.py 中断検知 + 自動再起動 watchdog

機能:
1. daily_predict.py 実行を起動 (subprocess)
2. 5分ごとに進捗監視 (data/daily_predictions/YYYYMMDD.csv の race数)
3. 中断検知:
   - プロセス死亡 + race数 < 30 → Discord red アラート
   - Cookie切れ検知 (logにそのキーワード) → refresh_cookie.py 自動実行
   - 自動再起動 (max 3回、--resume付き)
4. 完了:
   - race数 >= 30 で正常完了 → Discord green
   - 障害除外でも >= 30 を期待 (5/3 35races実績)
5. ログ: logs/daily_predict_watchdog_YYYYMMDD.log

Usage:
  python tools/daily_predict_watchdog.py [--date YYYYMMDD]
"""
import sys, os, io, subprocess, time, argparse, glob
sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None

import pandas as pd
from datetime import datetime, timedelta

BASE_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
os.chdir(BASE_DIR)

MIN_RACES_FALLBACK = 30              # JRDB開催R数が取れない時のフォールバック floor
RACE_COMPLETION_RATIO = 0.9          # 開催R数のこの割合以上 予測できれば正常(取消/障害/parse失敗を許容)
CHECK_INTERVAL_SEC = 300             # 5分ごと監視
MAX_RESTARTS = 3                     # 最大再起動回数
PROCESS_HARD_TIMEOUT_SEC = 3600      # 1h でハード停止 (cycle あたり)


def expected_race_count(date_str):
    """その日の実開催R数(JRDB出走表 KYI 由来・netkeiba非アクセス)。取得不可なら0。"""
    try:
        sys.path.insert(0, os.path.join(BASE_DIR, 'tools'))
        from race_day_check import jrdb_race_day_info
        info = jrdb_race_day_info(date_str, BASE_DIR)
        if info.get('is_race_day') and info.get('n_races', 0) > 0:
            return int(info['n_races'])
    except Exception as e:
        print(f"[WARN] 開催R数取得失敗(fallback使用): {e}", flush=True)
    return 0


def compute_min_races_ok(date_str):
    """誤FATAL根絶: 固定30でなく その日の実開催R数 × 割合 を閾値に。
    2場(24R)→22 / 3場(36R)→33。 JRDB不可時は固定30へfallback。"""
    n = expected_race_count(date_str)
    if n > 0:
        return max(1, round(n * RACE_COMPLETION_RATIO)), n
    return MIN_RACES_FALLBACK, 0


def count_races(date_str):
    fp = f'data/daily_predictions/{date_str}.csv'
    if not os.path.exists(fp):
        return 0
    try:
        with open(fp, 'r', encoding='utf-8-sig') as f:
            lines = f.readlines()
        return max(0, len(lines) - 1)
    except Exception:
        return 0


def detect_cookie_failure(log_path):
    if not os.path.exists(log_path):
        return False
    try:
        with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
            tail = f.read()[-5000:]
        keys = ['Cookie無効', 'ログインセッション切れ', 'Cookie expired',
                'COOKIE WARN', '認証情報未保存']
        return any(k in tail for k in keys)
    except Exception:
        return False


def send_discord_alert(title, msg, color="yellow"):
    try:
        sys.path.insert(0, os.path.join(BASE_DIR, 'tools'))
        from notify import send_discord
        send_discord(title=title, message=msg, color=color, channel='updates')
    except Exception as e:
        print(f"[WARN] Discord送信失敗: {e}", flush=True)


def refresh_cookie():
    print("=== Cookie 自動更新 試行 ===", flush=True)
    try:
        r = subprocess.run([sys.executable, 'tools/refresh_cookie.py', '--auto'],
                           capture_output=True, text=True, timeout=180,
                           encoding='utf-8', errors='replace')
        ok = ('Cookie有効' in (r.stdout or '') or 'Premium認証OK' in (r.stdout or ''))
        print(f"[Cookie] result OK={ok}\n{r.stdout[-500:]}", flush=True)
        return ok
    except Exception as e:
        print(f"[Cookie] 例外: {e}", flush=True)
        return False


def run_daily_predict(date_str, resume=False, log_path=None):
    """daily_predict.py を subprocess で起動。"""
    cmd = [sys.executable, '-u', 'tools/daily_predict.py', '--date', date_str]
    if resume:
        cmd.append('--resume')
    print(f"[RUN] {' '.join(cmd)}", flush=True)
    log_f = open(log_path, 'a', encoding='utf-8', errors='replace') if log_path else None
    try:
        p = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT,
                              cwd=BASE_DIR, text=True, encoding='utf-8', errors='replace')
        return p, log_f
    except Exception as e:
        print(f"[ERROR] subprocess起動失敗: {e}", flush=True)
        if log_f: log_f.close()
        return None, None


def watchdog(date_str):
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    sub_log_path = os.path.join(log_dir, f'daily_predict_watchdog_{date_str}_subproc.log')
    main_log_path = os.path.join(log_dir, f'daily_predict_watchdog_{date_str}.log')

    min_races_ok, expected_n = compute_min_races_ok(date_str)
    exp_disp = expected_n if expected_n > 0 else f"{MIN_RACES_FALLBACK}(fallback)"

    print(f"=== daily_predict_watchdog START ===", flush=True)
    print(f"  Date: {date_str}", flush=True)
    print(f"  開催R数(JRDB): {exp_disp}  Min races OK(連動): {min_races_ok}", flush=True)
    print(f"  Check interval: {CHECK_INTERVAL_SEC}s", flush=True)
    print(f"  Max restarts: {MAX_RESTARTS}", flush=True)
    print(f"  Subprocess log: {sub_log_path}", flush=True)

    initial_races = count_races(date_str)
    print(f"  既存予測: {initial_races} レース", flush=True)

    restarts = 0
    last_race_count = initial_races
    prev_finish_races = -1   # 直前の subprocess 終了時の race数(無進捗 resume の検知用)
    stale_check_count = 0
    cycle_start = datetime.now()

    proc, log_f = run_daily_predict(date_str, resume=(initial_races > 0), log_path=sub_log_path)
    if proc is None:
        send_discord_alert("[Watchdog ERROR]", f"{date_str} subprocess起動失敗", color="red")
        return 1

    while True:
        time.sleep(CHECK_INTERVAL_SEC)

        races = count_races(date_str)
        rc = proc.poll()
        elapsed = (datetime.now() - cycle_start).total_seconds()

        print(f"[CHECK] races={races} (前回{last_race_count}) proc.rc={rc} elapsed={elapsed:.0f}s", flush=True)

        # 進捗 stuck 検知
        if races == last_race_count:
            stale_check_count += 1
        else:
            stale_check_count = 0
        last_race_count = races

        # プロセス終了チェック
        if rc is not None:
            print(f"[FINISH] subprocess exited with rc={rc}, races={races}", flush=True)
            if log_f: log_f.close()

            # Cookie切れ検知
            if detect_cookie_failure(sub_log_path) and races < min_races_ok:
                print("[DETECT] Cookie切れ → refresh試行", flush=True)
                if refresh_cookie() and restarts < MAX_RESTARTS:
                    restarts += 1
                    send_discord_alert(
                        f"[Watchdog] Cookie切れ → 自動再起動 #{restarts}",
                        f"{date_str} races={races}/{min_races_ok} - Cookie復活、--resume で再起動",
                        color="yellow",
                    )
                    proc, log_f = run_daily_predict(date_str, resume=True, log_path=sub_log_path)
                    cycle_start = datetime.now()
                    stale_check_count = 0
                    prev_finish_races = races
                    continue

            # 中断検知 (閾値=開催R数連動)
            if races < min_races_ok:
                # 無進捗 resume の検知: 直前終了時から1レースも増えていない=これ以上 resume しても無駄
                #   (daily_predict は rc=0 で「自分は完了」と判断している)。 無駄な再起動を避け即判定。
                no_progress = (races <= prev_finish_races)
                if restarts < MAX_RESTARTS and not no_progress:
                    restarts += 1
                    prev_finish_races = races
                    print(f"[RESTART {restarts}/{MAX_RESTARTS}] race数不足 ({races} < {min_races_ok})", flush=True)
                    send_discord_alert(
                        f"[Watchdog] daily_predict 中断検知 → 再起動 #{restarts}",
                        f"{date_str} races={races}/{min_races_ok} (rc={rc})\n--resume で再起動",
                        color="yellow",
                    )
                    proc, log_f = run_daily_predict(date_str, resume=True, log_path=sub_log_path)
                    cycle_start = datetime.now()
                    stale_check_count = 0
                    continue
                else:
                    reason = "再起動しても無進捗(resume不能)" if no_progress else f"{restarts}回再起動後も改善せず"
                    msg = f"{date_str} {reason}。 {races}/{min_races_ok} レースのみ完了(開催{exp_disp}R)。手動対応必要。"
                    print(f"[FATAL] {msg}", flush=True)
                    send_discord_alert(
                        f"❌ [Watchdog FATAL] daily_predict 修復失敗",
                        msg, color="red",
                    )
                    return 1

            # 正常完了
            print(f"[OK] daily_predict 正常完了 races={races}", flush=True)
            send_discord_alert(
                f"✅ [Watchdog] daily_predict 完了",
                f"{date_str} races={races} 再起動回数={restarts}",
                color="green",
            )
            return 0

        # ハードタイムアウト (1h)
        if elapsed > PROCESS_HARD_TIMEOUT_SEC:
            print(f"[TIMEOUT] {elapsed:.0f}s 経過、kill", flush=True)
            try: proc.kill()
            except Exception: pass
            send_discord_alert(
                f"[Watchdog] daily_predict タイムアウト",
                f"{date_str} 1h 経過 races={races} → kill して --resume 再起動",
                color="yellow",
            )
            if log_f: log_f.close()
            if restarts < MAX_RESTARTS:
                restarts += 1
                proc, log_f = run_daily_predict(date_str, resume=True, log_path=sub_log_path)
                cycle_start = datetime.now()
                stale_check_count = 0
                continue
            else:
                send_discord_alert(
                    f"❌ [Watchdog FATAL] timeout 連続",
                    f"{date_str} {restarts}回 timeout、手動対応必要",
                    color="red",
                )
                return 1

        # 進捗 stuck 警告 (3回連続で同じ races数)
        if stale_check_count >= 3 and rc is None:
            print(f"[WARN] 進捗停滞 {stale_check_count*CHECK_INTERVAL_SEC}s races={races}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', default=datetime.now().strftime('%Y%m%d'))
    args = parser.parse_args()
    rc = watchdog(args.date)
    sys.exit(rc)


if __name__ == '__main__':
    main()
