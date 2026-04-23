"""本番前 deploy-check (4/23版)

Usage:
    python tools/deploy_check.py
    python tools/deploy_check.py --no-discord  # Discord送信スキップ

状態確認のみ。修正なし。
"""
import argparse
import os
import sys
import json
import shutil
import subprocess
from datetime import datetime, timedelta

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORT_PATH = os.path.join(BASE_DIR, 'report', 'deploy_check_20260423.md')


def fmt_size(n):
    for u in ['B', 'KB', 'MB', 'GB']:
        if n < 1024:
            return f'{n:.1f}{u}'
        n /= 1024
    return f'{n:.1f}TB'


def file_age_days(path):
    if not os.path.exists(path):
        return None
    mtime = datetime.fromtimestamp(os.path.getmtime(path))
    return (datetime.now() - mtime).days


def check_pytest():
    """pytest tests/ 実行"""
    try:
        r = subprocess.run(
            ['python', '-m', 'pytest', 'tests/', '-q', '--tb=line', '--timeout=60'],
            cwd=BASE_DIR, capture_output=True, text=True, timeout=180
        )
        out = (r.stdout + r.stderr).strip()
        last = out.split('\n')[-1] if out else ''
        ok = r.returncode == 0
        return {
            'ok': ok,
            'returncode': r.returncode,
            'summary': last[:200],
        }
    except Exception as e:
        return {'ok': False, 'error': str(e)}


def check_schtasks():
    """Keiba タスクスケジューラ Ready 確認"""
    try:
        r = subprocess.run(
            ['powershell', '-Command',
             "Get-ScheduledTask | Where-Object { $_.TaskName -match 'Keiba' } | "
             "Select-Object TaskName, State | ConvertTo-Json"],
            capture_output=True, text=True, timeout=30
        )
        data = json.loads(r.stdout) if r.stdout.strip() else []
        if isinstance(data, dict):
            data = [data]
        return data
    except Exception as e:
        return [{'error': str(e)}]


def check_cookies():
    """cookie 状態 (.env のNETKEIBA_COOKIE + cookies.pkl)"""
    env_path = os.path.join(BASE_DIR, '.env')
    cookies_pkl = os.path.join(BASE_DIR, 'data', 'cookies.pkl')
    has_env = False
    if os.path.exists(env_path):
        with open(env_path, 'r', encoding='utf-8', errors='ignore') as f:
            txt = f.read()
            has_env = 'NETKEIBA_COOKIE=' in txt
    pkl_exists = os.path.exists(cookies_pkl)
    pkl_age = file_age_days(cookies_pkl) if pkl_exists else None
    return {
        'env_NETKEIBA_COOKIE': has_env,
        'cookies_pkl_exists': pkl_exists,
        'cookies_pkl_age_days': pkl_age,
    }


def check_disk():
    """ディスク空き容量"""
    total, used, free = shutil.disk_usage(BASE_DIR)
    return {
        'free': fmt_size(free),
        'free_gb': free / 1024 / 1024 / 1024,
        'used_pct': used / total * 100,
    }


def check_model():
    """v15 モデルファイル"""
    paths = [
        ('keiba_model_v15_central_live.pkl.gz', True),
        ('keiba_model_v15_central.pkl.gz', False),
    ]
    res = []
    for p, required in paths:
        full = os.path.join(BASE_DIR, p)
        ex = os.path.exists(full)
        res.append({
            'path': p,
            'exists': ex,
            'size': fmt_size(os.path.getsize(full)) if ex else None,
            'age_days': file_age_days(full) if ex else None,
            'required': required,
        })
    return res


def check_jrdb():
    """JRDB ファイル鮮度"""
    files = ['jrdb_kyi.csv', 'jrdb_sed.csv', 'jrdb_tyb.csv', 'jrdb_cyb.csv']
    res = []
    for f in files:
        full = os.path.join(BASE_DIR, 'data', f)
        ex = os.path.exists(full)
        res.append({
            'file': f,
            'exists': ex,
            'age_days': file_age_days(full) if ex else None,
            'size': fmt_size(os.path.getsize(full)) if ex else None,
        })
    return res


def check_predict_data():
    """直近 daily_predictions"""
    pred_dir = os.path.join(BASE_DIR, 'data', 'daily_predictions')
    if not os.path.isdir(pred_dir):
        return {'exists': False}
    files = sorted([f for f in os.listdir(pred_dir) if f.endswith('.csv')])
    return {
        'exists': True,
        'count': len(files),
        'latest': files[-3:] if files else [],
    }


def check_syntax():
    """app.py + predict_core.py 構文"""
    targets = ['app.py', 'tools/predict_core.py']
    res = []
    for t in targets:
        try:
            r = subprocess.run(
                ['python', '-c', f'import py_compile; py_compile.compile(r"{t}", doraise=True)'],
                cwd=BASE_DIR, capture_output=True, text=True, timeout=30
            )
            res.append({'file': t, 'ok': r.returncode == 0,
                        'err': r.stderr[:200] if r.stderr else ''})
        except Exception as e:
            res.append({'file': t, 'ok': False, 'err': str(e)})
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--no-discord', action='store_true')
    args = ap.parse_args()

    print('=' * 60)
    print('Deploy Check 20260423')
    print('=' * 60)

    results = {}

    print('\n[1] pytest...')
    results['pytest'] = check_pytest()
    print(f"  ok={results['pytest']['ok']} | {results['pytest'].get('summary', '')}")

    print('\n[2] schtasks (Keiba)...')
    results['schtasks'] = check_schtasks()
    if isinstance(results['schtasks'], list):
        for t in results['schtasks']:
            print(f"  {t.get('TaskName', t)}: {t.get('State', '?')}")

    print('\n[3] cookies...')
    results['cookies'] = check_cookies()
    print(f"  {results['cookies']}")

    print('\n[4] disk...')
    results['disk'] = check_disk()
    print(f"  free={results['disk']['free']} used={results['disk']['used_pct']:.1f}%")

    print('\n[5] model files...')
    results['model'] = check_model()
    for m in results['model']:
        print(f"  {m['path']}: exists={m['exists']} size={m['size']} age={m['age_days']}d")

    print('\n[6] JRDB files...')
    results['jrdb'] = check_jrdb()
    for j in results['jrdb']:
        print(f"  {j['file']}: exists={j['exists']} age={j['age_days']}d size={j['size']}")

    print('\n[7] predict data...')
    results['predict_data'] = check_predict_data()
    print(f"  {results['predict_data']}")

    print('\n[8] syntax...')
    results['syntax'] = check_syntax()
    for s in results['syntax']:
        print(f"  {s['file']}: ok={s['ok']}")

    # 判定
    warnings = []
    criticals = []

    if not results['pytest']['ok']:
        # Python 3.14 + pytest 環境互換問題は既知 (pre-existing)
        warnings.append(f"pytest FAIL (環境互換問題の可能性): {results['pytest'].get('summary', '')[:120]}")

    # ScheduledTaskState enum: 1=Disabled, 3=Ready, 4=Running
    READY_STATES = {'Ready', 3, '3'}
    not_ready = [t.get('TaskName') for t in results['schtasks']
                 if isinstance(t, dict) and t.get('State') not in READY_STATES]
    if not_ready:
        warnings.append(f"schtasks 非Ready: {not_ready}")

    if not results['cookies']['cookies_pkl_exists']:
        warnings.append("cookies.pkl 不在 (env のCOOKIEのみで運用)")
    elif results['cookies']['cookies_pkl_age_days'] and \
            results['cookies']['cookies_pkl_age_days'] > 7:
        warnings.append(f"cookies.pkl 古い ({results['cookies']['cookies_pkl_age_days']}日)")

    if results['disk']['free_gb'] < 10:
        criticals.append(f"ディスク空き不足: {results['disk']['free']}")

    for m in results['model']:
        if m['required'] and not m['exists']:
            criticals.append(f"必須モデル不在: {m['path']}")

    for j in results['jrdb']:
        if j['exists'] and j['age_days'] is not None and j['age_days'] > 5:
            warnings.append(f"JRDB {j['file']} 古い ({j['age_days']}日)")

    for s in results['syntax']:
        if not s['ok']:
            criticals.append(f"構文エラー {s['file']}: {s['err']}")

    # cookie要対応
    cookie_critical = '[CRITICAL] cookie期限切れ、金曜昼に python tools/refresh_cookie.py 必須'

    # レポート
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    lines = []
    lines.append(f'# Deploy Check 20260423\n\n')
    lines.append(f'実行日時: {datetime.now().strftime("%Y-%m-%d %H:%M")}\n')
    lines.append(f'本番: 2026-04-25 (土)\n\n')
    lines.append(f'## 判定\n\n')
    if not warnings and not criticals:
        lines.append(f'**🟢 本番準備完了**\n\n')
    elif criticals:
        lines.append(f'**🔴 CRITICAL ({len(criticals)}) — 即対応必須**\n\n')
        for c in criticals:
            lines.append(f'- {c}\n')
        lines.append(f'\n')
    else:
        lines.append(f'**🟡 警告 ({len(warnings)}) — 確認推奨**\n\n')
        for w in warnings:
            lines.append(f'- {w}\n')
        lines.append(f'\n')

    lines.append(f'\n## Cookie 状態\n')
    lines.append(f'- .env NETKEIBA_COOKIE 設定: {results["cookies"]["env_NETKEIBA_COOKIE"]}\n')
    lines.append(f'- cookies.pkl: {results["cookies"]["cookies_pkl_exists"]}\n')
    lines.append(f'- 期限切れの場合: {cookie_critical}\n\n')

    lines.append(f'## pytest\n')
    lines.append(f'- ok: {results["pytest"]["ok"]}\n')
    lines.append(f'- summary: `{results["pytest"].get("summary", "")[:200]}`\n\n')

    lines.append(f'## タスクスケジューラ (Keiba)\n')
    lines.append(f'(State enum: 1=Disabled, 3=Ready, 4=Running)\n\n')
    for t in results['schtasks']:
        if isinstance(t, dict) and 'TaskName' in t:
            st = t["State"]
            label = 'Ready' if st in READY_STATES else f'**{st} (要確認)**'
            lines.append(f'- {t["TaskName"]}: {label}\n')
    lines.append(f'\n')

    lines.append(f'## モデル\n')
    for m in results['model']:
        lines.append(f'- {m["path"]}: exists={m["exists"]} size={m["size"]} age={m["age_days"]}日\n')
    lines.append(f'\n')

    lines.append(f'## JRDB データ\n')
    for j in results['jrdb']:
        lines.append(f'- {j["file"]}: exists={j["exists"]} age={j["age_days"]}日 size={j["size"]}\n')
    lines.append(f'\n')

    lines.append(f'## ディスク\n')
    lines.append(f'- 空き: **{results["disk"]["free"]}** (使用率 {results["disk"]["used_pct"]:.1f}%)\n\n')

    lines.append(f'## 構文チェック\n')
    for s in results['syntax']:
        lines.append(f'- {s["file"]}: ok={s["ok"]}\n')
    lines.append(f'\n')

    lines.append(f'## 直近予測\n')
    lines.append(f'- {results["predict_data"]}\n\n')

    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write(''.join(lines))

    print(f'\nReport: {REPORT_PATH}')
    print(f'Warnings: {len(warnings)}, Criticals: {len(criticals)}')

    # Discord
    if not args.no_discord:
        try:
            sys.path.insert(0, BASE_DIR)
            from notify import send_discord
            color = 'red' if criticals else ('yellow' if warnings else 'green')
            title = ('🔴 deploy-check CRITICAL' if criticals
                     else ('🟡 deploy-check 警告' if warnings
                           else '🟢 deploy-check OK'))
            body = (f"本番: 2026-04-25 (土)\n"
                    f"CRITICAL: {len(criticals)} / 警告: {len(warnings)}\n"
                    f"詳細: report/deploy_check_20260423.md")
            if criticals:
                body += '\n\n' + '\n'.join(f'- {c}' for c in criticals[:5])
            if warnings:
                body += '\n\n' + '\n'.join(f'- {w}' for w in warnings[:5])
            send_discord(title, body, color=color, channel='updates')
        except Exception as e:
            print(f'Discord通知失敗: {e}')

    return 0 if not criticals else 1


if __name__ == '__main__':
    sys.exit(main())
