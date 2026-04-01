#!/usr/bin/env python3
"""netkeiba IPブロック解除を5分おきにリトライし、解除後に自動スクレイプ実行"""
import subprocess, sys, time, requests, os, io

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CHECK_URLS = [
    'https://race.netkeiba.com/top/',
    'https://db.netkeiba.com/',
]

def check_netkeiba():
    """race.netkeiba.com と db.netkeiba.com の両方がアクセス可能か確認"""
    for url in CHECK_URLS:
        try:
            r = requests.get(url,
                             headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'},
                             timeout=10)
            if r.status_code != 200:
                return False, f"{url} -> {r.status_code}"
        except Exception as e:
            return False, f"{url} -> {e}"
    return True, "OK"

def run_scraper(cmd, label):
    print(f"\n{'='*60}")
    print(f"  START: {label}")
    print(f"  CMD: {cmd}")
    print(f"{'='*60}", flush=True)
    result = subprocess.run(cmd, shell=True, cwd=BASE_DIR)
    status = "OK" if result.returncode == 0 else f"FAIL(exit={result.returncode})"
    print(f"\n  DONE: {label} [{status}]", flush=True)
    return result.returncode

def wait_for_unblock():
    """5分おきにリトライ、解除されたらTrue"""
    attempt = 0
    while True:
        attempt += 1
        ok, detail = check_netkeiba()
        ts = time.strftime('%Y-%m-%d %H:%M:%S')
        print(f"  [{ts}] attempt {attempt}: {detail}", flush=True)
        if ok:
            return True
        time.sleep(300)  # 5分

def main():
    print("=" * 60)
    print("  netkeiba IPブロック解除待機 → 自動スクレイプ")
    print("  リトライ間隔: 5分")
    print("=" * 60, flush=True)

    # 即座にチェック
    ok, detail = check_netkeiba()
    if ok:
        print(f"  アクセス可能! そのまま開始します。", flush=True)
    else:
        print(f"  現在ブロック中: {detail}", flush=True)
        print(f"  解除されるまで5分おきにリトライ...\n", flush=True)
        wait_for_unblock()
        print(f"\n  ブロック解除確認! 30秒待機後にスクレイプ開始...", flush=True)
        time.sleep(30)

    # ジョブ定義: (コマンド, ラベル)
    jobs = [
        ('python -u tools/scrape_training_bulk.py --years 2020 2021 2022 2023 2024',
         '調教タイム 2020-2024'),
        ('python -u tools/scrape_comments_bulk.py --years 2021 2022 2023',
         '厩舎コメント 2021-2023'),
    ]

    results = []
    for i, (cmd, label) in enumerate(jobs):
        # 各ジョブ前にアクセス確認
        ok, detail = check_netkeiba()
        if not ok:
            print(f"\n  再ブロック検出: {detail}", flush=True)
            print(f"  5分待機後にリトライ...", flush=True)
            wait_for_unblock()
            time.sleep(30)

        rc = run_scraper(cmd, label)
        results.append((label, rc))

        if i < len(jobs) - 1:
            time.sleep(10)  # ジョブ間10秒

    # 結果サマリー
    print(f"\n{'='*60}")
    print(f"  ALL JOBS COMPLETE")
    for label, rc in results:
        status = "OK" if rc == 0 else f"FAIL(exit={rc})"
        print(f"    {label}: {status}")
    print(f"{'='*60}", flush=True)

    # 完了通知音
    try:
        subprocess.run(['powershell', '-Command',
                        '[System.Media.SystemSounds]::Exclamation.Play()'])
    except:
        pass

    # Discord通知
    try:
        msg = "\\n".join(f"  {l}: {'OK' if rc==0 else 'FAIL'}" for l, rc in results)
        subprocess.run([
            'python', 'tools/notify_done.py',
            'Bulk Scrape完了',
            f"調教タイム+厩舎コメント一括取得完了\\n{msg}"
        ], cwd=BASE_DIR)
    except:
        pass

if __name__ == '__main__':
    main()
