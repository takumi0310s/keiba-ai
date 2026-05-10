#!/usr/bin/env python
"""netkeiba Cookie自動更新ツール

Playwrightでnetkeibaにログインし、Cookieを取得して.envを更新する。

Usage:
    python tools/refresh_cookie.py                    # 対話式（ID/PW入力）
    python tools/refresh_cookie.py --check            # Cookie有効性チェックのみ
    python tools/refresh_cookie.py --headless=false    # ブラウザ表示モード

環境変数:
    NETKEIBA_LOGIN_ID   ログインID（メールアドレス）
    NETKEIBA_PASSWORD   パスワード

初回は対話式でID/PWを入力。--save-credentialsで.envに保存可能。
"""
import os
import sys
import re
import argparse
import getpass
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_PATH = os.path.join(BASE_DIR, '.env')

LOGIN_URL = 'https://regist.netkeiba.com/account/?pid=login'
TEST_URL = 'https://race.netkeiba.com/race/oikiri.html?race_id=202609010901'


def _read_env():
    """Read .env file as dict."""
    env = {}
    if os.path.exists(ENV_PATH):
        with open(ENV_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if '=' in line and not line.startswith('#'):
                    key, val = line.split('=', 1)
                    env[key.strip()] = val.strip().strip('"').strip("'")
    return env


def _write_env(env):
    """Write env dict back to .env file."""
    lines = []
    for key, val in env.items():
        # Quote values with special characters
        if any(c in val for c in ' ;='):
            lines.append(f'{key}="{val}"')
        else:
            lines.append(f'{key}={val}')
    with open(ENV_PATH, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')


def _get_credentials(save=False):
    """Get login credentials from env or interactive input."""
    env = _read_env()
    login_id = env.get('NETKEIBA_LOGIN_ID', '') or env.get('NETKEIBA_EMAIL', '')
    password = env.get('NETKEIBA_PASSWORD', '')

    if not login_id:
        login_id = input('netkeiba ログインID (メールアドレス): ').strip()
    else:
        print(f'ログインID: {login_id[:3]}***')

    if not password:
        password = getpass.getpass('netkeiba パスワード: ')

    if save and (login_id and password):
        env['NETKEIBA_LOGIN_ID'] = login_id
        env['NETKEIBA_PASSWORD'] = password
        _write_env(env)
        print('[INFO] 認証情報を.envに保存しました')

    return login_id, password


def refresh_cookie(headless=True, save_credentials=False):
    """Playwrightでnetkeibaにログインし、Cookieを更新する。

    Returns:
        tuple: (success: bool, message: str)
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return False, 'Playwrightがインストールされていません。\n  python -m pip install playwright && python -m playwright install chromium'

    login_id, password = _get_credentials(save=save_credentials)
    if not login_id or not password:
        return False, 'ログインID/パスワードが未入力です'

    print(f'[{datetime.now().strftime("%H:%M:%S")}] Playwright起動中...')

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        context = browser.new_context(
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        )
        page = context.new_page()

        try:
            # 1. ログインページに移動
            print(f'[{datetime.now().strftime("%H:%M:%S")}] ログインページに移動...')
            page.goto(LOGIN_URL, wait_until='domcontentloaded', timeout=15000)

            # 2. ログインフォームに入力
            page.fill('input[name="login_id"]', login_id)
            page.fill('input[name="pswd"]', password)

            # 3. ログインボタンをクリック
            submit = page.query_selector('input[type="image"]')
            if submit:
                submit.click()
            else:
                # Fallback: form submit
                page.evaluate('document.querySelector("form[action*=\\"account\\"]").submit()')

            # 4. ログイン完了を待つ
            page.wait_for_load_state('domcontentloaded', timeout=15000)
            page.wait_for_timeout(2000)

            # 5. ログイン成功確認
            current_url = page.url
            if 'login' in current_url and 'error' in current_url:
                browser.close()
                return False, 'ログイン失敗: IDまたはパスワードが間違っています'

            # 6. Premium確認ページにアクセスしてCookieを確定
            page.goto(TEST_URL, wait_until='domcontentloaded', timeout=15000)
            page.wait_for_timeout(1000)

            # 7. Cookieを取得
            cookies = context.cookies()
            if not cookies:
                browser.close()
                return False, 'Cookie取得失敗: ブラウザからCookieが取得できませんでした'

            # 重要なCookieをフィルタリング
            important_domains = ['netkeiba.com', '.netkeiba.com', 'race.netkeiba.com']
            cookie_parts = []
            key_cookies = set()
            for c in cookies:
                domain = c.get('domain', '')
                if any(d in domain for d in ['netkeiba.com']):
                    name = c['name']
                    value = c['value']
                    cookie_parts.append(f'{name}={value}')
                    key_cookies.add(name)

            # nkauthがあるか確認（Premium認証の証）
            if 'nkauth' not in key_cookies:
                browser.close()
                return False, 'Cookie取得失敗: nkauth cookieが見つかりません。Premium会員でない可能性があります'

            cookie_str = '; '.join(cookie_parts)

            # 8. .envに書き込み
            env = _read_env()
            env['NETKEIBA_COOKIE'] = cookie_str
            _write_env(env)

            # 8b. data/cookies.json にも Playwright 形式で export (paddock_video_capture 等で使用)
            try:
                import json as _json
                cookie_json_path = os.path.join(BASE_DIR, 'data', 'cookies.json')
                os.makedirs(os.path.dirname(cookie_json_path), exist_ok=True)
                with open(cookie_json_path, 'w', encoding='utf-8') as cf:
                    _json.dump(cookies, cf, ensure_ascii=False, indent=2)
                print(f'[INFO] data/cookies.json updated ({len(cookies)} cookies)')
            except Exception as e:
                print(f'[WARN] cookies.json export failed: {e}')

            browser.close()

            # 9. 検証
            ok, msg = _verify_cookie(cookie_str)
            if ok:
                return True, f'Cookie更新成功 ({len(cookie_parts)}個のCookie取得)\n  {msg}'
            else:
                return True, f'Cookie更新完了（検証未確認: {msg}）'

        except Exception as e:
            browser.close()
            return False, f'ログイン中にエラー: {e}'


def _verify_cookie(cookie_str):
    """更新したCookieが有効か検証する。"""
    import requests

    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    })
    for part in cookie_str.split(';'):
        part = part.strip()
        if '=' in part:
            key, val = part.split('=', 1)
            session.cookies.set(key.strip(), val.strip())

    try:
        # Use same check as scrape_training.check_premium_access()
        resp = session.get('https://db.netkeiba.com?pid=horse_training&id=2023102916', timeout=10)
        resp.encoding = 'EUC-JP'
        if 'プレミアムサービス案内' in resp.text:
            return False, 'Cookie無効: ログインセッション切れ'
        if re.search(r'\d{2}\.\d', resp.text):
            return True, 'Premium認証OK: 調教タイムデータ取得確認'
        return True, 'ページアクセス可（データ有無は未確認）'
    except Exception as e:
        return False, f'検証エラー: {e}'


def check_cookie():
    """現在のCookieの有効性をチェックする。"""
    env = _read_env()
    cookie_str = env.get('NETKEIBA_COOKIE', '')
    if not cookie_str or cookie_str == 'nkauth=XXXX; _nk_session=XXXX; ...':
        return False, 'Cookie未設定'

    ok, msg = _verify_cookie(cookie_str)
    return ok, msg


def auto_refresh_if_expired(headless=True):
    """Cookieが期限切れの場合のみ自動更新する。

    daily_predict.pyやrace_auto_notify.pyから呼び出す想定。
    認証情報が.envに保存されている場合のみ自動実行。

    Returns:
        tuple: (refreshed: bool, message: str)
    """
    ok, msg = check_cookie()
    if ok:
        return False, f'Cookie有効: {msg}'

    env = _read_env()
    login_id = env.get('NETKEIBA_LOGIN_ID', '') or env.get('NETKEIBA_EMAIL', '')
    password = env.get('NETKEIBA_PASSWORD', '')

    if not login_id or not password:
        return False, f'Cookie期限切れだが認証情報未保存。手動で refresh_cookie.py を実行してください。\n  {msg}'

    print(f'[INFO] Cookie期限切れを検出。自動更新中...')
    success, refresh_msg = refresh_cookie(headless=headless)
    if success:
        return True, f'Cookie自動更新成功: {refresh_msg}'
    else:
        return False, f'Cookie自動更新失敗: {refresh_msg}'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='netkeiba Cookie自動更新')
    parser.add_argument('--check', action='store_true', help='Cookie有効性チェックのみ')
    parser.add_argument('--headless', default='true',
                        help='ヘッドレスモード (true/false, デフォルト: true)')
    parser.add_argument('--save-credentials', action='store_true',
                        help='ログインID/PWを.envに保存')
    parser.add_argument('--auto', action='store_true',
                        help='期限切れ時のみ自動更新（認証情報が.envにある場合）')
    args = parser.parse_args()

    if args.check:
        ok, msg = check_cookie()
        status = 'OK' if ok else 'NG'
        print(f'[{status}] {msg}')
        sys.exit(0 if ok else 1)

    if args.auto:
        refreshed, msg = auto_refresh_if_expired(headless=args.headless.lower() != 'false')
        print(msg)
        sys.exit(0)

    headless = args.headless.lower() != 'false'
    success, msg = refresh_cookie(headless=headless, save_credentials=args.save_credentials)
    print(f'\n{"[OK]" if success else "[NG]"} {msg}')
    sys.exit(0 if success else 1)
