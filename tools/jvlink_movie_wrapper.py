#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""JV-Link MovieType wrapper (JRA レーシングビュアー access for V21 features).

user 加入: JRA-VAN DataLab + JRAレーシングビュアー (¥550/月、 既加入確認)
利用可能: 全レース映像 + 調教映像 + GI 特集 + ダートグレード

【V15 投資保護】 V15 production 一切 不変、 動画 download + frame extract のみ

【32-bit Python 必須】
C:\\Users\\takum\\jvlink-venv\\Scripts\\activate.bat

【JV-Link MovieType API】
- JVMVCheck(race_id, horse_id, kind) → movie 利用可否
- JVMVPlay(race_id, horse_id, kind) → movie URL or 直接 stream
- kind: 1=本馬場入場, 2=パドック, 3=レース, 4=調教

Usage:
    # JVLink COM 動作確認 + 利用可能 movie 確認
    python tools/jvlink_movie_wrapper.py --probe --race-id 202603010112 --horse-id 2022106229

    # 調教動画 取得
    python tools/jvlink_movie_wrapper.py --kind oikiri --race-id 202603010112 --horse-id 2022106229

    # レース動画
    python tools/jvlink_movie_wrapper.py --kind race --race-id 202603010112

Output:
    data/jvlink_movies/{kind}/{race_id}_{horse_id}.mp4 (or .ts)
    + frame extraction (cv2 で 後処理)
"""
import argparse
import os
import sys

try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# JV-Link MovieType kind mapping
MOVIE_KINDS = {
    'main': 1,      # 本馬場入場
    'paddock': 2,   # パドック (LIVE)
    'race': 3,      # レース実況
    'oikiri': 4,    # 調教
    'return': 5,    # 返し馬 (推定)
}


def check_32bit():
    """32-bit Python verify."""
    if sys.maxsize > 2**32:
        print('[ERROR] 64-bit Python detected. JV-Link requires 32-bit Python.')
        print('  Run: C:\\Users\\takum\\jvlink-venv\\Scripts\\activate.bat')
        return False
    return True


def init_jvlink():
    """JV-Link COM 初期化 (32-bit Python only)."""
    if not check_32bit():
        return None
    try:
        import win32com.client
        jv = win32com.client.Dispatch('JVDTLab.JVLink')
        # JVInit: SID 設定
        sid = 'UNKNOWN/0.1'  # 任意の identifier
        rc = jv.JVInit(sid)
        if rc != 0:
            print(f'[ERROR] JVInit failed rc={rc}')
            return None
        print(f'[OK] JVInit success')
        return jv
    except ImportError:
        print('[ERROR] pywin32 not installed: pip install pywin32')
        return None
    except Exception as e:
        print(f'[ERROR] {e}')
        return None


def probe_movie(jv, race_id, horse_id):
    """利用可能な movie kind 列挙."""
    print(f'\n=== Movie probe: race_id={race_id}, horse_id={horse_id} ===')
    for name, kind in MOVIE_KINDS.items():
        try:
            # JVMVCheck シグネチャ (実装は要 確認):
            # rc = jv.JVMVCheck(kind, raceid, umaban)
            rc = jv.JVMVCheck(kind, race_id, horse_id or '0')
            status = 'AVAILABLE' if rc == 0 else f'NA (rc={rc})'
            print(f'  {name:<10} (kind={kind}): {status}')
        except AttributeError:
            print(f'  {name:<10}: JVMVCheck method not exposed')
            break
        except Exception as e:
            print(f'  {name:<10}: ERROR {e}')


def download_movie(jv, kind, race_id, horse_id, out_dir):
    """1 movie download. JVMVPlay の動作は要 実 確認."""
    os.makedirs(out_dir, exist_ok=True)
    print(f'\n=== Download: kind={kind}, race_id={race_id}, horse_id={horse_id} ===')
    try:
        # JVMVPlay は通常 関連ソフトを起動するため direct 取得は困難
        # 別 method: JVMVGetMovieURL or similar (実装 によって異なる)
        # ここでは skeleton
        rc = jv.JVMVPlay(kind, race_id, horse_id or '0')
        print(f'  JVMVPlay rc={rc}')
        if rc == 0:
            print('  ※ 注意: JVMVPlay は 通常 関連 player 起動、 直接 download 不可')
            print('  ※ frame extract には Selenium / OBS / 画面録画 が必要かも')
    except AttributeError:
        print('  JVMVPlay method not exposed')
    except Exception as e:
        print(f'  ERROR {e}')


def main():
    ap = argparse.ArgumentParser(description='JV-Link MovieType wrapper (skeleton)')
    ap.add_argument('--probe', action='store_true', help='利用可能 movie 確認')
    ap.add_argument('--kind', choices=list(MOVIE_KINDS.keys()), default='oikiri')
    ap.add_argument('--race-id', dest='race_id', required=True)
    ap.add_argument('--horse-id', dest='horse_id', default='0')
    args = ap.parse_args()

    print('=== JV-Link MovieType wrapper ===')
    print(f'Python: {sys.maxsize > 2**32 and "64-bit" or "32-bit"}')

    jv = init_jvlink()
    if jv is None:
        return 1

    try:
        if args.probe:
            probe_movie(jv, args.race_id, args.horse_id)
        else:
            kind_code = MOVIE_KINDS[args.kind]
            out_dir = os.path.join(BASE_DIR, 'data', 'jvlink_movies', args.kind)
            download_movie(jv, kind_code, args.race_id, args.horse_id, out_dir)
    finally:
        try:
            jv.JVClose()
            print('\n[OK] JVClose')
        except Exception:
            pass

    return 0


if __name__ == '__main__':
    sys.exit(main())
