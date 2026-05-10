#!/usr/bin/env python
"""netkeiba マスターコース scraper — Phase 13 (2026-05-10).

netkeiba ¥4,980/月 マスターコース 加入後、 4 系統 25 features を取得する。

★★★ 重要 caveat ★★★
- ToS risk あり (ユーザー受け入れ済、 個人利用範囲)。
- 自動 fetch は KILL_SWITCH 経由で即停止可能。
- rate limit 3 sec interval 厳守。
- data 再配布禁止 (data/netkeiba_master/ 直下保存、 git ignore)。

実 fetch tooling 用 caller (V15 production には影響しない):
    python tools/netkeiba_master_scraper.py --enable --date 20260510
    python tools/netkeiba_master_scraper.py --race 202605020611

無効化 (kill switch):
    touch data/netkeiba_master/.disabled
    → 全 fetch skip、 default fill のみ返す。

Phase 13 で skeleton + parser stub 実装、 Phase 13.5 で実 DOM 検証 + 本稼働。
"""
from __future__ import annotations
import os
import sys
import re
import json
import time
import argparse
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Any, Optional

import requests
from bs4 import BeautifulSoup

# =========================================================================
# Path / Constants
# =========================================================================

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MASTER_DIR = os.path.join(DATA_DIR, 'netkeiba_master')
KILL_SWITCH_FILE = os.path.join(MASTER_DIR, '.disabled')

# rate limit (絶対遵守 — 3 秒 interval)
FETCH_INTERVAL_SEC = 3.0

# 最終 fetch 時刻 (rate limit 計算用)
_last_fetch_ts: float = 0.0

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 keiba-ai/Phase13"
)

# =========================================================================
# Feature schema — 4 系統 計 25 features
# =========================================================================

# B. AI 展開予測 (7 features) — race/compatibility.html
AI_TENKAI_FEATURES: List[str] = [
    'master_pace_pred',           # AI 予測ペース (slow=0/medium=1/high=2)
    'master_pred_winner_score',   # AI 予測 1 着馬 score (0-100)
    'master_pred_first3f_avg',    # AI 予測 前半 3F 平均 (秒)
    'master_pred_last3f_avg',     # AI 予測 後半 3F 平均 (秒)
    'master_pred_finish_time',    # AI 予測 走破タイム (秒)
    'master_horse_aitenkai_score', # 当該馬の AI 展開適性 score (0-100、 馬別)
    'master_horse_pred_pos',      # AI 予測 通過順位 (4 角想定、 1-18)
]

# C1. AI 波乱度 (3 features)
AI_HARAN_FEATURES: List[str] = [
    'master_haran_score',         # 波乱度 score (0-100、 高=荒れ)
    'master_top_pop_trust',       # 上位人気 信頼度 (0-100)
    'master_haran_meter',         # 波乱メーター 5 段階 (1-5)
]

# C2. 個別ラップ (10 features) — race/lap.html (推定)
LAP_FEATURES: List[str] = [
    'master_horse_lap_avg_first3f', # 当該馬 前 3 走 前半 3F 平均
    'master_horse_lap_avg_last3f',  # 当該馬 前 3 走 後半 3F 平均
    'master_horse_lap_best_last3f', # 前 3 走 後半 3F best
    'master_horse_lap_consistency', # ラップ安定性 (std)
    'master_horse_lap_best_3f',     # 全期間 ベスト後半 3F
    'master_horse_lap_pos_change_avg', # 平均位置取り変化
    'master_horse_lap_finish_speed',   # 終速指標 (last 1F speed)
    'master_horse_lap_acc_phase',      # 加速 phase 数
    'master_horse_lap_dec_phase',      # 減速 phase 数
    'master_horse_lap_distance_factor', # 距離適応 factor
]

# C3. トラックバイアス (5 features) — race/track_bias.html (推定)
TRACK_BIAS_FEATURES: List[str] = [
    'master_track_inner_outer_bias',  # 内 vs 外 (-1 内有利、 +1 外有利、 0 中立)
    'master_track_front_back_bias',   # 前 vs 後 (-1 逃げ有利、 +1 差し有利)
    'master_track_corner_bias',       # コーナー有利不利 (-1〜+1)
    'master_track_pace_bias_score',   # ペース bias score
    'master_track_today_severity',    # 当日 馬場 severity (0-100)
]

ALL_PHASE13_FEATURES: List[str] = (
    AI_TENKAI_FEATURES
    + AI_HARAN_FEATURES
    + LAP_FEATURES
    + TRACK_BIAS_FEATURES
)
assert len(ALL_PHASE13_FEATURES) == 25, f"expected 25, got {len(ALL_PHASE13_FEATURES)}"

# Default fill values
PHASE13_DEFAULTS: Dict[str, Any] = {
    # AI 展開予測
    'master_pace_pred': 1,                # medium default
    'master_pred_winner_score': 50.0,
    'master_pred_first3f_avg': 35.5,
    'master_pred_last3f_avg': 35.5,
    'master_pred_finish_time': 100.0,
    'master_horse_aitenkai_score': 50.0,
    'master_horse_pred_pos': 9,           # 中位 default
    # 波乱度
    'master_haran_score': 50.0,
    'master_top_pop_trust': 50.0,
    'master_haran_meter': 3,
    # ラップ
    'master_horse_lap_avg_first3f': 35.5,
    'master_horse_lap_avg_last3f': 35.5,
    'master_horse_lap_best_last3f': 34.5,
    'master_horse_lap_consistency': 1.0,
    'master_horse_lap_best_3f': 34.0,
    'master_horse_lap_pos_change_avg': 0.0,
    'master_horse_lap_finish_speed': 12.0,
    'master_horse_lap_acc_phase': 1,
    'master_horse_lap_dec_phase': 1,
    'master_horse_lap_distance_factor': 0.5,
    # トラックバイアス
    'master_track_inner_outer_bias': 0.0,
    'master_track_front_back_bias': 0.0,
    'master_track_corner_bias': 0.0,
    'master_track_pace_bias_score': 0.0,
    'master_track_today_severity': 50.0,
}
assert set(PHASE13_DEFAULTS.keys()) == set(ALL_PHASE13_FEATURES)


# =========================================================================
# URL templates
# =========================================================================

URL_AI_TENKAI = "https://race.sp.netkeiba.com/race/compatibility.html?race_id={race_id}"
URL_LAP = "https://race.sp.netkeiba.com/race/lap.html?race_id={race_id}"
URL_TRACK_BIAS = "https://race.sp.netkeiba.com/race/track_bias.html?kaisai_id={kaisai_id}"
URL_AI_HARAN = "https://race.sp.netkeiba.com/race/upset.html?race_id={race_id}"


# =========================================================================
# Cookie / Session 管理
# =========================================================================

def _load_cookie() -> Optional[str]:
    """`.env` から NETKEIBA_COOKIE を読み込む (既存 system 共有)."""
    env_path = os.path.join(BASE_DIR, '.env')
    if not os.path.exists(env_path):
        return None
    try:
        with open(env_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line.startswith('NETKEIBA_COOKIE='):
                    val = line[len('NETKEIBA_COOKIE='):]
                    val = val.strip('"').strip("'")
                    if val and 'XXXX' not in val:
                        return val
    except Exception:
        pass
    return None


def _make_session() -> Optional[requests.Session]:
    cookie_str = _load_cookie()
    if not cookie_str:
        return None
    sess = requests.Session()
    sess.headers.update({"User-Agent": USER_AGENT})
    for part in cookie_str.split(';'):
        part = part.strip()
        if '=' in part:
            k, v = part.split('=', 1)
            sess.cookies.set(k.strip(), v.strip())
    return sess


# =========================================================================
# Kill switch + rate limit
# =========================================================================

def is_disabled() -> bool:
    return os.path.exists(KILL_SWITCH_FILE)


def _wait_rate_limit():
    global _last_fetch_ts
    now = time.time()
    elapsed = now - _last_fetch_ts
    if elapsed < FETCH_INTERVAL_SEC:
        time.sleep(FETCH_INTERVAL_SEC - elapsed)
    _last_fetch_ts = time.time()


def _fetch(url: str, session: requests.Session) -> Optional[str]:
    """rate-limited GET、 失敗時 None."""
    if is_disabled():
        return None
    _wait_rate_limit()
    try:
        r = session.get(url, timeout=20)
        if r.status_code == 200:
            r.encoding = r.apparent_encoding or 'utf-8'
            return r.text
    except Exception as e:
        print(f"[netkeiba_master] fetch fail {url}: {e}", file=sys.stderr)
    return None


# =========================================================================
# Parser stubs (Phase 13 = skeleton、 実 DOM 検証 + 本稼働 は Phase 13.5)
# =========================================================================

def _parse_ai_tenkai(html: str, umaban: int) -> Dict[str, Any]:
    """AI 展開予測 parser (compatibility.html).

    Phase 13 では DOM 構造を best-effort で抽出。 取れない field は default。
    実 DOM 検証 + selector 確定 は user 初回 fetch 時 (Phase 13.5)。
    """
    out = {f: PHASE13_DEFAULTS[f] for f in AI_TENKAI_FEATURES}
    if not html:
        return out
    soup = BeautifulSoup(html, 'html.parser')

    # ペース予測 — 想定 selector: .RaceData_PacePred / .pace_pred
    pace_text = ''
    for sel in ['.RaceData_PacePred', '.pace_pred', '#pace_prediction']:
        el = soup.select_one(sel)
        if el and el.get_text(strip=True):
            pace_text = el.get_text(strip=True)
            break
    if pace_text:
        if 'スロー' in pace_text or 'slow' in pace_text.lower():
            out['master_pace_pred'] = 0
        elif 'ハイ' in pace_text or 'high' in pace_text.lower():
            out['master_pace_pred'] = 2

    # 全体予測 タイム — m{XX}.{X} pattern
    for el in soup.select('.RaceData_FinishTime, .pred_finish_time, .race_pred_time'):
        m = re.search(r'(\d+)[:m]?(\d+\.\d+)', el.get_text(strip=True))
        if m:
            out['master_pred_finish_time'] = int(m.group(1)) * 60 + float(m.group(2))
            break

    # 馬別 score / 通過順 — table 行から umaban で照合
    for tr in soup.select('table tr'):
        umaban_el = tr.select_one('.umaban, .horse_num, td[class*="umaban"]')
        if not umaban_el:
            continue
        try:
            row_umaban = int(re.sub(r'\D', '', umaban_el.get_text()))
        except Exception:
            continue
        if row_umaban != umaban:
            continue
        score_el = tr.select_one('.score, .ai_score, .aitenkai_score')
        if score_el:
            m = re.search(r'(\d+\.?\d*)', score_el.get_text())
            if m:
                out['master_horse_aitenkai_score'] = float(m.group(1))
        pos_el = tr.select_one('.pred_pos, .pass_pos, .pos_pred')
        if pos_el:
            m = re.search(r'(\d+)', pos_el.get_text())
            if m:
                out['master_horse_pred_pos'] = int(m.group(1))
        break

    return out


def _parse_ai_haran(html: str) -> Dict[str, Any]:
    """AI 波乱度 parser (upset.html 推定)."""
    out = {f: PHASE13_DEFAULTS[f] for f in AI_HARAN_FEATURES}
    if not html:
        return out
    soup = BeautifulSoup(html, 'html.parser')

    for sel in ['.haran_score', '.upset_score', '#upset_score']:
        el = soup.select_one(sel)
        if el:
            m = re.search(r'(\d+\.?\d*)', el.get_text())
            if m:
                out['master_haran_score'] = float(m.group(1))
                break

    # メーター 5 段階 — class 内に 'lv1'-'lv5' 想定
    for cls in ['lv1', 'lv2', 'lv3', 'lv4', 'lv5']:
        if soup.select_one(f'.haran_meter.{cls}, .upset_meter.{cls}'):
            out['master_haran_meter'] = int(cls.replace('lv', ''))
            break

    return out


def _parse_lap(html: str, umaban: int) -> Dict[str, Any]:
    """個別ラップ parser (lap.html 推定).

    netkeiba master は 各馬の前走〜過去 3 走 個別ラップを表示。
    """
    out = {f: PHASE13_DEFAULTS[f] for f in LAP_FEATURES}
    if not html:
        return out
    soup = BeautifulSoup(html, 'html.parser')

    # 馬番 row 探索
    target_rows = []
    for tr in soup.select('table tr, .horse_lap_row'):
        umaban_el = tr.select_one('.umaban, .horse_num')
        if not umaban_el:
            continue
        try:
            if int(re.sub(r'\D', '', umaban_el.get_text())) == umaban:
                target_rows.append(tr)
        except Exception:
            continue

    if not target_rows:
        return out

    # 直近 3 走の前後半 3F 抽出
    first3f_vals: List[float] = []
    last3f_vals: List[float] = []
    for tr in target_rows[:3]:
        for cell in tr.select('.first3f, .last3f, .lap_first, .lap_last'):
            m = re.search(r'(\d+\.\d+)', cell.get_text())
            if not m:
                continue
            v = float(m.group(1))
            cls = ' '.join(cell.get('class', []))
            if 'first' in cls.lower():
                first3f_vals.append(v)
            elif 'last' in cls.lower():
                last3f_vals.append(v)

    if first3f_vals:
        out['master_horse_lap_avg_first3f'] = sum(first3f_vals) / len(first3f_vals)
    if last3f_vals:
        out['master_horse_lap_avg_last3f'] = sum(last3f_vals) / len(last3f_vals)
        out['master_horse_lap_best_last3f'] = min(last3f_vals)
        out['master_horse_lap_best_3f'] = min(last3f_vals)
        if len(last3f_vals) > 1:
            mu = sum(last3f_vals) / len(last3f_vals)
            var = sum((v - mu) ** 2 for v in last3f_vals) / len(last3f_vals)
            out['master_horse_lap_consistency'] = var ** 0.5

    return out


def _parse_track_bias(html: str) -> Dict[str, Any]:
    """トラックバイアス parser (track_bias.html 推定)."""
    out = {f: PHASE13_DEFAULTS[f] for f in TRACK_BIAS_FEATURES}
    if not html:
        return out
    soup = BeautifulSoup(html, 'html.parser')

    # 内外 bias — 「内有利」「外有利」「中立」
    text = soup.get_text()
    if '内有利' in text:
        out['master_track_inner_outer_bias'] = -1.0
    elif '外有利' in text:
        out['master_track_inner_outer_bias'] = 1.0
    if '逃げ有利' in text or '前残り' in text:
        out['master_track_front_back_bias'] = -1.0
    elif '差し有利' in text or '差し決まる' in text:
        out['master_track_front_back_bias'] = 1.0

    # severity score
    for sel in ['.bias_severity', '.track_severity', '#severity']:
        el = soup.select_one(sel)
        if el:
            m = re.search(r'(\d+\.?\d*)', el.get_text())
            if m:
                out['master_track_today_severity'] = float(m.group(1))
                break

    return out


# =========================================================================
# Top-level fetcher
# =========================================================================

@dataclass
class MasterFeatureBundle:
    race_id: str
    umaban: int
    fetched_at: str
    features: Dict[str, Any] = field(default_factory=dict)
    fetch_status: Dict[str, str] = field(default_factory=dict)  # category → 'ok' / 'fail' / 'default'


def fetch_master_features(
    race_id: str,
    umaban: int,
    kaisai_id: Optional[str] = None,
    session: Optional[requests.Session] = None,
) -> MasterFeatureBundle:
    """4 系統 25 features を 1 馬 分 取得する.

    各 category の fetch fail / kill switch ON 時は default 値を埋め、
    fetch_status に 'fail' / 'default' を記録。
    """
    bundle = MasterFeatureBundle(
        race_id=race_id,
        umaban=umaban,
        fetched_at=datetime.now().isoformat(timespec='seconds'),
        features=dict(PHASE13_DEFAULTS),
        fetch_status={'tenkai': 'default', 'haran': 'default', 'lap': 'default', 'bias': 'default'},
    )

    if is_disabled():
        return bundle

    if session is None:
        session = _make_session()
    if session is None:
        # cookie 未設定 → 全 default
        return bundle

    if kaisai_id is None:
        kaisai_id = race_id[:10] if len(race_id) >= 12 else race_id

    # B. AI 展開予測
    html = _fetch(URL_AI_TENKAI.format(race_id=race_id), session)
    if html:
        try:
            bundle.features.update(_parse_ai_tenkai(html, umaban))
            bundle.fetch_status['tenkai'] = 'ok'
        except Exception as e:
            print(f"[netkeiba_master] tenkai parse error: {e}", file=sys.stderr)
            bundle.fetch_status['tenkai'] = 'fail'

    # C1. AI 波乱度
    html = _fetch(URL_AI_HARAN.format(race_id=race_id), session)
    if html:
        try:
            bundle.features.update(_parse_ai_haran(html))
            bundle.fetch_status['haran'] = 'ok'
        except Exception as e:
            print(f"[netkeiba_master] haran parse error: {e}", file=sys.stderr)
            bundle.fetch_status['haran'] = 'fail'

    # C2. 個別ラップ
    html = _fetch(URL_LAP.format(race_id=race_id), session)
    if html:
        try:
            bundle.features.update(_parse_lap(html, umaban))
            bundle.fetch_status['lap'] = 'ok'
        except Exception as e:
            print(f"[netkeiba_master] lap parse error: {e}", file=sys.stderr)
            bundle.fetch_status['lap'] = 'fail'

    # C3. トラックバイアス (kaisai 単位、 同 kaisai 内 R 共通)
    html = _fetch(URL_TRACK_BIAS.format(kaisai_id=kaisai_id), session)
    if html:
        try:
            bundle.features.update(_parse_track_bias(html))
            bundle.fetch_status['bias'] = 'ok'
        except Exception as e:
            print(f"[netkeiba_master] bias parse error: {e}", file=sys.stderr)
            bundle.fetch_status['bias'] = 'fail'

    return bundle


def fetch_race_master_features(race_id: str, umaban_list: List[int]) -> Dict[int, Dict[str, Any]]:
    """1 R 全頭分 一括 fetch.

    トラックバイアス + 展開予測 race-level data は 1 回 fetch、
    馬別 data は umaban list 分 parser に渡す (HTML は共有可能なため、
    実装簡易化のため Phase 13 では 馬別に fetch_master_features を再実行)。
    """
    os.makedirs(MASTER_DIR, exist_ok=True)
    session = _make_session()
    out: Dict[int, Dict[str, Any]] = {}
    for umaban in umaban_list:
        bundle = fetch_master_features(race_id, umaban, session=session)
        out[umaban] = bundle.features
    # cache 保存
    cache_path = os.path.join(MASTER_DIR, f'{race_id}_master.json')
    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump({str(k): v for k, v in out.items()}, f, ensure_ascii=False, indent=2)
    return out


def get_phase13_feature_names() -> List[str]:
    return list(ALL_PHASE13_FEATURES)


def get_phase13_defaults() -> Dict[str, Any]:
    return dict(PHASE13_DEFAULTS)


# =========================================================================
# CLI
# =========================================================================

def _cli():
    p = argparse.ArgumentParser(description='netkeiba マスター scraper (Phase 13)')
    p.add_argument('--race', help='race_id (12 桁)')
    p.add_argument('--umaban', type=int, default=1)
    p.add_argument('--enable', action='store_true', help='kill switch を解除')
    p.add_argument('--disable', action='store_true', help='kill switch を有効化')
    p.add_argument('--status', action='store_true', help='kill switch 状態 確認')
    p.add_argument('--list', action='store_true', help='25 features list 表示')
    args = p.parse_args()

    os.makedirs(MASTER_DIR, exist_ok=True)

    if args.disable:
        with open(KILL_SWITCH_FILE, 'w', encoding='utf-8') as f:
            f.write(datetime.now().isoformat())
        print(f"[KILL_SWITCH] enabled: {KILL_SWITCH_FILE}")
        return

    if args.enable:
        if os.path.exists(KILL_SWITCH_FILE):
            os.remove(KILL_SWITCH_FILE)
        print("[KILL_SWITCH] disabled (fetch 可能 状態)")
        return

    if args.status:
        print(f"disabled: {is_disabled()}")
        print(f"cookie loaded: {_load_cookie() is not None}")
        return

    if args.list:
        print(f"Phase 13 全 {len(ALL_PHASE13_FEATURES)} features:")
        print(f"  B. AI 展開予測 ({len(AI_TENKAI_FEATURES)}): {AI_TENKAI_FEATURES}")
        print(f"  C1. AI 波乱度 ({len(AI_HARAN_FEATURES)}): {AI_HARAN_FEATURES}")
        print(f"  C2. 個別ラップ ({len(LAP_FEATURES)}): {LAP_FEATURES}")
        print(f"  C3. トラックバイアス ({len(TRACK_BIAS_FEATURES)}): {TRACK_BIAS_FEATURES}")
        return

    if args.race:
        bundle = fetch_master_features(args.race, args.umaban)
        print(json.dumps(asdict(bundle), ensure_ascii=False, indent=2))
        return

    p.print_help()


if __name__ == '__main__':
    _cli()
