"""Discord通知モジュール（チャンネル振り分け対応）

Usage:
    from tools.notify import send_discord, build_rich_bet_message
    send_discord("予測完了", "...", color="green", channel="bets")
    send_discord("スクレイピング完了", "...", color="blue", channel="updates")

    # リッチ買い目通知
    title, msg, color = build_rich_bet_message(df, race_name, race_info, cond_key,
                                                cond_profile, bets, odds_dict, horses)
    send_discord(title, msg, color=color, channel="bets")

Channels:
    "bets"    → DISCORD_WEBHOOK_BETS (買い目通知)
    "updates" → DISCORD_WEBHOOK_UPDATES (システム通知)
    未指定     → DISCORD_WEBHOOK_URL (フォールバック)

Dedup (Session #59):
    send_discord() は default で 5 分以内の同一 (channel + title + message) を
    silent skip する。 dedup_window_sec=0 で無効化可能。
    cache は data/discord_send_cache.json (JSON、 best-effort、 lock なし)。
"""
import os
import re
import json
import hashlib
import time
from datetime import datetime
import requests

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

COLORS = {"green": 0x4ade80, "yellow": 0xf0c040, "red": 0xff4060, "blue": 0x60b0ff}

_ENV_CACHE = None

# === Session #59: dedup cache (5min default) ===
_DEDUP_CACHE_PATH = os.path.join(BASE_DIR, 'data', 'discord_send_cache.json')
_DEDUP_DEFAULT_WINDOW_SEC = 300


def _hash_message(channel: str, title: str, message: str) -> str:
    """Stable sha256 hash for dedup key. message は先頭 500 文字のみ使用 (bet 通知など
    末尾のオッズ更新で hash がブレるのを防ぐため)."""
    s = f"{channel}\x1f{title}\x1f{(message or '')[:500]}"
    return hashlib.sha256(s.encode('utf-8')).hexdigest()


def _load_dedup_cache() -> dict:
    if not os.path.exists(_DEDUP_CACHE_PATH):
        return {}
    try:
        with open(_DEDUP_CACHE_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}


def _save_dedup_cache(cache: dict) -> None:
    try:
        os.makedirs(os.path.dirname(_DEDUP_CACHE_PATH), exist_ok=True)
        # prune: keep only entries within last 1h to avoid unbounded growth
        cutoff = time.time() - 3600
        pruned = {k: v for k, v in cache.items() if isinstance(v, (int, float)) and v >= cutoff}
        with open(_DEDUP_CACHE_PATH, 'w', encoding='utf-8') as f:
            json.dump(pruned, f)
    except Exception:
        pass  # best-effort, failure must not block notify


def _load_env():
    global _ENV_CACHE
    if _ENV_CACHE is not None:
        return _ENV_CACHE
    _ENV_CACHE = {}
    env_path = os.path.join(BASE_DIR, '.env')
    if not os.path.exists(env_path):
        return _ENV_CACHE
    with open(env_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if '=' in line and not line.startswith('#'):
                key, val = line.split('=', 1)
                _ENV_CACHE[key.strip()] = val.strip('"').strip("'")
    return _ENV_CACHE


def _get_webhook_url(channel="updates"):
    env = _load_env()
    if channel == "bets":
        url = env.get('DISCORD_WEBHOOK_BETS', '')
        if url.startswith('https://'):
            return url
    if channel == "updates":
        url = env.get('DISCORD_WEBHOOK_UPDATES', '')
        if url.startswith('https://'):
            return url
    if channel == "v21_paper":
        url = env.get('DISCORD_WEBHOOK_V21_PAPER', '')
        if url.startswith('https://'):
            return url
        # fallback to updates
        url = env.get('DISCORD_WEBHOOK_UPDATES', '')
        if url.startswith('https://'):
            return url
    # Fallback
    url = env.get('DISCORD_WEBHOOK_URL', '')
    return url if url.startswith('https://') else None


def _log_failure(title: str, message: str, channel: str, reason: str) -> None:
    """Discord 失敗を logs/discord_failures.log に記録 (Session #31 A2 強化)"""
    try:
        log_dir = os.path.join(BASE_DIR, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, 'discord_failures.log')
        ts = datetime.now().isoformat()
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(f"[{ts}] channel={channel} reason={reason}\n")
            f.write(f"  title: {title[:200]}\n")
            f.write(f"  msg: {message[:300]}\n")
            f.write("---\n")
    except Exception:
        pass  # ログ失敗は silent fail (recursion 防止)


def send_discord(title, message, color="green", fields=None, channel="updates",
                 max_retries=3, retry_backoff=(1, 2, 4),
                 dedup_window_sec=_DEDUP_DEFAULT_WINDOW_SEC):
    """Discord Webhook通知を送信 (retry + log、Session #31 A2 強化、 #59 dedup)。
    URLが未設定ならスキップ。

    Args:
        channel: "bets" (買い目) or "updates" (システム通知)
        max_retries: 最大 retry 回数 (default 3)
        retry_backoff: 各 retry 間の sleep 秒 (default 1/2/4)
        dedup_window_sec: 同一 (channel + title + message[:500]) を秒数以内 silent skip。
                          default 300 (5分)、 0 で無効化。
    """
    import time as _time

    # === Session #59: dedup check ===
    h = None
    if dedup_window_sec and dedup_window_sec > 0:
        h = _hash_message(channel, title, message)
        cache = _load_dedup_cache()
        last_ts = cache.get(h)
        now = _time.time()
        if last_ts and (now - last_ts) < dedup_window_sec:
            # silent skip — 重複送信防止
            return True

    url = _get_webhook_url(channel)
    if not url:
        _log_failure(title, message, channel, "url_not_set")
        return False

    embed = {
        "title": title[:256],
        "description": message[:2000],
        "color": COLORS.get(color, COLORS["blue"]),
    }
    if fields:
        embed["fields"] = [{"name": str(k)[:256], "value": str(v)[:200], "inline": True}
                           for k, v in list(fields.items())[:10]]

    last_reason = "unknown"
    for attempt in range(max_retries):
        try:
            resp = requests.post(url, json={"embeds": [embed]}, timeout=10)
            if resp.status_code in (200, 204):
                # === Session #59: 成功時のみ dedup cache 記録 ===
                if h is not None:
                    cache = _load_dedup_cache()
                    cache[h] = _time.time()
                    _save_dedup_cache(cache)
                return True
            last_reason = f"http_{resp.status_code}"
            # 429 (rate limit) は backoff、4xx の他は retry 無意味なので break
            if 400 <= resp.status_code < 500 and resp.status_code != 429:
                break
        except Exception as e:
            last_reason = f"exc:{type(e).__name__}"
        # backoff (最後の試行は不要)
        if attempt < max_retries - 1:
            sleep_sec = retry_backoff[attempt] if attempt < len(retry_backoff) else 4
            _time.sleep(sleep_sec)

    _log_failure(title, message, channel, last_reason)
    return False


_V16_CALIB_CACHE = None


def _load_v16_calib():
    global _V16_CALIB_CACHE
    if _V16_CALIB_CACHE is not None:
        return _V16_CALIB_CACHE
    path = os.path.join(BASE_DIR, 'data', 'v16_calibration.json')
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        _V16_CALIB_CACHE = data.get('buckets', [])
    except Exception:
        _V16_CALIB_CACHE = []
    return _V16_CALIB_CACHE


def _score_to_top3r(score: float, buckets: list) -> float:
    """V16スコア → キャリブレーション済み馬券内率 (オッズ不使用)。"""
    for b in buckets:
        if b['lo'] <= score < b['hi']:
            return b['top3r']
    return buckets[-1]['top3r'] if buckets and score >= 1.0 else 0.0


def _myomi_label(calib_prob: float, odds: float) -> str:
    """妙味判定: 能力馬券内率 vs 単勝オッズの乖離。表示補助のみ — 投票判断には使わない。"""
    if calib_prob >= 0.40 and odds >= 7.0:
        return '★穴'
    if calib_prob >= 0.40 and odds < 7.0:
        return '〇順'
    if calib_prob < 0.25 and odds <= 4.0:
        return '△危'
    return ''


def _build_all_scores_table(df, v16_scores=None, odds_dict=None):
    """全馬スコア表 (Discord 用 compact テキスト)。

    v16_scores: {馬番(int): float} または None
    odds_dict: {馬番(int): float} 単勝オッズ または None
    能力馬券内率 = V16スコアをキャリブレーション変換 (オッズは混入しない)。
    """
    has_v16 = bool(v16_scores)
    has_odds = bool(odds_dict)
    has_top3r = 'horse_career_top3r' in df.columns

    calib_buckets = _load_v16_calib() if has_v16 else []
    has_calib = has_v16 and bool(calib_buckets)

    rows = []
    for i, (_, row) in enumerate(df.iterrows()):
        uma = int(row.get('馬番', 0))
        name = str(row.get('馬名', '?'))
        if len(name) > 7:
            name = name[:6] + '…'
        v15s = float(row.get('スコア', 0))
        pop = int(row.get('pop_rank', 0) or 0)
        top3r = float(row.get('horse_career_top3r', 0) or 0) if has_top3r else 0

        marker = '▶' if i < 3 else '  '

        if has_calib and has_odds:
            v16s = v16_scores.get(uma, 0)
            calib_prob = _score_to_top3r(v16s, calib_buckets)
            odds = float(odds_dict.get(uma, 0) or 0)
            myomi = _myomi_label(calib_prob, odds) if odds > 0 else ''
            odds_str = f"O{odds:.1f}" if odds > 0 else 'O--'
            myomi_str = f" {myomi}" if myomi else ''
            line = (f"{marker}{i+1:2}番{uma:2} {name} {v15s:.2f}"
                    f" 能:{calib_prob*100:.0f}% {odds_str}{myomi_str}")
        elif has_v16:
            v16s = v16_scores.get(uma, 0)
            pop_str = f"{pop}人" if pop > 0 else ' -'
            line = f"{marker}{i+1:2}番{uma:2} {name} V15:{v15s:.2f} V16:{v16s:.2f} {pop_str}"
        else:
            pop_str = f"{pop}人" if pop > 0 else ' -'
            top3r_str = f"{top3r * 100:.0f}%" if top3r > 0 else '-'
            line = f"{marker}{i+1:2}番{uma:2} {name} {v15s:.2f} {pop_str} 複:{top3r_str}"
        rows.append(line)

    sep_idx = 3
    parts = []
    if has_calib and has_odds:
        parts.append("📊 **全馬スコア** (V15 | 能力馬券内率 | オッズ | 妙味)")
    elif has_v16:
        parts.append("📊 **全馬スコア** (V15 | V16 | 人気)")
    else:
        parts.append("📊 **全馬スコア** (V15 | 人気 | 過去複勝率)")
    parts.extend(rows[:sep_idx])
    parts.append("─" * 26)
    parts.extend(rows[sep_idx:])
    if has_calib and has_odds:
        parts.append("_★能=V16キャリブ馬券内率(能力のみ)・オッズ参考・投票はV15_")
    elif has_v16:
        parts.append("_★V16=能力ベースpaper参考のみ・投票はV15_")
    return "\n".join(parts)


def build_rich_bet_message(df, race_name, race_info, cond_key, cond_profile,
                           bets, odds_dict=None, horses=None, date_str=None,
                           upset_data=None, newspaper_data=None,
                           pp_stars=0, pp_matched=None,
                           par_df=None, v16_scores=None):
    """リッチな買い目通知メッセージを構築。全通知元で共通フォーマット。

    Args:
        df: スコアでソート済みの予測DataFrame (columns: 馬番, 馬名, スコア, etc.)
        race_name: レース名
        race_info: dict with keys: course, race_num, distance, surface, condition,
                   start_time(optional), weather(optional), grade(optional)
        cond_key: 条件キー (A-E, X)
        cond_profile: CONDITION_PROFILESの値
        v16_scores: {馬番(int): float} V16 ability-only スコア (optional、paper参考)
        bets: 買い目リスト
        odds_dict: {馬番: 単勝オッズ} (optional)
        horses: 出走馬リスト (premium data用, optional)
        date_str: 日付文字列 YYYYMMDD (optional, default=today)

    Returns:
        (title, message, color) tuple
    """
    if date_str is None:
        date_str = datetime.now().strftime('%Y%m%d')

    # 日付フォーマット: 3/28(土)
    weekday_map = {0: '月', 1: '火', 2: '水', 3: '木', 4: '金', 5: '土', 6: '日'}
    try:
        dt = datetime.strptime(date_str, '%Y%m%d')
        date_disp = f"{dt.month}/{dt.day}({weekday_map[dt.weekday()]})"
    except Exception:
        date_disp = date_str

    course = race_info.get('course', '')
    race_num = race_info.get('race_num', '')
    start_time = race_info.get('start_time', '')
    distance = race_info.get('distance', 0)
    surface = race_info.get('surface', '')
    condition = race_info.get('condition', '')
    num_horses = len(df)
    bet_type = cond_profile.get('bet_type', 'trio')
    roi = cond_profile.get('roi', 0)
    hit_rate = cond_profile.get('hit_rate', 0)
    investment = cond_profile.get('investment', 700)

    stars = '★★★' if roi >= 200 else ('★★' if roi >= 100 else '★')

    # Title: 🏇 3/28(土) 中山1R 10:00発走
    time_part = f" {start_time}発走" if start_time else ""
    title = f"🏇 {date_disp} {course}{race_num}{time_part}"

    # Line 1: レース名 surface+distance condition 頭数
    lines = [f"**{race_name}** {surface}{distance}m {condition} {num_horses}頭"]
    # Line 2: 条件 + stars + ROI
    lines.append(f"条件{cond_key} {stars} ROI {roi:.1f}% (的中{hit_rate:.1f}%)")
    # 収益パターンマッチ
    if pp_stars >= 3:
        lines.append("⭐⭐⭐ 収益パターン: 強く推奨")
    elif pp_stars == 2:
        lines.append("⭐⭐ 収益パターン: 推奨")
    elif pp_stars == 1:
        lines.append("⭐ 収益パターン: 参考")
    if pp_matched:
        _best = pp_matched[0]
        _br = _best.get('roi', 0)
        _bh = _best.get('hit_rate', 0) * 100
        _bn = _best.get('n', 0)
        lines.append(f"  Best: ROI {_br:.0f}% 的中{_bh:.0f}% N={_bn}")
    lines.append("")

    # ━ V15 本番・投票対象 ━ (V16 参考情報と区別、誤投票防止)
    lines.append("**━【V15 本番・投票対象】━**")
    lines.append("")

    # 買い目
    if bet_type == 'umaren':
        n1 = int(df.iloc[0]['馬番'])
        n2 = int(df.iloc[1]['馬番'])
        n3 = int(df.iloc[2]['馬番'])
        lines.append(f"馬連 1軸2流し")
        lines.append(f"軸: {n1} → {n2}, {n3}")
    else:
        top6 = df.head(6)
        n1 = int(top6.iloc[0]['馬番'])
        col2 = sorted([int(top6.iloc[1]['馬番']), int(top6.iloc[2]['馬番'])])
        col3 = sorted([int(top6.iloc[i]['馬番']) for i in range(1, min(6, len(top6)))])
        lines.append(f"三連複フォーメーション {len(bets)}点")
        lines.append(f"1列目: {n1}")
        lines.append(f"2列目: {', '.join(str(n) for n in col2)}")
        lines.append(f"3列目: {', '.join(str(n) for n in col3)}")
    lines.append("")

    # TOP3
    for i in range(min(3, len(df))):
        row = df.iloc[i]
        if par_df is not None:
            try:
                from tools.popularity_rank import format_par_horse_line
                par_rows = par_df[par_df['馬番'].astype(int) == int(row['馬番'])]
                par_row = par_rows.iloc[0] if len(par_rows) > 0 else row
                lines.append(format_par_horse_line(par_row, i + 1))
                continue
            except Exception:
                pass
        num = int(row['馬番'])
        name = row.get('馬名', '?')
        score = row.get('スコア', 0)
        rank_label = ['軸', '2位', '3位'][i]
        lines.append(f"{rank_label}: {num} {name} (スコア{score:.2f})")
    lines.append("")

    # 配当レンジ (V15 section 内)
    try:
        ro = odds_dict or {}
        if ro and bet_type != 'umaren' and len(bets) > 0:
            payouts_est = []
            for b in bets:
                o = [ro.get(int(x), 10.0) for x in b]
                est = o[0] * o[1] * o[2] * 0.6
                payouts_est.append(max(100, int(est * 100)))
            if payouts_est:
                lines.append(f"💰 配当レンジ: {min(payouts_est):,}円〜{max(payouts_est):,}円")
        elif ro and bet_type == 'umaren':
            n1v = int(df.iloc[0]['馬番'])
            n2v = int(df.iloc[1]['馬番'])
            n3v = int(df.iloc[2]['馬番'])
            o1 = ro.get(n1v, 10.0)
            o2 = ro.get(n2v, 10.0)
            o3 = ro.get(n3v, 10.0)
            est1 = max(100, int(o1 * o2 * 5))
            est2 = max(100, int(o1 * o3 * 5))
            lines.append(f"💰 配当目安: {est1:,}円 / {est2:,}円")
    except Exception:
        pass
    lines.append(f"投資額: {investment}円")

    # ━ V16 能力ベース・参考のみ・投票しない ━
    if v16_scores:
        lines.append("")
        lines.append("─" * 22)
        lines.append("**【V16 能力ベース・参考のみ・投票しない】**")

    # 全馬スコアテーブル
    try:
        lines.append(_build_all_scores_table(df, v16_scores=v16_scores, odds_dict=odds_dict))
        lines.append("")
    except Exception:
        pass

    # Premium data
    premium_parts = []
    if horses and len(horses) > 0:
        h0 = horses[0] if isinstance(horses[0], dict) else {}
        si = h0.get('タイム指数', 0) or h0.get('speed_index', 0)
        if si and si > 1000:
            premium_parts.append(f"指数: {si}")
        rank = h0.get('調教ランク', '') or h0.get('training_rank', '')
        if rank:
            premium_parts.append(f"調教: {rank}")
        cs = h0.get('厩舎スコア', 0) or h0.get('stable_score', 0)
        if cs and cs > 0:
            premium_parts.append("厩舎: 好調")
    # Also check df columns for premium data
    if not premium_parts and len(df) > 0:
        top1 = df.iloc[0]
        si = top1.get('タイム指数', 0)
        if si and si > 1000:
            premium_parts.append(f"指数: {si}")
    if premium_parts:
        lines.append(f"\n📊 Premium ✓  {' / '.join(premium_parts)}")

    # JRDB指数（TOP3馬のIDM・パドック指数・オッズ指数）
    jrdb_lines = []
    if horses and len(horses) > 0:
        for i in range(min(3, len(df))):
            row = df.iloc[i]
            uma = int(row.get('馬番', 0))
            # horses dictからJRDB指数を取得
            h_jrdb = None
            for h in horses:
                if isinstance(h, dict) and int(h.get('馬番', 0)) == uma:
                    h_jrdb = h
                    break
            if h_jrdb is None:
                continue
            idm = h_jrdb.get('JRDB_IDM', 0)
            paddock = h_jrdb.get('JRDB_パドック指数', 0)
            odds_idx = h_jrdb.get('JRDB_オッズ指数', 0)
            # デフォルト値(50.0)以外なら表示
            parts_j = []
            if idm and float(idm) != 50.0:
                parts_j.append(f"IDM:{idm:.0f}")
            if paddock and float(paddock) != 50.0:
                parts_j.append(f"パド:{paddock:.0f}")
            if odds_idx and float(odds_idx) != 50.0:
                parts_j.append(f"ｵｯﾂﾞ:{odds_idx:.0f}")
            if parts_j:
                name = row.get('馬名', '?')
                jrdb_lines.append(f"  {uma} {name}: {' / '.join(parts_j)}")
    if jrdb_lines:
        lines.append("\n🎯 JRDB指数")
        lines.extend(jrdb_lines)

    # 新馬評価（新馬戦の場合）
    race_name_str = str(race_name) if race_name else ''
    if '新馬' in race_name_str and horses:
        shinba_lines = []
        for h in (horses[:3] if len(horses) >= 3 else horses):
            if not isinstance(h, dict):
                continue
            se = h.get('新馬厩舎評価', '') or h.get('shinba_stable_eval', '')
            tr = h.get('新馬調教ランク', '') or h.get('shinba_training_rank', '')
            cs = h.get('新馬スコア', None)
            if cs is None:
                cs = h.get('shinba_comment_score', None)
            if se or tr:
                name = h.get('馬名', '?')
                parts_s = []
                if se:
                    parts_s.append(f"厩舎{se}")
                if tr:
                    parts_s.append(f"調教{tr}")
                if cs is not None:
                    sign = '+' if cs > 0 else ''
                    parts_s.append(f"スコア{sign}{cs}")
                shinba_lines.append(f"  {name}: {'/'.join(parts_s)}")
        if shinba_lines:
            lines.append("\n🐴 新馬評価")
            lines.extend(shinba_lines)

    # 波乱度・AI予測タイム
    if upset_data:
        _u_lv = upset_data.get('upset_level')
        _u_rel = upset_data.get('top_popularity_reliability')
        if _u_lv is not None:
            _u_labels = {1: '堅い', 2: 'やや堅い', 3: '標準', 4: '波乱含み', 5: '大荒れ'}
            _u_bar = '🟩' * _u_lv + '⬜' * (5 - _u_lv) if _u_lv <= 3 else '🟥' * _u_lv + '⬜' * (5 - _u_lv)
            _u_line = f"\n🎲 波乱度 Lv{_u_lv} {_u_labels.get(_u_lv, '')} {_u_bar}"
            if _u_rel is not None:
                _u_line += f" (信頼度{_u_rel:.0f}%)"
            lines.append(_u_line)
            if _u_lv >= 4:
                lines.append("⚠️ 荒れやすいレース — 見送り推奨")

    if newspaper_data:
        _ai_opinion = newspaper_data.get('ai_opinion', '')
        if _ai_opinion:
            lines.append(f"\n🤖 AI見解: {_ai_opinion[:100]}{'...' if len(_ai_opinion) > 100 else ''}")

    msg = "\n".join(lines)
    color = "green" if roi >= 200 else ("blue" if roi >= 100 else "yellow")
    return title, msg, color
