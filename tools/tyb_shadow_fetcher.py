"""
tyb_shadow_fetcher.py — JRDB TYB 直前情報 shadow fetch (観測専用)

★★★ SHADOW MODE ONLY ★★★
- TYB は V15 145 features に含まれない → V15 inference に一切使用しない
- 投票ロジック・買い目生成に干渉しない
- default disabled (TYB_SHADOW_ENABLED = False)
- 5/23 初運用日は必ず disabled のまま起動される

How to enable for 6/1+ observation:
    fetch_tyb_shadow(..., enabled=True)
    or set TYB_SHADOW_ENABLED = True (file-level)
"""
from __future__ import annotations

import csv
import json
import logging
import os
import subprocess
import sys
import tempfile
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# ★ DEFAULT DISABLED — 5/23 fire 安全保証 ★
# ---------------------------------------------------------------------------
TYB_SHADOW_ENABLED: bool = False

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO = Path(__file__).resolve().parents[1]
TYB_SHADOW_DIR = REPO / "data" / "tyb_shadow"
TYB_SHADOW_LOG = REPO / "data" / "tyb_shadow_log.csv"

# ---------------------------------------------------------------------------
# Logger (module-level, never propagates exceptions to caller)
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# JRDB auth (from .env via dotenv)
# ---------------------------------------------------------------------------
def _load_env() -> tuple[str, str]:
    """JRDB_USER / JRDB_PASSWORD を env から取得。dotenv がなければ environ 直読み。"""
    try:
        from dotenv import load_dotenv
        load_dotenv(REPO / ".env", override=False)
    except ImportError:
        pass
    user = os.environ.get("JRDB_USER", "")
    password = os.environ.get("JRDB_PASSWORD", "")
    return user, password


# ---------------------------------------------------------------------------
# TYB fixed-width parser (uses TYB_COLUMNS from parse_jrdb.py logic)
# ---------------------------------------------------------------------------
# TYB record length = 128 bytes (cp932), 1-based positions
TYB_COLUMNS = [
    ("basho_code",   1,  2),
    ("year",         3,  2),
    ("kai",          5,  1),
    ("nichi",        6,  1),
    ("race_num",     7,  2),
    ("umaban",       9,  2),
    ("idm",         11,  5),
    ("jockey_idx",  16,  5),
    ("info_idx",    21,  5),
    ("odds_idx",    26,  5),
    ("padock_idx",  31,  5),
    ("reserve1",    36,  5),
    ("sogo_idx",    41,  5),
    ("bagu_change", 46,  1),
    ("ashimoto",    47,  1),
    ("cancel_flag", 48,  1),
    ("jockey_code", 49,  5),
    ("jockey_name", 54, 12),
    ("weight_carry",66,  3),
    ("minarai",     69,  1),
    ("baba_code",   70,  2),
    ("weather_code",72,  1),
    ("tansho_odds", 73,  6),
    ("fukusho_odds",79,  6),
    ("odds_time",   85,  4),
    ("horse_weight",89,  3),
    ("weight_diff", 92,  3),
    ("odds_mark",   95,  1),
    ("padock_mark", 96,  1),
    ("sogo_mark",   97,  1),
    ("batai_code",  98,  1),
    ("kehai_code",  99,  1),
    ("start_time", 100,  4),
]


def _field(line_bytes: bytes, start: int, length: int) -> str:
    """1-based fixed-width field → decoded string."""
    s = start - 1
    return line_bytes[s:s + length].decode("cp932", errors="replace")


def _safe_float(s: str) -> Optional[float]:
    s = s.strip()
    if not s or s == "-":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _safe_int(s: str) -> Optional[int]:
    s = s.strip()
    if not s or s == "-":
        return None
    try:
        return int(s)
    except ValueError:
        return None


def _hex_day(c: str) -> int:
    try:
        return int(c, 16)
    except (ValueError, TypeError):
        return 0


def _parse_tyb_file(tyb_path: Path, target_race_num_str: str) -> list[dict]:
    """TYB 固定長ファイルをパースして target race_num の行だけ返す。"""
    rows: list[dict] = []
    with open(tyb_path, "rb") as f:
        for raw in f:
            line_bytes = raw.rstrip(b"\r\n")
            if len(line_bytes) < 95:
                continue
            race_num_field = _field(line_bytes, 7, 2).strip()
            if race_num_field != target_race_num_str:
                continue
            row: dict = {}
            for name, start, length in TYB_COLUMNS:
                row[name] = _field(line_bytes, start, length)
            # numeric conversions
            for col in ("odds_idx", "padock_idx", "sogo_idx", "idm",
                        "jockey_idx", "info_idx", "tansho_odds",
                        "fukusho_odds", "weight_carry"):
                row[col] = _safe_float(row[col])
            for col in ("umaban", "bagu_change", "ashimoto",
                        "cancel_flag", "horse_weight"):
                row[col] = _safe_int(row[col])
            # weight_diff (signed integer with leading +/-)
            wd = row.get("weight_diff", "")
            if isinstance(wd, str):
                wd = wd.strip()
                row["weight_diff"] = _safe_int(wd) if wd else None
            # strip strings
            for col in ("jockey_name", "jockey_code", "odds_mark",
                        "padock_mark", "sogo_mark", "batai_code",
                        "kehai_code", "odds_time", "baba_code",
                        "weather_code", "minarai", "start_time",
                        "basho_code", "year", "kai", "nichi", "race_num"):
                if col in row:
                    row[col] = str(row[col]).strip()
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Shadow log helpers
# ---------------------------------------------------------------------------
def _append_shadow_log(record: dict) -> None:
    """data/tyb_shadow_log.csv に 1 行追記。"""
    try:
        TYB_SHADOW_LOG.parent.mkdir(parents=True, exist_ok=True)
        file_exists = TYB_SHADOW_LOG.exists()
        fields = [
            "race_id", "fetch_time", "delta_to_start_min",
            "num_horses", "status", "note",
        ]
        with open(TYB_SHADOW_LOG, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            if not file_exists:
                writer.writeheader()
            writer.writerow(record)
    except Exception as e:
        logger.warning("tyb_shadow_log append failed: %s", e)


def _save_shadow_json(race_id: str, date_str: str, result: dict) -> None:
    """data/tyb_shadow/{date}/{race_id}_tyb.json に保存。"""
    try:
        dest_dir = TYB_SHADOW_DIR / date_str
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / f"{race_id}_tyb.json"
        with open(dest, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        logger.info("TYB shadow saved: %s", dest)
    except Exception as e:
        logger.warning("TYB shadow JSON save failed: %s", e)


# ---------------------------------------------------------------------------
# Main public API
# ---------------------------------------------------------------------------
def fetch_tyb_shadow(
    race_id: str,
    start_time_str: str,
    enabled: bool = TYB_SHADOW_ENABLED,
) -> Optional[dict]:
    """
    TYB 直前情報を shadow fetch する。

    Parameters
    ----------
    race_id : str
        12桁 race_id (例: "202605020211")
    start_time_str : str
        発走時刻 HHMM (例: "1530")
    enabled : bool
        False (default) → 即 None 返却、ネットワーク呼び出しなし
        True → fetch + parse + save

    Returns
    -------
    dict | None
        fetch 成功時は観測データ dict。
        disabled / 失敗時は None。
        ★ 返り値は V15 inference に一切渡してはならない ★
    """
    # ★ SAFETY GATE: disabled → immediately return None, no side effects ★
    if not enabled:
        return None

    fetch_time = datetime.now()

    try:
        # --- parse race_id ---
        # race_id format: YYYYBBKKNNRR  (YYYY=year, BB=basho, KK=kai, NN=nichi, RR=race_num)
        # e.g. "202605020211" → date ~20260502, basho=02, race_num=11
        if len(race_id) != 12 or not race_id.isdigit():
            raise ValueError(f"Invalid race_id: {race_id!r}")

        yyyy = race_id[:4]
        mm_dd = race_id[4:8]  # BBKK not calendar month, use fetch_time for date
        race_num_int = int(race_id[10:12])
        race_num_str = f"{race_num_int:02d}"

        # For URL we need the actual calendar date.
        # Convention: use today's date (fetch is called on race day)
        date_str = fetch_time.strftime("%Y%m%d")
        yymmdd = date_str[2:]  # 6 digits: yymmdd
        YYYYMMDD = date_str

        # --- download TYB .lzh ---
        jrdb_user, jrdb_password = _load_env()
        if not jrdb_user or not jrdb_password:
            raise RuntimeError("JRDB_USER / JRDB_PASSWORD not set")

        url = (
            f"http://www.jrdb.com/member/{YYYYMMDD}/tyokuzen/"
            f"TYB{yymmdd}.lzh"
        )
        logger.info("TYB fetch URL: %s", url)

        with tempfile.TemporaryDirectory(prefix="tyb_shadow_") as tmpdir:
            tmp_path = Path(tmpdir)
            lzh_path = tmp_path / f"TYB{yymmdd}.lzh"

            # download
            import base64
            cred = base64.b64encode(
                f"{jrdb_user}:{jrdb_password}".encode()
            ).decode()
            req = urllib.request.Request(
                url,
                headers={"Authorization": f"Basic {cred}"},
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                lzh_path.write_bytes(resp.read())

            logger.info("TYB downloaded: %s (%d bytes)", lzh_path, lzh_path.stat().st_size)

            # extract .lzh with 7z
            extract_dir = tmp_path / "extracted"
            extract_dir.mkdir()
            result_7z = subprocess.run(
                ["7z", "e", str(lzh_path), f"-o{extract_dir}", "-y"],
                capture_output=True,
                timeout=30,
            )
            if result_7z.returncode != 0:
                raise RuntimeError(
                    f"7z extraction failed: {result_7z.stderr.decode(errors='replace')}"
                )

            # find extracted TYB txt file
            tyb_files = list(extract_dir.glob("TYB*.txt")) + list(
                extract_dir.glob("TYB*.TXT")
            )
            if not tyb_files:
                # try case-insensitive
                tyb_files = [
                    p for p in extract_dir.iterdir()
                    if p.name.upper().startswith("TYB") and p.suffix.upper() in (".TXT", "")
                ]
            if not tyb_files:
                raise FileNotFoundError(
                    f"No TYB txt file found after extraction in {extract_dir}"
                )
            tyb_file = tyb_files[0]
            logger.info("TYB extracted: %s", tyb_file)

            # parse
            rows = _parse_tyb_file(tyb_file, race_num_str)

        if not rows:
            raise ValueError(
                f"No TYB rows found for race_num={race_num_str} in {tyb_file.name}"
            )

        # --- compute delta to start ---
        delta_min: Optional[float] = None
        try:
            if start_time_str and len(start_time_str) == 4:
                sh, sm = int(start_time_str[:2]), int(start_time_str[2:])
                fh, fm = fetch_time.hour, fetch_time.minute
                delta_min = (sh * 60 + sm) - (fh * 60 + fm)
        except Exception:
            pass

        # --- aggregate result ---
        odds_idx_list = [r.get("odds_idx") for r in rows]
        padock_idx_list = [r.get("padock_idx") for r in rows]
        tansho_odds_list = [r.get("tansho_odds") for r in rows]
        horse_weight_list = [r.get("horse_weight") for r in rows]
        padock_mark_list = [r.get("padock_mark") for r in rows]

        tyb_result = {
            "race_id": race_id,
            "odds_idx_list": odds_idx_list,
            "padock_idx_list": padock_idx_list,
            "tansho_odds_list": tansho_odds_list,
            "horse_weight_list": horse_weight_list,
            "padock_mark_list": padock_mark_list,
            "fetch_time": fetch_time.isoformat(),
            "delta_to_start_min": delta_min,
            "num_horses": len(rows),
            "raw_rows": rows,
        }

        # save
        _save_shadow_json(race_id, date_str, tyb_result)
        _append_shadow_log({
            "race_id": race_id,
            "fetch_time": fetch_time.isoformat(),
            "delta_to_start_min": delta_min,
            "num_horses": len(rows),
            "status": "OK",
            "note": "",
        })

        return tyb_result

    except Exception as e:
        logger.exception("TYB shadow fetch failed for %s: %s", race_id, e)
        try:
            _append_shadow_log({
                "race_id": race_id,
                "fetch_time": fetch_time.isoformat(),
                "delta_to_start_min": None,
                "num_horses": 0,
                "status": "ERROR",
                "note": str(e)[:200],
            })
        except Exception:
            pass
        # ★ NEVER propagate exception to caller ★
        return None


# ---------------------------------------------------------------------------
# Discord supplement formatter (optional, NOT auto-sent)
# ---------------------------------------------------------------------------
def format_tyb_discord_supplement(tyb_result: dict) -> str:
    """
    TYB 観測データを Discord 補足文字列にフォーマット。
    呼び出しは任意 — 自動送信はしない。

    Parameters
    ----------
    tyb_result : dict
        fetch_tyb_shadow() の返り値

    Returns
    -------
    str
        フォーマット済み補足文字列
    """
    if not tyb_result:
        return ""

    lines: list[str] = ["【参考】直前情報 (TYB shadow)"]

    # 馬体重変化サマリー
    wt_list = [w for w in tyb_result.get("horse_weight_list", []) if w is not None]
    if wt_list:
        # weight_diff は raw_rows から
        diffs = [
            r.get("weight_diff")
            for r in tyb_result.get("raw_rows", [])
            if r.get("weight_diff") is not None
        ]
        if diffs:
            big = [d for d in diffs if abs(d) >= 10]
            if big:
                lines.append(f"  馬体重大幅変化: {len(big)}頭 (±10kg以上)")
            avg_diff = sum(diffs) / len(diffs)
            lines.append(f"  馬体重変化平均: {avg_diff:+.1f}kg")

    # オッズ指数サマリー
    odds_idx = [v for v in tyb_result.get("odds_idx_list", []) if v is not None]
    if odds_idx:
        top3_avg = sum(sorted(odds_idx, reverse=True)[:3]) / min(3, len(odds_idx))
        lines.append(f"  オッズ指数 TOP3平均: {top3_avg:.1f}")

    # パドック指数サマリー
    padock_idx = [v for v in tyb_result.get("padock_idx_list", []) if v is not None]
    if padock_idx:
        top3_avg = sum(sorted(padock_idx, reverse=True)[:3]) / min(3, len(padock_idx))
        lines.append(f"  パドック指数 TOP3平均: {top3_avg:.1f}")

    # 取消馬
    cancels = [
        r.get("umaban")
        for r in tyb_result.get("raw_rows", [])
        if r.get("cancel_flag") == 1
    ]
    if cancels:
        lines.append(f"  取消: {cancels}")

    # delta
    delta = tyb_result.get("delta_to_start_min")
    if delta is not None:
        lines.append(f"  取得タイミング: 発走{delta:.0f}分前")

    lines.append(f"  ★ shadow only — V15 投票に非影響 ★")
    return "\n".join(lines)
