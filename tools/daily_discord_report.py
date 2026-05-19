"""
tools/daily_discord_report.py
Daily 20:00 cumulative stats report to Discord.

Usage:
    python tools/daily_discord_report.py               # use today's date
    python tools/daily_discord_report.py 20260519      # specify date
"""
import os
import sys
import pandas as pd
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CUMUL_CSV = os.path.join(BASE_DIR, "data", "cumulative_results.csv")
WITHDRAWAL_LIMIT = 50_000  # 撤退ライン: 累計 -50,000 円


def load_cumulative_stats(csv_path: str = None, date_str: str = None) -> dict:
    """
    Load data/cumulative_results.csv and compute cumulative + today's stats.

    Returns dict with keys:
        total_n, total_pnl, total_roi,
        today_n, today_pnl, today_roi,
        strat_c_n, strat_c_pnl, strat_c_roi,  (0 / None if col absent)
        has_strat_c,
        withdrawal_margin
    """
    empty = {
        "total_n": 0, "total_pnl": 0, "total_roi": 0.0,
        "today_n": 0, "today_pnl": 0, "today_roi": 0.0,
        "strat_c_n": 0, "strat_c_pnl": 0, "strat_c_roi": 0.0,
        "has_strat_c": False,
        "withdrawal_margin": WITHDRAWAL_LIMIT,
    }

    path = csv_path or CUMUL_CSV
    if not os.path.exists(path):
        return empty

    try:
        df = pd.read_csv(path)
    except Exception:
        return empty

    # Use only settled rows
    settled = df[df["status"] == "settled"].copy()
    if len(settled) == 0:
        return empty

    # Numeric columns
    profit_s = pd.to_numeric(settled["profit"], errors="coerce").fillna(0)
    inv_s = pd.to_numeric(settled["investment"], errors="coerce").fillna(0)

    # Cumulative stats
    total_pnl = int(profit_s.sum())
    total_inv = int(inv_s.sum())
    total_n = len(settled)
    total_roi = (total_inv + total_pnl) / total_inv * 100 if total_inv > 0 else 0.0

    # Today's stats (date column stored as float yyyymmdd)
    if date_str is None:
        date_str = datetime.now().strftime("%Y%m%d")
    date_num = float(date_str)
    date_col = pd.to_numeric(settled["date"], errors="coerce")
    today_rows = settled[date_col == date_num]
    today_profit_s = pd.to_numeric(today_rows["profit"], errors="coerce").fillna(0)
    today_inv_s = pd.to_numeric(today_rows["investment"], errors="coerce").fillna(0)
    today_pnl = int(today_profit_s.sum())
    today_inv = int(today_inv_s.sum())
    today_n = len(today_rows)
    today_roi = (today_inv + today_pnl) / today_inv * 100 if today_inv > 0 else 0.0

    # Strat-C stats (戦略⑦案C: column 'strat_c' が True/1/'True' の行)
    has_strat_c = "strat_c" in settled.columns
    strat_c_n = 0
    strat_c_pnl = 0
    strat_c_roi = 0.0
    if has_strat_c:
        sc_mask = settled["strat_c"].astype(str).str.lower().isin(["true", "1", "yes"])
        sc_rows = settled[sc_mask]
        if len(sc_rows) > 0:
            sc_profit = pd.to_numeric(sc_rows["profit"], errors="coerce").fillna(0)
            sc_inv = pd.to_numeric(sc_rows["investment"], errors="coerce").fillna(0)
            strat_c_n = len(sc_rows)
            strat_c_pnl = int(sc_profit.sum())
            sc_inv_total = int(sc_inv.sum())
            strat_c_roi = (sc_inv_total + strat_c_pnl) / sc_inv_total * 100 if sc_inv_total > 0 else 0.0

    withdrawal_margin = WITHDRAWAL_LIMIT + total_pnl

    return {
        "total_n": total_n,
        "total_pnl": total_pnl,
        "total_roi": total_roi,
        "today_n": today_n,
        "today_pnl": today_pnl,
        "today_roi": today_roi,
        "strat_c_n": strat_c_n,
        "strat_c_pnl": strat_c_pnl,
        "strat_c_roi": strat_c_roi,
        "has_strat_c": has_strat_c,
        "withdrawal_margin": withdrawal_margin,
    }


def build_report_message(stats: dict, date_str: str = None) -> str:
    """Build Discord embed description from stats dict."""
    if date_str is None:
        date_str = datetime.now().strftime("%Y%m%d")

    # Format date as YYYY/MM/DD
    try:
        d = datetime.strptime(date_str, "%Y%m%d")
        date_label = d.strftime("%Y/%m/%d")
    except ValueError:
        date_label = date_str

    today_sign = "+" if stats["today_pnl"] >= 0 else ""
    total_sign = "+" if stats["total_pnl"] >= 0 else ""
    margin = stats["withdrawal_margin"]
    margin_sign = "+" if margin >= 0 else ""

    lines = [
        f"**{date_label} 日次収支レポート**",
        "",
        f"本日: **{stats['today_n']}R** / **{today_sign}{stats['today_pnl']:,}円** / ROI **{stats['today_roi']:.1f}%**",
        f"累計: **{stats['total_n']}R** / **{total_sign}{stats['total_pnl']:,}円** / ROI **{stats['total_roi']:.1f}%**",
    ]

    if stats.get("has_strat_c") and stats["strat_c_n"] > 0:
        sc_sign = "+" if stats["strat_c_pnl"] >= 0 else ""
        lines.append(
            f"(戦略⑦案C: {stats['strat_c_n']}R / {sc_sign}{stats['strat_c_pnl']:,}円 / ROI {stats['strat_c_roi']:.1f}%)"
        )

    lines.append(f"撤退余裕: **{margin_sign}{margin:,}円** (上限 -50,000円)")

    return "\n".join(lines)


def send_daily_report(webhook_url: str = None, date_str: str = None) -> bool:
    """
    Load stats, build message, send to Discord #updates channel.
    Returns True on success, False otherwise (missing webhook, send error, etc.).
    """
    if date_str is None:
        date_str = datetime.now().strftime("%Y%m%d")

    stats = load_cumulative_stats(date_str=date_str)
    message = build_report_message(stats, date_str=date_str)

    # Determine webhook URL
    url = webhook_url
    if not url:
        # Try importing notify module to use the same env-loading logic
        try:
            sys.path.insert(0, BASE_DIR)
            from tools.notify import _get_webhook_url
            url = _get_webhook_url(channel="updates")
        except Exception:
            pass

    if not url:
        # Manual env / .env fallback
        env_path = os.path.join(BASE_DIR, ".env")
        for key in ("DISCORD_WEBHOOK_UPDATES", "DISCORD_WEBHOOK_URL"):
            val = os.environ.get(key, "")
            if val.startswith("https://"):
                url = val
                break
        if not url and os.path.exists(env_path):
            try:
                with open(env_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if "=" in line and not line.startswith("#"):
                            k, v = line.split("=", 1)
                            if k.strip() in ("DISCORD_WEBHOOK_UPDATES", "DISCORD_WEBHOOK_URL"):
                                v = v.strip().strip('"').strip("'")
                                if v.startswith("https://"):
                                    url = v
                                    break
            except Exception:
                pass

    if not url:
        print("[daily_discord_report] DISCORD webhook URL not set -- skipping send.")
        return False

    try:
        import requests

        try:
            d = datetime.strptime(date_str, "%Y%m%d")
            title = f"日次収支レポート {d.strftime('%Y/%m/%d')}"
        except ValueError:
            title = f"日次収支レポート {date_str}"

        embed = {
            "title": title,
            "description": message,
            "color": 0x60b0ff,  # blue
        }
        payload = {"embeds": [embed]}
        resp = requests.post(url, json=payload, timeout=10)
        if resp.status_code in (200, 204):
            print(f"[daily_discord_report] Sent OK ({resp.status_code})")
            return True
        else:
            print(f"[daily_discord_report] Send failed: HTTP {resp.status_code}")
            return False
    except Exception as e:
        print(f"[daily_discord_report] Send error: {e}")
        return False


if __name__ == "__main__":
    date_arg = sys.argv[1] if len(sys.argv) > 1 else None
    success = send_daily_report(date_str=date_arg)
    sys.exit(0 if success else 1)
