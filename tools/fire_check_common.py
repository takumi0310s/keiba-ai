"""Reverse-Watchdog 共通ロジック.

タスク完了ログの鮮度・サイズ・エラーキーワードで ok/warning/critical を判定する。

Usage:
    from tools.fire_check_common import FireCheckConfig, check_fire, notify_result, save_result

    cfg = FireCheckConfig(
        task_name="DailyJrdbKyi",
        log_candidates=[BASE / f"logs/jrdb_kyi_auto_{today}.log"],
        expected_time=datetime(2026, 4, 20, 6, 0),
        min_size=500,
        error_keywords=["Traceback", "ERROR"],
        recovery_command="python tools/daily_jrdb_kyi.py",
    )
    r = check_fire(cfg)
    save_result(cfg.task_name, r)
    notify_result(cfg.task_name, r)
"""
from __future__ import annotations

import datetime
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

BASE = Path(r"C:/Users/takum/keiba-ai")


@dataclass
class FireCheckConfig:
    task_name: str
    log_candidates: list[Path]
    expected_time: datetime.datetime
    min_size: int = 2000
    max_age_min: int = 30  # mtime が「now - max_age_min」より古ければ警告
    error_keywords: list[str] = field(default_factory=lambda: [
        "SCRAPER-GUARD", "Traceback", "Exception", "IP banned", "ERROR"
    ])
    recovery_command: str = ""
    # ログの代わりに CSV で完了判定するケース用
    csv_candidates: list[Path] = field(default_factory=list)
    min_csv_rows: int = 0


def _check_log(cfg: FireCheckConfig, now: datetime.datetime) -> dict:
    log_file: Path | None = None
    for c in cfg.log_candidates:
        if c.exists():
            log_file = c
            break

    if log_file is None:
        return {
            "status": "critical",
            "message": f"{cfg.task_name}: ログファイル未検出",
            "candidates": [str(p) for p in cfg.log_candidates],
            "recovery": cfg.recovery_command,
        }

    stat = log_file.stat()
    size = stat.st_size
    mtime = datetime.datetime.fromtimestamp(stat.st_mtime)

    if mtime < cfg.expected_time:
        return {
            "status": "critical",
            "message": f"{cfg.task_name}: ログ未更新 (mtime {mtime.isoformat()}, 期待 {cfg.expected_time.isoformat()}+)",
            "size": size,
            "recovery": cfg.recovery_command,
        }

    if size < cfg.min_size:
        try:
            tail = log_file.read_text(encoding="utf-8", errors="replace")[-500:]
        except Exception as e:
            tail = f"(read err: {e})"
        return {
            "status": "critical",
            "message": f"{cfg.task_name}: ログサイズ異常 {size}B (min {cfg.min_size}B)",
            "size": size,
            "mtime": mtime.isoformat(),
            "log_tail": tail,
            "recovery": cfg.recovery_command,
        }

    try:
        tail = log_file.read_text(encoding="utf-8", errors="replace")[-3000:]
    except Exception as e:
        tail = f"(read err: {e})"

    for kw in cfg.error_keywords:
        if kw in tail:
            return {
                "status": "warning",
                "message": f"{cfg.task_name}: ログに '{kw}' 検出",
                "keyword": kw,
                "size": size,
                "mtime": mtime.isoformat(),
                "log_tail": tail[-500:],
            }

    return {
        "status": "ok",
        "message": f"{cfg.task_name} 正常発火",
        "size": size,
        "mtime": mtime.isoformat(),
        "log": str(log_file),
    }


def _check_csv(cfg: FireCheckConfig, now: datetime.datetime) -> dict:
    """CSV で成功判定するケース (DailyPredict の daily_predictions/YYYYMMDD.csv 等)."""
    csv_file: Path | None = None
    for c in cfg.csv_candidates:
        if c.exists():
            csv_file = c
            break
    if csv_file is None:
        return {
            "status": "critical",
            "message": f"{cfg.task_name}: CSV未生成",
            "candidates": [str(p) for p in cfg.csv_candidates],
            "recovery": cfg.recovery_command,
        }
    try:
        rows = sum(1 for _ in csv_file.open("r", encoding="utf-8", errors="replace")) - 1
    except Exception as e:
        return {"status": "critical", "message": f"CSV read err: {e}", "recovery": cfg.recovery_command}
    if rows < cfg.min_csv_rows:
        return {
            "status": "warning",
            "message": f"{cfg.task_name}: CSV 行数不足 {rows} < {cfg.min_csv_rows}",
            "csv": str(csv_file),
            "rows": rows,
        }
    return {
        "status": "ok",
        "message": f"{cfg.task_name} CSV OK ({rows} rows)",
        "csv": str(csv_file),
        "rows": rows,
    }


def check_fire(cfg: FireCheckConfig, now: datetime.datetime | None = None) -> dict:
    """ログ / CSV 両方で判定。ログ優先、無い場合 CSV で補完。"""
    if now is None:
        now = datetime.datetime.now()

    r = _check_log(cfg, now)
    if r["status"] == "ok":
        return r
    # ログが critical かつ CSV 候補があれば CSV チェックも試す
    if cfg.csv_candidates:
        r_csv = _check_csv(cfg, now)
        if r_csv["status"] == "ok":
            return r_csv
        # 両方とも NG なら log 結果を返す
    return r


def save_result(task_name: str, result: dict) -> Path:
    """data/fire_check_results/{YYYYMMDD}.json に追記保存."""
    results_dir = BASE / "data" / "fire_check_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    ymd = datetime.date.today().strftime("%Y%m%d")
    path = results_dir / f"{ymd}.json"
    data = {}
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            data = {}
    data[task_name] = {"timestamp": datetime.datetime.now().isoformat(), **result}
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return path


def notify_result(task_name: str, result: dict) -> None:
    """Discord 投稿。tools/discord_notifier.py があれば経由、なければ notify_done.py 直結."""
    status = result.get("status", "critical")
    if status == "ok":
        title = f"{task_name} 正常発火"
        subtitle = "OK"
        body = f"size={result.get('size', 'n/a')} mtime={result.get('mtime', 'n/a')}"
        color = "green"
        dedup_key = f"fire_check_{task_name}_{datetime.date.today().strftime('%Y%m%d')}_ok"
    elif status == "warning":
        title = f"{task_name} 警告"
        subtitle = "要確認"
        body = result.get("message", "")
        tail = result.get("log_tail", "")
        if tail:
            body += "\n\nlog_tail:\n" + tail[:400]
        color = "yellow"
        dedup_key = f"fire_check_{task_name}_{datetime.date.today().strftime('%Y%m%d')}_warn"
    else:
        title = f"CRITICAL: {task_name} 失敗"
        subtitle = "要手動介入"
        body = result.get("message", "")
        if result.get("recovery"):
            body += "\n\nリカバリ:\n" + result["recovery"]
        color = "red"
        dedup_key = None  # critical は dedup 無視

    # discord_notifier.py 経由 (存在すれば)
    notifier = BASE / "tools" / "discord_notifier.py"
    if notifier.exists():
        try:
            subprocess.run(
                [sys.executable, str(notifier),
                 "--title", title, "--subtitle", subtitle, "--body", body,
                 "--severity", status, "--color", color,
                 *(["--dedup-key", dedup_key] if dedup_key else [])],
                check=False, timeout=30,
                env={**os.environ, "PYTHONIOENCODING": "utf-8"},
            )
            return
        except Exception:
            pass

    # フォールバック: notify_done.py 直結
    try:
        subprocess.run(
            [sys.executable, str(BASE / "tools/notify_done.py"),
             title, subtitle, body, "--color", color],
            check=False, timeout=30,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )
    except Exception as e:
        print(f"[WARN] Discord 通知失敗: {e}", file=sys.stderr)
