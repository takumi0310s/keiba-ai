"""Session #60 A: .env NETKEIBA_COOKIE -> data/cookies.json (yt-dlp 用).

dev/training-poc 用の helper。 既存 cookie 文字列を JSON list 形式に変換。
"""
import os
import json
from pathlib import Path

BASE = Path(r"C:/Users/takum/keiba-ai")
ENV = BASE / ".env"
OUT = BASE / "data" / "cookies.json"


def main() -> int:
    if not ENV.exists():
        print(f"[ERR] .env not found: {ENV}")
        return 1
    cookie_str = None
    for line in ENV.read_text(encoding="utf-8").splitlines():
        if line.startswith("NETKEIBA_COOKIE="):
            cookie_str = line.split("=", 1)[1].strip().strip('"').strip("'")
            break
    if not cookie_str:
        print("[ERR] NETKEIBA_COOKIE not in .env")
        return 1

    cookies = []
    for part in cookie_str.split(";"):
        part = part.strip()
        if "=" not in part:
            continue
        name, _, value = part.partition("=")
        cookies.append({
            "name": name.strip(),
            "value": value.strip(),
            "domain": ".netkeiba.com",
            "path": "/",
            "secure": False,
            "expiry": 9999999999,
        })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(cookies, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] wrote {len(cookies)} cookies -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
