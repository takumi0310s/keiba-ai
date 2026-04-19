"""事故ナレッジベース クエリツール.

症状から類似事故を検索し、過去の対応 (fix_commit / recovery) を返す。

Usage:
    python tools/knowledge_base_query.py "SCRAPER-GUARD で止まった"
    python tools/knowledge_base_query.py --list                  # 全件表示
    python tools/knowledge_base_query.py --id inc-20260419-001   # ID指定
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BASE = Path(r"C:/Users/takum/keiba-ai")
KB_PATH = BASE / "data" / "incident_knowledge_base.json"


def load_kb() -> dict:
    if not KB_PATH.exists():
        return {"incidents": []}
    return json.loads(KB_PATH.read_text(encoding="utf-8"))


def query_by_symptom(query: str, kb: dict | None = None) -> list[dict]:
    """部分一致で検索。ヒット件数順にソート (単純なキーワード出現数)."""
    if kb is None:
        kb = load_kb()
    incidents = kb.get("incidents", [])
    q = query.lower()
    qwords = [w for w in q.split() if w]

    scored = []
    for inc in incidents:
        text = " ".join([
            inc.get("symptom", ""),
            inc.get("root_cause", ""),
            inc.get("detection_rule", ""),
        ]).lower()
        score = 0
        # 完全一致 > 単語一致
        if q in text:
            score += 10
        for w in qwords:
            if w in text:
                score += 1
        if score > 0:
            scored.append((score, inc))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [inc for _, inc in scored]


def get_by_id(inc_id: str, kb: dict | None = None) -> dict | None:
    if kb is None:
        kb = load_kb()
    for inc in kb.get("incidents", []):
        if inc.get("incident_id") == inc_id:
            return inc
    return None


def format_incident(inc: dict) -> str:
    lines = [
        f"[{inc.get('incident_id')}] {inc.get('date')} {inc.get('time')} "
        f"({inc.get('severity')})",
        f"  症状: {inc.get('symptom')}",
        f"  原因: {inc.get('root_cause')}",
        f"  修正: commit {inc.get('fix_commit')}",
        f"  回帰テスト: {inc.get('regression_test')}",
        f"  検知: {inc.get('detection_rule')}",
        f"  リカバリ: {inc.get('recovery')}",
    ]
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("query", nargs="?", default=None)
    p.add_argument("--list", action="store_true", help="全件表示")
    p.add_argument("--id", type=str, default=None, help="特定IDで検索")
    p.add_argument("--limit", type=int, default=5, help="表示件数上限")
    args = p.parse_args()

    kb = load_kb()

    if args.id:
        inc = get_by_id(args.id, kb)
        if inc is None:
            print(f"[not found] {args.id}")
            return 1
        print(format_incident(inc))
        return 0

    if args.list:
        for inc in kb.get("incidents", []):
            print(format_incident(inc))
            print()
        return 0

    if not args.query:
        p.print_help()
        return 1

    hits = query_by_symptom(args.query, kb)
    if not hits:
        print(f"(検索結果なし: '{args.query}')")
        return 1
    print(f"=== {len(hits)} 件ヒット (top {min(args.limit, len(hits))}) ===\n")
    for inc in hits[:args.limit]:
        print(format_incident(inc))
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
