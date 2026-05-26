"""全馬スコア HTML 生成ツール。

data/allscores/YYYYMMDD/*.json から HTML 表形式のレポートを生成。
結果が出たら actual_finish を JSON に追記し HTML を再生成する。

usage:
    python tools/build_allscores_html.py [--date YYYYMMDD]
    from tools.build_allscores_html import build_allscores_html, update_allscores_result
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
ALLSCORES_DIR = BASE_DIR / "data" / "allscores"

_CSS = """
body{font-family:'Segoe UI',sans-serif;font-size:13px;background:#111;color:#ddd;margin:10px 14px}
h1{color:#eee;font-size:17px;margin:4px 0 8px}
.meta{font-size:11px;color:#666;margin-bottom:14px}
.race{margin-bottom:20px}
.rhdr{background:#1e1e1e;padding:6px 10px;font-size:13px;font-weight:bold;
      border-left:3px solid #555;color:#ccc}
table{border-collapse:collapse;width:100%;font-size:12px;margin-top:1px}
th{background:#252525;color:#888;padding:4px 9px;text-align:right;
   white-space:nowrap;cursor:pointer;user-select:none}
th:hover{color:#ddd}
th:nth-child(1),th:nth-child(2){text-align:left}
td{padding:3px 9px;border-bottom:1px solid #1a1a1a;text-align:right;white-space:nowrap}
td:nth-child(1),td:nth-child(2){text-align:left}
tr.t1{background:#153515}
tr.t1 td{color:#a0f0a0}
tr.t3{background:#0e2a0e}
tr.t3 td{color:#80d080}
tr.dv{background:#2c2900}
tr.dv td{color:#e0d870}
tr.t1 td.rnk-diff,tr.t3 td.rnk-diff{color:inherit}
th.asc::after{content:" ▲"}
th.desc::after{content:" ▼"}
.badge-go{color:#90ee90;font-size:10px;margin-left:4px}
.badge-no{color:#ee9090;font-size:10px;margin-left:4px}
"""

_JS = """
function sortTable(th, tid) {
  const tbl = document.getElementById(tid);
  const ths = tbl.querySelectorAll('th');
  const ci = Array.from(ths).indexOf(th);
  const asc = th.classList.contains('asc');
  ths.forEach(h => h.classList.remove('asc','desc'));
  th.classList.add(asc ? 'desc' : 'asc');
  const tbody = tbl.querySelector('tbody');
  const rows = Array.from(tbody.querySelectorAll('tr'));
  rows.sort((a,b) => {
    const va = a.cells[ci].dataset.v !== undefined ? +a.cells[ci].dataset.v : a.cells[ci].textContent.trim();
    const vb = b.cells[ci].dataset.v !== undefined ? +b.cells[ci].dataset.v : b.cells[ci].textContent.trim();
    if (!isNaN(va) && !isNaN(vb)) return asc ? vb - va : va - vb;
    return asc ? String(vb).localeCompare(String(va)) : String(va).localeCompare(String(vb));
  });
  rows.forEach(r => tbody.appendChild(r));
}
"""


def _rnk_diff_html(v15r: int, v16r: int) -> str:
    diff = v15r - v16r  # positive = V16 thinks horse is better than V15
    if diff == 0:
        return '<span style="color:#555">—</span>'
    if diff > 0:
        return f'<span style="color:#90ee90">▲{diff}</span>'
    return f'<span style="color:#ee9090">▼{abs(diff)}</span>'


def build_allscores_html(date_str: str | None = None) -> str | None:
    """指定日の全馬スコア HTML を生成。生成したファイルパスを返す。"""
    if date_str is None:
        date_str = datetime.now().strftime("%Y%m%d")

    race_dir = ALLSCORES_DIR / date_str
    if not race_dir.exists():
        print(f"[build_allscores_html] No data dir: {race_dir}")
        return None

    race_files = sorted(race_dir.glob("*.json"))
    if not race_files:
        print(f"[build_allscores_html] No race JSON files in {race_dir}")
        return None

    races_html: list[str] = []
    races_loaded = 0

    for rf in race_files:
        try:
            with open(rf, encoding="utf-8") as f:
                rec = json.load(f)
        except Exception as e:
            print(f"  [WARN] {rf.name}: {e}")
            continue

        race_name = rec.get("race_name", rf.stem)
        course = rec.get("course", "")
        race_num = rec.get("race_num", "")
        distance = rec.get("distance", "")
        surface = rec.get("surface", "")
        condition = rec.get("condition", "")
        cond_key = rec.get("cond_key", "")
        num_horses = rec.get("num_horses", 0)
        actual_finish: dict[str, int] = rec.get("actual_finish", {})

        v15_scores: dict[str, float] = {str(k): float(v) for k, v in rec.get("v15_scores", {}).items()}
        v16_scores: dict[str, float] = {str(k): float(v) for k, v in rec.get("v16_scores", {}).items()}
        horses: list[dict] = rec.get("horses", [])
        name_map = {
            str(h.get("馬番", h.get("horse_num", ""))): h.get("馬名", h.get("name", ""))
            for h in horses
        }

        # rank maps (1-based, smaller = better)
        v15_sorted = sorted(v15_scores.items(), key=lambda x: x[1], reverse=True)
        v16_sorted = sorted(v16_scores.items(), key=lambda x: x[1], reverse=True)
        v15_rank = {uma: i + 1 for i, (uma, _) in enumerate(v15_sorted)}
        v16_rank = {uma: i + 1 for i, (uma, _) in enumerate(v16_sorted)}

        has_results = bool(actual_finish)
        result_badge = ""
        if has_results:
            winner = next((u for u, f in actual_finish.items() if f == 1), None)
            if winner:
                wr = v15_rank.get(winner)
                result_badge = f'<span class="badge-go">✓ {wr}位的中</span>' if wr and wr <= 3 else '<span class="badge-no">×</span>'

        tid = f"t{rf.stem}"
        hdr = (f'【{course} {race_num}R】{race_name} '
               f'<span style="font-weight:normal;color:#999;font-size:11px">'
               f'{surface}{distance}m {condition} 条件{cond_key} {num_horses}頭'
               f'</span>{result_badge}')

        rows_html: list[str] = []
        for uma, v15sc in v15_sorted:
            name = name_map.get(uma, "")
            v16sc = v16_scores.get(uma)
            v15r = v15_rank.get(uma, 0)
            v16r = v16_rank.get(uma, 0) if v16sc is not None else 0
            actual = actual_finish.get(uma) or actual_finish.get(int(uma) if uma.isdigit() else uma)

            # row class
            cls = ""
            if actual == 1:
                cls = "t1"
            elif actual in (2, 3):
                cls = "t3"
            elif v16sc is not None and abs(v15r - v16r) >= 3:
                cls = "dv"

            actual_str = str(actual) if actual is not None else "—"
            actual_dv = str(actual) if actual is not None else "999"  # sort nulls last
            diff_html = _rnk_diff_html(v15r, v16r) if v16sc is not None else "—"
            v16_str = f"{v16sc:.4f}" if v16sc is not None else "—"
            v16r_str = str(v16r) if v16r else "—"

            rows_html.append(
                f'<tr class="{cls}">'
                f'<td data-v="{uma}">{uma}</td>'
                f'<td>{name}</td>'
                f'<td data-v="{v15sc:.6f}">{v15sc:.4f}</td>'
                f'<td data-v="{v15r}">{v15r}</td>'
                f'<td data-v="{v16sc if v16sc is not None else 0:.6f}">{v16_str}</td>'
                f'<td data-v="{v16r if v16r else 999}">{v16r_str}</td>'
                f'<td class="rnk-diff">{diff_html}</td>'
                f'<td data-v="{actual_dv}">{actual_str}</td>'
                f'</tr>'
            )

        def _th(label, idx):
            return f'<th onclick="sortTable(this,\'{tid}\')">{label}</th>'

        thead = (
            "<tr>"
            + _th("馬番", 0)
            + _th("馬名", 1)
            + _th("V15スコア", 2)
            + _th("V15順位", 3)
            + _th("V16スコア", 4)
            + _th("V16順位", 5)
            + _th("V16-V15", 6)
            + _th("実着順", 7)
            + "</tr>"
        )

        race_html = (
            f'<div class="race">'
            f'<div class="rhdr">{hdr}</div>'
            f'<table id="{tid}"><thead>{thead}</thead>'
            f'<tbody>{"".join(rows_html)}</tbody></table>'
            f'</div>'
        )
        races_html.append(race_html)
        races_loaded += 1

    if not races_html:
        print(f"[build_allscores_html] No valid races for {date_str}")
        return None

    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    legend = (
        '<div class="meta">'
        '<span style="background:#153515;padding:1px 6px;border-radius:3px">■ 1着</span> '
        '<span style="background:#0e2a0e;padding:1px 6px;border-radius:3px">■ 2-3着</span> '
        '<span style="background:#2c2900;padding:1px 6px;border-radius:3px">■ V16-V15 乖離≥3</span>'
        '&nbsp;&nbsp;V16=参考(投票しない)&nbsp;&nbsp;列ヘッダでソート'
        f'&nbsp;&nbsp;<span style="color:#444">生成: {generated_at}</span>'
        '</div>'
    )

    html = (
        f'<!DOCTYPE html><html lang="ja"><head>'
        f'<meta charset="utf-8">'
        f'<title>全馬スコア {date_str}</title>'
        f'<style>{_CSS}</style>'
        f'</head><body>'
        f'<h1>全馬スコア {date_str} ({races_loaded}R)</h1>'
        f'{legend}'
        f'{"".join(races_html)}'
        f'<script>{_JS}</script>'
        f'</body></html>'
    )

    out_path = ALLSCORES_DIR / f"allscores_{date_str}.html"
    ALLSCORES_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"[build_allscores_html] {races_loaded}R → {out_path}")
    return str(out_path)


def update_allscores_result(
    date_str: str,
    race_id: str,
    finish_order: dict,
) -> bool:
    """レース結果を allscores JSON に追記して HTML を再生成。

    finish_order: {umaban(int or str): finish_position(int)}
    戻り値: 対象 JSON が存在して更新成功したら True
    """
    json_path = ALLSCORES_DIR / date_str / f"{race_id}.json"
    if not json_path.exists():
        return False

    try:
        with open(json_path, encoding="utf-8") as f:
            rec = json.load(f)
    except Exception as e:
        print(f"  [allscores_html] JSON read error {race_id}: {e}")
        return False

    # normalise keys to str
    rec["actual_finish"] = {str(k): int(v) for k, v in finish_order.items()}

    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"  [allscores_html] JSON write error {race_id}: {e}")
        return False

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="全馬スコア HTML 生成")
    parser.add_argument("--date", default=None, help="YYYYMMDD (default: today)")
    args = parser.parse_args()
    path = build_allscores_html(args.date)
    if path:
        print(f"Generated: {path}")
    else:
        print("No data.")
        sys.exit(1)
