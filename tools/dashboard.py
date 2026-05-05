"""累計収支 + Phase 2.5 進捗 ダッシュボード.

Outputs:
    data/dashboard/data.json      生データ
    data/dashboard/dashboard.html plotly 静的 HTML

Usage:
    python tools/dashboard.py                # build + open in browser hint
    python tools/dashboard.py --no-discord   # Discord 通知 skip
    python tools/dashboard.py --silent       # print 抑制
"""
from __future__ import annotations

import argparse
import csv
import datetime
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Windows cp932 stdout で ¥ や絵文字が落ちないよう utf-8 に切替
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

BASE = Path(r"C:/Users/takum/keiba-ai")
DASH_DIR = BASE / "data" / "dashboard"
DASH_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# USER 実投資 累計収支 (HANDOFF v2 / docs/HANDOFF_5_5_TO_5_9.md 由来)
# 注: 4/12 までの累計は「USER 報告」、5/2/5/3 は USER 実投資レース分のみ。
# BATCH 仮想 (data/daily_results/*.csv) とは厳密に分離する。
# ---------------------------------------------------------------------------
USER_CUMULATIVE: list[tuple[str, int, str]] = [
    # (date, cumulative_jpy, label)
    ("2026-04-12", 23_480, "GW前 累計"),
    ("2026-05-02", 14_660, "5/2 USER 投資 -8,820"),
    ("2026-05-03", 14_140, "5/3 USER 投資 -520"),
    ("2026-05-09", 14_140, "5/9 投資前 (現在)"),
]
WITHDRAW_LINE_JPY = -50_000

# ---------------------------------------------------------------------------
# Phase 2.5 残タスク (HANDOFF section 4 由来、優先度別)
# status: 'done' | 'pending' | 'manual'
# ---------------------------------------------------------------------------
PHASE_25_TASKS: list[dict] = [
    # 高 H1-H3
    {"prio": "高", "id": "H1", "task": "tools/scrape_nar_today.py 実装", "status": "done"},
    {"prio": "高", "id": "H2", "task": "tools/scrape_nar_results.py 実装", "status": "done"},
    {"prio": "高", "id": "H3", "task": "5/8 21:00 後 12R race_name 確認", "status": "manual"},
    # 中 M1-M7
    {"prio": "中", "id": "M1", "task": "TYB publish 観測 完了判定", "status": "pending"},
    {"prio": "中", "id": "M2", "task": "feature distribution shift 調査 (BT vs prod)", "status": "pending"},
    {"prio": "中", "id": "M3", "task": "race-level normalize 本番統合 (predict_core.py)", "status": "pending"},
    {"prio": "中", "id": "M4", "task": "chihou_races_2020_2025.csv 生成", "status": "pending"},
    {"prio": "中", "id": "M5", "task": "条件別 NAR ROI 計算", "status": "pending"},
    {"prio": "中", "id": "M6", "task": "v18/v19 5/16 試行 (条件達成後)", "status": "pending"},
    {"prio": "中", "id": "M7", "task": "NAR 5/12-5/15 paper → 5/16 試行", "status": "pending"},
    # 低 L1-L4
    {"prio": "低", "id": "L1", "task": "v15.1 features 拡張 (ra_score/sc_score 復活)", "status": "pending"},
    {"prio": "低", "id": "L2", "task": "v20 統合モデル 設計 (JRA + NAR 共通 52+)", "status": "pending"},
    {"prio": "低", "id": "L3", "task": "古いモデル削除 (v9/v12/v134)", "status": "pending"},
    {"prio": "低", "id": "L4", "task": "predict_v20.py 統合 inference", "status": "pending"},
]

# ---------------------------------------------------------------------------
# 過去 Session 累計 完了 (Session #1-#18, sessions_5_3_5_5_recap.md 等)
# ---------------------------------------------------------------------------
SESSION_DONE_COUNT = 33  # Session #1-#18 で完了したタスク数 (推定)
PHASE_25_TOTAL_PLANNED = SESSION_DONE_COUNT + len(PHASE_25_TASKS)  # 33 + 14 = 47


# ---------------------------------------------------------------------------
def load_batch_daily_roi() -> list[dict]:
    """BATCH 仮想 ROI (全レース 700 円投資想定) を日別集計。
    USER 実投資ではない、参考値。"""
    rows: list[dict] = []
    pattern = re.compile(r"^(\d{8})\.csv$")
    for p in sorted((BASE / "data" / "daily_results").glob("2026*.csv")):
        m = pattern.match(p.name)
        if not m:
            continue
        ymd = m.group(1)
        date = f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:]}"
        n = inv = pay = 0
        with p.open(encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                try:
                    inv += int(row.get("investment") or 0)
                    pay += int(row.get("actual_payout") or 0)
                    n += 1
                except Exception:
                    pass
        if inv == 0:
            continue
        roi = pay / inv * 100
        rows.append({
            "date": date,
            "races": n,
            "investment": inv,
            "payout": pay,
            "profit": pay - inv,
            "roi_pct": round(roi, 1),
        })
    return rows


def list_keiba_schtasks() -> dict:
    """schtasks 一覧から Keiba 系を抽出。"""
    try:
        proc = subprocess.run(
            ["schtasks", "/query", "/fo", "csv", "/nh"],
            capture_output=True, text=True, timeout=30,
            encoding="cp932", errors="replace",
        )
        out = proc.stdout
    except Exception as e:
        return {"error": str(e), "tasks": []}
    tasks: list[dict] = []
    for line in out.splitlines():
        if not line.strip():
            continue
        parts = [p.strip().strip('"') for p in line.split(",")]
        if len(parts) < 3:
            continue
        name, nxt, status = parts[0], parts[1], parts[2]
        if not re.search(r"keiba|Daily(Jrdb|Predict|Premium|Results)|JrdbHealth|RaceAuto|WeeklyReport|ProcessWatchdog", name, re.IGNORECASE):
            continue
        is_nar = "Nar" in name
        tasks.append({
            "name": name.lstrip("\\"),
            "next": nxt,
            "status": status,
            "kind": "NAR" if is_nar else "JRA",
        })
    nar = [t for t in tasks if t["kind"] == "NAR"]
    jra = [t for t in tasks if t["kind"] == "JRA"]
    return {
        "total": len(tasks),
        "nar_count": len(nar),
        "jra_count": len(jra),
        "ready": sum(1 for t in tasks if t["status"].lower() == "ready"),
        "disabled": sum(1 for t in tasks if t["status"].lower() == "disabled"),
        "tasks": tasks,
    }


def build_data() -> dict:
    """全データ集計 → dict。"""
    cumulative_now = USER_CUMULATIVE[-1][1]
    margin = cumulative_now - WITHDRAW_LINE_JPY  # +64,140
    done = sum(1 for t in PHASE_25_TASKS if t["status"] == "done")
    pending = sum(1 for t in PHASE_25_TASKS if t["status"] == "pending")
    manual = sum(1 for t in PHASE_25_TASKS if t["status"] == "manual")
    return {
        "generated": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "user_cumulative": [
            {"date": d, "cumulative_jpy": v, "label": lbl}
            for d, v, lbl in USER_CUMULATIVE
        ],
        "withdraw": {
            "line_jpy": WITHDRAW_LINE_JPY,
            "current_jpy": cumulative_now,
            "margin_jpy": margin,
            "margin_ratio": margin / abs(WITHDRAW_LINE_JPY - 23_480),  # 起点比
        },
        "phase_25": {
            "total_planned": PHASE_25_TOTAL_PLANNED,
            "session_done": SESSION_DONE_COUNT,
            "remaining_total": len(PHASE_25_TASKS),
            "remaining_done": done,
            "remaining_pending": pending,
            "remaining_manual": manual,
            "completion_pct": round((SESSION_DONE_COUNT + done) / PHASE_25_TOTAL_PLANNED * 100, 1),
            "tasks": PHASE_25_TASKS,
        },
        "batch_daily": load_batch_daily_roi(),
        "schtasks": list_keiba_schtasks(),
    }


# ---------------------------------------------------------------------------
def fig_cumulative(data: dict) -> go.Figure:
    cum = data["user_cumulative"]
    dates = [c["date"] for c in cum]
    values = [c["cumulative_jpy"] for c in cum]
    labels = [c["label"] for c in cum]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=values, mode="lines+markers+text",
        text=[f"¥{v:+,}" for v in values], textposition="top center",
        line=dict(color="#1f77b4", width=3),
        marker=dict(size=12, color="#1f77b4"),
        hovertext=labels, name="USER 実投資 累計",
    ))
    # 撤退ライン
    fig.add_hline(y=WITHDRAW_LINE_JPY, line_dash="dash", line_color="red",
                  annotation_text=f"撤退ライン ¥{WITHDRAW_LINE_JPY:+,}",
                  annotation_position="bottom right")
    fig.add_hline(y=0, line_dash="dot", line_color="gray")
    fig.update_layout(
        title="累計収支推移 (USER 実投資ベース)",
        xaxis_title="日付", yaxis_title="累計 (円)",
        yaxis_tickformat=",d", height=400,
    )
    return fig


def fig_phase25(data: dict) -> go.Figure:
    p25 = data["phase_25"]
    done = p25["session_done"] + p25["remaining_done"]
    pending = p25["remaining_pending"]
    manual = p25["remaining_manual"]
    fig = go.Figure(data=[go.Pie(
        labels=["完了 (Session#1-18 + H1-H2)", "残 pending", "手動 (人手)"],
        values=[done, pending, manual],
        hole=0.5,
        marker_colors=["#2ca02c", "#ff7f0e", "#7f7f7f"],
        textinfo="label+value+percent",
    )])
    fig.update_layout(
        title=f"Phase 2.5 進捗 ({done}/{p25['total_planned']} 完了, {p25['completion_pct']}%)",
        height=400,
        annotations=[dict(text=f"{p25['completion_pct']}%", x=0.5, y=0.5, font_size=24, showarrow=False)],
    )
    return fig


def fig_withdraw_gauge(data: dict) -> go.Figure:
    w = data["withdraw"]
    cur = w["current_jpy"]
    line = w["line_jpy"]
    # gauge: 撤退ライン = 0%, 起点 (+23,480) = 100%
    base = 23_480
    rng_low, rng_high = line, base
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=cur,
        number={"prefix": "¥", "valueformat": ",d"},
        delta={"reference": 0, "valueformat": ",d", "suffix": "円"},
        title={"text": f"現在累計 (撤退まで余裕 ¥{w['margin_jpy']:+,})"},
        gauge={
            "axis": {"range": [rng_low, rng_high], "tickformat": ",d"},
            "bar": {"color": "#1f77b4"},
            "steps": [
                {"range": [rng_low, -25_000], "color": "#ffcccc"},
                {"range": [-25_000, 0],       "color": "#ffe9b3"},
                {"range": [0, rng_high],      "color": "#cce6cc"},
            ],
            "threshold": {
                "line": {"color": "red", "width": 3},
                "thickness": 0.9,
                "value": line,
            },
        },
    ))
    fig.update_layout(height=400)
    return fig


def fig_batch_roi(data: dict) -> go.Figure:
    bd = data["batch_daily"]
    if not bd:
        fig = go.Figure()
        fig.update_layout(title="BATCH 仮想 ROI (データなし)", height=400)
        return fig
    dates = [b["date"] for b in bd]
    rois = [b["roi_pct"] for b in bd]
    profits = [b["profit"] for b in bd]
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(
        x=dates, y=profits, name="日次損益 (円)",
        marker_color=["#2ca02c" if p >= 0 else "#d62728" for p in profits],
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=dates, y=rois, name="ROI (%)", mode="lines+markers",
        line=dict(color="#1f77b4", width=2),
    ), secondary_y=True)
    fig.add_hline(y=100, line_dash="dot", line_color="gray", secondary_y=True)
    fig.update_yaxes(title_text="日次損益 (円)", tickformat=",d", secondary_y=False)
    fig.update_yaxes(title_text="ROI (%)", secondary_y=True)
    fig.update_layout(
        title="BATCH 仮想 全レース投資 (参考、USER 実投資ではない)",
        xaxis_title="日付", height=400,
    )
    return fig


def render_html(data: dict) -> str:
    """4 figures + schtasks 表 を 1 HTML にまとめる。"""
    figs = [fig_cumulative(data), fig_phase25(data),
            fig_withdraw_gauge(data), fig_batch_roi(data)]
    fig_html = [f.to_html(full_html=False, include_plotlyjs=("cdn" if i == 0 else False))
                for i, f in enumerate(figs)]
    # schtasks 表
    st = data["schtasks"]
    rows_html = "".join(
        f"<tr><td>{t['name']}</td><td>{t['next']}</td>"
        f"<td>{t['status']}</td><td>{t['kind']}</td></tr>"
        for t in st.get("tasks", [])
    )
    # Phase 2.5 残タスク表
    p25_rows = "".join(
        f"<tr><td>{t['prio']}</td><td>{t['id']}</td><td>{t['task']}</td>"
        f"<td class='s-{t['status']}'>{t['status']}</td></tr>"
        for t in PHASE_25_TASKS
    )
    style = """
    body { font-family: -apple-system, "Segoe UI", sans-serif; max-width: 1200px;
           margin: 20px auto; padding: 0 20px; color: #333; }
    h1 { border-bottom: 2px solid #1f77b4; padding-bottom: 8px; }
    h2 { color: #1f77b4; margin-top: 32px; }
    .meta { color: #666; font-size: 14px; }
    table { border-collapse: collapse; width: 100%; margin-top: 12px; font-size: 13px; }
    th, td { padding: 6px 10px; border: 1px solid #ddd; text-align: left; }
    th { background: #f4f4f4; }
    .s-done { color: #2ca02c; font-weight: bold; }
    .s-pending { color: #ff7f0e; }
    .s-manual { color: #7f7f7f; }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
    @media (max-width: 900px) { .grid { grid-template-columns: 1fr; } }
    """
    nar_summary = (
        f"NAR タスク {st.get('nar_count', 0)} 件 / JRA タスク {st.get('jra_count', 0)} 件 / "
        f"全 {st.get('total', 0)} 件 (Ready {st.get('ready', 0)}, Disabled {st.get('disabled', 0)})"
    )
    w = data["withdraw"]
    p25 = data["phase_25"]
    return f"""<!DOCTYPE html>
<html lang="ja"><head><meta charset="utf-8">
<title>Keiba-AI Dashboard</title>
<style>{style}</style></head><body>
<h1>累計収支 + Phase 2.5 進捗 ダッシュボード</h1>
<p class="meta">生成: {data['generated']} / 撤退ライン ¥{w['line_jpy']:+,} / 余裕 ¥{w['margin_jpy']:+,} /
Phase 2.5 完了率 {p25['completion_pct']}%</p>

<div class="grid">
  <div>{fig_html[0]}</div>
  <div>{fig_html[1]}</div>
  <div>{fig_html[2]}</div>
  <div>{fig_html[3]}</div>
</div>

<h2>Phase 2.5 残タスク ({len(PHASE_25_TASKS)} 件)</h2>
<table><thead><tr><th>優先度</th><th>ID</th><th>タスク</th><th>状態</th></tr></thead>
<tbody>{p25_rows}</tbody></table>

<h2>NAR pipeline + 全タスクスケジューラ</h2>
<p class="meta">{nar_summary}</p>
<table><thead><tr><th>Task 名</th><th>次回起動</th><th>状態</th><th>分類</th></tr></thead>
<tbody>{rows_html}</tbody></table>

<h2>BATCH 仮想 日別 (参考)</h2>
<p class="meta">USER 実投資とは別、全レース 700 円投資した場合の仮想シミュレーション。</p>
<table><thead><tr><th>日付</th><th>R 数</th><th>投資</th><th>払戻</th><th>損益</th><th>ROI%</th></tr></thead>
<tbody>{''.join(
    f"<tr><td>{b['date']}</td><td>{b['races']}</td><td>¥{b['investment']:,}</td>"
    f"<td>¥{b['payout']:,}</td><td>¥{b['profit']:+,}</td><td>{b['roi_pct']}</td></tr>"
    for b in data['batch_daily']
)}</tbody></table>
</body></html>
"""


def send_discord(msg: str, title: str = "Keiba Dashboard 更新") -> None:
    notify = BASE / "tools" / "notify_done.py"
    if not notify.exists():
        return
    try:
        subprocess.run(
            [sys.executable, str(notify), title, msg, "--color", "blue"],
            check=False, timeout=30,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )
    except Exception as e:
        print(f"[WARN] Discord 送信失敗: {e}", file=sys.stderr)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--no-discord", action="store_true")
    p.add_argument("--silent", action="store_true")
    args = p.parse_args()

    data = build_data()
    json_path = DASH_DIR / "data.json"
    json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    html_path = DASH_DIR / "dashboard.html"
    html_path.write_text(render_html(data), encoding="utf-8")

    w = data["withdraw"]
    p25 = data["phase_25"]
    summary = (
        f"累計 ¥{w['current_jpy']:+,} / 撤退まで ¥{w['margin_jpy']:+,} / "
        f"Phase 2.5 {p25['completion_pct']}% ({p25['session_done'] + p25['remaining_done']}/{p25['total_planned']})"
    )
    if not args.silent:
        print(f"[OK] {html_path}")
        print(f"[OK] {json_path}")
        print(summary)
    if not args.no_discord:
        send_discord(f"{summary}\nHTML: {html_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
