"""V15 Outcome Dashboard - Before/After visualization for each task.

run:
  streamlit run dashboard/outcome_dashboard.py --server.port 8502

production V15 app.py (port 8501) and independent. production impact = 0.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import streamlit as st

REPO = Path(__file__).resolve().parents[1]
OUTCOMES_DIR = REPO / "data" / "task_outcomes"
BASELINE_PATH = OUTCOMES_DIR / "baseline_v15.json"

st.set_page_config(page_title="V15 Outcome Dashboard", layout="wide")
st.title("V15 Outcome Dashboard")
st.caption("each task implementation Before vs After. V15 production unchanged.")


@st.cache_data(ttl=30)
def load_baseline() -> dict:
    if not BASELINE_PATH.exists():
        return {}
    with open(BASELINE_PATH, encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(ttl=30)
def load_outcomes() -> list[dict]:
    outcomes = []
    if not OUTCOMES_DIR.exists():
        return outcomes
    for p in sorted(OUTCOMES_DIR.glob("*.json")):
        if p.name == "baseline_v15.json":
            continue
        try:
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
                data["_file"] = p.name
                outcomes.append(data)
        except Exception as e:
            st.warning(f"failed to load {p.name}: {e}")
    return outcomes


baseline = load_baseline()
outcomes = load_outcomes()

# baseline summary in sidebar
with st.sidebar:
    st.header("V15 baseline")
    if baseline:
        m = baseline.get("metrics", {})
        st.write(f"calc date: {baseline.get('calculation_date')}")
        st.write(f"production: {baseline.get('production_state')}")
        wf_auc = m.get("wf_auc", {}).get("value")
        if wf_auc is not None:
            st.metric("WF AUC", f"{wf_auc:.4f}")
        roi_cands = m.get("actual_roi", {}).get("candidate_values", {})
        if "raw_cumulative_5_16_evening" in roi_cands:
            v = roi_cands["raw_cumulative_5_16_evening"]["value"]
            st.metric("raw ROI (5/16 ev)", f"{v:.2f}%")
        hr = m.get("hit_rate_top3", {}).get("value")
        if hr is not None:
            st.metric("top3 hit rate", f"{hr*100:.2f}%")
        st.write(f"n settled: {m.get('n_settled')}")
        st.divider()
        st.caption("honest notes:")
        for note in baseline.get("honest_notes", []):
            st.caption(f"- {note}")
    else:
        st.error("baseline_v15.json not found")

tab1, tab2, tab3, tab4 = st.tabs([
    "timeline",
    "task cards",
    "cumulative graph",
    "limit gauge",
])

with tab1:
    st.subheader("task implementation timeline")
    if outcomes:
        rows = []
        for o in outcomes:
            rows.append({
                "task_id": o.get("task_id"),
                "task_name": o.get("task_name"),
                "phase": o.get("phase"),
                "implemented_at": o.get("implemented_at"),
                "before_roi": o.get("before", {}).get("roi"),
                "after_roi": o.get("after", {}).get("roi"),
                "delta_roi": o.get("delta", {}).get("roi"),
                "p_value": (o.get("statistical_test") or {}).get("p_value"),
                "significant": (o.get("statistical_test") or {}).get("significant_at_0_05"),
                "status": o.get("status"),
            })
        df = pd.DataFrame(rows).sort_values("implemented_at", ascending=False)
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("no task record yet. baseline_v15.json only.")

with tab2:
    st.subheader("task cards")
    if outcomes:
        for o in outcomes:
            with st.expander(f"[{o.get('task_id')}] {o.get('task_name')} - {o.get('status')}", expanded=False):
                cols = st.columns(4)
                before = o.get("before", {})
                after = o.get("after", {})
                delta = o.get("delta", {})
                with cols[0]:
                    st.metric(
                        "ROI",
                        f"{after.get('roi')}%" if after.get("roi") is not None else "N/A",
                        delta=f"{delta.get('roi')} pt" if delta.get("roi") is not None else None,
                    )
                with cols[1]:
                    st.metric(
                        "hit3",
                        f"{(after.get('hit_rate_top3') or 0)*100:.2f}%",
                        delta=f"{(delta.get('hit_rate_top3') or 0)*100:.2f} pt" if delta.get("hit_rate_top3") is not None else None,
                    )
                with cols[2]:
                    st.metric("n (after)", after.get("n"))
                with cols[3]:
                    test = o.get("statistical_test") or {}
                    if test.get("available"):
                        st.metric("p-value", f"{test.get('p_value')}")
                        st.caption(f"CI95: {test.get('ci_95')}")
                    else:
                        st.caption(f"stat test: {test.get('reason', 'N/A')}")
                exp = o.get("expected")
                if exp:
                    st.caption(f"expected: {exp.get('description', exp)}")
                for note in o.get("notes", []) or []:
                    st.caption(f"note: {note}")
                st.caption(f"implemented_at: {o.get('implemented_at')}")
    else:
        st.info("no task record yet")

with tab3:
    st.subheader("cumulative metric trend")
    if outcomes:
        rows = []
        base_roi = None
        if baseline:
            roi_cands = baseline.get("metrics", {}).get("actual_roi", {}).get("candidate_values", {})
            base_roi = roi_cands.get("raw_cumulative_5_16_evening", {}).get("value")
            rows.append({
                "implemented_at": baseline.get("calculation_date"),
                "task_id": "baseline_v15",
                "roi": base_roi,
                "hit_rate_top3": baseline.get("metrics", {}).get("hit_rate_top3", {}).get("value"),
            })
        for o in outcomes:
            rows.append({
                "implemented_at": o.get("implemented_at", "")[:10],
                "task_id": o.get("task_id"),
                "roi": o.get("after", {}).get("roi"),
                "hit_rate_top3": o.get("after", {}).get("hit_rate_top3"),
            })
        df = pd.DataFrame(rows).sort_values("implemented_at")
        c1, c2 = st.columns(2)
        with c1:
            st.write("ROI trend")
            st.line_chart(df.set_index("implemented_at")[["roi"]])
        with c2:
            st.write("hit3 trend")
            st.line_chart(df.set_index("implemented_at")[["hit_rate_top3"]])
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("no trend data yet")

with tab4:
    st.subheader("limit gauge (video-less ceiling)")
    st.caption("target: AUC 0.9020+ / ROI 110%+ / cum pnl +50K yen")
    col1, col2, col3 = st.columns(3)
    cur_auc = baseline.get("metrics", {}).get("wf_auc", {}).get("value") if baseline else None
    cur_roi = None
    if baseline:
        rc = baseline.get("metrics", {}).get("actual_roi", {}).get("candidate_values", {})
        cur_roi = rc.get("raw_cumulative_5_16_evening", {}).get("value")
    cur_pnl = None
    if baseline:
        pc = baseline.get("metrics", {}).get("cumulative_pnl_yen", {}).get("value_candidates", {})
        cur_pnl = pc.get("cumulative_csv_5_16_plus_5240")

    with col1:
        if cur_auc is not None:
            st.metric("AUC", f"{cur_auc:.4f}", delta=f"target 0.9020+ ({0.9020 - cur_auc:+.4f})")
        else:
            st.metric("AUC", "N/A")
    with col2:
        if cur_roi is not None:
            st.metric("ROI", f"{cur_roi:.2f}%", delta=f"target 110%+ ({110 - cur_roi:+.2f}pt) (P0-1 update pending)")
        else:
            st.metric("ROI", "TBD (P0-1 pending)")
    with col3:
        if cur_pnl is not None:
            st.metric("cum pnl (yen)", f"{cur_pnl:+,}", delta=f"target +50,000 ({50000 - cur_pnl:+,})")
        else:
            st.metric("cum pnl", "TBD (P0-1 pending)")

st.caption("refresh: 30 sec auto. independent from V15 dashboard (port 8501).")

# auto refresh (sleep then rerun)
time.sleep(30)
try:
    st.rerun()
except Exception:
    # streamlit older may not have rerun
    pass
