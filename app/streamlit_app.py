"""Four pages: Overview, Lens, Radar, Discovery Copilot."""

from __future__ import annotations

import html
import json
import sys
import textwrap
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config.settings import settings
from pipeline.dashboard import build_dashboard, cite
from pipeline.derived import job_segment
from pipeline.extract.extractor import _load_jsonl
from app.views.v5_evidence_copilot import render as render_evidence
from pipeline.label import PAIN_LABELS

# Modern vibrant colors
NAVY = "#0B1120"
CARD = "rgba(15, 23, 42, 0.6)"
TEAL = "#2DD4BF"
CORAL = "#F43F5E"
PURPLE = "#A855F7"
YELLOW = "#FBBF24"
BLUE = "#3B82F6"
INK = "#F8FAFC"
MUTED = "#64748B"

PAGES = ["Overview", "Lens", "Radar", "Discovery Copilot"]

QUADRANT = {
    "Core Strategic Bet": "First priority",
    "Critical Safety Net": "Urgent, fewer people",
    "Everyday Papercut": "Common, milder",
    "Low-Priority Nuance": "Later",
}

REMEMBER_LABELS = {
    "a rough time": "A rough time",
    "how it looked": "How the photo looked",
    "who was there": "Who was in the photo",
    "how it felt": "How the moment felt",
}

MISSING_LABELS = {
    "the exact date": "The exact date",
    "the place name": "The place name",
    "the person's name": "The person's name",
    "the object's name": "The name of the object",
    "the album name": "The album name",
}

DOT_COLORS = [CORAL, TEAL, YELLOW, BLUE, PURPLE, "#34D399", "#F472B6", "#94A3B8", "#FDA4AF"]


def _share(count: int, total: int) -> str:
    """Never print 0% when at least one review is in the group."""
    if count <= 0 or total <= 0:
        return "0%"
    pct = 100.0 * count / total
    places = 0
    while places < 6 and round(pct, places) == 0:
        places += 1
    return f"{pct:.{places}f}%"


def _esc(text: object) -> str:
    return html.escape(str(text or ""))


def _load() -> list[dict]:
    return _load_jsonl(settings.processed_dir / "labeled.jsonl")


def _builds() -> list[dict]:
    path = settings.processed_dir / "dashboard.json"
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("builds") or []


def _navigation() -> str:
    st.sidebar.markdown(
        textwrap.dedent("""
        <div style='text-align: center; margin-bottom: 30px; margin-top: -10px;'>
            <h2 style='color: #2DD4BF; border: none; font-size: 1.4rem; font-weight: 800;'>User Pain Discovery</h2>
        </div>
        """), 
        unsafe_allow_html=True
    )
    return st.sidebar.radio("Navigate", PAGES, label_visibility="collapsed")


def _apply_filters(records: list[dict], key_prefix: str, use_source=True, use_pain=True, use_screen=True, use_who=True, use_dates=True) -> list[dict]:
    if not records:
        return []
    
    st.markdown("<h4 style='margin-bottom: 15px; color: #94A3B8; font-weight: 600;'>🔍 Filter Data</h4>", unsafe_allow_html=True)
    cols = st.columns(5)
    col_idx = 0
    
    source = "All"
    if use_source:
        sources = ["All", *sorted({row.get("source") or "" for row in records if row.get("source")})]
        source = cols[col_idx % 5].selectbox("Source", sources, key=f"{key_prefix}_source")
        col_idx += 1
        
    pain = "All"
    if use_pain:
        pains = ["All", *sorted({pain for row in records for pain in (row.get("label") or {}).get("pains") or []})]
        pain = cols[col_idx % 5].selectbox("Pain", pains, format_func=lambda key: "All pains" if key == "All" else PAIN_LABELS.get(key, key), key=f"{key_prefix}_pain")
        col_idx += 1
        
    screen = "All"
    if use_screen:
        screens = ["All", *sorted({(row.get("label") or {}).get("screen") or "" for row in records})]
        screen = cols[col_idx % 5].selectbox("Screen", screens, key=f"{key_prefix}_screen")
        col_idx += 1
        
    person = "All"
    if use_who:
        who = ["All", *sorted({(row.get("label") or {}).get("who") or "" for row in records})]
        person = cols[col_idx % 5].selectbox("Who", who, format_func=lambda key: "Anyone" if key == "All" else key.replace("_", " "), key=f"{key_prefix}_who")
        col_idx += 1
        
    start = end = None
    if use_dates:
        stamps = []
        for row in records:
            stamp = (row.get("timestamp") or "")[:10]
            if len(stamp) == 10:
                try:
                    stamps.append(date.fromisoformat(stamp))
                except ValueError:
                    continue
        if stamps:
            picked = cols[col_idx % 5].date_input("Dates", value=(min(stamps), max(stamps)), key=f"{key_prefix}_dates")
            if isinstance(picked, tuple) and len(picked) == 2:
                start, end = (str(picked[0]), str(picked[1]))
                
    st.markdown("<hr style='margin: 10px 0 25px 0; border-color: rgba(255,255,255,0.05);'>", unsafe_allow_html=True)
    
    chosen = []
    for row in records:
        if use_source and source != "All" and row.get("source") != source: continue
        label = row.get("label") or {}
        if use_pain and pain != "All" and pain not in (label.get("pains") or []): continue
        if use_screen and screen != "All" and label.get("screen") != screen: continue
        if use_who and person != "All" and label.get("who") != person: continue
        if use_dates:
            stamp = (row.get("timestamp") or "")[:10]
            if start and stamp and stamp < start: continue
            if end and stamp and stamp > end: continue
        chosen.append(row)
    
    return chosen


def _quote(sentence: str, attribution: str, url: str | None = None, color: str = TEAL) -> None:
    if not sentence:
        return
    link = ""
    if url and str(url).startswith("http"):
        link = f"<div class='meta'><a href='{_esc(url)}' target='_blank'>↗ Open source</a></div>"
    st.markdown(
        f"<div class='qcard' style='border-left-color: {color};'><q>{_esc(sentence)}</q><div class='meta'><span>{_esc(attribution)}</span>{link}</div></div>",
        unsafe_allow_html=True,
    )


def _donut(rows: list[dict], title: str) -> None:
    if not rows:
        return
    fig = go.Figure(go.Pie(
        labels=[row["label"] for row in rows],
        values=[row["count"] for row in rows],
        hole=0.6,
        marker=dict(colors=DOT_COLORS),
        textinfo="label+percent",
        showlegend=False,
    ))
    fig.update_layout(
        margin=dict(t=30, b=10, l=10, r=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=INK),
        title=dict(text=title, x=0.5),
        height=360,
    )
    st.plotly_chart(fig, use_container_width=True, config={"staticPlot": True})


def _bars(rows: list[dict], name: str, value: str, color: str = TEAL) -> None:
    if not rows:
        return
    frame = pd.DataFrame(rows)
    fig = go.Figure(go.Bar(
        y=frame[name][::-1],
        x=frame[value][::-1],
        orientation="h",
        marker_color=color,
        marker_line_width=0,
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": INK, "family": "Inter"},
        height=max(180, 42 * len(frame)),
        margin={"l": 8, "r": 16, "t": 8, "b": 24},
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.1)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
    )
    st.plotly_chart(fig, use_container_width=True, config={'staticPlot': True})


def _top_row(rows: list[dict] | None) -> dict | None:
    if not rows:
        return None
    return rows[0]


def _pain_row(data: dict, pain_id: str) -> dict | None:
    for row in data.get("pains_by_count") or []:
        if row.get("id") == pain_id and row.get("count"):
            return row
    return None


def _failure_line(row: dict | None, name: str, total: int) -> str:
    if row is None:
        return f"{_esc(name)} does not appear in this selection."
    abandoned = round(100 * float(row.get("failed_share") or 0))
    return (
        f"{_esc(name)} is <b>{int(row['count']):,}</b> reviews, "
        f"{_share(int(row['count']), total)} of this selection, "
        f"and {abandoned}% did not get the photo back."
    )


_JOB_CARDS = (
    {
        "id": "objects",
        "letter": "A",
        "name": "Object searchers",
        "verdict": "Why chosen",
        "who": "People with a large library who already know the thing: a dog, a car, a truck.",
        "behavior": "They type one noun and stop. Search returns nothing, or a hit that is still a pile.",
        "why": "Thickest evidence. Make that noun match, then let them narrow by person, date, or album.",
        "color": TEAL,
    },
    {
        "id": "memories",
        "letter": "B",
        "name": "Memory keepers",
        "verdict": "Why not first",
        "who": "Parents, travelers, and people with older libraries. They want one person, event, or trip.",
        "behavior": "They still have how it looked or a rough time. The exact date, or the right face, is what they lack.",
        "why": "The job is real, and it is the second build. Fewer reviews name it.",
        "color": BLUE,
    },
    {
        "id": "documents",
        "letter": "C",
        "name": "Document keepers",
        "verdict": "Why not first",
        "who": "People keeping receipts, labels, serial numbers, and screenshots.",
        "behavior": "The word is already on the photo. Search does not bring that word back.",
        "why": "Smallest group. The miss is text inside the photo, so it is a safety net, not the first bet.",
        "color": YELLOW,
    },
)


def _overview_segments(records: list[dict]) -> None:
    counts = {"objects": 0, "memories": 0, "documents": 0}
    for row in records:
        job = job_segment(row)
        if job:
            counts[job] += 1
    placed = sum(counts.values())
    aside = len(records) - placed
    cols = st.columns(3)
    for col, card in zip(cols, _JOB_CARDS):
        count = counts[card["id"]]
        share = _share(count, placed) if placed else "0%"
        col.markdown(
            f"""
            <div class="segment-card" style="border-top: 3px solid {card['color']};">
                <div class="kicker">Segment {card['letter']} · {share}</div>
                <div class="name">{_esc(card['name'])}</div>
                <div class="count">{count:,} reviews</div>
                <p><b>Who.</b> {_esc(card['who'])}</p>
                <p><b>Search.</b> {_esc(card['behavior'])}</p>
                <p><b>{_esc(card['verdict'])}.</b> {_esc(card['why'])}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.caption(
        f"Share is of the {placed:,} reviews in this selection about finding a photo. "
        f"{aside:,} are about backup, layout, or storage, so they are not a segment."
    )


def _overview_reading(data: dict) -> None:
    """Three sentences under the metric cards, counted from the current selection."""
    total = int(data.get("total") or 0)
    target = _top_row(data.get("looking_for"))
    remembered = _top_row(data.get("remembers"))
    missing = _top_row(data.get("missing"))
    doing: list[str] = []
    if target:
        doing.append(
            f"The most common named target is <b>{_esc(target['label'])}</b> ({int(target['count']):,})."
        )
    else:
        doing.append("These reviews do not name what they were trying to find.")
    if remembered:
        label = REMEMBER_LABELS.get(remembered["label"], remembered["label"])
        doing.append(
            f"The clue they still have is <b>{_esc(label.lower())}</b> ({int(remembered['count']):,})."
        )
    else:
        doing.append("These reviews do not say what they still remember.")
    if missing:
        label = MISSING_LABELS.get(missing["label"], missing["label"])
        doing.append(
            f"The detail they say is missing is <b>{_esc(label.lower())}</b> ({int(missing['count']):,})."
        )
    else:
        doing.append("These reviews do not say what is missing.")
    quoted = int(data.get("described_query") or 0)
    doing.append(f"<b>{quoted:,}</b> of {total:,} reviews quote the words they typed.")

    empty = _pain_row(data, "search_returns_nothing")
    flood = _pain_row(data, "library_flood")
    failures = [
        _failure_line(empty, "Empty search", total),
        _failure_line(flood, "The library flood", total),
    ]
    if empty and flood:
        failures.append(
            "Empty search means the ordinary word should have matched. "
            "After a match, the flood needs a way to narrow by person, date, or album."
        )

    unfit = int((data.get("does_not_fit") or {}).get("count") or 0)
    unnamed = int(data.get("who_not_said") or 0)
    limit = (
        f"<b>{unfit:,}</b> reviews in this selection are not about finding a photo. "
        f"<b>{unnamed:,}</b> never say who the person is."
    )

    st.markdown(
        "<div class='closing'>"
        "<b>What this selection says</b>"
        f"<p><b>What they are doing.</b> {' '.join(doing)}</p>"
        f"<p><b>The two failures.</b> {' '.join(failures)}</p>"
        f"<p><b>The limit.</b> {limit}</p>"
        "</div>",
        unsafe_allow_html=True,
    )


def _metric_card(title: str, value: str, desc: str, color: str = TEAL, height: str = "150px") -> str:
    return f"""
    <div class="custom-metric-card" style="border-top: 3px solid {color}; height: {height};">
        <div class="title">{_esc(title)}</div>
        <div class="value">{_esc(value)}</div>
        <div class="desc">{_esc(desc)}</div>
    </div>
    """


def _overview(records: list[dict], corpus: int) -> None:
    st.markdown("<h1>Overview <span style='font-size: 1.2rem; font-weight: 500; color: #94A3B8; margin-left: 12px;'>High-level metrics and trends</span></h1>", unsafe_allow_html=True)
    selected = _apply_filters(records, "ov")
    if not selected:
        st.info("No reviews match these filters.")
        return
    data = build_dashboard(selected)
    
    st.caption(f"This selection: **{data['total']:,}** reviews. Full set: **{corpus:,}** retrieval reviews from **{data['raw_ingested']:,}** collected.")
    
    top = data.get("top_pain") or {}
    c1, c2, c3, c4 = st.columns(4)
    failed = 0
    if data["total"]:
        failed = round(100 * sum(row["count"] * row["failed_share"] for row in data["pains_by_count"]) / data["total"])
    
    c1.markdown(_metric_card("Reviews", f"{data['total']:,}", "Total reviews in this selection.", BLUE), unsafe_allow_html=True)
    c2.markdown(_metric_card("Never got photo back", f"{failed}%", "User quit without finding the photo.", PURPLE), unsafe_allow_html=True)
    c3.markdown(_metric_card("Wrote search words", f"{data['described_query']:,}", "Included actual words they typed.", YELLOW), unsafe_allow_html=True)
    c4.markdown(_metric_card("Leading pain severity", f"{top.get('severity', 0):.0f}" if top else "—", "Scale of 100 for the top problem.", CORAL), unsafe_allow_html=True)

    st.markdown("<h3 style='margin-top: 8px;'>Who the reviews are for</h3>", unsafe_allow_html=True)
    _overview_segments(selected)
    _overview_reading(data)
    st.divider()

    st.subheader("Where the reviews came from")
    if data["sources"]:
        cols = st.columns(len(data["sources"]))
        for col, row, color in zip(cols, data["sources"], [TEAL, BLUE, PURPLE, YELLOW, CORAL]):
            span = f"{row['year_min']}–{row['year_max']}" if row.get("year_min") else "undated"
            col.markdown(_metric_card(row["label"], f"{row['count']:,}", f"{span} · {row['share'] * 100:.0f}%", color, height="120px"), unsafe_allow_html=True)
            
    st.divider()

    st.subheader("The pains")
    st.caption("A review can name more than one pain, so the counts can overlap.")
    
    pains_data = []
    for row in data["pains_by_count"]:
        if row["id"] in {"does_not_fit", "search_worked"}:
            continue
        pains_data.append({
            "Pain": row['label'],
            "Reviews": row['count'],
            "Share": _share(row['count'], data['total']),
            "Severity": int(round(row['severity']))
        })
    if pains_data:
        max_reviews = max([row['Reviews'] for row in pains_data]) if pains_data else 0
        html_table = "<table class='modern-table'><thead><tr><th>Pain</th><th>Reviews</th><th>Share</th><th>Severity</th></tr></thead><tbody>"
        for row in pains_data:
            is_max = row['Reviews'] == max_reviews
            row_class = "highlight-row" if is_max else ""
            html_table += f"<tr class='{row_class}'>"
            html_table += f"<td>{_esc(row['Pain'])}</td>"
            html_table += f"<td>{row['Reviews']:,}</td>"
            html_table += f"<td>{row['Share']}</td>"
            html_table += f"<td>{row['Severity']}</td>"
            html_table += "</tr>"
        html_table += "</tbody></table>"
        st.markdown(html_table, unsafe_allow_html=True)

    st.divider()

    col_t, col_w = st.columns(2)
    with col_t:
        if top:
            st.markdown(f"### Top Pain Point Identified", unsafe_allow_html=True)
            st.write(f"**{top['label']}**: {top['count']:,} reviews, most often in {top['top_region']}. Severity **{top['severity']:.0f}** out of 100.")
            _quote(top.get("sentence") or "", top.get("cite") or "", top.get("url"), color=CORAL)
            
    with col_w:
        worked = data.get("search_worked")
        if worked and worked["count"]:
            st.markdown(f"### Contrast: Search Actually Worked", unsafe_allow_html=True)
            st.caption(f"{worked['count']:,} reviews contrast the failures.")
            _quote(worked.get("sentence") or "", worked.get("cite") or "", worked.get("url"), color=BLUE)

    st.divider()

    left, right = st.columns(2)
    with left:
        st.subheader("By year")
        years = data["years"]
        if years:
            fig = go.Figure(go.Bar(x=[row["year"] for row in years], y=[row["count"] for row in years], marker_color=BLUE, marker_line_width=0))
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={"color": INK, "family": "Inter"}, height=280,
                              yaxis=dict(gridcolor="rgba(255,255,255,0.05)"), margin=dict(t=10, b=30, l=30, r=10))
            st.plotly_chart(fig, use_container_width=True, config={'staticPlot': True})
    with right:
        st.subheader("Stars")
        ratings = data["ratings"]
        if ratings["average"] is None:
            st.caption("No star ratings in this selection.")
        else:
            hist = ratings["histogram"]
            fig = go.Figure(go.Bar(
                x=[f"{row['star']} star" for row in hist],
                y=[row["count"] for row in hist],
                marker_color=[CORAL, CORAL, YELLOW, TEAL, TEAL],
                marker_line_width=0,
            ))
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={"color": INK, "family": "Inter"}, height=280,
                              yaxis=dict(gridcolor="rgba(255,255,255,0.05)"), margin=dict(t=10, b=30, l=30, r=10))
            st.plotly_chart(fig, use_container_width=True, config={'staticPlot': True})

    st.divider()
    st.subheader("Sentiment from the star rating")
    st.caption("1 and 2 stars count as frustrated. 3 stars, and reviews with no stars, count as neutral. 4 and 5 stars count as positive.")
    _donut(data.get("sentiment") or [], "Sentiment")

    st.divider()
    st.subheader("Recent reviews")
    cols = st.columns(2)
    for i, row in enumerate(data["recent"][:6]):
        with cols[i % 2]:
            _quote(row.get("sentence") or "", row.get("cite") or "", row.get("url"), color=PURPLE)
            



def _lens(records: list[dict]) -> None:
    st.markdown("<h1>Lens <span style='font-size: 1.2rem; font-weight: 500; color: #94A3B8; margin-left: 12px;'>Clues, Intent & Impact</span></h1>", unsafe_allow_html=True)
    selected = _apply_filters(records, "lens")
    if not selected:
        st.info("No reviews match these filters.")
        return
    data = build_dashboard(selected)
    
    st.caption(f"**{data['total']:,}** reviews in this selection.")
    
    st.subheader("What they still remember")
    st.caption("The clues the review still has: a rough time, how the photo looked, who was in it, or how the moment felt.")
    remembered = [{"label": REMEMBER_LABELS.get(row["label"], row["label"]), "count": row["count"]} for row in data["remembers"]]
    _bars(remembered, "label", "count", color=BLUE)
    
    st.subheader("What they can no longer name")
    st.caption("The detail the review says is gone: the exact date, the place, the person’s name, the object’s name, or the album.")
    missing = [{"label": MISSING_LABELS.get(row["label"], row["label"]), "count": row["count"]} for row in data["missing"]]
    _bars(missing, "label", "count", color=CORAL)
        
    st.divider()
    
    if data["pairs"]:
        st.subheader("Remembered together")
        st.caption("Two clues named in the same review. People rarely remember only one fact.")
        pairs_data = []
        for row in data["pairs"]:
            left, _, right = row["label"].partition(" + ")
            pretty = f"{REMEMBER_LABELS.get(left, left)} and {REMEMBER_LABELS.get(right, right)}" if right else row["label"]
            pairs_data.append({"Combination": pretty, "Count": row["count"]})
        df_pairs = pd.DataFrame(pairs_data)
        st.dataframe(df_pairs, use_container_width=True, hide_index=True, height=len(df_pairs)*36+42)
            
    st.subheader("How severe each pain is")
    st.caption("This is how bad the outcome is, not how many people said it. A higher bar is a worse experience.")
    sharp = [
        {"label": row["label"], "impact": row["impact"]}
        for row in data["failures_by_opportunity"]
    ]
    _bars(sharp, "label", "impact", color=PURPLE)

    friction = data.get("friction") or {}
    st.subheader("Friction score")
    st.caption("Each review is scored 0 to 100: the average of its stars, the feeling already stored on it, and how hard the search outcome was. 60 or above is high friction.")
    f1, f2 = st.columns(2)
    f1.markdown(_metric_card("Average friction", f"{friction.get('average', 0)}", "Out of 100.", CORAL), unsafe_allow_html=True)
    f2.markdown(_metric_card("High friction", f"{friction.get('high_count', 0):,}", "Reviews scored 60 or above.", YELLOW), unsafe_allow_html=True)
    _bars(friction.get("bands") or [], "label", "count", color=CORAL)
    by_pain = [{"label": row["label"], "average": row["average"]} for row in friction.get("by_pain") or []]
    if by_pain:
        st.caption("Average friction inside each pain. A review that names two pains is counted in both.")
        _bars(by_pain, "label", "average", color=PURPLE)

    st.divider()

    st.subheader("Context & Behavior")
    c_feel, c_look, c_next = st.columns(3)
    
    with c_feel:
        felt = list(data["feelings"])
        if data.get("feeling_not_clear"):
            felt.append({"label": "Not clear", "count": data["feeling_not_clear"]})
        st.caption("One feeling is already stored on every review. Not clear means the words did not say.")
        _donut(felt, "How it felt")
    with c_look:
        _donut(data["looking_for"], "What they tried to find")
    with c_next:
        _donut(data["did_next"], "After search failed")
        
    st.divider()
    
    st.subheader("Which part of Google Photos they mentioned")
    cols = st.columns(3)
    for i, row in enumerate(data["screens"]):
        with cols[i % 3]:
            st.markdown(f"""
            <div class='screen-card'>
                <h4>{row['screen']}</h4>
                <div style='font-size: 1.2em; font-weight: bold; color: {TEAL};'>{row['count']:,} mentions</div>
                <div style='color: {MUTED};'>{row.get('frustrated_share', row['hard_share']) * 100:.0f}% read as frustrated</div>
            </div>
            """, unsafe_allow_html=True)
            if row.get("sentence"):
                _quote(row.get("sentence") or "", row.get("cite") or "", row.get("url"), color=YELLOW)

    st.divider()
    st.subheader("Details they say are missing")
    st.caption("Grouped from the missing-detail text already stored on each review. A review is counted once per detail.")
    _bars(data.get("missing_details") or [], "label", "count", color=CORAL)

    st.subheader("Words they still use")
    st.caption("Phrases named in at least three reviews. Product complaints, such as frustration with the app, are left out.")
    kind_name = {
        "temporal": "a time",
        "sensory": "how it looked",
        "emotional": "how it felt",
        "social": "who was there",
    }
    phrases = [
        {
            "label": f"{row['label']} · {kind_name.get(row.get('kind') or '', row.get('kind') or '')}".rstrip(" ·"),
            "count": row["count"],
        }
        for row in data.get("named_phrases") or []
    ]
    _bars(phrases, "label", "count", color=TEAL)

    st.divider()
    st.subheader("How their search words changed")
    if data["query_steps"]:
        s_data = [{"From": row['source'], "To": row['target'], "Count": row['value']} for row in data["query_steps"][:8]]
        df_query = pd.DataFrame(s_data)
        st.dataframe(df_query, use_container_width=True, hide_index=True, height=len(df_query)*36+42)


def _radar(records: list[dict]) -> None:
    st.markdown("<h1>Radar <span style='font-size: 1.2rem; font-weight: 500; color: #94A3B8; margin-left: 12px;'>Opportunities and Priorities</span></h1>", unsafe_allow_html=True)
    selected = _apply_filters(records, "radar", use_who=False, use_dates=False)
    if not selected:
        st.info("No reviews match these filters.")
        return
    data = build_dashboard(selected)
    
    st.caption("Right means more people said it. Up means it hurt more. Each number is one pain, listed under the chart.")
    rows = data["failures_by_opportunity"]
    if rows:
        frame = pd.DataFrame(rows).reset_index(drop=True)
        numbers = [str(index) for index in range(1, len(frame) + 1)]
        colors = [DOT_COLORS[index % len(DOT_COLORS)] for index in range(len(frame))]
        sizes = (frame["count"] / frame["count"].max() * 32 + 24).tolist()
        fig = go.Figure(go.Scatter(
            x=frame["frequency"],
            y=frame["impact"],
            mode="markers+text",
            text=numbers,
            textposition="middle center",
            textfont={"color": "#0F172A", "size": 14, "family": "Inter", "weight": "bold"},
            marker={"size": sizes, "color": colors, "line": {"width": 0}},
            hovertext=frame["label"],
            hovertemplate="<b>%{hovertext}</b><br>How common: %{x:.2f}<br>How severe: %{y:.2f}<extra></extra>",
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": INK, "family": "Inter", "size": 13},
            height=540,
            xaxis_title="How common",
            yaxis_title="How severe",
            xaxis=dict(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.1)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.1)"),
            margin=dict(l=40, r=40, t=20, b=40)
        )
        st.plotly_chart(fig, use_container_width=True, config={'staticPlot': True})
        
        l_cols = st.columns(3)
        for index, row in enumerate(rows, start=1):
            l_cols[(index-1) % 3].markdown(f"**{index}.** {row['label']}")
            
    st.divider()
    
    st.subheader("By how common and how severe")
    st.caption("Impact is 40% stakes, 30% not getting the photo, 30% how hard the next step was. Opportunity is how common times how severe.")
    df_opp = pd.DataFrame([
        {
            "Pain": row["label"],
            "Impact": row["impact"],
            "Evidence": row["count"],
            "Opportunity": row["opportunity"],
            "Priority": QUADRANT.get(row["quadrant"], row["quadrant"]),
        }
        for row in data["pains_by_opportunity"]
        if row["id"] not in {"does_not_fit", "search_worked"}
    ])
    st.dataframe(df_opp, use_container_width=True, hide_index=True, height=len(df_opp)*36+42, column_config={
        "Impact": st.column_config.ProgressColumn("Impact", help="Severity of the pain", min_value=0, max_value=100, format="%.1f"),
        "Opportunity": st.column_config.ProgressColumn("Opportunity", min_value=0, max_value=float(max(df_opp["Opportunity"].max(), 1)), format="%.1f"),
    })
    
    st.divider()
    
    st.subheader("What to build")
    builds = {row.get("id"): row for row in _builds()}
    for row in data["failures_by_opportunity"]:
        suggestion = builds.get(row["id"]) or {}
        with st.container():
            st.markdown(f"<div style='background: rgba(45, 212, 191, 0.05); padding: 16px; border-radius: 12px; border: 1px solid rgba(45, 212, 191, 0.2); margin-bottom: 16px;'>", unsafe_allow_html=True)
            st.markdown(f"### {row['label']}")
            st.caption(f"**{row['count']:,}** reviews ({_share(row['count'], data['total'])})")
            if suggestion:
                st.write(f"**Cause:** {suggestion.get('cause') or ''}")
                st.write(f"**Build:** {suggestion.get('build') or ''}")
                st.write(f"**Why:** {suggestion.get('why') or ''}")
            _quote(row.get("sentence") or "", row.get("cite") or "", row.get("url"), color=TEAL)
            st.markdown("</div>", unsafe_allow_html=True)
            
    st.divider()
    segments = data.get("segments") or []
    if segments:
        st.subheader("Who is hit")
        st.caption(
            f"{data.get('segment_unplaced', 0):,} reviews never say who they are, so they are not a segment. "
            "Problem rate is the share whose review is about failing to find a photo."
        )
        df_seg = pd.DataFrame([
            {
                "Who": row["label"],
                "Reviews": row["count"],
                "Problem rate": f"{row['problem_rate'] * 100:.0f}%",
                "Top pain": row["top_pain"],
                "What it implies": row["implication"],
            }
            for row in segments
        ])
        st.dataframe(df_seg, use_container_width=True, hide_index=True, height=len(df_seg) * 36 + 42)


def _proof(records: list[dict]) -> None:
    st.markdown("<h1>Proof <span style='font-size: 1.2rem; font-weight: 500; color: #94A3B8; margin-left: 12px;'>Verbatim user quotes</span></h1>", unsafe_allow_html=True)
    selected = _apply_filters(records, "proof")
    if not selected:
        st.info("No reviews match these filters.")
        return
    data = build_dashboard(selected)
    
    st.caption(f"**{data['total']:,}** reviews in this selection.")
    ordered = sorted(selected, key=lambda row: row.get("timestamp") or "", reverse=True)
    limit = st.slider("Number of reviews to show", 4, 100, 12, 4)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    cols = st.columns(2)
    for i, row in enumerate(ordered[:limit]):
        col = cols[i % 2]
        label = row.get("label") or {}
        names = ", ".join(PAIN_LABELS.get(item, item) for item in label.get("pains") or [])
        with col:
            st.caption(f"**{names}** · {label.get('screen') or 'Unknown screen'} · {label.get('looking_for') or 'Unknown intent'}")
            _quote(label.get("sentence") or "", cite(row), row.get("source_url"), color=BLUE)


def main() -> None:
    st.set_page_config(page_title="Photo Analysis", page_icon="✨", layout="wide", initial_sidebar_state="expanded")
    st.markdown(
        textwrap.dedent(f"""
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Outfit:wght@500;700;800&display=swap" rel="stylesheet">
        <style>
        :root {{
            --bg-color: {NAVY};
            --card-bg: {CARD};
            --border-color: rgba(255, 255, 255, 0.08);
            --text-primary: {INK};
            --text-muted: {MUTED};
        }}
        .stApp {{ 
            background-color: var(--bg-color); 
            background-image: 
                radial-gradient(at 0% 0%, rgba(59, 130, 246, 0.08) 0px, transparent 40%),
                radial-gradient(at 100% 0%, rgba(168, 85, 247, 0.08) 0px, transparent 40%),
                radial-gradient(at 100% 100%, rgba(244, 63, 94, 0.08) 0px, transparent 40%);
            color: var(--text-primary); 
            font-family: 'Inter', sans-serif;
        }}
        /* Sidebar styling to make it thinner and cleaner */
        [data-testid="stSidebar"] {{ 
            min-width: 220px !important;
            max-width: 220px !important;
            background-color: rgba(11, 17, 32, 0.95); 
            border-right: 1px solid var(--border-color);
        }}
        /* Totally hide radio circles and style the labels as modern buttons */
        [data-testid="stSidebar"] div[role="radiogroup"] > label span[data-baseweb="radio"] {{
            display: none;
        }}
        [data-testid="stSidebar"] div[role="radiogroup"] > label {{
            background: rgba(255,255,255,0.03);
            padding: 12px 16px;
            border-radius: 8px;
            margin-bottom: 8px;
            border: 1px solid transparent;
            transition: all 0.2s;
            cursor: pointer;
        }}
        [data-testid="stSidebar"] div[role="radiogroup"] > label:hover {{
            background: rgba(255,255,255,0.08);
            border-color: rgba(255,255,255,0.1);
        }}
        [data-testid="stSidebar"] div[role="radiogroup"] > label [data-testid="stMarkdownContainer"] p {{
            font-size: 1.05rem;
            font-weight: 600;
            color: var(--text-primary);
        }}
        h1, h2, h3, h4, h5, h6 {{ 
            font-family: 'Outfit', sans-serif; 
            letter-spacing: -0.02em;
            color: var(--text-primary);
        }}
        h1 {{ font-weight: 800; margin-bottom: 0.5rem; }}
        h3 {{ margin-top: 1.5rem; }}
        .custom-metric-card {{
            background: var(--card-bg); 
            backdrop-filter: blur(12px);
            border: 1px solid var(--border-color); 
            border-radius: 12px; 
            padding: 20px; 
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
            display: flex;
            flex-direction: column;
            justify-content: flex-start;
        }}
        .custom-metric-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
        }}
        .segment-card {{
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 16px 16px 8px 16px;
            min-height: 100%;
        }}
        .segment-card .kicker {{
            color: var(--text-muted);
            font-size: 0.8rem;
            font-weight: 600;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            margin-bottom: 6px;
        }}
        .segment-card .name {{
            color: var(--text-primary);
            font-size: 1.15rem;
            font-weight: 700;
            margin-bottom: 2px;
        }}
        .segment-card .count {{
            color: var(--text-muted);
            font-size: 0.85rem;
            margin-bottom: 10px;
        }}
        .segment-card p {{
            color: var(--text-muted);
            font-size: 0.92rem;
            line-height: 1.45;
            margin: 0 0 8px 0;
        }}
        .segment-card b {{ color: var(--text-primary); }}
        .custom-metric-card .title {{
            font-size: 0.9rem;
            color: var(--text-muted);
            margin-bottom: 8px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .custom-metric-card .value {{
            font-family: 'Outfit', sans-serif; 
            font-weight: 800; 
            color: var(--text-primary);
            font-size: 2.2rem;
            margin-bottom: 8px;
            line-height: 1.1;
        }}
        .custom-metric-card .desc {{
            font-size: 0.85rem;
            color: var(--text-muted);
            line-height: 1.4;
        }}
        .screen-card {{
            background: rgba(30, 41, 59, 0.4);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 16px;
            margin-bottom: 16px;
        }}
        .screen-card h4 {{ margin-top: 0; margin-bottom: 8px; }}
        .qcard {{ 
            background: linear-gradient(145deg, rgba(30, 41, 59, 0.6), rgba(15, 23, 42, 0.8));
            border-left: 4px solid {TEAL}; 
            border-radius: 12px; 
            padding: 18px 24px; 
            margin: 0 0 16px 0; 
            box-shadow: 0 2px 8px rgba(0,0,0,0.15);
            transition: all 0.2s ease;
        }}
        .qcard:hover {{
            transform: translateX(4px);
        }}
        .qcard q {{ 
            display: block; 
            font-style: italic; 
            font-size: 1.05em;
            line-height: 1.5;
            color: var(--text-primary);
            margin-bottom: 12px;
        }}
        .qcard .meta {{ 
            display: flex;
            justify-content: space-between;
            align-items: center;
            color: var(--text-muted); 
            font-size: 0.85em;
            font-weight: 500;
        }}
        .qcard a {{ color: {TEAL}; text-decoration: none; display: flex; align-items: center; gap: 4px; }}
        .qcard a:hover {{ text-decoration: underline; }}
        .closing {{ 
            background: rgba(30, 41, 59, 0.4); 
            border: 1px solid rgba(255, 255, 255, 0.1); 
            border-radius: 12px; 
            padding: 20px; 
            font-size: 1.05em;
            line-height: 1.6;
            color: var(--text-muted);
        }}
        .closing b {{ color: var(--text-primary); }}
        .closing p {{ margin: 12px 0 0 0; }}
        .modern-table {{
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 20px;
            background: var(--card-bg);
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            border-style: hidden;
        }}
        .modern-table thead {{
            background: rgba(15, 23, 42, 0.8);
        }}
        .modern-table th {{
            color: var(--text-muted);
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            padding: 16px;
            text-align: left;
            font-size: 0.85rem;
            border-bottom: 2px solid var(--border-color);
        }}
        .modern-table td {{
            padding: 16px;
            color: var(--text-primary);
            border-bottom: 1px solid rgba(255,255,255,0.05);
            font-size: 0.95rem;
        }}
        .modern-table tr.highlight-row td {{
            background: rgba(45, 212, 191, 0.1);
            color: #2DD4BF;
            font-weight: 600;
            border-bottom: 1px solid rgba(45, 212, 191, 0.2);
        }}
        .modern-table tr.highlight-row td:first-child {{
            border-left: 4px solid #2DD4BF;
        }}
        hr {{ border-color: var(--border-color); margin: 2rem 0; }}
        </style>
        """),
        unsafe_allow_html=True,
    )
    
    page = _navigation()
    
    records = _load()
    if not records:
        st.title("Photo Analysis")
        st.error("The reviews are still being labeled. Run the label step, then refresh.")
        return
        
    if page == "Overview":
        _overview(records, len(records))
    elif page == "Lens":
        _lens(records)
    elif page == "Radar":
        _radar(records)
    else:
        render_evidence()


if __name__ == "__main__":
    main()
