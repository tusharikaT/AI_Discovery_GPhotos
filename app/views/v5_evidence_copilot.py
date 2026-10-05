"""Discovery Copilot: the four saved answers and one typed question."""

from __future__ import annotations

import html
import json

import streamlit as st

from app.rag import PRESET_QUESTIONS, answer_question
from config.settings import settings

HEADINGS = (
    ("executive_finding", "Executive finding"),
    ("key_behaviors", "Key behaviors"),
    ("memory_gap", "Where memory and search miss each other"),
    ("who", "Who the reviews describe"),
)


def _esc(value: object) -> str:
    return html.escape(str(value or ""))


def _presets() -> list[dict]:
    path = settings.processed_dir / "copilot_presets.json"
    if not path.exists():
        return []
    saved = json.loads(path.read_text(encoding="utf-8"))
    by_question = {row.get("question"): row for row in saved}
    return [by_question[question] for question in PRESET_QUESTIONS if question in by_question]


def _quote(sentence: str, attribution: str, url: str) -> None:
    if not sentence:
        return
    link = ""
    if url and str(url).startswith("http"):
        link = f"<div class='meta'><a href='{_esc(url)}' target='_blank'>Open source</a></div>"
    st.markdown(
        f"<div class='qcard'><q>{_esc(sentence)}</q><div class='meta'><span>{_esc(attribution)}</span>{link}</div></div>",
        unsafe_allow_html=True,
    )


def _show_saved(row: dict, key_prefix: str) -> None:
    if not row.get("ok"):
        st.write(row.get("line") or "")
        return
    answer = row["answer"]
    for key, title in HEADINGS:
        st.markdown(f"**{title}**")
        value = answer.get(key)
        if key == "key_behaviors" and isinstance(value, list):
            for item in value:
                st.markdown(f"- {item}")
        else:
            st.write(value or "")
    reviews = answer.get("reviews") or []
    st.markdown(f"**Reviews backing this answer ({len(reviews)} reviews)**")
    more_key = f"evidence-more-{key_prefix}"
    show_all = st.session_state.get(more_key, False)
    visible = reviews if show_all else reviews[:3]
    for review in visible:
        _quote(review.get("sentence") or review.get("text") or "", review.get("cite") or "", review.get("url") or "")
    if len(reviews) > 3:
        label = "See less" if show_all else f"See more ({len(reviews) - 3})"
        if st.button(label, key=f"see-{key_prefix}"):
            st.session_state[more_key] = not show_all
            st.rerun()


def render() -> None:
    st.markdown(
        "<h1>Discovery Copilot <span style='font-size: 1.2rem; font-weight: 500; color: #94A3B8; margin-left: 12px;'>Answers from the reviews</span></h1>",
        unsafe_allow_html=True,
    )
    presets = _presets()
    if len(presets) != 4:
        st.error("The four saved answers are not on disk yet.")
        return

    open_at = st.session_state.get("evidence_open")
    for index, row in enumerate(presets):
        if st.button(row["question"], key=f"evidence-q-{index}", use_container_width=True):
            open_at = None if open_at == index else index
            st.session_state["evidence_open"] = open_at
            st.rerun()
        if st.session_state.get("evidence_open") == index:
            _show_saved(row, f"preset-{index}")

    st.markdown("### Ask about these reviews")
    typed = st.text_input("Question", label_visibility="collapsed", placeholder="Ask about these reviews", key="evidence-typed")
    if st.button("Ask", key="evidence-ask"):
        with st.spinner("Reading the reviews…"):
            st.session_state["evidence_typed_question"] = typed.strip()
            st.session_state["evidence_typed_answer"] = answer_question(typed.strip())
    shown = st.session_state.get("evidence_typed_answer")
    if shown:
        if not shown.get("ok"):
            st.write(shown.get("line") or "")
        else:
            _show_saved(shown, "typed")
