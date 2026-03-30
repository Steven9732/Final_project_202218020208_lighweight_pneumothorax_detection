from __future__ import annotations

from pathlib import Path
from typing import Iterable

import streamlit as st

from ui_constants import CASE_IMAGE_KEYS, REPORT_HEADINGS


def fmt_pct(value: float) -> str:
    return f"{value:.2f}%"


def summary_label(y_pred: int, narrative: str) -> str:
    if narrative == "indeterminate":
        return "Indeterminate"
    return "Suggestive of Pneumothorax" if y_pred == 1 else "Not Suggestive of Pneumothorax"


def render_metric_card(label: str, value: str, note: str = "") -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="muted">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_feature_metric_card(label: str, value: str, note: str = "") -> None:
    st.markdown(
        f"""
        <div class="metric-card-feature">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_compact_metric_card(label: str, value: str, note: str = "") -> None:
    st.markdown(
        f"""
        <div class="metric-card-compact">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_decision_banner(decision: str, p_cal: float, confidence: str) -> None:
    st.markdown(
        f"""
        <div class="decision-banner">
            <div class="decision-title">Clinical Impression</div>
            <div class="decision-value">{decision}</div>
            <div class="muted">
                Calibrated probability: <b>{p_cal:.3f}</b> &nbsp;&nbsp;|&nbsp;&nbsp;
                Confidence: <b>{confidence.title()}</b>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_report_sections(report_text: str) -> None:
    parsed: dict[str, str] = {}
    current: str | None = None

    for raw_line in report_text.split("\n"):
        line = raw_line.strip()
        if not line:
            continue

        matched = next((heading for heading in REPORT_HEADINGS if line.startswith(heading)), None)
        if matched is not None:
            current = matched.replace(":", "")
            parsed[current] = line[len(matched):].strip()
            continue

        if current is not None:
            parsed[current] += f" {line}"
        else:
            parsed.setdefault("Report", "")
            parsed["Report"] += f" {line}"

    for title, body in parsed.items():
        st.markdown(f'<div class="report-heading">{title}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="report-body">{body.strip()}</div>', unsafe_allow_html=True)


def render_badges(items: Iterable[str] | None) -> None:
    badge_items = list(items or [])
    if badge_items:
        html = "".join(f'<span class="badge">{item}</span>' for item in badge_items)
        st.markdown(html, unsafe_allow_html=True)
        return
    st.caption("None")


def render_prompt_block(title: str, content: str) -> None:
    with st.expander(title, expanded=False):
        if content:
            st.code(content, language="text")
        else:
            st.info("Not available.")


def resolve_case_image_path(case: dict, assets_dir: str | None = None) -> str | None:
    for key in CASE_IMAGE_KEYS:
        raw_path = case.get(key)
        if not raw_path:
            continue

        path = Path(str(raw_path))
        if path.exists():
            return str(path)

        if assets_dir is None:
            continue

        joined = Path(assets_dir) / path
        if joined.exists():
            return str(joined)

    return None


def render_case_card(
    case: dict,
    idx: int,
    used_case_ids: Iterable[str] | None = None,
    assets_dir: str | None = None,
) -> None:
    used_ids = {str(x) for x in (used_case_ids or [])}

    case_id = case.get("case_id", f"Case {idx}")
    sim = case.get("sim")
    label = case.get("label", case.get("y_true", "N/A"))
    pred = case.get("pred_label", case.get("y_pred", "N/A"))
    sim_text = f"{sim:.4f}" if isinstance(sim, (int, float)) else "N/A"
    image_path = resolve_case_image_path(case, assets_dir=assets_dir)

    col_img, col_meta = st.columns([1.0, 1.35], gap="large")

    with col_img:
        if image_path and Path(image_path).exists():
            st.image(image_path, use_container_width=True)
        else:
            st.info("No displayable image found for this retrieved case.")

    with col_meta:
        st.markdown(f"**Retrieved Case {idx}:** {case_id}")
        st.markdown(f"**Similarity:** {sim_text}")
        st.markdown(f"**Reference label:** {label}")
        st.markdown(f"**Retrieved prediction:** {pred}")

        if str(case_id) in used_ids:
            st.success("Used in report")


def render_chunk_card(chunk: dict, used_chunk_ids: Iterable[str] | None = None) -> None:
    used_ids = {str(x) for x in (used_chunk_ids or [])}

    chunk_id = chunk.get("chunk_id", "N/A")
    tags = ", ".join(chunk.get("tags", [])) if chunk.get("tags") else "N/A"
    score = float(chunk.get("score", 0))
    sim = float(chunk.get("sim", 0))
    text = chunk.get("text", "")

    st.markdown(f"**Chunk ID:** {chunk_id}")
    st.markdown(f"**Tags:** {tags}")
    st.markdown(f"**Score:** {score:.4f} | **Similarity:** {sim:.4f}")

    if str(chunk_id) in used_ids:
        st.success("Used in report")

    st.write(text)


def render_gradcam_panel(gradcam_overlay_path: str | None, note: str = "") -> None:
    if not gradcam_overlay_path:
        return

    overlay = Path(gradcam_overlay_path)
    if not overlay.exists():
        return

    st.markdown('<div class="section-title">Model Explainability</div>', unsafe_allow_html=True)
    st.image(str(overlay), use_container_width=True)
    if note:
        st.caption(note)
