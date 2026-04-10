from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import streamlit as st
from PIL import Image

from rag_pipeline import PneumoRAGPipeline
from report_generator import ReportGenerator
from ui_components import render_decision_banner, render_gradcam_panel, render_metric_card, summary_label
from ui_constants import (
    APP_ICON,
    APP_LAYOUT,
    APP_SIDEBAR_STATE,
    APP_TITLE,
    NAVIGATION_PAGES,
)
from ui_sections import (
    render_disclaimer,
    render_export_tab,
    render_hero,
    render_image_retrieval_tab,
    render_model_performance_page,
    render_prompt_tab,
    render_rag_overview_tab,
    render_structured_report_tab,
    render_text_evidence_tab,
    render_training_dynamics_page,
)
from ui_styles import inject_css


st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout=APP_LAYOUT,
    initial_sidebar_state=APP_SIDEBAR_STATE,
)


@dataclass(slots=True)
class SidebarSettings:
    page: str
    assets_dir: str
    topk_img: int
    topk_text: int
    save_report: bool
    defense_mode: bool
    show_prompt_tab: bool


@st.cache_resource(show_spinner=False)
def load_pipeline(assets_dir: str) -> PneumoRAGPipeline:
    return PneumoRAGPipeline(assets_dir=assets_dir)


@st.cache_resource(show_spinner=False)
def load_report_generator() -> ReportGenerator:
    return ReportGenerator()


def get_default_assets_dir() -> str:
    default_dir = Path(__file__).resolve().parent / "assets" / "baseline_lsmf"
    return os.getenv("PNEUMO_ASSETS_DIR", str(default_dir))


def render_sidebar() -> SidebarSettings:
    default_assets = get_default_assets_dir()

    with st.sidebar:
        st.markdown("## PneumoAssist")
        st.caption("Clinical decision-support interface")

        page = st.radio(
            "Navigation",
            NAVIGATION_PAGES,
            index=0,
        )

        with st.expander("Advanced Settings", expanded=False):
            assets_dir = st.text_input("Assets directory", value=default_assets)
            topk_img = st.slider("Top-k image retrieval", min_value=3, max_value=10, value=5, step=1)
            topk_text = st.slider("Top-k text evidence", min_value=3, max_value=6, value=6, step=1)
            save_report = st.toggle("Save report JSON", value=True)
            defense_mode = st.toggle("Defense mode", value=True)
            show_prompt_tab = st.toggle("Show prompt tab", value=True)

    return SidebarSettings(
        page=page,
        assets_dir=assets_dir,
        topk_img=topk_img,
        topk_text=topk_text,
        save_report=save_report,
        defense_mode=defense_mode,
        show_prompt_tab=show_prompt_tab,
    )


def render_input_panel() -> tuple[object | None, bool, st.delta_generator.DeltaGenerator]:
    st.subheader("Input Image")

    uploaded = st.file_uploader(
        "Upload chest X-ray",
        type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
        accept_multiple_files=False,
        label_visibility="collapsed",
    )

    run_clicked = st.button(
        "Run Analysis",
        type="primary",
        use_container_width=True,
        disabled=uploaded is None,
    )

    if uploaded is not None:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, use_container_width=True)

    gradcam_slot = st.empty()
    return uploaded, run_clicked, gradcam_slot


def run_analysis(
    pipeline: PneumoRAGPipeline,
    reporter: ReportGenerator,
    uploaded,
    settings: SidebarSettings,
) -> tuple[dict, dict, str | None]:
    temp_path = pipeline.save_uploaded_bytes(uploaded.name, uploaded.getvalue())

    with st.status("Running pipeline...", expanded=True) as status:
        st.write("Loading image and executing inference...")
        report, payload, out_path = reporter.generate_pneumo_report(
            pipeline=pipeline,
            image_path=temp_path,
            topk_img=settings.topk_img,
            topk_text=settings.topk_text,
            save=settings.save_report,
        )
        st.write("Retrieving similar cases and evidence chunks...")
        st.write("Generating structured report...")
        status.update(label="Analysis complete", state="complete", expanded=False)

    return report, payload, out_path


def render_prediction_panel(report: dict, payload: dict) -> tuple[list[str], list[str], dict]:
    prediction = payload["prediction"]
    y_pred = int(prediction["y_pred"])
    p_cal = float(prediction["p_calibrated"])
    threshold = float(prediction["threshold"])
    confidence = prediction["confidence_band"]
    narrative = prediction["narrative_label"]

    decision_text = summary_label(y_pred, narrative)
    prompt_debug = report.get("prompt_debug", {})
    used_chunk_ids = report.get("evidence", {}).get("text_chunk_ids", [])
    used_case_ids = report.get("evidence", {}).get("retrieved_case_ids", [])

    render_decision_banner(decision_text, p_cal, confidence)

    c1, c2, c3 = st.columns(3)
    with c1:
        render_metric_card(
            "Final Decision",
            decision_text,
            "Threshold-aware narrative classification",
        )
    with c2:
        render_metric_card(
            "Calibrated Probability",
            f"{p_cal:.4f}",
            f"Decision threshold = {threshold:.2f}",
        )
    with c3:
        render_metric_card(
            "Confidence Band",
            confidence.title(),
            "Derived from calibrated decision margin",
        )

    overlay_path = report.get("visual_support", {}).get("overlay_path")
    if overlay_path and Path(overlay_path).exists():
        st.markdown('<div class="section-title">Visual Support</div>', unsafe_allow_html=True)
        st.image(overlay_path, use_container_width=True)
        note = report.get("visual_support", {}).get("overlay_note", "")
        if note:
            st.caption(note)

    return used_chunk_ids, used_case_ids, prompt_debug


def render_tabs(
    report: dict,
    payload: dict,
    out_path: str | None,
    uploaded_name: str,
    settings: SidebarSettings,
    used_chunk_ids: list[str],
    used_case_ids: list[str],
    prompt_debug: dict,
) -> None:
    tabs = st.tabs([
        "Structured Report",
        "RAG Overview",
        "Image Retrieval",
        "Text Evidence",
        "Prompt",
        "Export",
    ])

    with tabs[0]:
        render_structured_report_tab(report)
    with tabs[1]:
        render_rag_overview_tab(payload, report)
    with tabs[2]:
        render_image_retrieval_tab(payload, used_case_ids=used_case_ids, assets_dir=settings.assets_dir)
    with tabs[3]:
        render_text_evidence_tab(payload, used_chunk_ids=used_chunk_ids)
    with tabs[4]:
        render_prompt_tab(prompt_debug, show_prompt_tab=settings.show_prompt_tab)
    with tabs[5]:
        render_export_tab(
            report=report,
            payload=payload,
            prompt_debug=prompt_debug,
            saved_json_path=out_path,
            uploaded_name=uploaded_name,
        )


def render_diagnosis_page(settings: SidebarSettings) -> None:
    if not Path(settings.assets_dir).exists():
        st.error(f"Assets directory not found: {settings.assets_dir}")
        st.stop()

    pipeline = load_pipeline(settings.assets_dir)
    reporter = load_report_generator()
    st.caption(f"Loaded model: {getattr(pipeline, 'variant_name', 'Unknown variant')}")

    left, right = st.columns([1.1, 1.4], gap="large")

    with left:
        uploaded, run_clicked, gradcam_slot = render_input_panel()

    if uploaded is None or not run_clicked:
        return

    report, payload, out_path = run_analysis(pipeline, reporter, uploaded, settings)

    explain = report.get("explainability", {}) or {}
    with gradcam_slot.container():
        render_gradcam_panel(
            explain.get("gradcam_overlay_path"),
            explain.get("gradcam_note", ""),
        )

    with right:
        used_chunk_ids, used_case_ids, prompt_debug = render_prediction_panel(report, payload)

    render_tabs(
        report=report,
        payload=payload,
        out_path=out_path,
        uploaded_name=uploaded.name,
        settings=settings,
        used_chunk_ids=used_chunk_ids,
        used_case_ids=used_case_ids,
        prompt_debug=prompt_debug,
    )

def main() -> None:
    inject_css()

    settings = render_sidebar()
    render_hero(settings.page)
    render_disclaimer()

    if settings.page == "Diagnosis":
        render_diagnosis_page(settings)

    elif settings.page == "Model Performance":
        render_model_performance_page()

    elif settings.page == "Training Dynamics":
        render_training_dynamics_page()


if __name__ == "__main__":
    main()
