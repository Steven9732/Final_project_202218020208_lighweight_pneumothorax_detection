from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

from rag_pipeline import PneumoRAGPipeline
from report_generator import ReportGenerator


st.set_page_config(
    page_title="PneumoAssist: Pneumothorax Detection & Reporting",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

FINAL_MODEL_INFO = {
    "model_name": "ConvNeXtV2 + UNet++",
    "innovation": "ECA-enhancedLesion-Sensitive Multi-Scale Fusion (ECA-LSMF)",
    "evaluation": "Temp+Threshold model",
    "temperature": 0.716,
    "threshold": 0.750,
}

FINAL_MODEL_METRICS = {
    "Accuracy": 91.53526970954357,
    "Precision": 79.15194346289752,
    "Recall": 83.89513108614233,
    "F1-Score": 81.45454545454545,
    "Specificity": 93.71002132096258,
    "ROC-AUC": 96.41120241489183,
    "PR-AUC": 89.08149789724244,
}

def inject_css():
    st.markdown(
        """
        <style>
        .perf-wrap {
            background: linear-gradient(180deg, #121A23 0%, #111821 100%);
            border: 1px solid #22303E;
            border-radius: 20px;
            padding: 22px 22px 18px 22px;
            box-shadow: 0 10px 28px rgba(0,0,0,0.24);
            margin-bottom: 18px;
        }

        .perf-head {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            gap: 18px;
            flex-wrap: wrap;
            margin-bottom: 16px;
            padding-bottom: 14px;
            border-bottom: 1px solid #22303E;
        }

        .perf-head-left {
            min-width: 280px;
        }

        .perf-title {
            font-size: 1.12rem;
            font-weight: 750;
            color: #EAF2F7;
            margin-bottom: 0.35rem;
        }

        .perf-subtitle {
            color: #9CB0BF;
            font-size: 0.95rem;
            line-height: 1.7;
        }

        .perf-meta {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            align-items: center;
            justify-content: flex-start;
        }

        .perf-chip {
            display: inline-block;
            padding: 0.38rem 0.75rem;
            border-radius: 999px;
            border: 1px solid #304252;
            background: #17212B;
            color: #C6D7E3;
            font-size: 0.82rem;
            font-weight: 600;
        }

        .perf-row-gap {
            height: 14px;
        }

        .metric-card-feature {
            background: linear-gradient(180deg, #172331 0%, #14202B 100%);
            border: 1px solid #35516A;
            border-radius: 18px;
            padding: 22px 22px;
            min-height: 142px;
            box-shadow: 0 8px 22px rgba(0,0,0,0.22);
        }

        .metric-card-feature .metric-label {
            color: #9CB6C9;
            font-size: 0.95rem;
            font-weight: 700;
            margin-bottom: 10px;
        }

        .metric-card-feature .metric-value {
            color: #F5FAFD;
            font-size: 2rem;
            font-weight: 800;
            line-height: 1.12;
            margin-bottom: 10px;
        }

        .metric-card-feature .metric-note {
            color: #9CB0BF;
            font-size: 0.90rem;
            line-height: 1.55;
        }

        .metric-card-compact {
            background: linear-gradient(180deg, #141D27 0%, #121A23 100%);
            border: 1px solid #243241;
            border-radius: 16px;
            padding: 18px 18px;
            min-height: 122px;
            box-shadow: 0 6px 18px rgba(0,0,0,0.18);
        }

        .metric-card-compact .metric-label {
            color: #8EA1B1;
            font-size: 0.92rem;
            font-weight: 700;
            margin-bottom: 8px;
        }

        .metric-card-compact .metric-value {
            color: #F2F7FB;
            font-size: 1.45rem;
            font-weight: 780;
            line-height: 1.15;
            margin-bottom: 8px;
        }

        .metric-card-compact .metric-note {
            color: #93A6B6;
            font-size: 0.88rem;
            line-height: 1.5;
        }

        .perf-divider {
            height: 1px;
            background: linear-gradient(90deg, transparent 0%, #233243 20%, #233243 80%, transparent 100%);
            margin: 14px 0 6px 0;
        }
        .stApp {
            background:
                radial-gradient(circle at top right, rgba(77,163,217,0.10), transparent 22%),
                radial-gradient(circle at left top, rgba(79,182,168,0.08), transparent 18%),
                linear-gradient(180deg, #0B1117 0%, #0D141C 100%);
            color: #EAF2F7;
        }

        [data-testid="stHeader"] {
            background: rgba(11, 17, 23, 0.88);
            border-bottom: 1px solid #1E2A36;
            backdrop-filter: blur(8px);
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0E151D 0%, #101922 100%);
            border-right: 1px solid #1E2A36;
        }

        .top-hero {
            background: linear-gradient(180deg, rgba(18,26,35,0.96) 0%, rgba(16,24,32,0.96) 100%);
            border: 1px solid #233243;
            border-radius: 22px;
            padding: 24px 26px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.28);
            margin-bottom: 18px;
        }

        .title-xl {
            font-size: 2rem;
            font-weight: 800;
            color: #EAF2F7;
            letter-spacing: -0.02em;
            margin-bottom: 0.2rem;
        }

        .subtitle {
            color: #9FB0BF;
            font-size: 1rem;
            line-height: 1.65;
        }

        .section-card {
            background: linear-gradient(180deg, #121A23 0%, #111821 100%);
            border: 1px solid #22303E;
            border-radius: 18px;
            padding: 18px 20px;
            box-shadow: 0 8px 24px rgba(0,0,0,0.22);
            margin-bottom: 16px;
        }

        .section-title {
            font-size: 1.06rem;
            font-weight: 700;
            color: #DCE8F2;
            margin-bottom: 0.75rem;
        }

        .metric-card {
            background: linear-gradient(180deg, #141D27 0%, #121A23 100%);
            border-radius: 16px;
            padding: 16px 18px;
            border: 1px solid #243241;
            min-height: 112px;
            box-shadow: 0 6px 18px rgba(0,0,0,0.18);
        }

        .metric-label {
            color: #8EA1B1;
            font-size: 0.92rem;
            margin-bottom: 8px;
            font-weight: 600;
        }

        .metric-value {
            color: #F2F7FB;
            font-size: 1.42rem;
            font-weight: 750;
            line-height: 1.2;
        }

        .muted {
            color: #93A6B6;
            font-size: 0.92rem;
            line-height: 1.6;
        }

        .decision-banner {
            background: linear-gradient(180deg, rgba(20,33,45,0.98) 0%, rgba(15,24,33,0.98) 100%);
            border: 1px solid #274258;
            border-left: 6px solid #4FB6A8;
            border-radius: 18px;
            padding: 18px 20px;
            margin-bottom: 16px;
            box-shadow: 0 8px 22px rgba(0,0,0,0.22);
        }

        .decision-title {
            font-size: 0.88rem;
            color: #89A0B2;
            font-weight: 700;
            margin-bottom: 6px;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .decision-value {
            font-size: 1.58rem;
            font-weight: 800;
            color: #F3F8FC;
            line-height: 1.2;
            margin-bottom: 8px;
        }

        .badge {
            display: inline-block;
            padding: 0.30rem 0.72rem;
            border-radius: 999px;
            border: 1px solid #304252;
            margin-right: 0.45rem;
            margin-bottom: 0.45rem;
            background: #17212B;
            color: #BCD0DE;
            font-size: 0.82rem;
        }

        .report-heading {
            font-size: 0.96rem;
            font-weight: 700;
            color: #D8E7F2;
            margin-top: 1rem;
            margin-bottom: 0.35rem;
        }

        .report-body {
            color: #B7C6D2;
            line-height: 1.72;
            font-size: 0.97rem;
        }

        .case-card {
            background: linear-gradient(180deg, #141D27 0%, #121A23 100%);
            border: 1px solid #243241;
            border-radius: 16px;
            padding: 16px 18px;
            margin-bottom: 12px;
            box-shadow: 0 6px 16px rgba(0,0,0,0.18);
        }

        .case-title {
            font-weight: 700;
            color: #E4EEF6;
            margin-bottom: 0.45rem;
        }

        .case-meta {
            color: #98AABA;
            font-size: 0.92rem;
            line-height: 1.65;
        }

        .chunk-card {
            background: linear-gradient(180deg, #141D27 0%, #121A23 100%);
            border: 1px solid #243241;
            border-radius: 16px;
            padding: 16px 18px;
            margin-bottom: 12px;
            box-shadow: 0 6px 16px rgba(0,0,0,0.18);
        }

        .disclaimer {
            background: rgba(214,162,74,0.10);
            border: 1px solid rgba(214,162,74,0.28);
            color: #E4C987;
            border-radius: 14px;
            padding: 14px 16px;
            font-size: 0.92rem;
            line-height: 1.6;
            margin-top: 10px;
        }

        .stButton > button {
            border-radius: 12px;
            height: 44px;
            border: 1px solid #33506A;
            background: linear-gradient(180deg, #215172 0%, #1B4663 100%);
            color: #F4F8FB;
            font-weight: 700;
            box-shadow: 0 6px 16px rgba(0,0,0,0.24);
        }

        .stButton > button:hover {
            border: 1px solid #41759B;
            background: linear-gradient(180deg, #276286 0%, #205675 100%);
            color: #FFFFFF;
        }

        .stDownloadButton > button {
            border-radius: 12px;
            height: 42px;
            border: 1px solid #304252;
            background: #17212B;
            color: #EAF2F7;
            font-weight: 600;
        }

        div[data-baseweb="tab-list"] {
            gap: 8px;
        }

        button[data-baseweb="tab"] {
            background: #131B24;
            border: 1px solid #22303E;
            border-radius: 12px;
            color: #9EB0BF;
            padding: 10px 16px;
        }

        button[data-baseweb="tab"][aria-selected="true"] {
            background: #1A2733;
            border: 1px solid #3A5368;
            color: #F2F7FB;
        }

        .stTextInput input {
            background: #111923 !important;
            color: #EAF2F7 !important;
            border: 1px solid #2A3947 !important;
            border-radius: 10px !important;
        }

        .stAlert {
            border-radius: 14px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def load_pipeline(assets_dir: str):
    return PneumoRAGPipeline(assets_dir=assets_dir)


@st.cache_resource(show_spinner=False)
def load_report_generator():
    return ReportGenerator()


def render_metric_card(label: str, value: str, note: str = ""):
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

def fmt_pct(x: float) -> str:
    return f"{x:.2f}%"

def render_model_performance_section():
    info = FINAL_MODEL_INFO
    metrics = FINAL_MODEL_METRICS

    # st.markdown('<div class="perf-wrap">', unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class="perf-head">
            <div class="perf-head-left">
                <div class="perf-title">Model Performance</div>
                <div class="perf-subtitle">
                    <b>Final model:</b> {info['model_name']}<br>
                    <b>Innovation:</b> {info['innovation']}<br>
                    <b>Evaluation:</b> {info['evaluation']}
                </div>
            </div>
            <div class="perf-meta">
                <span class="perf-chip">Temperature = {info['temperature']:.3f}</span>
                <span class="perf-chip">Threshold = {info['threshold']:.3f}</span>
                <span class="perf-chip">Final Test Result</span>
                <span class="perf-chip">Calibrated Inference</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2, gap="large")
    with c1:
        render_feature_metric_card(
            "Accuracy",
            fmt_pct(metrics["Accuracy"]),
            "Overall classification accuracy on the final test setting"
        )
    with c2:
        render_feature_metric_card(
            "ROC-AUC",
            fmt_pct(metrics["ROC-AUC"]),
            "Overall separability between pneumothorax and non-pneumothorax cases"
        )

    st.markdown('<div class="perf-row-gap"></div>', unsafe_allow_html=True)

    c3, c4, c5 = st.columns(3, gap="large")
    with c3:
        render_compact_metric_card(
            "Precision",
            fmt_pct(metrics["Precision"]),
            "Correctness of positive predictions"
        )
    with c4:
        render_compact_metric_card(
            "Recall",
            fmt_pct(metrics["Recall"]),
            "Sensitivity to positive cases"
        )
    with c5:
        render_compact_metric_card(
            "F1-Score",
            fmt_pct(metrics["F1-Score"]),
            "Balanced precision-recall performance"
        )

    st.markdown('<div class="perf-row-gap"></div>', unsafe_allow_html=True)

    c6, c7 = st.columns(2, gap="large")
    with c6:
        render_compact_metric_card(
            "Specificity",
            fmt_pct(metrics["Specificity"]),
            "Recognition of negative cases"
        )
    with c7:
        render_compact_metric_card(
            "PR-AUC",
            fmt_pct(metrics["PR-AUC"]),
            "Positive-class retrieval quality under imbalance"
        )

    st.markdown("</div>", unsafe_allow_html=True)

def render_feature_metric_card(label: str, value: str, note: str = ""):
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


def render_compact_metric_card(label: str, value: str, note: str = ""):
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
def summary_label(y_pred: int, narrative: str) -> str:
    if narrative == "indeterminate":
        return "Indeterminate"
    return "Suggestive of Pneumothorax" if y_pred == 1 else "Not Suggestive of Pneumothorax"

def render_decision_banner(decision: str, p_cal: float, confidence: str):
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

def render_report_sections(report_text: str):
    headings = [
        "Clinical context:",
        "Technique:",
        "Findings:",
        "Impression:",
        "Recommendations:",
        "Limitations:",
    ]

    parsed = {}
    current = None

    for raw_line in report_text.split("\n"):
        line = raw_line.strip()
        if not line:
            continue

        matched = None
        for h in headings:
            if line.startswith(h):
                matched = h
                break

        if matched is not None:
            current = matched.replace(":", "")
            content = line[len(matched):].strip()
            parsed[current] = content
        elif current is not None:
            parsed[current] += " " + line
        else:
            parsed.setdefault("Report", "")
            parsed["Report"] += " " + line

    for title, body in parsed.items():
        st.markdown(f'<div class="report-heading">{title}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="report-body">{body.strip()}</div>', unsafe_allow_html=True)

def resolve_case_image_path(case: dict, assets_dir: str | None = None):
    candidate_keys = [
        "image_path",
        "img_path",
        "path",
        "file_path",
        "png_path",
        "source_image_path",
        "retrieved_image_path",
    ]

    for key in candidate_keys:
        p = case.get(key)
        if not p:
            continue

        p = Path(str(p))

        if p.exists():
            return str(p)

        if assets_dir is not None:
            p2 = Path(assets_dir) / p
            if p2.exists():
                return str(p2)

    return None

def render_case_card(case: dict, idx: int, used_case_ids=None, assets_dir: str | None = None):
    used_case_ids = set(map(str, used_case_ids or []))

    case_id = case.get("case_id", f"Case {idx}")
    sim = case.get("sim", None)
    label = case.get("label", case.get("y_true", "N/A"))
    pred = case.get("pred_label", case.get("y_pred", "N/A"))

    sim_text = f"{sim:.4f}" if isinstance(sim, (int, float)) else "N/A"
    img_path = resolve_case_image_path(case, assets_dir=assets_dir)

    col_img, col_meta = st.columns([1.0, 1.35], gap="large")

    with col_img:
        if img_path and Path(img_path).exists():
            st.image(img_path, use_container_width=True)
        else:
            st.info("No displayable image found for this retrieved case.")

    with col_meta:
        st.markdown(f"**Retrieved Case {idx}:** {case_id}")
        st.markdown(f"**Similarity:** {sim_text}")
        st.markdown(f"**Reference label:** {label}")
        st.markdown(f"**Retrieved prediction:** {pred}")

        if str(case_id) in used_case_ids:
            st.success("Used in report")

def render_chunk_card(chunk: dict, used_chunk_ids=None):
    used_chunk_ids = set(map(str, used_chunk_ids or []))

    chunk_id = chunk.get("chunk_id", "N/A")
    tags = ", ".join(chunk.get("tags", [])) if chunk.get("tags") else "N/A"
    score = float(chunk.get("score", 0))
    sim = float(chunk.get("sim", 0))
    text = chunk.get("text", "")

    st.markdown(f"**Chunk ID:** {chunk_id}")
    st.markdown(f"**Tags:** {tags}")
    st.markdown(f"**Score:** {score:.4f} | **Similarity:** {sim:.4f}")

    if str(chunk_id) in used_chunk_ids:
        st.success("Used in report")

    st.write(text)

def render_badges(items):
    if items:
        st.markdown(
            "".join([f'<span class="badge">{x}</span>' for x in items]),
            unsafe_allow_html=True,
        )
    else:
        st.caption("None")

def render_rag_summary(payload: dict, report: dict):
    image_rag = payload.get("image_rag", {}) or {}
    text_rag = payload.get("text_rag", {}) or {}
    summary = image_rag.get("summary", {}) or {}
    ctx = image_rag.get("behaviour_context", {}) or {}
    ev = report.get("evidence", {}) or {}

    num_cases = summary.get("num_cases", 0)
    mean_sim = summary.get("mean_similarity", None)
    text_chunks = text_rag.get("evidence_chunks", []) or []
    used_count = len(ev.get("text_chunk_ids", []) or []) + len(ev.get("retrieved_case_ids", []) or [])

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        render_metric_card("Retrieved Cases", str(num_cases), "Top-k image retrieval results")

    with c2:
        render_metric_card(
            "Mean Similarity",
            f"{mean_sim:.3f}" if isinstance(mean_sim, (int, float)) else "N/A",
            f"State = {ctx.get('retrieval_state', 'n/a')}"
        )

    with c3:
        render_metric_card("Text Chunks", str(len(text_chunks)), "Retrieved evidence text")

    with c4:
        render_metric_card("Evidence Used", str(used_count), "Items cited by final report")

def render_prompt_block(title: str, content: str):
    with st.expander(title, expanded=False):
        if content:
            st.code(content, language="text")
        else:
            st.info("Not available.")

def render_pipeline_overview(payload: dict, report: dict):
    text_rag = payload.get("text_rag", {}) or {}
    image_rag = payload.get("image_rag", {}) or {}
    pred = payload.get("prediction", {}) or {}
    ev = report.get("evidence", {}) or {}

    st.markdown("**Pipeline Flow**")
    st.caption(
        "Input image → CNN inference → image retrieval → text retrieval → prompt assembly → LLM report → safety check"
    )

    render_rag_summary(payload, report)

    st.markdown("**Prediction Context**")
    st.json(
        {
            "y_pred": pred.get("y_pred"),
            "narrative_label": pred.get("narrative_label"),
            "confidence_band": pred.get("confidence_band"),
        },
        expanded=True,
    )

    st.markdown("**Text Retrieval Query**")
    st.code(text_rag.get("query", "N/A"), language="text")

    st.markdown("**Retrieval Scenario**")
    st.json(text_rag.get("scenario", {}), expanded=True)

    st.markdown("**Evidence Used in Final Report**")
    st.markdown("Text chunk IDs")
    render_badges(ev.get("text_chunk_ids", []))
    st.markdown("Retrieved case IDs")
    render_badges(ev.get("retrieved_case_ids", []))

    st.markdown("**Image Retrieval Behaviour Context**")
    st.json(image_rag.get("behaviour_context", {}), expanded=False)

def render_gradcam_panel(gradcam_overlay_path: str, note: str = ""):
    if not gradcam_overlay_path or not Path(gradcam_overlay_path).exists():
        return

    # st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Model Explainability</div>', unsafe_allow_html=True)
    st.image(gradcam_overlay_path, use_container_width=True)
    if note:
        st.caption(note)
    # st.markdown("</div>", unsafe_allow_html=True)

def main():
    inject_css()

    st.markdown(
        """
        <div class="top-hero">
            <div class="title-xl">PneumoAssist</div>
            <div class="subtitle">
                Lightweight pneumothorax detection with retrieval-augmented evidence and structured LLM reporting.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    default_assets = os.getenv("PNEUMO_ASSETS_DIR", str(Path(__file__).resolve().parent))
    with st.sidebar:
        st.markdown("## PneumoAssist")
        st.caption("Clinical decision-support interface")

        with st.expander("Advanced Settings", expanded=False):
            assets_dir = st.text_input("Assets directory", value=default_assets)
            topk_img = st.slider("Top-k image retrieval", min_value=3, max_value=10, value=5, step=1)
            topk_text = st.slider("Top-k text evidence", min_value=3, max_value=6, value=6, step=1)
            save_report = st.toggle("Save report JSON", value=True)

            defense_mode = st.toggle("Defense mode", value=True)
            show_prompt_tab = st.toggle("Show prompt tab", value=True)
            # show_payload_json = st.toggle("Show payload JSON", value=False)

    st.markdown(
        """
        <div class="disclaimer">
            This tool is for decision support only and does not replace clinical judgement,
            radiologist review, or formal diagnosis.
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_model_performance_section()
        

    if not Path(assets_dir).exists():
        st.error(f"Assets directory not found: {assets_dir}")
        st.stop()

    pipeline = load_pipeline(assets_dir)
    reporter = load_report_generator()

    left, right = st.columns([1.1, 1.4], gap="large")

    with left:
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
            img = Image.open(uploaded).convert("RGB")
            st.image(img, use_container_width=True)

        gradcam_slot = st.empty()

    if uploaded is not None and run_clicked:
        temp_path = pipeline.save_uploaded_bytes(uploaded.name, uploaded.getvalue())

        with st.status("Running pipeline...", expanded=True) as status:
            st.write("Loading image and executing inference...")
            report, payload, out_path = reporter.generate_pneumo_report(
                pipeline=pipeline,
                image_path=temp_path,
                topk_img=topk_img,
                topk_text=topk_text,
                save=save_report,
            )
            st.write("Retrieving similar cases and evidence chunks...")
            st.write("Generating structured report...")
            status.update(label="Analysis complete", state="complete", expanded=False)

        pred = payload["prediction"]
        y_pred = int(pred["y_pred"])
        p_cal = float(pred["p_calibrated"])
        threshold = float(pred["threshold"])
        confidence = pred["confidence_band"]
        narrative = pred["narrative_label"]
        decision_text = summary_label(y_pred, narrative)
        prompt_debug = report.get("prompt_debug", {})
        used_chunk_ids = report.get("evidence", {}).get("text_chunk_ids", [])
        used_case_ids = report.get("evidence", {}).get("retrieved_case_ids", [])
        explain = report.get("explainability", {}) or {}
        gradcam_overlay_path = explain.get("gradcam_overlay_path")
        gradcam_note = explain.get("gradcam_note", "")

        with gradcam_slot.container():
            render_gradcam_panel(gradcam_overlay_path, gradcam_note)

        with right:
            render_decision_banner(decision_text, p_cal, confidence)

            c1, c2, c3 = st.columns(3)
            with c1:
                render_metric_card(
                    "Final Decision",
                    decision_text,
                    "Threshold-aware narrative classification"
                )
            with c2:
                render_metric_card(
                    "Calibrated Probability",
                    f"{p_cal:.4f}",
                    f"Decision threshold = {threshold:.2f}"
                )
            with c3:
                render_metric_card(
                    "Confidence Band",
                    confidence.title(),
                    "Derived from calibrated decision margin"
                )

            overlay_path = report.get("visual_support", {}).get("overlay_path")
            if overlay_path and Path(overlay_path).exists():
                # st.markdown('<div class="section-card">', unsafe_allow_html=True)
                st.markdown('<div class="section-title">Visual Support</div>', unsafe_allow_html=True)
                st.image(overlay_path, use_container_width=True)
                note = report.get("visual_support", {}).get("overlay_note", "")
                if note:
                    st.caption(note)
                # st.markdown("</div>", unsafe_allow_html=True)

        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
            ["Structured Report", "RAG Overview", "Image Retrieval", "Text Evidence", "Prompt", "Export"]
        )

        with tab1:
            # st.markdown('<div class="glass">', unsafe_allow_html=True)
            st.subheader("Diagnostic Report")

            report_text = report["diagnostic_report"]

            headings = [
                "Clinical context:",
                "Technique:",
                "Findings:",
                "Impression:",
                "Recommendations:",
                "Limitations:",
            ]

            for h in headings:
                report_text = report_text.replace(h, f"\n{h}")

            parts = [p.strip() for p in report_text.split("\n") if p.strip()]

            for p in parts:
                st.markdown(f"- {p}")

            ev = report.get("evidence", {})
            txt_ids = ev.get("text_chunk_ids", [])
            case_ids = ev.get("retrieved_case_ids", [])
            st.markdown("**Evidence Trace**")
            if txt_ids:
                st.markdown("".join([f'<span class="badge">{x}</span>' for x in txt_ids]), unsafe_allow_html=True)
            if case_ids:
                st.markdown("".join([f'<span class="badge">{x}</span>' for x in case_ids]), unsafe_allow_html=True)

            if report.get("fail_safe"):
                st.warning(f"Fail-safe report used: {report.get('fail_reason', 'unknown')}")
            if report.get("fallback_mode") == "template":
                st.info("Template report was used because no live LLM API key was available or the API call failed.")
            if report.get("llm_error"):
                st.caption(f"LLM note: {report['llm_error']}")

        with tab2:
            # st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">RAG Overview</div>', unsafe_allow_html=True)

            render_pipeline_overview(payload, report)

            # st.markdown("</div>", unsafe_allow_html=True)

        with tab3:
            st.markdown('<div class="section-title">Retrieved Similar Cases</div>', unsafe_allow_html=True)

            sim_cases = payload.get("image_rag", {}).get("similar_cases", [])

            if sim_cases:
                for i, case in enumerate(sim_cases, start=1):
                    render_case_card(
                        case,
                        i,
                        used_case_ids=used_case_ids,
                        assets_dir=assets_dir,
                    )
                    st.divider()
            else:
                st.info("No similar cases available.")

            ctx = payload.get("image_rag", {}).get("behaviour_context", {})
            st.caption(
                f"Retrieval state: {ctx.get('retrieval_state', 'n/a')} | "
                f"Mean similarity: {ctx.get('mean_similarity', 'n/a')} | "
                f"Agreement rate: {ctx.get('agreement_rate', 'n/a')}"
            )

        with tab4:
            # st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Text Evidence</div>', unsafe_allow_html=True)

            text_chunks = payload.get("text_rag", {}).get("evidence_chunks", [])
            if text_chunks:
                for chunk in text_chunks:
                    render_chunk_card(chunk, used_chunk_ids=used_chunk_ids)
                    st.divider()
            else:
                st.info("No text evidence retrieved.")

            # st.markdown("</div>", unsafe_allow_html=True)

        with tab5:
            # st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Prompt Transparency</div>', unsafe_allow_html=True)

            if show_prompt_tab:
                render_prompt_block("System Prompt", prompt_debug.get("system_prompt", ""))
                render_prompt_block("User Prompt Template", prompt_debug.get("user_template", ""))
                render_prompt_block("Final User Prompt for This Case", prompt_debug.get("final_user_prompt", ""))
            else:
                st.info("Prompt display is disabled.")

            # st.markdown("</div>", unsafe_allow_html=True)

        with tab6:
            # st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Export</div>', unsafe_allow_html=True)

            export_obj = {
                "report": report,
                "payload": payload,
                "prompt_debug": prompt_debug,
                "saved_json_path": out_path,
            }
            export_str = json.dumps(export_obj, ensure_ascii=False, indent=2)

            st.download_button(
                "Download analysis JSON",
                data=export_str,
                file_name=f"{Path(uploaded.name).stem}_analysis.json",
                mime="application/json",
                use_container_width=True,
            )

            if out_path:
                st.success(f"Saved to: {out_path}")

            # if show_payload_json:
            #     st.markdown("**Payload JSON**")
            #     st.json(payload, expanded=False)

            # st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
