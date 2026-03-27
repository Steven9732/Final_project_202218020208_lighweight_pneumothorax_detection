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


def inject_css():
    st.markdown(
        """
        <style>
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

def render_case_card(case: dict, idx: int):
    case_id = case.get("case_id", f"Case {idx}")
    sim = case.get("sim", None)
    label = case.get("label", case.get("y_true", "N/A"))
    pred = case.get("pred_label", case.get("y_pred", "N/A"))

    sim_text = f"{sim:.4f}" if isinstance(sim, (int, float)) else "N/A"

    st.markdown(
        f"""
        <div class="case-card">
            <div class="case-title">Retrieved Case {idx}: {case_id}</div>
            <div class="case-meta">
                Similarity: <b>{sim_text}</b><br>
                Reference label: <b>{label}</b><br>
                Retrieved prediction: <b>{pred}</b>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def render_chunk_card(chunk: dict):
    chunk_id = chunk.get("chunk_id", "N/A")
    tags = ", ".join(chunk.get("tags", [])) if chunk.get("tags") else "N/A"
    score = float(chunk.get("score", 0))
    sim = float(chunk.get("sim", 0))
    text = chunk.get("text", "")

    st.markdown(
        f"""
        <div class="chunk-card">
            <div class="case-title">{chunk_id}</div>
            <div class="case-meta">
                Tags: <b>{tags}</b><br>
                Score: <b>{score:.4f}</b> &nbsp;&nbsp;|&nbsp;&nbsp;
                Similarity: <b>{sim:.4f}</b>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write(text)


def main():
    inject_css()

    st.markdown(
        """
        <div class="glass">
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

    st.markdown(
        """
        <div class="disclaimer">
            This tool is for decision support only and does not replace clinical judgement,
            radiologist review, or formal diagnosis.
        </div>
        """,
        unsafe_allow_html=True,
    )
        

    if not Path(assets_dir).exists():
        st.error(f"Assets directory not found: {assets_dir}")
        st.stop()

    pipeline = load_pipeline(assets_dir)
    reporter = load_report_generator()

    left, right = st.columns([1.1, 1.4], gap="large")

    with left:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Input Image</div>', unsafe_allow_html=True)

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

        st.markdown("</div>", unsafe_allow_html=True)

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
                st.markdown('<div class="section-card">', unsafe_allow_html=True)
                st.markdown('<div class="section-title">Visual Support</div>', unsafe_allow_html=True)
                st.image(overlay_path, use_container_width=True)
                note = report.get("visual_support", {}).get("overlay_note", "")
                if note:
                    st.caption(note)
                st.markdown("</div>", unsafe_allow_html=True)

        tab1, tab2, tab3, tab4 = st.tabs(["Structured Report", "Image Retrieval", "Text Evidence", "Export"])

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
            st.markdown("</div>", unsafe_allow_html=True)

        with tab2:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Retrieved Similar Cases</div>', unsafe_allow_html=True)

            sim_cases = payload.get("image_rag", {}).get("similar_cases", [])
            if sim_cases:
                for i, case in enumerate(sim_cases, start=1):
                    render_case_card(case, i)
            else:
                st.info("No similar cases available.")

            ctx = payload.get("image_rag", {}).get("behaviour_context", {})
            st.caption(
                f"Retrieval state: {ctx.get('retrieval_state', 'n/a')} | "
                f"Mean similarity: {ctx.get('mean_similarity', 'n/a')} | "
                f"Agreement rate: {ctx.get('agreement_rate', 'n/a')}"
            )

            st.markdown("</div>", unsafe_allow_html=True)

        with tab3:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Text Evidence</div>', unsafe_allow_html=True)

            text_chunks = payload.get("text_rag", {}).get("evidence_chunks", [])
            if text_chunks:
                for chunk in text_chunks:
                    render_chunk_card(chunk)
                    st.divider()
            else:
                st.info("No text evidence retrieved.")

            st.markdown("</div>", unsafe_allow_html=True)

        with tab4:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Export</div>', unsafe_allow_html=True)

            export_obj = {
                "report": report,
                "payload": payload,
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

            st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
