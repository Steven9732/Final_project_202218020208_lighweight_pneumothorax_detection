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
    page_title="PneuInsight Studio",
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
                radial-gradient(circle at top left, rgba(65, 105, 225, 0.10), transparent 28%),
                radial-gradient(circle at top right, rgba(0, 191, 166, 0.10), transparent 22%),
                linear-gradient(180deg, #0b1220 0%, #0f172a 45%, #111827 100%);
            color: #f8fafc;
        }
        [data-testid="stHeader"] {background: rgba(0,0,0,0);}
        [data-testid="stSidebar"] {
            background: rgba(15, 23, 42, 0.82);
            border-right: 1px solid rgba(148, 163, 184, 0.15);
        }
        .glass {
            background: rgba(15, 23, 42, 0.72);
            border: 1px solid rgba(148, 163, 184, 0.16);
            border-radius: 22px;
            padding: 20px 22px;
            backdrop-filter: blur(10px);
            box-shadow: 0 10px 35px rgba(0,0,0,0.22);
            margin-bottom: 16px;
        }
        .metric-card {
            background: linear-gradient(180deg, rgba(30, 41, 59, 0.88), rgba(15, 23, 42, 0.95));
            border-radius: 18px;
            padding: 16px 18px;
            border: 1px solid rgba(148, 163, 184, 0.14);
            min-height: 114px;
        }
        .metric-label {
            color: #94a3b8;
            font-size: 0.92rem;
            margin-bottom: 8px;
        }
        .metric-value {
            color: #f8fafc;
            font-size: 1.6rem;
            font-weight: 700;
            line-height: 1.2;
        }
        .muted {
            color: #cbd5e1;
            font-size: 0.95rem;
        }
        .badge {
            display:inline-block;
            padding: 0.3rem 0.7rem;
            border-radius: 999px;
            border: 1px solid rgba(148, 163, 184, 0.18);
            margin-right: 0.4rem;
            margin-bottom: 0.4rem;
            background: rgba(30, 41, 59, 0.85);
            color: #e2e8f0;
            font-size: 0.84rem;
        }
        .title-xl {
            font-size: 2.2rem;
            font-weight: 800;
            color: #f8fafc;
            letter-spacing: -0.02em;
        }
        .subtitle {
            color: #cbd5e1;
            font-size: 1.02rem;
            margin-top: 0.4rem;
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


def main():
    inject_css()

    st.markdown(
        """
        <div class="glass">
            <div class="title-xl">PneuInsight Studio</div>
            <div class="subtitle">
                Lightweight pneumothorax detection with retrieval-augmented evidence and structured LLM reporting.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    default_assets = os.getenv("PNEUMO_ASSETS_DIR", str(Path(__file__).resolve().parent))
    with st.sidebar:
        st.markdown("## Configuration")
        assets_dir = st.text_input("Assets directory", value=default_assets)
        topk_img = st.slider("Top-k image retrieval", min_value=3, max_value=10, value=5, step=1)
        topk_text = st.slider("Top-k text evidence", min_value=3, max_value=6, value=6, step=1)
        save_report = st.toggle("Save report JSON", value=True)
        st.caption("Set `DEEPSEEK_API_KEY` to enable live LLM generation. Without it, the app uses a safe fallback template report.")

    if not Path(assets_dir).exists():
        st.error(f"Assets directory not found: {assets_dir}")
        st.stop()

    pipeline = load_pipeline(assets_dir)
    reporter = load_report_generator()

    left, right = st.columns([1.1, 1.4], gap="large")

    with left:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.subheader("Input")
        uploaded = st.file_uploader(
            "Upload chest X-ray",
            type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
            accept_multiple_files=False,
        )
        run_clicked = st.button("Run Analysis", type="primary", use_container_width=True, disabled=uploaded is None)
        st.markdown("</div>", unsafe_allow_html=True)

        if uploaded is not None:
            img = Image.open(uploaded).convert("RGB")
            st.markdown('<div class="glass">', unsafe_allow_html=True)
            st.subheader("Original Image")
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

        with right:
            st.markdown('<div class="glass">', unsafe_allow_html=True)
            st.subheader("Result Summary")
            c1, c2, c3 = st.columns(3)
            with c1:
                render_metric_card("Final Decision", summary_label(y_pred, narrative), "Threshold-aware narrative label")
            with c2:
                render_metric_card("Calibrated Probability", f"{p_cal:.4f}", f"Decision threshold = {threshold:.2f}")
            with c3:
                render_metric_card("Confidence Band", confidence.title(), "Derived from |p - threshold|")
            st.markdown("</div>", unsafe_allow_html=True)

            overlay_path = report.get("visual_support", {}).get("overlay_path")
            if overlay_path and Path(overlay_path).exists():
                st.markdown('<div class="glass">', unsafe_allow_html=True)
                st.subheader("Model Visual Support")
                st.image(overlay_path, use_container_width=True)
                note = report.get("visual_support", {}).get("overlay_note", "")
                if note:
                    st.caption(note)
                st.markdown("</div>", unsafe_allow_html=True)

        tab1, tab2, tab3, tab4 = st.tabs(["Structured Report", "Image Retrieval", "Text Evidence", "Export"])

        with tab1:
            st.markdown('<div class="glass">', unsafe_allow_html=True)
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
            st.markdown('<div class="glass">', unsafe_allow_html=True)
            st.subheader("Retrieved Similar Cases")
            sim_cases = payload.get("image_rag", {}).get("similar_cases", [])
            if sim_cases:
                sim_df = pd.DataFrame(sim_cases)
                st.dataframe(sim_df, use_container_width=True, hide_index=True)
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
            st.markdown('<div class="glass">', unsafe_allow_html=True)
            st.subheader("Text Evidence Chunks")
            text_chunks = payload.get("text_rag", {}).get("evidence_chunks", [])
            if text_chunks:
                for chunk in text_chunks:
                    st.markdown(
                        f"**{chunk['chunk_id']}**  \n"
                        f"Tags: {', '.join(chunk.get('tags', []))}  \n"
                        f"Score: {chunk.get('score', 0):.4f} | Similarity: {chunk.get('sim', 0):.4f}"
                    )
                    st.write(chunk["text"])
                    st.divider()
            else:
                st.info("No text evidence retrieved.")
            st.markdown("</div>", unsafe_allow_html=True)

        with tab4:
            st.markdown('<div class="glass">', unsafe_allow_html=True)
            st.subheader("Export")
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
