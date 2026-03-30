from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
PERFORMANCE_DIR = BASE_DIR / "performance"

from ui_components import (
    fmt_pct,
    render_badges,
    render_case_card,
    render_chunk_card,
    render_compact_metric_card,
    render_feature_metric_card,
    render_metric_card,
    render_prompt_block,
    render_report_sections,
)
from ui_constants import (
    FINAL_MODEL_INFO,
    FINAL_MODEL_METRICS,
    PERFORMANCE_IMAGE_DIR,
    ROC_CURVE_FILE,
    PR_CURVE_FILE,
    LOSS_CURVE_FILE,
    LEARNING_CURVE_FILE,
)


def render_hero(page: str = "Diagnosis") -> None:
    subtitle_map = {
        "Diagnosis": "Upload chest X-ray images and generate calibrated prediction, explainability, and structured report.",
        "Model Performance": "Review summary metrics together with ROC and PR analysis for the final calibrated model.",
        "Training Dynamics": "Inspect convergence behaviour, loss trend, and learning stability across training.",
    }

    st.markdown(
        f"""
        <div class="top-hero">
            <div class="title-xl">PneumoAssist</div>
            <div class="subtitle">
                {subtitle_map.get(page, subtitle_map["Diagnosis"])}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_disclaimer() -> None:
    st.markdown(
        """
        <div class="disclaimer">
            This tool is for decision support only and does not replace clinical judgement,
            radiologist review, or formal diagnosis.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_model_performance_section() -> None:
    info = FINAL_MODEL_INFO
    metrics = FINAL_MODEL_METRICS

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
            "Overall classification accuracy on the final test setting",
        )
    with c2:
        render_feature_metric_card(
            "ROC-AUC",
            fmt_pct(metrics["ROC-AUC"]),
            "Overall separability between pneumothorax and non-pneumothorax cases",
        )

    st.markdown('<div class="perf-row-gap"></div>', unsafe_allow_html=True)

    c3, c4, c5 = st.columns(3, gap="large")
    with c3:
        render_compact_metric_card(
            "Precision",
            fmt_pct(metrics["Precision"]),
            "Correctness of positive predictions",
        )
    with c4:
        render_compact_metric_card(
            "Recall",
            fmt_pct(metrics["Recall"]),
            "Sensitivity to positive cases",
        )
    with c5:
        render_compact_metric_card(
            "F1-Score",
            fmt_pct(metrics["F1-Score"]),
            "Balanced precision-recall performance",
        )

    st.markdown('<div class="perf-row-gap"></div>', unsafe_allow_html=True)

    c6, c7 = st.columns(2, gap="large")
    with c6:
        render_compact_metric_card(
            "Specificity",
            fmt_pct(metrics["Specificity"]),
            "Recognition of negative cases",
        )
    with c7:
        render_compact_metric_card(
            "PR-AUC",
            fmt_pct(metrics["PR-AUC"]),
            "Positive-class retrieval quality under imbalance",
        )


def render_rag_summary(payload: dict[str, Any], report: dict[str, Any]) -> None:
    image_rag = payload.get("image_rag", {}) or {}
    text_rag = payload.get("text_rag", {}) or {}
    summary = image_rag.get("summary", {}) or {}
    ctx = image_rag.get("behaviour_context", {}) or {}
    evidence = report.get("evidence", {}) or {}

    num_cases = summary.get("num_cases", 0)
    mean_sim = summary.get("mean_similarity")
    text_chunks = text_rag.get("evidence_chunks", []) or []
    used_count = len(evidence.get("text_chunk_ids", []) or []) + len(evidence.get("retrieved_case_ids", []) or [])

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_metric_card("Retrieved Cases", str(num_cases), "Top-k image retrieval results")
    with c2:
        render_metric_card(
            "Mean Similarity",
            f"{mean_sim:.3f}" if isinstance(mean_sim, (int, float)) else "N/A",
            f"State = {ctx.get('retrieval_state', 'n/a')}",
        )
    with c3:
        render_metric_card("Text Chunks", str(len(text_chunks)), "Retrieved evidence text")
    with c4:
        render_metric_card("Evidence Used", str(used_count), "Items cited by final report")


def render_pipeline_overview(payload: dict[str, Any], report: dict[str, Any]) -> None:
    text_rag = payload.get("text_rag", {}) or {}
    image_rag = payload.get("image_rag", {}) or {}
    prediction = payload.get("prediction", {}) or {}
    evidence = report.get("evidence", {}) or {}

    st.markdown("**Pipeline Flow**")
    st.caption(
        "Input image → CNN inference → image retrieval → text retrieval → prompt assembly → LLM report → safety check"
    )

    render_rag_summary(payload, report)

    st.markdown("**Prediction Context**")
    st.json(
        {
            "y_pred": prediction.get("y_pred"),
            "narrative_label": prediction.get("narrative_label"),
            "confidence_band": prediction.get("confidence_band"),
        },
        expanded=True,
    )

    st.markdown("**Text Retrieval Query**")
    st.code(text_rag.get("query", "N/A"), language="text")

    st.markdown("**Retrieval Scenario**")
    st.json(text_rag.get("scenario", {}), expanded=True)

    st.markdown("**Evidence Used in Final Report**")
    st.markdown("Text chunk IDs")
    render_badges(evidence.get("text_chunk_ids", []))
    st.markdown("Retrieved case IDs")
    render_badges(evidence.get("retrieved_case_ids", []))

    st.markdown("**Image Retrieval Behaviour Context**")
    st.json(image_rag.get("behaviour_context", {}), expanded=False)


def render_structured_report_tab(report: dict[str, Any]) -> None:
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


def render_rag_overview_tab(payload: dict[str, Any], report: dict[str, Any]) -> None:
    st.markdown('<div class="section-title">RAG Overview</div>', unsafe_allow_html=True)
    render_pipeline_overview(payload, report)


def render_image_retrieval_tab(payload: dict[str, Any], used_case_ids: list[str], assets_dir: str) -> None:
    st.markdown('<div class="section-title">Retrieved Similar Cases</div>', unsafe_allow_html=True)
    similar_cases = payload.get("image_rag", {}).get("similar_cases", [])

    if similar_cases:
        for idx, case in enumerate(similar_cases, start=1):
            render_case_card(case, idx, used_case_ids=used_case_ids, assets_dir=assets_dir)
            st.divider()
    else:
        st.info("No similar cases available.")

    ctx = payload.get("image_rag", {}).get("behaviour_context", {})
    st.caption(
        f"Retrieval state: {ctx.get('retrieval_state', 'n/a')} | "
        f"Mean similarity: {ctx.get('mean_similarity', 'n/a')} | "
        f"Agreement rate: {ctx.get('agreement_rate', 'n/a')}"
    )


def render_text_evidence_tab(payload: dict[str, Any], used_chunk_ids: list[str]) -> None:
    st.markdown('<div class="section-title">Text Evidence</div>', unsafe_allow_html=True)
    text_chunks = payload.get("text_rag", {}).get("evidence_chunks", [])

    if text_chunks:
        for chunk in text_chunks:
            render_chunk_card(chunk, used_chunk_ids=used_chunk_ids)
            st.divider()
    else:
        st.info("No text evidence retrieved.")


def render_prompt_tab(prompt_debug: dict[str, Any], show_prompt_tab: bool) -> None:
    st.markdown('<div class="section-title">Prompt Transparency</div>', unsafe_allow_html=True)
    if not show_prompt_tab:
        st.info("Prompt display is disabled.")
        return

    render_prompt_block("System Prompt", prompt_debug.get("system_prompt", ""))
    render_prompt_block("User Prompt Template", prompt_debug.get("user_template", ""))
    render_prompt_block("Final User Prompt for This Case", prompt_debug.get("final_user_prompt", ""))


def render_export_tab(
    report: dict[str, Any],
    payload: dict[str, Any],
    prompt_debug: dict[str, Any],
    saved_json_path: str | None,
    uploaded_name: str,
) -> None:
    st.markdown('<div class="section-title">Export</div>', unsafe_allow_html=True)

    export_obj = {
        "report": report,
        "payload": payload,
        "prompt_debug": prompt_debug,
        "saved_json_path": saved_json_path,
    }
    export_str = json.dumps(export_obj, ensure_ascii=False, indent=2)

    st.download_button(
        "Download analysis JSON",
        data=export_str,
        file_name=f"{Path(uploaded_name).stem}_analysis.json",
        mime="application/json",
        use_container_width=True,
    )

    if saved_json_path:
        st.success(f"Saved to: {saved_json_path}")

def render_image_if_exists(image_path: Path, empty_message: str) -> None:
    if image_path.exists():
        st.image(str(image_path), use_container_width=True)
    else:
        st.info(empty_message)

def render_model_performance_page() -> None:
    # st.markdown('<div class="section-title">Model Performance</div>', unsafe_allow_html=True)

    render_model_performance_section()

    st.markdown("### Confusion Matrix")
    render_curve_image("cm.png", "Confusion matrix image not found.")

    st.markdown("### ROC Curve")
    render_curve_image("roc_curve.png", "ROC curve image not found.")

    st.markdown("### PR Curve")
    render_curve_image("pr_curve.png", "PR curve image not found.")

def render_training_dynamics_page() -> None:
    st.markdown('<div class="section-title">Training Dynamics</div>', unsafe_allow_html=True)

    st.markdown("### Loss Curve")
    render_curve_image("loss_curve.png", "Loss curve image not found.")

    st.markdown("### Learning Curve")
    render_curve_image("learning_curve.png", "Learning curve image not found.")

def render_curve_image(image_name: str, empty_message: str) -> None:
    image_path = PERFORMANCE_DIR / image_name

    # st.caption(f"Looking for: {image_path}")
    if image_path.exists():
        st.image(str(image_path), use_container_width=True)
    else:
        st.info(empty_message)