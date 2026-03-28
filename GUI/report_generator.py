from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

import requests


SYSTEM = """You are a consultant radiologist drafting a safe AI-assisted diagnostic report for pneumothorax screening.

Write one concise professional English report using these inline headings:
Clinical context, Technique, Findings, Impression, Recommendations, Limitations.

Use the inputs as follows:
- prediction.narrative_label and prediction.confidence_band determine the overall wording strength.
- text_rag.evidence_chunks may be used ONLY for safe wording, limitation statements, uncertainty wording, and generic recommendations.
- image_rag.behaviour_context may be used ONLY to make the wording more cautious or more stable; it MUST NOT be treated as direct clinical evidence.

Strict rules:
1. For positive outputs, prefer wording such as 'suggestive of pneumothorax'.
2. For negative outputs, prefer wording such as 'not suggestive of pneumothorax'.
3. For borderline confidence or conflicting/limited contextual support, prefer 'indeterminate' or clearly cautious wording.
4. Never infer laterality, localization, size, extent, pleural line, collapse, or tension physiology.
5. It is acceptable to state that localization, laterality, size, extent, or tension assessment is not available from this output.
6. Do NOT mention model, probability, threshold, calibration, retrieval, RAG, chunk, case ID, prompt, code, or JSON.
7. Do NOT mention attached images, overlays, filenames, or file paths in the diagnostic report text.
8. Keep the report concise, clinically styled, and faithful to the provided evidence only.

Output MUST be valid JSON with exactly these keys:
{
  "diagnostic_report": "...",
  "evidence": {
    "text_chunk_ids": [],
    "retrieved_case_ids": []
  }
}
"""

USER_TMPL = """Input JSON:
{payload}

Return ONLY JSON in this exact format:
{{
  "diagnostic_report": "...",
  "evidence": {{
    "text_chunk_ids": [],
    "retrieved_case_ids": []
  }}
}}
"""


def extract_json_object(s: str) -> str:
    s = s.strip()
    l = s.find("{")
    r = s.rfind("}")
    if l == -1 or r == -1 or r <= l:
        raise ValueError("No JSON object found in LLM output.")
    return s[l : r + 1]


def normalize_text_for_safety(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower())


def unsupported_detail_flags(text: str) -> dict:
    s = normalize_text_for_safety(text)
    flags = {
        "unsupported_localization": False,
        "unsupported_size": False,
        "unsupported_tension": False,
        "unsupported_imaging_sign": False,
        "unsupported_other": False,
    }

    safe_localization_patterns = [
        r"no localization",
        r"localization is not available",
        r"laterality is not available",
        r"no laterality",
    ]
    safe_size_patterns = [
        r"no size assessment",
        r"size is not available",
        r"no extent assessment",
        r"extent is not available",
    ]
    safe_tension_patterns = [
        r"no tension assessment",
        r"tension assessment is not available",
    ]

    bad_localization_patterns = [
        r"\bleft pneumothorax\b",
        r"\bright pneumothorax\b",
        r"\bapical pneumothorax\b",
        r"\bbasal pneumothorax\b",
        r"\bleft-sided\b",
        r"\bright-sided\b",
    ]
    bad_size_patterns = [
        r"\bsmall pneumothorax\b",
        r"\blarge pneumothorax\b",
        r"\bmoderate pneumothorax\b",
        r"\btrace pneumothorax\b",
        r"\bsize of the pneumothorax\b",
        r"\bextent of the pneumothorax\b",
    ]
    bad_tension_patterns = [
        r"\btension pneumothorax\b",
        r"\bfindings suggest tension\b",
    ]
    bad_sign_patterns = [
        r"\bvisible pleural line\b",
        r"\bpleural line is seen\b",
        r"\blung collapse\b",
        r"\bpartial collapse\b",
    ]

    if any(re.search(p, s) for p in bad_localization_patterns):
        if not any(re.search(p, s) for p in safe_localization_patterns):
            flags["unsupported_localization"] = True
    if any(re.search(p, s) for p in bad_size_patterns):
        if not any(re.search(p, s) for p in safe_size_patterns):
            flags["unsupported_size"] = True
    if any(re.search(p, s) for p in bad_tension_patterns):
        if not any(re.search(p, s) for p in safe_tension_patterns):
            flags["unsupported_tension"] = True
    if any(re.search(p, s) for p in bad_sign_patterns):
        flags["unsupported_imaging_sign"] = True

    flags["unsupported_other"] = any(
        [
            flags["unsupported_localization"],
            flags["unsupported_size"],
            flags["unsupported_tension"],
            flags["unsupported_imaging_sign"],
        ]
    )
    return flags


def word_count_en(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?", str(text)))


def soft_too_long(text: str, max_words: int = 110) -> bool:
    return word_count_en(text) > max_words


def fail_safe_report(payload: dict, reason: str) -> dict:
    msg = (
        "Clinical context: AI-assisted pneumothorax screening output could not be safely summarized. "
        "Technique: Image-based automated screening output only. "
        "Findings: A reliable concise diagnostic summary could not be generated from the current processing pathway. "
        "Impression: Indeterminate automated report state. "
        "Recommendations: Recommend radiologist review and clinical correlation. "
        "Limitations: This fallback report was issued because the automated reporting safeguard was triggered."
    )

    visual = (payload or {}).get("visual_support", {}) or {}
    return {
        "diagnostic_report": msg,
        "evidence": {
            "text_chunk_ids": [],
            "retrieved_case_ids": [],
        },
        "visual_support": {
            "overlay_available": bool(visual.get("overlay_available", False)),
            "overlay_path": visual.get("overlay_path"),
            "overlay_filename": visual.get("overlay_filename"),
            "overlay_note": visual.get("overlay_note", ""),
        },
        "fail_safe": True,
        "fail_reason": str(reason),
    }


def evidence_ok(report: dict, payload: dict) -> bool:
    ev = report.get("evidence", {}) or {}
    txt_ids = ev.get("text_chunk_ids", []) or []
    case_ids = ev.get("retrieved_case_ids", []) or []

    if "text_rag" in payload:
        allowed_txt = {c["chunk_id"] for c in payload["text_rag"].get("evidence_chunks", [])}
        if not (isinstance(txt_ids, list) and set(txt_ids).issubset(allowed_txt)):
            return False
    elif txt_ids != []:
        return False

    if "image_rag" in payload:
        allowed_case = set(map(str, payload["image_rag"].get("retrieved_case_ids", []) or []))
        if not (isinstance(case_ids, list) and set(map(str, case_ids)).issubset(allowed_case)):
            return False
    elif case_ids != []:
        return False

    return True


def sanitize_report_structure(report: dict, payload: dict) -> dict:
    text = str((report or {}).get("diagnostic_report", "")).strip()
    ev = (report or {}).get("evidence", {}) or {}
    txt_ids = ev.get("text_chunk_ids", []) or []
    case_ids = ev.get("retrieved_case_ids", []) or []

    cleaned = {
        "diagnostic_report": text,
        "evidence": {
            "text_chunk_ids": list(map(str, txt_ids)) if isinstance(txt_ids, list) else [],
            "retrieved_case_ids": list(map(str, case_ids)) if isinstance(case_ids, list) else [],
        },
    }

    if not evidence_ok(cleaned, payload):
        cleaned["evidence"] = {"text_chunk_ids": [], "retrieved_case_ids": []}
    return cleaned


def build_user_prompt(payload: dict) -> str:
    payload_json = json.dumps(payload, ensure_ascii=False, indent=2)
    return USER_TMPL.format(payload=payload_json)

def build_prompt_debug(payload: dict) -> dict:
    return {
        "system_prompt": SYSTEM,
        "user_template": USER_TMPL,
        "final_user_prompt": build_user_prompt(payload),
    }

def template_report_from_payload(payload: dict) -> dict:
    pred = payload["prediction"]
    label = pred["narrative_label"]
    conf = pred["confidence_band"]

    if label == "positive":
        findings = "Findings: Automated screening output is suggestive of pneumothorax."
        impression = "Impression: Suggestive of pneumothorax on this automated screening result, pending qualified human review."
    elif label == "negative":
        findings = "Findings: Automated screening output is not suggestive of pneumothorax."
        impression = "Impression: Not suggestive of pneumothorax on this automated screening result, with continued need for human review."
    else:
        findings = "Findings: Automated screening output is near the decision boundary and should be treated as indeterminate."
        impression = "Impression: Indeterminate automated screening result."

    if conf == "borderline":
        rec = "Recommendations: Prioritise human review and do not use this output alone."
    else:
        rec = "Recommendations: Recommend radiologist review and correlation with the full clinical context."

    text = (
        "Clinical context: AI-assisted chest X-ray pneumothorax screening. "
        "Technique: Image-based automated classification output with retrieval-supported wording constraints. "
        f"{findings} "
        f"{impression} "
        f"{rec} "
        "Limitations: This output is classification-only and does not provide laterality, localization, size, extent, pleural line assessment, collapse assessment, or tension assessment."
    )

    text_ids = [c["chunk_id"] for c in payload.get("text_rag", {}).get("evidence_chunks", [])[:3]]
    case_ids = payload.get("image_rag", {}).get("retrieved_case_ids", [])[:3]
    return {
        "diagnostic_report": text,
        "evidence": {
            "text_chunk_ids": list(map(str, text_ids)),
            "retrieved_case_ids": list(map(str, case_ids)),
        },
        "fallback_mode": "template",
    }


class ReportGenerator:
    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        url: str | None = None,
    ):
        self.api_key = api_key or "sk-6de58e90d64a47ad9163e14cc50067aa直接写这里"
        self.model = model or "deepseek-chat"
        self.url = url or "https://api.deepseek.com/chat/completions"
    def call_llm(self, system: str, user: str) -> str:
        if not self.api_key:
            raise RuntimeError("Missing DEEPSEEK_API_KEY / OPENAI_API_KEY in environment.")
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "temperature": 0,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        r = requests.post(self.url, json=payload, headers=headers, timeout=180)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

    def generate_report(self, payload: dict) -> dict:
        user = build_user_prompt(payload)
        raw = self.call_llm(SYSTEM, user)

        text = ""
        evidence = {"text_chunk_ids": [], "retrieved_case_ids": []}

        try:
            obj = extract_json_object(raw)
            rep = json.loads(obj)
            if isinstance(rep, dict):
                text = str(rep.get("diagnostic_report", "")).strip()
                ev = rep.get("evidence", {}) or {}
                txt_ids = ev.get("text_chunk_ids", []) or []
                case_ids = ev.get("retrieved_case_ids", []) or []
                evidence = {
                    "text_chunk_ids": list(map(str, txt_ids)) if isinstance(txt_ids, list) else [],
                    "retrieved_case_ids": list(map(str, case_ids)) if isinstance(case_ids, list) else [],
                }
        except Exception:
            pass

        if not text:
            text = str(raw).strip()

        return {"diagnostic_report": text, "evidence": evidence}

    def generate_report_safe(self, payload: dict) -> dict:
        try:
            rep = self.generate_report(payload)
        except Exception as e:
            rep = template_report_from_payload(payload)
            rep["llm_error"] = str(e)

        rep = sanitize_report_structure(rep, payload)
        text = str(rep.get("diagnostic_report", "")).strip()
        if not text:
            return fail_safe_report(payload, "empty diagnostic_report")

        flags = unsupported_detail_flags(text)
        need_safety_rewrite = any(flags.values())
        need_length_rewrite = soft_too_long(text)

        if need_safety_rewrite or need_length_rewrite:
            if self.api_key:
                rewrite_instruction = """
                Rewrite the report to be concise, professional, and safe.
                Do NOT invent localization, laterality, size, extent, tension physiology, pleural line, collapse, or other unsupported imaging details.
                You may explicitly say that these assessments are not available from the current output.
                Return JSON only in this exact format:
                {
                "diagnostic_report": "...",
                "evidence": {
                    "text_chunk_ids": [],
                    "retrieved_case_ids": []
                }
                }
                """
                fix_user = rewrite_instruction + "\n\nOriginal report:\n" + text
                try:
                    raw2 = self.call_llm(SYSTEM, fix_user)
                    obj2 = extract_json_object(raw2)
                    rep2 = json.loads(obj2)
                    if isinstance(rep2, dict):
                        rep = {
                            "diagnostic_report": str(rep2.get("diagnostic_report", "")).strip(),
                            "evidence": rep2.get("evidence", {"text_chunk_ids": [], "retrieved_case_ids": []}),
                        }
                        rep = sanitize_report_structure(rep, payload)
                except Exception:
                    return fail_safe_report(payload, "unsafe or overlong report and rewrite failed")
            else:
                rep = template_report_from_payload(payload)

        if any(unsupported_detail_flags(rep.get("diagnostic_report", "")).values()):
            return fail_safe_report(payload, "unsafe after rewrite/template")

        return rep

    def generate_pneumo_report(
        self,
        pipeline,
        image_path: str,
        topk_img: int = 5,
        topk_text: int = 6,
        save: bool = True,
    ):
        payload = pipeline.build_payload(image_path, topk_img=topk_img, topk_text=topk_text)
        report = self.generate_report_safe(payload)

        report["prompt_debug"] = build_prompt_debug(payload)

        visual = payload.get("visual_support", {}) or {}
        if "visual_support" not in report:
            report["visual_support"] = {
                "overlay_available": bool(visual.get("overlay_available", False)),
                "overlay_path": visual.get("overlay_path"),
                "overlay_filename": visual.get("overlay_filename"),
                "overlay_note": visual.get("overlay_note", ""),
            }

        out_path = None
        if save:
            report_dir = Path(pipeline.report_dir) / "reports"
            report_dir.mkdir(parents=True, exist_ok=True)
            stem = Path(image_path).stem
            ts = time.strftime("%Y%m%d_%H%M%S")
            out_path = report_dir / f"{stem}_{ts}.json"
            save_obj = {
                "image_path": image_path,
                "payload": payload,
                "report": report,
            }
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(save_obj, f, ensure_ascii=False, indent=2)
            out_path = str(out_path)

        return report, payload, out_path
