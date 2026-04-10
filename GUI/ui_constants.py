from __future__ import annotations

APP_TITLE = "PneumoAssist: Pneumothorax Detection & Reporting"
APP_ICON = "🫁"
APP_LAYOUT = "wide"
APP_SIDEBAR_STATE = "expanded"

FINAL_MODEL_INFO = {
    "model_name": "ConvNeXtV2 + UNet++",
    "innovation": "Lesion-Sensitive Multi-Scale Fusion (LSMF)",
    "evaluation": "Temp+Threshold model",
    "temperature": 0.930,
    "threshold": 0.810,
}

FINAL_MODEL_METRICS = {
    "Accuracy": 92.03,
    "Precision": 85.77,
    "Recall": 76.78,
    "F1-Score": 81.03,
    "Specificity": 96.38,
    "ROC-AUC": 96.24,
    "PR-AUC": 89.72,
}

REPORT_HEADINGS = [
    "Clinical context:",
    "Technique:",
    "Findings:",
    "Impression:",
    "Recommendations:",
    "Limitations:",
]

CASE_IMAGE_KEYS = [
    "image_path",
    "img_path",
    "path",
    "file_path",
    "png_path",
    "source_image_path",
    "retrieved_image_path",
]

NAVIGATION_PAGES = [
    "Diagnosis",
    "Model Performance",
    "Training Dynamics",
]

PERFORMANCE_IMAGE_DIR = "performance"
PERFORMANCE_SUBDIR = "baseline_lsmf"

ROC_CURVE_FILE = "roc_curve.png"
PR_CURVE_FILE = "pr_curve.png"
LOSS_CURVE_FILE = "loss_curve.png"
LEARNING_CURVE_FILE = "learning_curve.png"