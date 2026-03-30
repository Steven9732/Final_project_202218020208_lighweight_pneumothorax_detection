from __future__ import annotations

APP_TITLE = "PneumoAssist: Pneumothorax Detection & Reporting"
APP_ICON = "🫁"
APP_LAYOUT = "wide"
APP_SIDEBAR_STATE = "expanded"

FINAL_MODEL_INFO = {
    "model_name": "ConvNeXtV2 + UNet++",
    "innovation": "ECA-enhanced Lesion-Sensitive Multi-Scale Fusion (ECA-LSMF)",
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
