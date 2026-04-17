from __future__ import annotations

APP_TITLE = "PneumoAssist: Pneumothorax Detection & Reporting"
APP_ICON = "🫁"
APP_LAYOUT = "wide"
APP_SIDEBAR_STATE = "expanded"

MODEL_VARIANTS = {
    "baseline": {
        "label": "Baseline",
        "assets_subdir": "baseline",
        "performance_subdir": "baseline",
        "model_name": "ConvNeXtV2 Tiny + U-Net++",
        "innovation": "Baseline",
        "evaluation": "Calibrated internal test result",
        "temperature": 0.734,
        "threshold": 0.760,

        "use_eca": False,
        "use_lsmf": False,
        "embedding_dim": 768,
        "preferred_feat_keys": [
            "gem_vec", "pooled_vec", "cls_vec", "feat_vec", "embedding", "global_feat"
        ],

        "metrics": {
            "Accuracy": 91.37,
            "Precision": 83.54,
            "Recall": 76.03,
            "F1-Score": 79.61,
            "Specificity": 95.74,
            "ROC-AUC": 95.74,
            "PR-AUC": 87.46,
        },
    },

    "baseline_eca": {
        "label": "Baseline + ECA",
        "assets_subdir": "baseline_eca",
        "performance_subdir": "baseline_eca",
        "model_name": "ConvNeXtV2 Tiny + U-Net++",
        "innovation": "Efficient Channel Attention (ECA)",
        "evaluation": "Calibrated internal test result",
        "temperature": 0.994,
        "threshold": 0.150,

        "use_eca": True,
        "use_lsmf": False,
        "embedding_dim": 768,
        "preferred_feat_keys": [
            "gem_vec", "pooled_vec", "cls_vec", "feat_vec", "embedding", "global_feat"
        ],

        "metrics": {
            "Accuracy": 91.04,
            "Precision": 76.41,
            "Recall": 86.14,
            "F1-Score": 80.99,
            "Specificity": 92.43,
            "ROC-AUC": 96.24,
            "PR-AUC": 88.45,
        },
    },

    "baseline_lsmf": {
        "label": "Baseline + LSMF",
        "assets_subdir": "baseline_lsmf",
        "performance_subdir": "baseline_lsmf",
        "model_name": "ConvNeXtV2 Tiny + U-Net++",
        "innovation": "Lesion-Sensitive Multi-Scale Fusion (LSMF)",
        "evaluation": "Calibrated internal test result",
        "temperature": 0.930,
        "threshold": 0.810,

        "use_eca": False,
        "use_lsmf": True,
        "embedding_dim": 256,
        "preferred_feat_keys": [
            "lsmf_vec", "gem_vec", "pooled_vec", "cls_vec", "feat_vec", "embedding", "global_feat"
        ],

        "metrics": {
            "Accuracy": 92.03,
            "Precision": 85.77,
            "Recall": 76.78,
            "F1-Score": 81.03,
            "Specificity": 96.38,
            "ROC-AUC": 96.24,
            "PR-AUC": 89.72,
        },
    },

    "baseline_eca_lsmf": {
        "label": "Baseline + ECA + LSMF",
        "assets_subdir": "baseline_eca_lsmf",
        "performance_subdir": "baseline_eca_lsmf",
        "model_name": "ConvNeXtV2 Tiny + U-Net++",
        "innovation": "ECA + LSMF",
        "evaluation": "Calibrated internal test result",
        "temperature": 0.716,
        "threshold": 0.750,

        "use_eca": True,
        "use_lsmf": True,
        "embedding_dim": 256,
        "preferred_feat_keys": [
            "lsmf_vec", "gem_vec", "pooled_vec", "cls_vec", "feat_vec", "embedding", "global_feat"
        ],

        "metrics": {
            "Accuracy": 91.54,
            "Precision": 79.15,
            "Recall": 83.90,
            "F1-Score": 81.45,
            "Specificity": 93.71,
            "ROC-AUC": 96.41,
            "PR-AUC": 89.08,
        },
    },
}

DEFAULT_MODEL_VARIANT = "baseline_lsmf"

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
    # "Training Dynamics",
]

ROC_CURVE_FILE = "roc_curve.png"
PR_CURVE_FILE = "pr_curve.png"
LOSS_CURVE_FILE = "loss_curve.png"
# LEARNING_CURVE_FILE = "learning_curve.png"