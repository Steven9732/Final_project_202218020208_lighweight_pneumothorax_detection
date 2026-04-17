
import os
import sys
import json
import re
from typing import Dict, List

import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

import torch
import torchvision.transforms as TV
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)

import matplotlib.pyplot as plt


# ============================================================
# 0) CONFIG
# ============================================================
PROJECT_DIR = r"C:\Users\Steven\Desktop\Final_project_202218020208_lighweight_pneumothorax_detection"
if PROJECT_DIR not in sys.path:
    sys.path.append(PROJECT_DIR)

from ECALSMFModel_compare4 import ConvNeXtV2TinyScratch  # put this file in PROJECT_DIR


MODEL_CONFIGS = [
    {
        "name": "Baseline",
        "ckpt_path": r"C:\Users\Steven\Desktop\Final_project_202218020208_lighweight_pneumothorax_detection\ConvUnet\AutoDL\Baseline_samplerUNetPP_GEMopt_AutoDL_Baseline_Outputcheckpoint_infer.pt",
        "override_kwargs": {"use_eca": False, "use_lsmf": False, "use_seg_guided": True},
    },
    {
        "name": "Baseline+ECA",
        "ckpt_path": r"C:\Users\Steven\Desktop\Final_project_202218020208_lighweight_pneumothorax_detection\ConvUnet\AutoDL\Baseline_ECA_samplerUNetPP_GEMopt_AutoDL\checkpoint_infer.pt",
        "override_kwargs": {"use_eca": True, "use_lsmf": False, "use_seg_guided": True},
    },
    {
        "name": "Baseline+LSMF",
        "ckpt_path": r"C:\Users\Steven\Desktop\Final_project_202218020208_lighweight_pneumothorax_detection\ConvUnet\AutoDL\Baseline_LSMF_samplerUNetPP_GEMopt_AutoDL\checkpoint_infer.pt",
        "override_kwargs": {"use_eca": False, "use_lsmf": True, "use_seg_guided": True},
    },
    {
        "name": "Baseline+ECA+LSMF",
        "ckpt_path": r"C:\Users\Steven\Desktop\Final_project_202218020208_lighweight_pneumothorax_detection\ConvUnet\AutoDL\Baseline_ECA_LSMF_samplerUNetPP_GEMopt_AutoDL_Output\checkpoint_infer.pt",
        "override_kwargs": {"use_eca": True, "use_lsmf": True, "use_seg_guided": True},
    },
]

DATASET2_DIR = r"C:\Users\Steven\Desktop\Final Project\Datasets\Dataset_2"
CSV_PATH = os.path.join(DATASET2_DIR, "pneumothorax_normal_balanced_subset.csv")
IMG_DIR = os.path.join(DATASET2_DIR, "pneumothorax_normal_balanced_images")

SAVE_DIR = os.path.join(PROJECT_DIR, "Generalization_check", "compare_4models_threshold_scan")
os.makedirs(SAVE_DIR, exist_ok=True)

BATCH_SIZE = 32
NUM_WORKERS = 0
PIN_MEMORY = True
TARGET_RECALL = 0.80
THRESHOLDS = np.linspace(0.01, 0.99, 99)
RANDOM_STATE = 42
TEST_SIZE = 0.80

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)


# ============================================================
# 1) DATASET
# ============================================================
class ExternalClsDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform):
        self.df = df.reset_index(drop=True).copy()
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["full_path"]).convert("L")
        img = self.transform(img)
        y = np.float32(row["has_pneumo"])
        return img, y


def build_eval_tfms(img_size: int, mean, std):
    return TV.Compose([
        TV.Resize((img_size, img_size)),
        TV.ToTensor(),
        TV.Normalize(mean=mean, std=std),
    ])


def safe_name(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_\-+]+", "_", text)
    return text.strip("_")


# ============================================================
# 2) METRICS
# ============================================================
def eval_at_threshold(y, p, t: float) -> Dict[str, float]:
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(float)
    yhat = (p >= t).astype(int)

    cm = confusion_matrix(y, yhat, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    out = {
        "t": float(t),
        "n": int(len(y)),
        "pos_rate": float(y.mean()),
        "acc": float(accuracy_score(y, yhat)),
        "precision": float(precision_score(y, yhat, zero_division=0)),
        "recall": float(recall_score(y, yhat, zero_division=0)),
        "f1": float(f1_score(y, yhat, zero_division=0)),
        "spec": float(tn / (tn + fp + 1e-12)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    if len(np.unique(y)) == 2:
        out["roc_auc"] = float(roc_auc_score(y, p))
        out["pr_auc"] = float(average_precision_score(y, p))
    else:
        out["roc_auc"] = np.nan
        out["pr_auc"] = np.nan
    return out


def sweep_thresholds(y, p, thresholds=THRESHOLDS) -> pd.DataFrame:
    rows = [eval_at_threshold(y, p, float(t)) for t in thresholds]
    return pd.DataFrame(rows)


def pick_threshold(calib_df: pd.DataFrame, mode="max_f1", target_recall=0.80):
    y = calib_df["has_pneumo"].to_numpy().astype(int)
    p = calib_df["p_calibrated"].to_numpy().astype(float)
    df = sweep_thresholds(y, p)

    if mode == "max_f1":
        best = df.loc[df["f1"].idxmax()].to_dict()
    elif mode == "recall_at_least":
        cand = df[df["recall"] >= target_recall].copy()
        if len(cand) == 0:
            best = df.loc[df["recall"].idxmax()].to_dict()
            best["note"] = f"No threshold reached recall>={target_recall}; fallback to max-recall."
        else:
            best = cand.loc[cand["precision"].idxmax()].to_dict()
    else:
        raise ValueError("mode must be 'max_f1' or 'recall_at_least'")

    return float(best["t"]), df, best


# ============================================================
# 3) INFERENCE
# ============================================================
@torch.no_grad()
def infer_probs(model, loader, T_cal=1.0, desc="Inference"):
    ys, ps = [], []
    pbar = tqdm(loader, total=len(loader), desc=desc)

    for xb, yb in pbar:
        xb = xb.to(device, non_blocking=True)
        cls_logits, _ = model(xb)
        p = torch.sigmoid(cls_logits / float(T_cal))

        ps.append(p.detach().cpu().float().numpy())
        ys.append(yb.numpy())

    y = np.concatenate(ys).astype(int)
    p = np.concatenate(ps).astype(float)
    return y, p


def load_checkpoint_model(cfg: Dict):
    ckpt_path = cfg["ckpt_path"]
    assert os.path.exists(ckpt_path), f"Checkpoint not found: {ckpt_path}"

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    model_kwargs = dict(ckpt.get("model_kwargs", {}))
    model_kwargs.update(cfg.get("override_kwargs", {}))

    model = ConvNeXtV2TinyScratch(**model_kwargs).to(device)

    state = ckpt.get("model_state", ckpt.get("state_dict"))
    if state is None:
        raise KeyError(f"No model_state/state_dict found in {ckpt_path}")

    load_msg = model.load_state_dict(state, strict=False)
    missing = list(load_msg.missing_keys)
    unexpected = list(load_msg.unexpected_keys)

    preprocess = ckpt.get("preprocess", {})
    postprocess = ckpt.get("postprocess", {})

    img_size = int(preprocess.get("img_size", 224))
    mean = preprocess.get("mean", [0.5])
    std = preprocess.get("std", [0.5])
    T_cal = float(postprocess.get("temperature_T", 1.0))
    internal_thr = float(postprocess.get("cls_threshold", 0.5))

    model.eval()

    return {
        "model": model,
        "ckpt": ckpt,
        "model_kwargs": model_kwargs,
        "img_size": img_size,
        "mean": mean,
        "std": std,
        "T_cal": T_cal,
        "internal_thr": internal_thr,
        "missing_keys": missing,
        "unexpected_keys": unexpected,
    }


# ============================================================
# 4) PLOTS
# ============================================================
def plot_threshold_metric_curves(sweep_long_df: pd.DataFrame, metric: str, save_path: str):
    fig = plt.figure(figsize=(8, 5))
    for model_name, sub in sweep_long_df.groupby("model"):
        sub = sub.sort_values("t")
        plt.plot(sub["t"].values, sub[metric].values, label=model_name)
    plt.xlabel("Threshold")
    plt.ylabel(metric)
    plt.title(f"{metric} vs threshold across four models")
    plt.ylim(0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_strategy_comparison(summary_df: pd.DataFrame, strategy: str, metric: str, save_path: str):
    sub = summary_df[summary_df["strategy"] == strategy].copy()
    if sub.empty:
        return

    fig = plt.figure(figsize=(8, 5))
    plt.bar(sub["model"], sub[metric])
    plt.ylim(0, 1.0)
    plt.ylabel(metric)
    plt.title(f"{metric} comparison | {strategy}")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 5) LOAD EXTERNAL DATA ONCE
# ============================================================
ext_df = pd.read_csv(CSV_PATH)
ext_df["full_path"] = ext_df["Image Index"].apply(lambda x: os.path.join(IMG_DIR, str(x)))
ext_df["has_pneumo"] = ext_df["Finding Labels"].astype(str).str.contains("Pneumothorax").astype(int)

ext_df = ext_df[ext_df["full_path"].apply(os.path.exists)].reset_index(drop=True)
assert len(ext_df) > 0, "No valid external images found."

all_indices = np.arange(len(ext_df))
calib_idx, test_idx = train_test_split(
    all_indices,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=ext_df["has_pneumo"].to_numpy(),
)

split_info = {
    "random_state": RANDOM_STATE,
    "test_size": TEST_SIZE,
    "calib_size": int(len(calib_idx)),
    "test_size_n": int(len(test_idx)),
}
print("split_info:", split_info)


# ============================================================
# 6) MAIN LOOP FOR FOUR MODELS
# ============================================================
summary_rows: List[Dict] = []
sweep_long_rows: List[pd.DataFrame] = []
load_log_rows: List[Dict] = []

for cfg in MODEL_CONFIGS:
    model_name = cfg["name"]
    print("\n" + "=" * 80)
    print("Running:", model_name)

    loaded = load_checkpoint_model(cfg)
    print("model_kwargs:", loaded["model_kwargs"])
    print("missing_keys:", loaded["missing_keys"][:10], "..." if len(loaded["missing_keys"]) > 10 else "")
    print("unexpected_keys:", loaded["unexpected_keys"][:10], "..." if len(loaded["unexpected_keys"]) > 10 else "")

    load_log_rows.append({
        "model": model_name,
        "ckpt_path": cfg["ckpt_path"],
        "img_size": loaded["img_size"],
        "T_cal": loaded["T_cal"],
        "internal_thr": loaded["internal_thr"],
        "missing_key_count": len(loaded["missing_keys"]),
        "unexpected_key_count": len(loaded["unexpected_keys"]),
        "missing_keys": json.dumps(loaded["missing_keys"], ensure_ascii=False),
        "unexpected_keys": json.dumps(loaded["unexpected_keys"], ensure_ascii=False),
    })

    tfms = build_eval_tfms(loaded["img_size"], loaded["mean"], loaded["std"])
    ds = ExternalClsDataset(ext_df, transform=tfms)
    dl = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    y_all, p_all = infer_probs(
        loaded["model"],
        dl,
        T_cal=loaded["T_cal"],
        desc=f"{model_name} | external inference",
    )

    pred_df = ext_df.copy()
    pred_df["p_calibrated"] = p_all
    pred_df["model"] = model_name

    calib_df = pred_df.iloc[calib_idx].copy().reset_index(drop=True)
    test_df = pred_df.iloc[test_idx].copy().reset_index(drop=True)

    internal_thr = float(loaded["internal_thr"])

    no_tuning_all = eval_at_threshold(y_all, p_all, internal_thr)
    no_tuning_test = eval_at_threshold(
        test_df["has_pneumo"].to_numpy().astype(int),
        test_df["p_calibrated"].to_numpy().astype(float),
        internal_thr,
    )

    t_f1, sweep_df, best_calib_f1 = pick_threshold(calib_df, mode="max_f1", target_recall=TARGET_RECALL)
    t_r80, sweep_df_r80, best_calib_r80 = pick_threshold(
        calib_df, mode="recall_at_least", target_recall=TARGET_RECALL
    )

    tuned_f1_test = eval_at_threshold(
        test_df["has_pneumo"].to_numpy().astype(int),
        test_df["p_calibrated"].to_numpy().astype(float),
        t_f1,
    )
    tuned_r80_test = eval_at_threshold(
        test_df["has_pneumo"].to_numpy().astype(int),
        test_df["p_calibrated"].to_numpy().astype(float),
        t_r80,
    )

    for strategy_name, metrics_dict in [
        ("internal_thr_all", no_tuning_all),
        ("internal_thr_test", no_tuning_test),
        ("tuned_maxf1_test", tuned_f1_test),
        ("tuned_recall80_test", tuned_r80_test),
    ]:
        row = {
            "model": model_name,
            "strategy": strategy_name,
            "T_cal": loaded["T_cal"],
            "internal_thr": internal_thr,
            "best_calib_t_f1": float(t_f1),
            "best_calib_t_r80": float(t_r80),
        }
        row.update(metrics_dict)
        summary_rows.append(row)

    sweep_df["model"] = model_name
    sweep_df["split"] = "external_calib"
    sweep_long_rows.append(sweep_df)

    pred_save = pred_df.copy()
    pred_save["split"] = "all"
    pred_save.to_csv(os.path.join(SAVE_DIR, f"{safe_name(model_name)}_all_predictions.csv"), index=False)

    calib_df.to_csv(os.path.join(SAVE_DIR, f"{safe_name(model_name)}_calib_predictions.csv"), index=False)
    test_df.to_csv(os.path.join(SAVE_DIR, f"{safe_name(model_name)}_test_predictions.csv"), index=False)
    sweep_df.to_csv(os.path.join(SAVE_DIR, f"{safe_name(model_name)}_calib_threshold_sweep.csv"), index=False)

    with open(os.path.join(SAVE_DIR, f"{safe_name(model_name)}_best_thresholds.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": model_name,
                "internal_thr": internal_thr,
                "best_calib_maxf1": best_calib_f1,
                f"best_calib_recall>={TARGET_RECALL:.2f}": best_calib_r80,
                "T_cal": loaded["T_cal"],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )


summary_df = pd.DataFrame(summary_rows)
sweep_long_df = pd.concat(sweep_long_rows, ignore_index=True)
load_log_df = pd.DataFrame(load_log_rows)

summary_df.to_csv(os.path.join(SAVE_DIR, "four_model_strategy_summary.csv"), index=False)
sweep_long_df.to_csv(os.path.join(SAVE_DIR, "four_model_calib_threshold_sweep_long.csv"), index=False)
load_log_df.to_csv(os.path.join(SAVE_DIR, "four_model_load_log.csv"), index=False)

with open(os.path.join(SAVE_DIR, "split_info.json"), "w", encoding="utf-8") as f:
    json.dump(split_info, f, ensure_ascii=False, indent=2)

print("\nSaved summary:", os.path.join(SAVE_DIR, "four_model_strategy_summary.csv"))
print("Saved sweep:", os.path.join(SAVE_DIR, "four_model_calib_threshold_sweep_long.csv"))
print("Saved load log:", os.path.join(SAVE_DIR, "four_model_load_log.csv"))


# ============================================================
# 7) PLOTS
# ============================================================
for metric in ["precision", "recall", "f1", "spec", "acc"]:
    plot_threshold_metric_curves(
        sweep_long_df=sweep_long_df,
        metric=metric,
        save_path=os.path.join(SAVE_DIR, f"threshold_curve_{metric}.png"),
    )

for strategy in ["internal_thr_test", "tuned_maxf1_test", "tuned_recall80_test"]:
    for metric in ["acc", "precision", "recall", "f1", "spec", "roc_auc", "pr_auc"]:
        plot_strategy_comparison(
            summary_df=summary_df,
            strategy=strategy,
            metric=metric,
            save_path=os.path.join(SAVE_DIR, f"{strategy}_{metric}.png"),
        )

print("\nDone.")
