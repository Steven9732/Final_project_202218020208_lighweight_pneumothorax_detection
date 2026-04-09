import re

import os
import json
import torch
import numpy as np
import pandas as pd
from PIL import Image

MODEL_CONFIGS = [
    {
        "name": "Baseline",
        "ckpt_path": r"C:\Users\Steven\Desktop\Final_project_202218020208_lighweight_pneumothorax_detection\ConvUnet\AutoDL\Baseline_samplerUNetPP_GEMopt_AutoDL_Baseline_Output\checkpoint_infer.pt",
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
CSV_PATH     = os.path.join(DATASET2_DIR, "pneumothorax_normal_balanced_subset.csv")
IMG_DIR      = os.path.join(DATASET2_DIR, "pneumothorax_normal_balanced_images")

SAVE_DIR = r"C:\Users\Steven\Desktop\Final_project_202218020208_lighweight_pneumothorax_detection\Generalization_check\result\four_model_threshold_scan_only"
os.makedirs(SAVE_DIR, exist_ok=True)

BATCH = 8
TARGET_RECALL = 0.80
THRESHOLDS = np.linspace(0.01, 0.99, 99)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)
print("save_dir:", SAVE_DIR)
print("num_models:", len(MODEL_CONFIGS))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as TV
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score
)
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = [
    "DropPath",
    "GRN",
    "LayerNorm2d",
    "ECALayer",
    "GeM",
    "LSMFHead",
    "ConvNeXtV2Block",
    "UpBlockUNetPP",
    "ConvNeXtV2Tiny",
    "ConvNeXtV2TinyScratch",
]


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x * random_tensor / keep_prob


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class GRN(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        nx = gx / (gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma.view(1, 1, 1, -1) * (x * nx) + self.beta.view(1, 1, 1, -1) + x


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = float(eps)

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight.view(1, -1, 1, 1) * x + self.bias.view(1, -1, 1, 1)


class ECALayer(nn.Module):
    def __init__(self, channels: int, k_size: int = 3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=k_size,
            padding=(k_size - 1) // 2,
            bias=False,
        )
        self.sigmoid = nn.Sigmoid()
        self.last_attn = None

    def forward(self, x):
        y = self.avg_pool(x)
        y = y.squeeze(-1).transpose(-1, -2)
        y = self.conv(y)
        y = self.sigmoid(y)
        y = y.transpose(-1, -2).unsqueeze(-1)
        self.last_attn = y.detach()
        return x * y


class GeM(nn.Module):
    def __init__(self, p: float = 3.0, eps: float = 1e-6, trainable: bool = True, relu_before: bool = True):
        super().__init__()
        self.eps = float(eps)
        self.relu_before = bool(relu_before)

        if trainable:
            self.p = nn.Parameter(torch.ones(1) * float(p))
        else:
            self.register_buffer("p", torch.ones(1) * float(p))

    def forward(self, x):
        if self.relu_before:
            x = F.relu(x)
        p = torch.clamp(self.p, 1.0, 6.0)
        x = x.clamp(min=self.eps).pow(p)
        x = x.mean(dim=(-1, -2)).pow(1.0 / p)
        return x


class LSMFHead(nn.Module):
    """
    Lesion-Sensitive Multi-scale Fusion Head.
    Fuse x3 (1/16, detail) and x4 (1/32, semantic) before classification.
    """
    def __init__(
        self,
        in_ch_x3: int = 384,
        in_ch_x4: int = 768,
        fuse_ch: int = 256,
        n_classes: int = 1,
        gem_p: float = 3.0,
        gn_groups: int = 8,
    ):
        super().__init__()

        self.proj_x3 = nn.Sequential(
            nn.Conv2d(in_ch_x3, fuse_ch, kernel_size=1, bias=False),
            nn.GroupNorm(gn_groups, fuse_ch),
            nn.GELU(),
        )

        self.proj_x4 = nn.Sequential(
            nn.Conv2d(in_ch_x4, fuse_ch, kernel_size=1, bias=False),
            nn.GroupNorm(gn_groups, fuse_ch),
            nn.GELU(),
        )

        self.gate = nn.Sequential(
            nn.Conv2d(fuse_ch * 2, fuse_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(gn_groups, fuse_ch),
            nn.GELU(),
            nn.Conv2d(fuse_ch, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

        self.refine = nn.Sequential(
            nn.Conv2d(fuse_ch, fuse_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(gn_groups, fuse_ch),
            nn.GELU(),
        )

        self.gem = GeM(p=gem_p, eps=1e-6, trainable=True, relu_before=True)
        self.norm = nn.LayerNorm(fuse_ch, eps=1e-6)
        self.fc = nn.Linear(fuse_ch, n_classes)

    def forward(self, x3, x4, feat_dict=None):
        x4_up = F.interpolate(x4, size=x3.shape[-2:], mode="bilinear", align_corners=False)

        x3_p = self.proj_x3(x3)
        x4_p = self.proj_x4(x4_up)

        gate = self.gate(torch.cat([x3_p, x4_p], dim=1))
        fused = gate * x3_p + (1.0 - gate) * x4_p
        fused = self.refine(fused)

        pooled = self.gem(fused)
        pooled = self.norm(pooled)
        logits = self.fc(pooled).squeeze(1)

        if feat_dict is not None:
            feat_dict["lsmf_x3_proj"] = x3_p.detach()
            feat_dict["lsmf_x4_proj"] = x4_p.detach()
            feat_dict["lsmf_gate"] = gate.detach()
            feat_dict["lsmf_fused"] = fused.detach()
            feat_dict["lsmf_vec"] = pooled.detach()

        return logits


class ConvNeXtV2Block(nn.Module):
    def __init__(
        self,
        dim: int,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        layer_scale_init_value: float = 0.0,
    ):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)

        hidden_dim = int(dim * mlp_ratio)
        self.pwconv1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.grn = GRN(hidden_dim)
        self.pwconv2 = nn.Linear(hidden_dim, dim)

        if layer_scale_init_value > 0:
            self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim))
        else:
            self.gamma = None

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.dwconv(x)

        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)

        if self.gamma is not None:
            x = self.gamma.view(1, 1, 1, -1) * x

        x = x.permute(0, 3, 1, 2)
        x = shortcut + self.drop_path(x)
        return x


class UpBlockUNetPP(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        self.conv1 = nn.Conv2d(in_ch + skip_ch, out_ch, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x, skips=None):
        x = self.upsample(x)

        if skips is None:
            skips = []

        cat_list = [x]
        for s in skips:
            if s is None:
                continue
            if s.shape[-2:] != x.shape[-2:]:
                s = F.interpolate(s, size=x.shape[-2:], mode="bilinear", align_corners=False)
            cat_list.append(s)

        x = torch.cat(cat_list, dim=1)
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        return x


class ConvNeXtV2Tiny(nn.Module):
    def __init__(
        self,
        in_chans: int = 3,
        num_classes: int = 1000,
        drop_path_rate: float = 0.0,
        layer_scale_init_value: float = 0.0,
    ):
        super().__init__()

        depths = [3, 3, 9, 3]
        dims = [96, 192, 384, 768]

        self.downsample_layers = nn.ModuleList()
        self.stages = nn.ModuleList()

        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm2d(dims[0]),
        )
        self.downsample_layers.append(stem)

        total_blocks = sum(depths)
        dpr_values = torch.linspace(0, drop_path_rate, total_blocks).tolist()
        block_idx = 0

        for stage_idx in range(4):
            depth = depths[stage_idx]
            dim = dims[stage_idx]

            if stage_idx > 0:
                down = nn.Sequential(
                    LayerNorm2d(dims[stage_idx - 1]),
                    nn.Conv2d(dims[stage_idx - 1], dim, kernel_size=2, stride=2),
                )
                self.downsample_layers.append(down)

            blocks = []
            for i in range(depth):
                blocks.append(
                    ConvNeXtV2Block(
                        dim=dim,
                        mlp_ratio=4.0,
                        drop_path=dpr_values[block_idx + i],
                        layer_scale_init_value=layer_scale_init_value,
                    )
                )
            block_idx += depth
            self.stages.append(nn.Sequential(*blocks))

        self.norm_head = nn.LayerNorm(dims[-1], eps=1e-6)
        self.head = nn.Linear(dims[-1], num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        x = x.mean(dim=[2, 3])
        x = self.norm_head(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


class ConvNeXtV2TinyScratch(nn.Module):
    def __init__(
        self,
        in_chans: int = 1,
        n_classes: int = 1,
        drop_path_rate: float = 0.1,
        use_seg_guided: bool = True,
        use_lsmf: bool = True,
        use_eca: bool = True,
        lsmf_fuse_ch: int = 256,
    ):
        super().__init__()

        self.backbone = ConvNeXtV2Tiny(
            in_chans=in_chans,
            num_classes=n_classes,
            drop_path_rate=drop_path_rate,
            layer_scale_init_value=0.0,
        )

        dims = [96, 192, 384, 768]
        c1, c2, c3, c4 = dims
        last_dim = c4

        self.use_eca = bool(use_eca)
        self.eca4 = ECALayer(last_dim, k_size=3) if self.use_eca else nn.Identity()

        self.use_seg_guided = bool(use_seg_guided)
        self.seg_guided_scale = 0.5

        self.use_lsmf = bool(use_lsmf)

        self.gem = GeM(p=3.0, eps=1e-6, trainable=True, relu_before=True)
        self.cls_head = nn.Linear(last_dim, n_classes)

        self.lsmf_head = (
            LSMFHead(
                in_ch_x3=c3,
                in_ch_x4=c4,
                fuse_ch=lsmf_fuse_ch,
                n_classes=n_classes,
                gem_p=3.0,
                gn_groups=8,
            )
            if self.use_lsmf else None
        )

        self.last_feats = {}

        self.up_2_1 = UpBlockUNetPP(in_ch=768, skip_ch=384, out_ch=384)
        self.up_1_1 = UpBlockUNetPP(in_ch=384, skip_ch=192, out_ch=192)
        self.up_0_1 = UpBlockUNetPP(in_ch=192, skip_ch=96, out_ch=96)

        self.up_1_2 = UpBlockUNetPP(in_ch=384, skip_ch=192 + 192, out_ch=192)
        self.up_0_2 = UpBlockUNetPP(in_ch=192, skip_ch=96 + 96, out_ch=96)

        self.up_0_3 = UpBlockUNetPP(in_ch=192, skip_ch=96 + 96 + 96, out_ch=96)

        self.up_half = UpBlockUNetPP(in_ch=96, skip_ch=0, out_ch=max(96 // 2, 32))
        self.seg_head = nn.Conv2d(max(96 // 2, 32), 1, kernel_size=1)

    def forward_backbone_pyramid(self, x):
        feats = []
        for i in range(4):
            x = self.backbone.downsample_layers[i](x)
            x = self.backbone.stages[i](x)

            if i == 3:
                self.last_feats["x4_pre_eca"] = x.detach()
                x = self.eca4(x)
                self.last_feats["x4_post_eca"] = x.detach()

            feats.append(x)
        return feats

    def get_global_embedding(self, x4_feat):
        gap = self.gem(x4_feat)
        vec = self.backbone.norm_head(gap)
        return vec

    def forward(self, x):
        self.last_feats = {}

        x1, x2, x3, x4 = self.forward_backbone_pyramid(x)

        x0_0 = x1
        x1_0 = x2
        x2_0 = x3
        x3_0 = x4

        x2_1 = self.up_2_1(x3_0, [x2_0])
        x1_1 = self.up_1_1(x2_0, [x1_0])
        x0_1 = self.up_0_1(x1_0, [x0_0])

        x1_2 = self.up_1_2(x2_1, [x1_0, x1_1])
        x0_2 = self.up_0_2(x1_1, [x0_0, x0_1])

        x0_3 = self.up_0_3(x1_2, [x0_0, x0_1, x0_2])

        d0 = self.up_half(x0_3, [])
        seg_logits = self.seg_head(d0)
        seg_logits = F.interpolate(seg_logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
        self.last_feats["seg_prob"] = torch.sigmoid(seg_logits).detach()

        x4_for_cls = x4
        if self.use_seg_guided:
            seg_prob = torch.sigmoid(seg_logits)
            seg_down = F.adaptive_avg_pool2d(seg_prob, output_size=x4.shape[-2:])
            x4_for_cls = x4 * (1.0 + self.seg_guided_scale * seg_down)

        self.last_feats["x3_for_cls"] = x3.detach()
        self.last_feats["x4_for_cls"] = x4_for_cls.detach()

        global_vec = self.get_global_embedding(x4_for_cls)
        self.last_feats["global_vec"] = global_vec.detach()

        if self.use_lsmf and self.lsmf_head is not None:
            cls_logits = self.lsmf_head(x3, x4_for_cls, feat_dict=self.last_feats)
        else:
            cls_logits = self.cls_head(global_vec).squeeze(1)

        return cls_logits, seg_logits

def safe_name(s: str) -> str:
    s = str(s).strip()
    s = re.sub(r"[^A-Za-z0-9._+-]+", "_", s)
    return s.strip("_")

def build_eval_tfms(img_size, mean, std):
    return TV.Compose([
        TV.Resize((img_size, img_size)),
        TV.ToTensor(),
        TV.Normalize(mean=mean, std=std),
    ])

def load_checkpoint_model(cfg):
    ckpt_path = cfg["ckpt_path"]
    assert os.path.exists(ckpt_path), f"Not found: {ckpt_path}"

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    model_kwargs = dict(ckpt.get("model_kwargs", {}))
    model_kwargs.update(cfg.get("override_kwargs", {}))

    if "use_eca" not in model_kwargs:
        model_kwargs["use_eca"] = True
    if "use_lsmf" not in model_kwargs:
        model_kwargs["use_lsmf"] = False
    if "use_seg_guided" not in model_kwargs:
        model_kwargs["use_seg_guided"] = True

    model = ConvNeXtV2TinyScratch(**model_kwargs).to(device)
    load_msg = model.load_state_dict(ckpt["model_state"], strict=False)
    model.eval()

    return {
        "model": model,
        "model_kwargs": model_kwargs,
        "img_size": int(ckpt["preprocess"]["img_size"]),
        "mean": ckpt["preprocess"]["mean"],
        "std": ckpt["preprocess"]["std"],
        "T_cal": float(ckpt["postprocess"]["temperature_T"]),
        "internal_thr": float(ckpt["postprocess"]["cls_threshold"]),
        "missing_keys": list(load_msg.missing_keys),
        "unexpected_keys": list(load_msg.unexpected_keys),
    }

class ExternalClsDataset(Dataset):
    def __init__(self, df, transform):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["full_path"]).convert("L")
        x = self.transform(img)
        y = int(row["has_pneumo"])
        return x, y

ext_df = pd.read_csv(CSV_PATH)
ext_df["full_path"] = ext_df["Image Index"].apply(lambda x: os.path.join(IMG_DIR, str(x)))
ext_df["has_pneumo"] = (
    ext_df["Finding Labels"].astype(str).str.contains("Pneumothorax")
).astype(int)

missing = (~ext_df["full_path"].apply(os.path.exists)).sum()
print("rows:", len(ext_df), "missing_files:", missing)

ext_df = ext_df[ext_df["full_path"].apply(os.path.exists)].reset_index(drop=True)
print(ext_df["has_pneumo"].value_counts())

split_index = np.arange(len(ext_df))
calib_idx, test_idx = train_test_split(
    split_index,
    test_size=0.8,
    random_state=42,
    stratify=ext_df["has_pneumo"],
)

print("split:", len(calib_idx), "calib |", len(test_idx), "test")

@torch.no_grad()
def infer_probs(model, loader, T_cal=1.0, desc="External inference"):
    ys, ps = [], []
    n_seen = 0

    pbar = tqdm(loader, total=len(loader), desc=desc)
    for xb, yb in pbar:
        xb = xb.to(device, non_blocking=True)

        cls_logits, _seg_logits = model(xb)
        p = torch.sigmoid(cls_logits / T_cal)

        p_np = p.detach().cpu().float().numpy()
        y_np = yb.numpy()

        ps.append(p_np)
        ys.append(y_np)

        n_seen += len(y_np)
        pbar.set_postfix(
            n=n_seen,
            p_mean=float(np.mean(p_np)),
            pos_rate=float(np.mean(np.concatenate(ys))) if len(ys) > 0 else 0.0
        )

    return np.concatenate(ys), np.concatenate(ps)

def eval_at_threshold(y, p, t: float):
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(float)
    yhat = (p >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat).ravel()
    return {
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

def sweep_thresholds(y, p, thresholds):
    rows = []
    for t in thresholds:
        row = eval_at_threshold(y, p, float(t))
        rows.append(row)
    return pd.DataFrame(rows)

def pick_threshold(calib_df, mode="max_f1", target_recall=0.80):
    y = calib_df["has_pneumo"].to_numpy().astype(int)
    p = calib_df["p_calibrated"].to_numpy().astype(float)
    df = sweep_thresholds(y, p, THRESHOLDS)

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

def auc_metrics(df):
    y = df["has_pneumo"].to_numpy().astype(int)
    p = df["p_calibrated"].to_numpy().astype(float)
    if len(np.unique(y)) == 2:
        return {
            "roc_auc": float(roc_auc_score(y, p)),
            "pr_auc": float(average_precision_score(y, p)),
        }
    return {
        "roc_auc": np.nan,
        "pr_auc": np.nan,
    }

summary_rows = []
best_threshold_rows = []
sweep_long_rows = []
load_log_rows = []

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

    eval_tfms = build_eval_tfms(loaded["img_size"], loaded["mean"], loaded["std"])
    ext_ds = ExternalClsDataset(ext_df, transform=eval_tfms)
    ext_dl = DataLoader(
        ext_ds,
        batch_size=BATCH,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    y_ext, p_ext = infer_probs(
        loaded["model"],
        ext_dl,
        T_cal=loaded["T_cal"],
        desc=f"{model_name} | external inference",
    )

    pred_df = ext_df.copy()
    pred_df["p_calibrated"] = p_ext
    pred_df["model"] = model_name

    calib_df = pred_df.iloc[calib_idx].copy().reset_index(drop=True)
    test_df = pred_df.iloc[test_idx].copy().reset_index(drop=True)

    t_f1, sweep_df_f1, best_calib_f1 = pick_threshold(
        calib_df,
        mode="max_f1",
        target_recall=TARGET_RECALL,
    )
    t_r80, sweep_df_r80, best_calib_r80 = pick_threshold(
        calib_df,
        mode="recall_at_least",
        target_recall=TARGET_RECALL,
    )

    sweep_df = sweep_df_f1.copy()
    sweep_df["model"] = model_name
    sweep_df["split"] = "external_calib"
    sweep_long_rows.append(sweep_df)

    test_auc = auc_metrics(test_df)
    tuned_f1_test = eval_at_threshold(
        test_df["has_pneumo"].to_numpy().astype(int),
        test_df["p_calibrated"].to_numpy().astype(float),
        t_f1,
    ) | test_auc
    tuned_r80_test = eval_at_threshold(
        test_df["has_pneumo"].to_numpy().astype(int),
        test_df["p_calibrated"].to_numpy().astype(float),
        t_r80,
    ) | test_auc

    best_threshold_rows.append({
        "model": model_name,
        "T_cal": loaded["T_cal"],
        "internal_thr": loaded["internal_thr"],
        "best_calib_t_f1": float(t_f1),
        "best_calib_t_r80": float(t_r80),
        "calib_f1_at_best_t_f1": float(best_calib_f1["f1"]),
        "calib_precision_at_best_t_f1": float(best_calib_f1["precision"]),
        "calib_recall_at_best_t_f1": float(best_calib_f1["recall"]),
        "calib_spec_at_best_t_f1": float(best_calib_f1["spec"]),
        "calib_precision_at_best_t_r80": float(best_calib_r80["precision"]),
        "calib_recall_at_best_t_r80": float(best_calib_r80["recall"]),
        "calib_f1_at_best_t_r80": float(best_calib_r80["f1"]),
        "calib_spec_at_best_t_r80": float(best_calib_r80["spec"]),
    })

    for strategy_name, metrics_dict in [
        ("tuned_maxf1_test", tuned_f1_test),
        ("tuned_recall80_test", tuned_r80_test),
    ]:
        row = {
            "model": model_name,
            "strategy": strategy_name,
            "T_cal": loaded["T_cal"],
            "internal_thr": loaded["internal_thr"],
            "best_calib_t_f1": float(t_f1),
            "best_calib_t_r80": float(t_r80),
        }
        row.update(metrics_dict)
        summary_rows.append(row)

    sweep_save = sweep_df.copy()
    sweep_save.to_csv(
        os.path.join(SAVE_DIR, f"{safe_name(model_name)}_external_calib_threshold_sweep.csv"),
        index=False
    )

summary_df = pd.DataFrame(summary_rows)
best_threshold_df = pd.DataFrame(best_threshold_rows)
sweep_long_df = pd.concat(sweep_long_rows, ignore_index=True)
load_log_df = pd.DataFrame(load_log_rows)

summary_df.to_csv(os.path.join(SAVE_DIR, "four_model_threshold_scan_test_summary.csv"), index=False)
best_threshold_df.to_csv(os.path.join(SAVE_DIR, "four_model_best_thresholds.csv"), index=False)
sweep_long_df.to_csv(os.path.join(SAVE_DIR, "four_model_external_calib_threshold_sweep_long.csv"), index=False)
load_log_df.to_csv(os.path.join(SAVE_DIR, "four_model_load_log.csv"), index=False)

print("\n[Saved]", os.path.join(SAVE_DIR, "four_model_threshold_scan_test_summary.csv"))
print("[Saved]", os.path.join(SAVE_DIR, "four_model_best_thresholds.csv"))
print("[Saved]", os.path.join(SAVE_DIR, "four_model_external_calib_threshold_sweep_long.csv"))
print("[Saved]", os.path.join(SAVE_DIR, "four_model_load_log.csv"))

display(best_threshold_df.round(4))
display(summary_df.round(4))