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
    Lesion-Sensitive Multi-scale Fusion Head
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

        self.eca4 = ECALayer(last_dim, k_size=3)

        self.use_seg_guided = use_seg_guided
        self.seg_guided_scale = 0.5

        self.use_lsmf = use_lsmf

        self.gem = GeM(p=3.0, eps=1e-6, trainable=True, relu_before=True)
        self.cls_head = nn.Linear(last_dim, n_classes)

        self.lsmf_head = LSMFHead(
            in_ch_x3=c3,
            in_ch_x4=c4,
            fuse_ch=lsmf_fuse_ch,
            n_classes=n_classes,
            gem_p=3.0,
            gn_groups=8,
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
        """
        Produce a 768-d global embedding through backbone.norm_head.

        This keeps compatibility with your current rag+llm notebook, because
        that notebook uses a forward hook on infer_model.backbone.norm_head.
        """
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

        if self.use_lsmf:
            cls_logits = self.lsmf_head(x3, x4_for_cls, feat_dict=self.last_feats)
        else:
            cls_logits = self.cls_head(global_vec).squeeze(1)

        return cls_logits, seg_logits
