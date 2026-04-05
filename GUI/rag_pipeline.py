from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms as tvT
from PIL import Image, ImageDraw
import matplotlib.cm as cm

from ECALSMFModel import ConvNeXtV2TinyScratch

try:
    import faiss  # type: ignore
    HAS_FAISS = True
except Exception:
    HAS_FAISS = False


def guess_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _as_list(v, default):
    if v is None:
        return [float(default)]
    if isinstance(v, (list, tuple)):
        return [float(x) for x in v]
    if isinstance(v, np.ndarray):
        return [float(x) for x in v.tolist()]
    return [float(v)]


def _l2norm(v: np.ndarray) -> np.ndarray:
    v = v.astype("float32")
    return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        self.hooks.append(
            self.target_layer.register_forward_hook(forward_hook)
        )

        if hasattr(self.target_layer, "register_full_backward_hook"):
            self.hooks.append(
                self.target_layer.register_full_backward_hook(backward_hook)
            )
        else:
            self.hooks.append(
                self.target_layer.register_backward_hook(backward_hook)
            )

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def __call__(self, x):
        self.model.zero_grad(set_to_none=True)

        out = self.model(x)
        if isinstance(out, tuple):
            cls_logits, _ = out
        else:
            cls_logits = out

        score = cls_logits.view(-1).sum()
        score.backward()

        grads = self.gradients
        activs = self.activations

        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activs).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = F.interpolate(
            cam,
            size=x.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        cam_min = cam.amin(dim=(2, 3), keepdim=True)
        cam_max = cam.amax(dim=(2, 3), keepdim=True)
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

        return cam.detach()


class PneumoRAGPipeline:
    def __init__(
        self,
        assets_dir: str | os.PathLike,
        report_dir: str | os.PathLike | None = None,
        device: str | None = None,
    ):
        self.assets_dir = Path(assets_dir)
        self.image_root = Path(
        os.getenv("PNEUMO_IMAGE_ROOT", r"C:\Users\Steven\Desktop\Final Project\Datasets\Dataset_1\Chest X-Ray Images with Pneumothorax Masks\png_images")
        )
        self.autodl_image_prefix = "/root/autodl-tmp/myproject/data/Datasets/Dataset_1/Chest X-Ray Images with Pneumothorax Masks/png_images"
        self.report_dir = Path(report_dir or (self.assets_dir / "streamlit_outputs"))
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.overlay_dir = self.report_dir / "overlays"
        self.overlay_dir.mkdir(parents=True, exist_ok=True)
        self.explain_dir = self.report_dir / "explainability"
        self.explain_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.meta = pd.read_csv(self.assets_dir / "rag_meta.csv").reset_index(drop=True)
        self.embs = np.load(self.assets_dir / "embeddings.npy").astype("float32")
        assert len(self.meta) == self.embs.shape[0], "rag_meta.csv row count does not match embeddings.npy"

        with open(self.assets_dir / "checkpoint_infer.json", "r", encoding="utf-8") as f:
            self.cfg_json = json.load(f)

        self.ckpt = torch.load(self.assets_dir / "checkpoint_infer.pt", map_location="cpu", weights_only=False)
        self._init_columns()
        self._init_model()
        self._init_retrieval()
        self._init_text_kb()

    def _init_columns(self):
        self.PATH_COL = guess_col(self.meta, ["full_path", "image_path", "img_path", "path", "filepath", "file_path", "png_path"])
        self.CASEID_COL = guess_col(self.meta, ["image_id", "ImageId", "case_id", "id", "new_filename", "filename", "file_name"])
        self.SPLIT_COL = guess_col(self.meta, ["split", "subset", "set", "stage", "partition"])
        self.YTRUE_COL = guess_col(self.meta, ["y_true", "label", "gt", "target", "truth"])
        self.PCOL = guess_col(self.meta, ["p_calibrated", "p", "prob", "probability", "pred_prob"])
        self.YPRED_COL = guess_col(self.meta, ["y_pred", "pred", "yhat", "prediction"])

        if self.PATH_COL is None:
            raise ValueError(f"Cannot find a path column in metadata. Columns: {list(self.meta.columns)}")

        post = self.ckpt.get("postprocess", {}) or self.cfg_json
        pre = self.ckpt.get("preprocess", {}) or self.cfg_json

        self.THR = float(post.get("cls_threshold", post.get("threshold", post.get("thr", 0.5))))
        self.T = float(post.get("temperature_T", post.get("T", 1.0)))
        self.MASK_THR = float(post.get("mask_threshold", 0.5))

        self.IMG_SIZE = int(pre.get("img_size", pre.get("image_size", 512)))
        self.MEAN = [_as_list(pre.get("mean", pre.get("img_mean", 0.0)), 0.0)[0]]
        self.STD = [_as_list(pre.get("std", pre.get("img_std", 1.0)), 1.0)[0]]

        self._path2row = {str(p): i for i, p in enumerate(self.meta[self.PATH_COL].astype(str).tolist())}

        self.preprocess = tvT.Compose(
            [
                tvT.Grayscale(num_output_channels=1),
                tvT.Resize((self.IMG_SIZE, self.IMG_SIZE)),
                tvT.ToTensor(),
                tvT.Normalize(mean=self.MEAN, std=self.STD),
            ]
        )

    def _init_model(self):
        model_kwargs = self.ckpt.get("model_kwargs", {}) or dict(
            in_chans=1,
            n_classes=1,
            drop_path_rate=0.1,
            use_seg_guided=False,
            use_lsmf=True,
            lsmf_fuse_ch=256,
        )

        self.model = ConvNeXtV2TinyScratch(**model_kwargs).to(self.device)
        state = (
            self.ckpt.get("model_state")
            or self.ckpt.get("model_state_dict")
            or self.ckpt.get("state_dict")
            or self.ckpt.get("model")
        )
        if state is None:
            raise ValueError(f"No state dict found. Checkpoint keys: {list(self.ckpt.keys())}")
        if any(k.startswith("module.") for k in state.keys()):
            state = {k.replace("module.", "", 1): v for k, v in state.items()}
        self.model.load_state_dict(state, strict=False)
        self.model.eval()

        target_layer = getattr(self.model, "eca4", None)
        if target_layer is None:
            raise RuntimeError(
                "Grad-CAM target layer 'eca4' not found in current model."
            )

        self.gradcam = GradCAM(self.model, target_layer)

    def _init_retrieval(self):
        self.embs_norm = _l2norm(self.embs)
        n = len(self.meta)
        if self.SPLIT_COL is None:
            self.train_rowids = np.arange(n, dtype=np.int64)
        else:
            split = self.meta[self.SPLIT_COL].astype(str).str.lower()
            self.train_rowids = np.flatnonzero(split.isin(["train", "tr"])).astype(np.int64)
            if self.train_rowids.size == 0:
                self.train_rowids = np.arange(n, dtype=np.int64)

        self.train_embs = self.embs_norm[self.train_rowids]
        if HAS_FAISS:
            self.index = faiss.IndexFlatIP(self.train_embs.shape[1])
            self.index.add(self.train_embs)
        else:
            self.index = None

    def _init_text_kb(self):
        kb_path = self.assets_dir / "knowledge_opt.jsonl"
        self.kb_chunks = []
        with open(kb_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.kb_chunks.append(json.loads(line))

        self.REPORT_EXCLUDE_TAGS = {
            "calibration", "threshold", "metrics", "evaluation", "engineering",
            "logging", "privacy", "research", "documentation", "schema",
            "versioning", "ablation", "system_level",
        }

        try:
            from sentence_transformers import SentenceTransformer  # type: ignore

            self.text_encoder = SentenceTransformer("all-MiniLM-L6-v2")
            kb_texts = [c["text"] for c in self.kb_chunks]
            self.kb_embs = self.text_encoder.encode(kb_texts, normalize_embeddings=True).astype("float32")
            if HAS_FAISS:
                self.text_index = faiss.IndexFlatIP(self.kb_embs.shape[1])
                self.text_index.add(self.kb_embs)
            else:
                self.text_index = None
            self.text_backend = "sentence-transformers"
        except Exception:
            from sklearn.feature_extraction.text import TfidfVectorizer

            kb_texts = [c["text"] for c in self.kb_chunks]
            self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
            self.kb_tfidf = self.vectorizer.fit_transform(kb_texts)
            self.text_encoder = None
            self.text_index = None
            self.text_backend = "tfidf"

    def save_uploaded_bytes(self, file_name: str, file_bytes: bytes) -> str:
        suffix = Path(file_name).suffix.lower() or ".png"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=self.report_dir) as f:
            f.write(file_bytes)
            return f.name

    def save_seg_overlay_png(
        self,
        image_path: str,
        seg_logits: torch.Tensor,
        out_png: str,
        thr: float | None = None,
        alpha: float = 0.35,
        draw_bbox: bool = True,
    ) -> str:
        base = Image.open(image_path).convert("RGB")
        w, h = base.size

        prob = torch.sigmoid(seg_logits)
        prob = F.interpolate(prob, size=(h, w), mode="bilinear", align_corners=False)[0, 0]
        prob_np = prob.detach().cpu().numpy()

        red = (prob_np * 255).clip(0, 255).astype(np.uint8)
        overlay_np = np.zeros((h, w, 3), dtype=np.uint8)
        overlay_np[..., 0] = red
        overlay_img = Image.fromarray(overlay_np, mode="RGB")
        blended = Image.blend(base, overlay_img, float(alpha))

        mask_thr = float(thr if thr is not None else self.MASK_THR)
        if draw_bbox:
            mask = prob_np >= mask_thr
            if mask.any():
                ys, xs = np.where(mask)
                x1, x2 = int(xs.min()), int(xs.max())
                y1, y2 = int(ys.min()), int(ys.max())
                draw = ImageDraw.Draw(blended)
                lw = max(2, int(min(w, h) * 0.005))
                draw.rectangle([x1, y1, x2, y2], outline=(255, 255, 0), width=lw)

        Path(out_png).parent.mkdir(parents=True, exist_ok=True)
        blended.save(out_png)
        return out_png

    def save_gradcam_overlay_png(
        self,
        image_path: str,
        cam_tensor: torch.Tensor,
        out_png: str,
        alpha: float = 0.45,
    ) -> str:
        base = Image.open(image_path).convert("RGB")
        w, h = base.size

        cam_np = cam_tensor.detach().cpu().numpy()
        cam_np = np.clip(cam_np, 0.0, 1.0)

        heat_rgb = (cm.jet(cam_np)[..., :3] * 255).astype(np.uint8)
        heat_img = Image.fromarray(heat_rgb, mode="RGB")
        heat_img = heat_img.resize((w, h), resample=Image.BILINEAR)

        blended = Image.blend(base, heat_img, float(alpha))

        Path(out_png).parent.mkdir(parents=True, exist_ok=True)
        blended.save(out_png)
        return out_png

    def infer_one(self, image_path: str):
        image_path = str(image_path)

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Input image not found: {image_path}")

        img = Image.open(image_path).convert("L")
        x = self.preprocess(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            cls_logits, seg_logits = self.model(x)

            if not hasattr(self.model, "last_feats"):
                raise RuntimeError("self.model has no attribute 'last_feats'")

            if "lsmf_vec" not in self.model.last_feats:
                raise RuntimeError(
                    "lsmf_vec not found in self.model.last_feats. "
                    "Please make sure the GUI is loading the LSMF+ECA model "
                    "and the model is running with use_lsmf=True."
                )

            emb = self.model.last_feats["lsmf_vec"]
            logits = cls_logits.view(-1).float()

            p = torch.sigmoid(logits / float(self.T)).item()
            yhat = int(p >= float(self.THR))

            emb_np = emb.detach().cpu().numpy().astype("float32")
            if emb_np.ndim == 1:
                emb_np = emb_np[None, :]

            if emb_np.shape[1] != self.embs.shape[1]:
                raise RuntimeError(
                    f"Embedding dim mismatch: query D={emb_np.shape[1]} vs library D={self.embs.shape[1]}"
                )

            seg_prob = torch.sigmoid(seg_logits)[0, 0].detach().cpu().numpy()

        with torch.enable_grad():
            cam = self.gradcam(x)[0, 0].cpu()

        stem = Path(image_path).stem
        ts = time.strftime("%Y%m%d_%H%M%S")

        overlay_path = str(self.overlay_dir / f"{stem}_{ts}_overlay.png")
        gradcam_overlay_path = str(self.explain_dir / f"{stem}_{ts}_gradcam_overlay.png")

        if yhat == 1:
            self.save_seg_overlay_png(
                image_path=image_path,
                seg_logits=seg_logits,
                out_png=overlay_path,
                thr=0.5,
                alpha=0.35,
                draw_bbox=True,
            )
        else:
            base = Image.open(image_path).convert("RGB")
            Path(overlay_path).parent.mkdir(parents=True, exist_ok=True)
            base.save(overlay_path)

        self.save_gradcam_overlay_png(
            image_path=image_path,
            cam_tensor=cam,
            out_png=gradcam_overlay_path,
            alpha=0.45,
        )

        return {
            "embedding": emb_np,
            "p_calibrated": float(p),
            "y_pred": int(yhat),
            "overlay_path": overlay_path,
            "gradcam_overlay_path": gradcam_overlay_path,
            "seg_prob_small": seg_prob,
            "cls_logits": float(logits.item()),
        }

    def embed_and_predict(self, image_path: str):
        return self.infer_one(image_path)

    def image_retrieve_by_vector(self, q_vec: np.ndarray, topk: int = 5, exclude_case_id=None) -> pd.DataFrame:
        q = _l2norm(q_vec)
        extra = 10
        if HAS_FAISS and self.index is not None:
            scores, idxs = self.index.search(q, topk + extra)
            row_ids = self.train_rowids[idxs[0]].astype(int)
            sims = scores[0]
        else:
            scores = self.train_embs @ q[0]
            idxs = np.argsort(-scores)[: topk + extra]
            row_ids = self.train_rowids[idxs].astype(int)
            sims = scores[idxs]

        hits = self.meta.iloc[row_ids].copy()
        hits["sim"] = sims

        if exclude_case_id is not None:
            if self.CASEID_COL and self.CASEID_COL in hits.columns:
                hits = hits[hits[self.CASEID_COL].astype(str) != str(exclude_case_id)]
        return hits.head(topk).reset_index(drop=True)

    def get_case_id_from_path(self, image_path: str) -> str:
        return Path(str(image_path)).name

    def shrink_hits_for_llm(self, hits_df: pd.DataFrame, topk: int):
        out = []

        for _, row in hits_df.head(topk).iterrows():
            raw_path = None
            if self.PATH_COL and self.PATH_COL in hits_df.columns:
                raw_path = row[self.PATH_COL]

            case_id = (
                str(row[self.CASEID_COL])
                if (self.CASEID_COL and self.CASEID_COL in hits_df.columns)
                else self.get_case_id_from_path(raw_path if raw_path is not None else "")
            )

            resolved_path = self.resolve_existing_image_path(raw_path)

            item = {
                "case_id": case_id,
                "sim": float(row["sim"]) if "sim" in hits_df.columns else None,
                # "raw_image_path": str(raw_path) if raw_path is not None and pd.notna(raw_path) else None,
                "image_path": resolved_path,
                # "image_exists": bool(resolved_path and Path(resolved_path).exists()),
            }

            if self.YTRUE_COL and self.YTRUE_COL in hits_df.columns:
                v = row[self.YTRUE_COL]
                try:
                    item["label"] = int(v)
                except Exception:
                    item["label"] = str(v)

            if self.YPRED_COL and self.YPRED_COL in hits_df.columns:
                v = row[self.YPRED_COL]
                try:
                    item["pred_label"] = int(v)
                except Exception:
                    item["pred_label"] = str(v)

            if self.PCOL and self.PCOL in hits_df.columns:
                v = row[self.PCOL]
                try:
                    item["retrieved_probability"] = float(v)
                except Exception:
                    item["retrieved_probability"] = str(v)

            out.append(item)

        return out
    
    def resolve_existing_image_path(self, raw_path) -> str | None:
        if raw_path is None:
            return None

        try:
            if pd.isna(raw_path):
                return None
        except Exception:
            pass

        raw = str(raw_path).strip()
        if not raw:
            return None

        p = Path(raw)
        if p.exists():
            return str(p)

        name = p.name
        if name:
            candidate = self.image_root / name
            if candidate.exists():
                return str(candidate)

        if self.image_root.exists() and name:
            matches = list(self.image_root.rglob(name))
            if matches:
                return str(matches[0])

        return None

    def image_rag_for_image(self, image_path: str, topk: int = 5):
        pack = self.embed_and_predict(image_path)
        emb = pack["embedding"]

        exclude_id = None
        key = str(image_path)
        if key in self._path2row and self.CASEID_COL and self.CASEID_COL in self.meta.columns:
            m = self.meta[self.meta[self.PATH_COL].astype(str) == key]
            exclude_id = str(m.iloc[0][self.CASEID_COL]) if len(m) > 0 else None

        hits = self.image_retrieve_by_vector(emb, topk=topk, exclude_case_id=exclude_id)
        similar_public = self.shrink_hits_for_llm(hits, topk=topk)
        retrieved_case_ids = [x.get("case_id") for x in similar_public if isinstance(x, dict)]

        # first_img = similar_public[0].get("image_path") if similar_public else None
        # first_exists = similar_public[0].get("image_exists") if similar_public else None

        return {
            "p_calibrated": float(pack["p_calibrated"]),
            "y_pred": int(pack["y_pred"]),
            "threshold": float(self.THR),
            "temperature_T": float(self.T),
            "overlay_path": pack["overlay_path"],
            "overlay_filename": os.path.basename(pack["overlay_path"]) if pack["overlay_path"] else None,
            "overlay_path": pack["overlay_path"],
            "overlay_filename": os.path.basename(pack["overlay_path"]) if pack["overlay_path"] else None,
            "gradcam_overlay_path": pack.get("gradcam_overlay_path"),
            "gradcam_overlay_filename": (
                os.path.basename(pack["gradcam_overlay_path"])
                if pack.get("gradcam_overlay_path") else None
            ),
            "similar_cases": similar_public,
            "retrieved_case_ids": retrieved_case_ids,
            # "debug": {
            #     "sim_mean": float(hits["sim"].mean()) if "sim" in hits.columns else None,
            #     "path_col": self.PATH_COL,
            #     "first_similar_image_path": first_img,
            #     "first_similar_image_exists": first_exists,
            # },
        }

    def chunk_allowed_for_report(self, item: dict) -> bool:
        tags = set(item.get("tags", []))
        return len(tags & self.REPORT_EXCLUDE_TAGS) == 0

    def scenario_tag_set(self, scenario: dict) -> set[str]:
        out = {"report_use"}
        if scenario.get("polarity"):
            out.add(str(scenario["polarity"]))
        if scenario.get("confidence"):
            out.add(str(scenario["confidence"]))
        if scenario.get("retrieval_state"):
            out.add(str(scenario["retrieval_state"]))
        if scenario.get("low_similarity", False):
            out.add("low_similarity")
        return out

    def chunk_bonus(self, item: dict, scenario: dict) -> float:
        tags = set(item.get("tags", []))
        sc_tags = self.scenario_tag_set(scenario)
        bonus = 0.0
        if "report_use" in tags:
            bonus += 0.15
        bonus += 0.06 * len(tags & sc_tags)
        if "uncertainty" in tags and scenario.get("confidence") == "borderline":
            bonus += 0.10
        if "conflict" in tags and scenario.get("retrieval_state") == "conflict":
            bonus += 0.10
        if "agreement" in tags and scenario.get("retrieval_state") == "agreement":
            bonus += 0.08
        if "low_similarity" in tags and scenario.get("low_similarity", False):
            bonus += 0.08
        return bonus

    def _retrieve_text_candidates(self, query: str, fetch_k: int = 20):
        if self.text_backend == "sentence-transformers" and self.text_encoder is not None:
            q = self.text_encoder.encode([query], normalize_embeddings=True).astype("float32")
            if HAS_FAISS and self.text_index is not None:
                scores, idxs = self.text_index.search(q, fetch_k)
                idxs = idxs[0]
                scores = scores[0]
            else:
                raw = self.kb_embs @ q[0]
                idxs = np.argsort(-raw)[:fetch_k]
                scores = raw[idxs]

            out = []
            for s, i in zip(scores, idxs):
                item = dict(self.kb_chunks[int(i)])
                item["sim"] = float(s)
                out.append(item)
            return out

        q = self.vectorizer.transform([query])
        scores = (self.kb_tfidf @ q.T).toarray().ravel()
        idxs = np.argsort(-scores)[:fetch_k]
        out = []
        for i in idxs:
            item = dict(self.kb_chunks[int(i)])
            item["sim"] = float(scores[i])
            out.append(item)
        return out

    def text_retrieve(self, query: str, scenario: dict, topk: int = 8, fetch_k: int = 20):
        candidates = self._retrieve_text_candidates(query, fetch_k=fetch_k)
        reranked = []
        for item in candidates:
            if not self.chunk_allowed_for_report(item):
                continue
            x = dict(item)
            x["score"] = float(x["sim"]) + self.chunk_bonus(x, scenario)
            reranked.append(x)
        reranked.sort(key=lambda z: z["score"], reverse=True)
        return reranked[:topk]

    @staticmethod
    def confidence_band(p: float, thr: float) -> str:
        d = abs(float(p) - float(thr))
        if d < 0.03:
            return "borderline"
        if d < 0.10:
            return "moderate"
        return "high"

    @staticmethod
    def narrative_label(p: float, thr: float) -> str:
        d = abs(float(p) - float(thr))
        if d < 0.03:
            return "indeterminate"
        return "positive" if float(p) >= float(thr) else "negative"

    def analyze_retrieval_context(self, similar_cases: list, yhat: int) -> dict:
        sims = []
        preds = []
        for x in similar_cases or []:
            if not isinstance(x, dict):
                continue
            if "sim" in x:
                try:
                    sims.append(float(x["sim"]))
                except Exception:
                    pass
            if "y_pred" in x:
                try:
                    preds.append(int(x["y_pred"]))
                except Exception:
                    pass

        mean_similarity = float(np.mean(sims)) if sims else None
        if not preds:
            agreement_rate = None
            retrieval_state = "limited"
        else:
            agreement_rate = float(np.mean([int(p == int(yhat)) for p in preds]))
            if agreement_rate >= 0.80:
                retrieval_state = "agreement"
            elif agreement_rate <= 0.60:
                retrieval_state = "conflict"
            else:
                retrieval_state = "mixed"

        low_similarity = mean_similarity is not None and mean_similarity < 0.35
        return {
            "num_cases": len(similar_cases or []),
            "mean_similarity": mean_similarity,
            "agreement_rate": agreement_rate,
            "retrieval_state": retrieval_state,
            "low_similarity": low_similarity,
        }

    def build_text_query(self, pack: dict, retrieval_ctx: dict) -> str:
        p = float(pack["p_calibrated"])
        thr = float(pack["threshold"])
        yhat = int(pack["y_pred"])
        polarity = "positive" if yhat == 1 else "negative"
        conf = self.confidence_band(p, thr)
        rstate = retrieval_ctx.get("retrieval_state", "limited")
        low_sim = retrieval_ctx.get("low_similarity", False)

        return (
            f"pneumothorax automated report wording; "
            f"{polarity} output; "
            f"{conf} confidence; "
            f"retrieval {rstate}; "
            f"{'low similarity context; ' if low_sim else ''}"
            f"findings impression recommendations limitations; "
            f"classification-only; "
            f"no laterality no localization no size no extent no tension; "
            f"safe concise professional wording"
        )

    @staticmethod
    def chunk_family(item: dict) -> str:
        tags = set(item.get("tags", []))
        if "template" in tags:
            return "template"
        if "findings" in tags:
            return "findings"
        if "impression" in tags:
            return "impression"
        if "recommendation" in tags:
            return "recommendation"
        if "limitation" in tags or "limitations" in tags:
            return "limitation"
        if "agreement" in tags or "conflict" in tags or "low_similarity" in tags:
            return "retrieval"
        if "uncertainty" in tags:
            return "uncertainty"
        return "other"

    def select_diverse_text_chunks(self, text_hits: list, max_chunks: int = 6) -> list:
        chosen = []
        used_ids = set()
        target_order = ["template", "findings", "impression", "recommendation", "limitation", "uncertainty", "retrieval"]

        for fam in target_order:
            for t in text_hits:
                cid = t["chunk_id"]
                if cid in used_ids:
                    continue
                if self.chunk_family(t) == fam:
                    chosen.append({
                        "chunk_id": t["chunk_id"],
                        "tags": t.get("tags", []),
                        "text": t["text"],
                        "sim": float(t.get("sim", 0.0)),
                        "score": float(t.get("score", t.get("sim", 0.0))),
                    })
                    used_ids.add(cid)
                    break
            if len(chosen) >= max_chunks:
                return chosen

        for t in text_hits:
            cid = t["chunk_id"]
            if cid in used_ids:
                continue
            chosen.append({
                "chunk_id": t["chunk_id"],
                "tags": t.get("tags", []),
                "text": t["text"],
                "sim": float(t.get("sim", 0.0)),
                "score": float(t.get("score", t.get("sim", 0.0))),
            })
            used_ids.add(cid)
            if len(chosen) >= max_chunks:
                break
        return chosen

    def summarize_similar_cases(self, similar_cases: list) -> dict:
        sims = []
        preds = []
        for x in similar_cases or []:
            if isinstance(x, dict):
                if "sim" in x:
                    try:
                        sims.append(float(x["sim"]))
                    except Exception:
                        pass
                if "y_pred" in x:
                    try:
                        preds.append(int(x["y_pred"]))
                    except Exception:
                        pass
        return {
            "num_cases": len(similar_cases or []),
            "mean_similarity": float(np.mean(sims)) if sims else None,
            "retrieved_pred_positive_rate": float(np.mean(preds)) if preds else None,
        }

    def build_payload(self, image_path: str, topk_img: int = 5, topk_text: int = 6) -> dict:
        pack = self.image_rag_for_image(image_path, topk=topk_img)

        similar_cases = pack.get("similar_cases", []) or []
        image_rag_summary = self.summarize_similar_cases(similar_cases)
        retrieval_ctx = self.analyze_retrieval_context(similar_cases, pack["y_pred"])

        scenario = {
            "polarity": "positive" if int(pack["y_pred"]) == 1 else "negative",
            "confidence": self.confidence_band(pack["p_calibrated"], pack["threshold"]),
            "narrative_label": self.narrative_label(pack["p_calibrated"], pack["threshold"]),
            "retrieval_state": retrieval_ctx["retrieval_state"],
            "low_similarity": retrieval_ctx["low_similarity"],
        }

        q = self.build_text_query(pack, retrieval_ctx)
        text_hits = self.text_retrieve(q, scenario=scenario, topk=max(topk_text * 3, 12), fetch_k=20)
        text_chunks = self.select_diverse_text_chunks(text_hits, max_chunks=min(topk_text, 6))

        overlay_path = pack.get("overlay_path")
        overlay_filename = pack.get("overlay_filename")
        gradcam_overlay_path = pack.get("gradcam_overlay_path")
        gradcam_overlay_filename = pack.get("gradcam_overlay_filename")

        return {
            "task": "pneumothorax_binary_classification",
            "prediction": {
                "p_calibrated": float(pack["p_calibrated"]),
                "y_pred": int(pack["y_pred"]),
                "threshold": float(pack["threshold"]),
                "temperature_T": float(pack["temperature_T"]),
                "confidence_band": scenario["confidence"],
                "narrative_label": scenario["narrative_label"],
            },
            "report_constraints": {
                "allow_localization": False,
                "allow_laterality": False,
                "allow_size": False,
                "allow_extent": False,
                "allow_tension": False,
                "allow_specific_imaging_signs": False,
            },
            "image_rag": {
                "topk": int(topk_img),
                "summary": image_rag_summary,
                "behaviour_context": retrieval_ctx,
                "retrieved_case_ids": pack.get("retrieved_case_ids", []),
                "similar_cases": similar_cases,
            },
            "text_rag": {
                "query": q,
                "scenario": scenario,
                "topk": len(text_chunks),
                "evidence_chunks": text_chunks,
            },
            "visual_support": {
                "overlay_available": bool(overlay_path),
                "overlay_path": overlay_path,
                "overlay_filename": overlay_filename,
                "overlay_note": (
                    "An accompanying visual overlay highlights the model-identified region of interest for review."
                    if overlay_path else ""
                ),
            },
            "explainability": {
                "gradcam_available": bool(gradcam_overlay_path),
                "gradcam_overlay_path": gradcam_overlay_path,
                "gradcam_overlay_filename": gradcam_overlay_filename,
                "gradcam_note": (
                    "Grad-CAM overlay highlights the image regions that most influenced the classification decision."
                    if gradcam_overlay_path else ""
                ),
            },
        }
