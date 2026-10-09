
import os, pickle, random, math, torch
import numpy as np
import cv2
import torch.nn.functional as F
import os.path as osp
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.data import DataManager
from utils.create_image import *
from clip_cam import clip
from tqdm import tqdm
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from torch.utils.data import DataLoader
from torchvision import transforms
_tokenizer = _Tokenizer()
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
try:
    from skimage.restoration import inpaint_biharmonic
except Exception:
    inpaint_biharmonic = None
try:
    from simple_lama_inpainting import SimpleLama
except Exception:
    SimpleLama = None

_LAMA_MODEL = None
            
def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    if backbone_name == "ViT-L/14" and backbone_name not in clip._MODELS:
        clip._MODELS[backbone_name] = "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt"
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)
    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model    


def fill_mask_region(image_orig, keep_mask, method, rng=None):
    method = method.lower()
    fill_mask = (1 - keep_mask).astype(np.uint8)
    removed = image_orig * np.expand_dims(keep_mask, axis=-1)

    if method == "telea":
        return cv2.inpaint(removed, fill_mask, 5, cv2.INPAINT_TELEA)
    if method == "ns":
        return cv2.inpaint(removed, fill_mask, 5, cv2.INPAINT_NS)
    if method == "biharmonic":
        if inpaint_biharmonic is None:
            raise RuntimeError("skimage.restoration.inpaint_biharmonic is unavailable")
        image_float = image_orig.astype(np.float32) / 255.0
        mask_bool = fill_mask.astype(bool)
        try:
            repaired = inpaint_biharmonic(image_float, mask_bool, channel_axis=-1)
        except TypeError:
            repaired = inpaint_biharmonic(image_float, mask_bool, multichannel=True)
        return np.clip(repaired * 255.0, 0, 255).astype(np.uint8)
    if method == "lama":
        if SimpleLama is None:
            raise RuntimeError("simple_lama_inpainting is unavailable")
        global _LAMA_MODEL
        if _LAMA_MODEL is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            _LAMA_MODEL = SimpleLama(device=device)
        image_pil = Image.fromarray(image_orig)
        mask_pil = Image.fromarray((fill_mask * 255).astype(np.uint8))
        repaired = _LAMA_MODEL(image_pil, mask_pil)
        return np.array(repaired.convert("RGB"))
    if method == "zero":
        return removed
    if method == "noise":
        rng = np.random.default_rng() if rng is None else rng
        noisy = removed.astype(np.int16)
        noise = rng.normal(0.0, 18.0, size=removed.shape).astype(np.int16)
        noisy[fill_mask.astype(bool)] += noise[fill_mask.astype(bool)]
        return np.clip(noisy, 0, 255).astype(np.uint8)
    raise ValueError(f"Unsupported inpaint method: {method}")
            
class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.COOP.N_CTX
        ctx_init = cfg.TRAINER.COOP.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            # random initialization
            if cfg.TRAINER.COOP.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.COOP.CLASS_TOKEN_POSITION

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        return prompts
class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.cfg = cfg
        self.image_encoder = clip_model.visual
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.clip_model = clip_model
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dm = DataManager(cfg)
        
    def forward(self, image):
        image_features = self.image_encoder(image.type(self.dtype))
        temp = "a photo of a {}."
        classnames = self.dm.dataset.classnames
        prompts = [temp.format(c.replace("_", " ")) for c in classnames]
        prompts = torch.cat([clip.tokenize(p) for p in prompts])
        prompts = prompts.to(self.device)
        text_features = self.clip_model.encode_text(prompts)
        
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits    


    @torch.no_grad()
    def visualize_text_clusters(self, K=50, save_path="./cluster_kmeans/vis_clip_clusters.png"):
        """
        在 ImageNet 或任意数据集 classnames 上，进行：
        - CLIP encode_text()
        - Cosine KMeans 聚类
        - t-SNE 可视化
        - 打印每个簇内的类别成员统计
        """
        print("=== [CLIP Visualization] Start CLIP text embedding clustering ===")
        device = self.device
        clip_model = self.clip_model
        classnames = self.dm.dataset.classnames

        # --- Step 1: 生成文本特征 ---
        print(f"Encoding {len(classnames)} text embeddings ...")
        temp = "a photo of a {}."
        prompts = [temp.format(c.replace("_", " ")) for c in classnames]
        prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(device)

        text_features = clip_model.encode_text(prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        text_features = text_features.cpu().numpy()

        # --- Step 2: 余弦距离 KMeans 聚类 ---
        print(f"Running cosine KMeans clustering (K={K}) ...")
        kmeans = KMeans(n_clusters=K, n_init="auto")
        labels = kmeans.fit_predict(text_features)
        centers = kmeans.cluster_centers_
        centers /= np.linalg.norm(centers, axis=1, keepdims=True)

        # --- Step 2.5: 打印每个簇的成员类别 ---
        print("\n=== [Cluster Composition Summary] ===")
        cluster_stats = {}
        for i, lbl in enumerate(labels):
            cluster_stats.setdefault(lbl, []).append(classnames[i])

        # 打印结果，按簇编号排序
        for cid in sorted(cluster_stats.keys()):
            members = cluster_stats[cid]
            print(f"Cluster {cid:03d} ({len(members)} classes): {', '.join(members[:10])}"
                  + (" ..." if len(members) > 10 else ""))

        # --- Step 3: t-SNE 降维 ---
        print("\nRunning t-SNE (may take a few minutes) ...")
        tsne = TSNE(
            n_components=2,
            metric="cosine"
        )
        emb_2d = tsne.fit_transform(text_features)

        # --- Step 4: 绘制聚类结果 ---
        plt.figure(figsize=(12, 10))
        colors = plt.cm.get_cmap("tab20", K)
        for i in range(len(classnames)):
            plt.scatter(
                emb_2d[i, 0],
                emb_2d[i, 1],
                color=colors(labels[i] % 20),
                s=20,
                alpha=0.9,
                edgecolor="none",
            )

        for k in range(K):
            cluster_center = emb_2d[labels == k].mean(axis=0)
            plt.text(
                cluster_center[0],
                cluster_center[1],
                f"C{k}",
                fontsize=7,
                color="black",
                ha="center",
                fontweight="bold",
            )

        plt.title(f"CLIP Text Embedding Clusters (K={K})", fontsize=13)
        plt.axis("off")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"\n✅ Saved CLIP cluster visualization to {save_path}")

        
    def create_images_from_split(
        self,
        split_file,
        neg_dir="neg_images",
        mask_ratio_range=(0.3, 0.5),
        mask_mode="ratio",
        cam_mass_levels=(0.5, 0.6, 0.7),
        batch_size=2,
        num_neg_per_image=3,
        num_workers=1,
        image_format="png",
        jpeg_quality=95,
        inpaint_method="telea",
        mask_source="cam",
        skip_existing=False,
        overwrite_labels=True,
        max_images=None,
    ):
        split_path = os.path.join(self.dm.dataset.split_fewshot_dir, split_file)
        print(f"Loading few-shot split from {split_path}")
        with open(split_path, "rb") as f:
            data = pickle.load(f)
            train_items = data["train"]
        if max_images is not None:
            train_items = train_items[:max_images]

        supported_inpaint_methods = {
            "telea", "ns", "lama", "biharmonic", "zero", "noise", "blur", "mean"
        }
        inpaint_methods = [method.strip().lower() for method in inpaint_method.split(",") if method.strip()]
        if not inpaint_methods:
            raise ValueError("At least one inpaint method is required")
        invalid_methods = sorted(set(inpaint_methods) - supported_inpaint_methods)
        if invalid_methods:
            raise ValueError(f"Unsupported inpaint method(s): {', '.join(invalid_methods)}")

        # A comma-separated method list shares attribution and masking work, while
        # writing each reconstruction variant to an independent dataset directory.
        multi_inpaint = len(inpaint_methods) > 1
        method_neg_dirs = {
            method: os.path.join(neg_dir, method) if multi_inpaint else neg_dir
            for method in inpaint_methods
        }
        neg_label_files = {}
        neg_label_lines = {method: [] for method in inpaint_methods}
        mask_stats_file = os.path.join(neg_dir, "mask_stats.csv")
        mask_stats_lines = []
        os.makedirs(neg_dir, exist_ok=True)
        if overwrite_labels and os.path.exists(mask_stats_file):
            os.remove(mask_stats_file)
        if not os.path.exists(mask_stats_file):
            with open(mask_stats_file, "w") as f:
                f.write("image_index,class_id,negative_index,evidence_level,masked_pixels,masked_fraction,mask_source\n")
        for method, method_neg_dir in method_neg_dirs.items():
            os.makedirs(method_neg_dir, exist_ok=True)
            label_file = os.path.join(method_neg_dir, "labels.txt")
            if overwrite_labels and os.path.exists(label_file):
                os.remove(label_file)
            neg_label_files[method] = label_file

        def flush_labels():
            for method in inpaint_methods:
                lines = neg_label_lines[method]
                if lines:
                    with open(neg_label_files[method], "a") as f:
                        f.writelines(lines)
                    lines.clear()
            if mask_stats_lines:
                with open(mask_stats_file, "a") as f:
                    f.writelines(mask_stats_lines)
                mask_stats_lines.clear()

        model = self.clip_model
        device = self.device
        model.eval()

        # classnames = [name.replace("_", " ") for name in self.dm.dataset.classnames]
        transform = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]
            )
        ])
        dataset = FewShotDataset(train_items, transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        image_format = image_format.lower().lstrip(".")
        if image_format not in {"png", "jpg", "jpeg"}:
            raise ValueError(f"Unsupported image_format: {image_format}")
        mask_source = mask_source.lower()
        supported_mask_sources = {
            "cam",
            "attention_rollout",
            "attention_rollout_area_matched",
            "gradcam_area_matched",
            "gradcampp_area_matched",
        }
        if mask_source not in supported_mask_sources:
            raise ValueError(f"Unsupported mask_source: {mask_source}")
        if mask_source != "cam" and hasattr(model.visual, "layer4"):
            raise ValueError(f"{mask_source} masks require a CLIP ViT backbone")
        mask_mode = mask_mode.lower()
        if mask_mode not in {"ratio", "mass"}:
            raise ValueError(f"Unsupported mask_mode: {mask_mode}")
        cam_mass_levels = tuple(float(x) for x in cam_mass_levels)
        if mask_mode == "mass":
            for level in cam_mass_levels:
                if level <= 0 or level >= 1:
                    raise ValueError(f"Evidence mass levels must be in (0, 1), got {level}")
            num_neg_per_image = len(cam_mass_levels)
        extension = "jpg" if image_format == "jpeg" else image_format
        imwrite_params = []
        if extension == "jpg":
            imwrite_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]

        global_idx = 0
        batch_count = 0
        for batch in tqdm(dataloader, desc="Generating negative and background-masked images"):
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            classnames_batch = batch["classname"]
            batch_size_ = images.size(0)
            texts = clip.tokenize(classnames_batch).to(device)
            with torch.cuda.amp.autocast():
                if mask_source == "attention_rollout":
                    _, R_image = interpret_batch_attention_rollout(
                        model=model,
                        image=images,
                        texts=texts,
                        device=device,
                        start_layer=0,
                    )
                    cam_relevance = R_image
                else:
                    _, cam_relevance = interpret_batch_paired(
                        model=model,
                        image=images,
                        texts=texts,
                        device=device,
                    )
                    if mask_source == "cam":
                        R_image = cam_relevance
                    elif mask_source == "attention_rollout_area_matched":
                        _, R_image = interpret_batch_attention_rollout(
                            model=model,
                            image=images,
                            texts=texts,
                            device=device,
                            start_layer=0,
                        )
                    elif mask_source == "gradcam_area_matched":
                        _, R_image = interpret_batch_vit_gradcam(
                            model=model,
                            image=images,
                            texts=texts,
                            device=device,
                            variant="gradcam",
                        )
                    else:
                        _, R_image = interpret_batch_vit_gradcam(
                            model=model,
                            image=images,
                            texts=texts,
                            device=device,
                            variant="gradcampp",
                        )

            for i in range(batch_size_):
                class_id = labels[i].item()
                class_folders = {}
                for method, method_neg_dir in method_neg_dirs.items():
                    class_folder = os.path.join(method_neg_dir, f"cls{class_id:03d}")
                    os.makedirs(class_folder, exist_ok=True)
                    class_folders[method] = class_folder

                relevance = R_image[i].cpu()
                side_length = int(math.sqrt(relevance.shape[0]))
                if relevance.numel() == 224 * 224:
                    # 确保形状是 [1, 1, 224, 224]
                    relevance = relevance.view(1, 1, 224, 224)

                # 2. 如果不是，说明是 ViT 的小特征图 (14x14 = 196)，需要 Reshape + 插值
                else:
                    # 自动计算边长 (针对 ViT-B/32 是 7，ViT-B/16 是 14)
                    side_length = int(relevance.numel() ** 0.5) 
                    relevance = relevance.view(1, 1, side_length, side_length)
                    relevance = torch.nn.functional.interpolate(
                        relevance, size=(224, 224), mode='bilinear', align_corners=False
                    )
                if mask_source == "attention_rollout":
                    # Rollout is already a non-negative attention distribution.
                    # Preserve its baseline mass when selecting a cumulative-mass mask.
                    relevance = relevance.clamp_min(0)
                else:
                    relevance = (relevance - relevance.min()) / (relevance.max() - relevance.min() + 1e-6)
                cam_reference = cam_relevance[i].detach().cpu()
                if cam_reference.numel() == 224 * 224:
                    cam_reference = cam_reference.view(1, 1, 224, 224)
                else:
                    cam_side = int(cam_reference.numel() ** 0.5)
                    cam_reference = cam_reference.view(1, 1, cam_side, cam_side)
                    cam_reference = torch.nn.functional.interpolate(
                        cam_reference, size=(224, 224), mode="bilinear", align_corners=False
                    )
                cam_reference = (cam_reference - cam_reference.min()) / (
                    cam_reference.max() - cam_reference.min() + 1e-6
                )
                image_orig = images[i].detach().cpu()
                mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
                std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
                image_orig = (image_orig * std + mean).permute(1, 2, 0).numpy()
                image_orig = np.clip(image_orig * 255, 0, 255).astype(np.uint8)
                flat = relevance.flatten()
                sorted_indices = torch.argsort(flat, descending=True).cpu().numpy()
                flat_np = flat.detach().cpu().numpy()
                cam_flat = cam_reference.flatten().detach().cpu().numpy()
                cam_order = np.argsort(cam_flat)[::-1]

                if mask_mode == "ratio":
                    base_ratios = np.linspace(mask_ratio_range[0], mask_ratio_range[1], num_neg_per_image)
                    noise = np.random.uniform(-0.05, 0.05, size=num_neg_per_image)
                    mask_specs = np.clip(base_ratios + noise, mask_ratio_range[0], mask_ratio_range[1])
                else:
                    sorted_values = flat_np[sorted_indices]
                    mass_cumsum = np.cumsum(sorted_values)
                    mass_total = float(mass_cumsum[-1]) + 1e-12
                    cam_sorted_values = cam_flat[cam_order]
                    cam_mass_cumsum = np.cumsum(cam_sorted_values)
                    cam_mass_total = float(cam_mass_cumsum[-1]) + 1e-12
                    mask_specs = cam_mass_levels

                for neg_idx, mask_spec in enumerate(mask_specs):
                    # 生成负样本
                    img_name_neg = f"img_{global_idx:06d}_cls{class_id}_neg{neg_idx}.{extension}"
                    if mask_mode == "ratio":
                        num_pixels_to_mask = int(224 * 224 * float(mask_spec))
                    elif mask_source.endswith("_area_matched"):
                        num_pixels_to_mask = int(
                            np.searchsorted(cam_mass_cumsum, float(mask_spec) * cam_mass_total) + 1
                        )
                    else:
                        num_pixels_to_mask = int(np.searchsorted(mass_cumsum, float(mask_spec) * mass_total) + 1)
                    num_pixels_to_mask = int(np.clip(num_pixels_to_mask, 1, 224 * 224))
                    target_mask_topk = np.ones(224 * 224, dtype=np.uint8)
                    target_mask_topk[sorted_indices[:num_pixels_to_mask]] = 0
                    target_mask_topk = target_mask_topk.reshape(224, 224)
                    mask_stats_lines.append(
                        f"{global_idx},{class_id},{neg_idx},{float(mask_spec):.6f},"
                        f"{num_pixels_to_mask},{num_pixels_to_mask / float(224 * 224):.8f},{mask_source}\n"
                    )
                    for method in inpaint_methods:
                        inpainted_topk = fill_mask_region(image_orig, target_mask_topk, method)
                        neg_path = os.path.join(class_folders[method], img_name_neg)
                        if not (skip_existing and os.path.exists(neg_path)):
                            cv2.imwrite(neg_path, inpainted_topk[..., ::-1], imwrite_params)
                        rel_path = os.path.relpath(neg_path, method_neg_dirs[method])
                        neg_label_lines[method].append(f"{rel_path},{class_id}\n")

                global_idx += 1

            batch_count += 1
            if batch_count % 10 == 0:
                flush_labels()

        # 保存最后剩余标签
        flush_labels()

    def create_images_from_split_sweep(
        self,
        split_file,
        neg_root="neg_images",
        mask_ratio_range=(0.3, 0.7),
        batch_size=4,
        num_workers=4,
        num_neg_list=(16,),
        methods=("telea",),
        image_format="jpg",
        jpeg_quality=95,
        skip_existing=False,
        overwrite_labels=True,
        max_images=None,
    ):
        split_path = os.path.join(self.dm.dataset.split_fewshot_dir, split_file)
        print(f"Loading few-shot split from {split_path}")
        with open(split_path, "rb") as f:
            data = pickle.load(f)
            train_items = data["train"]
        if max_images is not None:
            train_items = train_items[:max_images]

        methods = [m.lower() for m in methods]
        for method in methods:
            if method not in {"telea", "ns", "lama", "biharmonic", "zero"}:
                raise ValueError(f"Unsupported inpaint method: {method}")
        num_neg_list = [int(n) for n in num_neg_list]

        image_format = image_format.lower().lstrip(".")
        if image_format not in {"png", "jpg", "jpeg"}:
            raise ValueError(f"Unsupported image_format: {image_format}")
        extension = "jpg" if image_format == "jpeg" else image_format
        imwrite_params = []
        if extension == "jpg":
            imwrite_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]

        def flush_dir(label_file, label_lines):
            if label_lines:
                with open(label_file, "a") as f:
                    f.writelines(label_lines)
                label_lines.clear()

        run_specs = [(method, num_neg) for method in methods for num_neg in num_neg_list]
        spec_dirs = {}
        spec_labels = {}
        spec_files = {}
        for method, num_neg in run_specs:
            spec_dir = os.path.join(neg_root, f"seed{self.cfg.SEED}", f"{method}_k{num_neg}_{mask_ratio_range[0]:g}_{mask_ratio_range[1]:g}_{extension}")
            os.makedirs(spec_dir, exist_ok=True)
            label_file = os.path.join(spec_dir, "labels.txt")
            if overwrite_labels and os.path.exists(label_file):
                os.remove(label_file)
            spec_dirs[(method, num_neg)] = spec_dir
            spec_labels[(method, num_neg)] = []
            spec_files[(method, num_neg)] = label_file

        model = self.clip_model
        device = self.device
        model.eval()
        transform = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]
            )
        ])
        dataset = FewShotDataset(train_items, transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        global_idx = 0
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Generating negative variants")):
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            classnames_batch = batch["classname"]
            texts = clip.tokenize(classnames_batch).to(device)
            with torch.cuda.amp.autocast():
                _, R_image = interpret_batch_paired(model=model, image=images, texts=texts, device=device)

            for i in range(images.size(0)):
                class_id = int(labels[i].item())
                relevance = R_image[i].detach().cpu()
                if relevance.numel() == 224 * 224:
                    relevance = relevance.view(1, 1, 224, 224)
                else:
                    side_length = int(relevance.numel() ** 0.5)
                    relevance = relevance.view(1, 1, side_length, side_length)
                    relevance = torch.nn.functional.interpolate(
                        relevance, size=(224, 224), mode="bilinear", align_corners=False
                    )
                relevance = (relevance - relevance.min()) / (relevance.max() - relevance.min() + 1e-6)
                image_orig = images[i].detach().cpu()
                mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
                std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
                image_orig = (image_orig * std + mean).permute(1, 2, 0).numpy()
                image_orig = np.clip(image_orig * 255, 0, 255).astype(np.uint8)
                flat = relevance.flatten()
                sorted_indices = torch.argsort(flat, descending=True).cpu().numpy()
                image_rng = np.random.default_rng(self.cfg.SEED * 1000003 + global_idx * 10007 + class_id * 97)

                for method, num_neg in run_specs:
                    spec_dir = spec_dirs[(method, num_neg)]
                    class_folder = os.path.join(spec_dir, f"cls{class_id:03d}")
                    os.makedirs(class_folder, exist_ok=True)
                    base_ratios = np.linspace(mask_ratio_range[0], mask_ratio_range[1], num_neg)
                    noise = image_rng.uniform(-0.05, 0.05, size=num_neg)
                    mask_ratios = np.clip(base_ratios + noise, mask_ratio_range[0], mask_ratio_range[1])
                    rng = np.random.default_rng(self.cfg.SEED * 1000003 + global_idx * 10007 + class_id * 97 + num_neg)

                    for neg_idx, mask_ratio in enumerate(mask_ratios):
                        img_name = f"img_{global_idx:06d}_cls{class_id}_{method}_neg{neg_idx}.{extension}"
                        num_pixels_to_mask = int(224 * 224 * mask_ratio)
                        keep_mask = np.ones(224 * 224, dtype=np.uint8)
                        keep_mask[sorted_indices[:num_pixels_to_mask]] = 0
                        keep_mask = keep_mask.reshape(224, 224)
                        neg_image = fill_mask_region(image_orig, keep_mask, method, rng=rng)
                        neg_path = os.path.join(class_folder, img_name)
                        if not (skip_existing and os.path.exists(neg_path)):
                            cv2.imwrite(neg_path, neg_image[..., ::-1], imwrite_params)
                        rel_path = os.path.relpath(neg_path, spec_dir)
                        spec_labels[(method, num_neg)].append(f"{rel_path},{class_id}\n")

                global_idx += 1

            if (batch_idx + 1) % 10 == 0:
                for spec in run_specs:
                    flush_dir(spec_files[spec], spec_labels[spec])

        for spec in run_specs:
            flush_dir(spec_files[spec], spec_labels[spec])


@TRAINER_REGISTRY.register()
class CoOp(TrainerX):
    """Context Optimization (CoOp).

    Learning to Prompt for Vision-Language Models
    https://arxiv.org/abs/2109.01134
    """

    def check_cfg(self, cfg):
        assert cfg.TRAINER.COOP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        if cfg.TRAINER.COOP.PREC == "fp32" or cfg.TRAINER.COOP.PREC == "amp":
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        # print("Turning off gradients in both the image and the text encoder")
        # for name, param in self.model.named_parameters():
        #     if "prompt_learner" not in name:
        #         param.requires_grad_(False)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)

        self.scaler = GradScaler() if cfg.TRAINER.COOP.PREC == "amp" else None

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)

        prec = self.cfg.TRAINER.COOP.PREC
        if prec == "amp":
            with autocast():
                output = self.model(image)
                loss = F.cross_entropy(output, label)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            output = self.model(image)
            loss = F.cross_entropy(output, label)
            self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(output, label)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)

            
