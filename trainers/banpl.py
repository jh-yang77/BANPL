import torch.nn.functional as F
import os.path as osp
import torch.nn as nn
import math
from torch.amp import GradScaler, autocast
from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.data import DataManager
from torch.optim.lr_scheduler import LambdaLR
from utils.create_image import *
from utils.utils import *
from tqdm import tqdm
import os
from torch.utils.data import Dataset, DataLoader, Subset
import torch
from torch.utils.data import  DataLoader
from torchvision import transforms
import numpy as np
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()

import random, cv2, re
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from torch.cuda.amp import autocast

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


def build_banpl_train_transform(cfg, rrcrop_scale):
    """Build BANPL train-time image augmentation with a compact switch for ablations."""
    mode = getattr(cfg.TRAINER.BANPL, "AUG_MODE", "rrcrop")
    if mode == "resize_random_crop_pad":
        crop_padding = getattr(cfg.INPUT, "CROP_PADDING", 4)
        return transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomCrop(224, padding=crop_padding),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
        ])
    if mode == "rrcrop":
        return transforms.Compose([
            transforms.RandomResizedCrop(
                224,
                scale=rrcrop_scale,
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
        ])
    raise ValueError(f"Unsupported TRAINER.BANPL.AUG_MODE: {mode}")


def build_two_stage_warmup_cosine_scheduler(
    optimizer,
    total_epochs,
    first_stage_epochs,
    pos_lr,
    neg_lr,
    warmup_epochs=1,
    eta_min=1e-5,
):
    """Two independent epoch-level warmup+cosine schedules for POS and NEG stages."""
    base_lr = optimizer.param_groups[0]["lr"]
    warmup_epochs = max(0, int(warmup_epochs))
    eta_min = float(eta_min)

    def stage_factor(stage_epoch, stage_total, stage_lr):
        if stage_total <= 0:
            return stage_lr / base_lr
        if warmup_epochs > 0 and stage_epoch < warmup_epochs:
            warm = float(stage_epoch) / float(warmup_epochs)
            lr = eta_min + (stage_lr - eta_min) * warm
            return lr / base_lr
        decay_total = max(1, stage_total - warmup_epochs)
        progress = float(stage_epoch - warmup_epochs) / float(decay_total)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        lr = eta_min + (stage_lr - eta_min) * cosine
        return lr / base_lr

    def lr_lambda(epoch):
        if epoch < first_stage_epochs:
            return stage_factor(epoch, first_stage_epochs, pos_lr)
        return stage_factor(epoch - first_stage_epochs, total_epochs - first_stage_epochs, neg_lr)

    return LambdaLR(optimizer, lr_lambda)


def resolve_neg_dir(cfg):
    env_dir = os.environ.get("BANPL_NEG_DIR")
    if env_dir:
        label_file = osp.join(env_dir, "labels.txt")
        if osp.isfile(label_file):
            print(f"Using negative image directory from BANPL_NEG_DIR: {env_dir}")
            return env_dir
        raise FileNotFoundError(f"BANPL_NEG_DIR is set but labels.txt was not found: {env_dir}")

    raise FileNotFoundError("Set BANPL_NEG_DIR to the generated k=1 LaMa negatives directory containing labels.txt")

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
        x= self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x

# =========================================================
# === TextEncoder: 与 CLIP 完全一致 ===
# =========================================================
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

        eot_idx = tokenized_prompts.argmax(dim=-1)
        x = x[torch.arange(x.shape[0]), eot_idx] @ self.text_projection
        return x


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        print(n_cls)
        n_ctx = cfg.TRAINER.BANPL.N_CTX
        ctx_init = cfg.TRAINER.BANPL.CTX_INIT
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
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init
            neg_ctx_vectors = torch.empty(n_cls,n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(neg_ctx_vectors, std=0.02)

        else:
            # random initialization
            if cfg.TRAINER.BANPL.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            neg_ctx_vectors = torch.empty(n_cls,n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            nn.init.normal_(neg_ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized
        self.neg_ctx = nn.Parameter(neg_ctx_vectors)  # to be optimized
        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        neg_prompts = [prompt_prefix + " " + "not" + " " + name + "." for name in classnames]
        
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        neg_tokenized_prompts = torch.cat([clip.tokenize(p) for p in neg_prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
            neg_embedding = clip_model.token_embedding(neg_tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS
        self.register_buffer("neg_token_prefix", neg_embedding[:, :1, :])  # SOS
        self.register_buffer("neg_token_suffix", neg_embedding[:, 1 + n_ctx:, :])  # CLS, EOS
        
        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.neg_tokenized_prompts = neg_tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.BANPL.CLASS_TOKEN_POSITION

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        prefix = self.token_prefix
        suffix = self.token_suffix
        prompts = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)
                ctx,  # (n_cls, n_ctx, dim)
                suffix,  # (n_cls, *, dim)
            ],
            dim=1,
        )
        neg_ctx = self.neg_ctx
        if neg_ctx.dim() == 2:
            neg_ctx = neg_ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        neg_prefix = self.neg_token_prefix
        neg_suffix = self.neg_token_suffix
        neg_prompts = torch.cat(
            [
                neg_prefix,  # (n_cls, 1, dim)
                neg_ctx,  # (n_cls, n_ctx, dim)
                neg_suffix,  # (n_cls, *, dim)
            ],
            dim=1,
        )
        return prompts,neg_prompts


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.neg_tokenized_prompts = self.prompt_learner.neg_tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.clip_model = clip_model
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dm = DataManager(cfg)
        self.template ="a photo of a {}"
        self.classnames = classnames

    def encode_text_branch(self, prompts, tokenized_prompts, requires_grad):
        if requires_grad:
            return self.text_encoder(prompts.type(self.dtype), tokenized_prompts)
        with torch.no_grad():
            return self.text_encoder(prompts.type(self.dtype), tokenized_prompts)

    def forward(self, image, return_extra=False):
        image_features = self.image_encoder(image.type(self.dtype))

        prompts,neg_prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        neg_tokenized_prompts = self.neg_tokenized_prompts
        text_features = self.encode_text_branch(
            prompts,
            tokenized_prompts,
            self.prompt_learner.ctx.requires_grad,
        )
        neg_text_features = self.encode_text_branch(
            neg_prompts,
            neg_tokenized_prompts,
            self.prompt_learner.neg_ctx.requires_grad,
        )
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        combined_text_features = torch.cat([text_features, neg_text_features], dim=0)
        combined_text_features = combined_text_features / combined_text_features.norm(dim=-1, keepdim=True)
        text_features = combined_text_features[:text_features.size(0)]
        neg_text_features = combined_text_features[text_features.size(0):]
        
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()
        neg_logits = logit_scale * image_features @ neg_text_features.t()
        
        if return_extra:
            return logits, neg_logits, text_features, neg_text_features,image_features
        return logits, neg_logits


@TRAINER_REGISTRY.register()
class BANPL(TrainerX):

    def check_cfg(self, cfg):
        assert cfg.TRAINER.BANPL.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        self.classnames = classnames
        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.BANPL.PREC == "fp32" or cfg.TRAINER.BANPL.PREC == "amp":
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        if cfg.TRAINER.BANPL.TWO_STAGE_LR:
            self.sched = build_two_stage_warmup_cosine_scheduler(
                self.optim,
                total_epochs=cfg.OPTIM.MAX_EPOCH,
                first_stage_epochs=cfg.TRAINER.BANPL.STAGE_EPOCHS,
                pos_lr=cfg.TRAINER.BANPL.POS_LR,
                neg_lr=cfg.TRAINER.BANPL.NEG_LR,
                warmup_epochs=cfg.OPTIM.WARMUP_EPOCH,
                eta_min=cfg.TRAINER.BANPL.LEGACY_COSINE_ETA_MIN,
            )
            print(
                "BANPL two-stage LR: "
                f"pos_lr={cfg.TRAINER.BANPL.POS_LR}, "
                f"neg_lr={cfg.TRAINER.BANPL.NEG_LR}, "
                f"stage_epochs={cfg.TRAINER.BANPL.STAGE_EPOCHS}, "
                f"warmup_epochs={cfg.OPTIM.WARMUP_EPOCH}, "
                f"eta_min={cfg.TRAINER.BANPL.LEGACY_COSINE_ETA_MIN}"
            )
        else:
            self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)
        self.scaler = GradScaler(device='cuda') if cfg.TRAINER.BANPL.PREC == "amp" else None
        

        if cfg.TRAIN_MOD:
            print(f"BANPL train augmentation mode: {cfg.TRAINER.BANPL.AUG_MODE}")
            self._banpl_neg_loader_ready = False
            if cfg.TRAINER.BANPL.STAGE_EPOCHS > 0:
                self._build_pos_train_loader()
            else:
                self._build_neg_train_loader()

    def _build_pos_train_loader(self):
        pos_crop_scale = tuple(self.cfg.TRAINER.BANPL.POS_RRCROP_SCALE)
        if self.cfg.TRAINER.BANPL.AUG_MODE == "resize_random_crop_pad":
            print(
                "BANPL POS train augmentation: "
                f"mode=resize_random_crop_pad, padding={self.cfg.INPUT.CROP_PADDING}"
            )
        else:
            print(
                "BANPL POS train augmentation: "
                f"mode={self.cfg.TRAINER.BANPL.AUG_MODE}, pos_rrcrop={pos_crop_scale}"
            )
        transform_pos = build_banpl_train_transform(self.cfg, rrcrop_scale=pos_crop_scale)

        pos_dataset = CombinedDataset(
            orig_dataset=self.dm.dataset.train_x,
            neg_dataset=[],
            transform=transform_pos
        )
        self.train_loader_x = DataLoader(
            pos_dataset,
            batch_size=self.cfg.DATALOADER.TRAIN_X.BATCH_SIZE,
            shuffle=True,
            num_workers=self.cfg.DATALOADER.NUM_WORKERS
        )

    def _build_neg_train_loader(self):
        pos_crop_scale = tuple(self.cfg.TRAINER.BANPL.POS_RRCROP_SCALE)
        neg_crop_scale = tuple(self.cfg.TRAINER.BANPL.NEG_RRCROP_SCALE)
        if self.cfg.TRAINER.BANPL.AUG_MODE == "resize_random_crop_pad":
            print(
                "BANPL NEG train augmentation: "
                f"mode=resize_random_crop_pad, padding={self.cfg.INPUT.CROP_PADDING}"
            )
        else:
            print(
                "BANPL NEG train augmentation: "
                f"mode={self.cfg.TRAINER.BANPL.AUG_MODE}, "
                f"pos_rrcrop={pos_crop_scale}, neg_rrcrop={neg_crop_scale}"
            )

        transform_neg = build_banpl_train_transform(self.cfg, rrcrop_scale=neg_crop_scale)
        raw_neg = ImageDataset(
            dir=resolve_neg_dir(self.cfg),
            neg=True,
            transform=transform_neg
        )
        transform_pos = build_banpl_train_transform(self.cfg, rrcrop_scale=pos_crop_scale)

        topk_neg_indices = self.select_hardest_negatives(self.classnames)
        neg_dataset = Subset(raw_neg, topk_neg_indices)
        combined_dataset = CombinedDataset(
            orig_dataset=self.dm.dataset.train_x,
            neg_dataset=neg_dataset,
            transform=transform_pos
        )

        self.train_loader_x = DataLoader(
            combined_dataset,
            batch_size=self.cfg.DATALOADER.TRAIN_X.BATCH_SIZE,
            shuffle=True,
            num_workers=self.cfg.DATALOADER.NUM_WORKERS
        )
        self._banpl_neg_loader_ready = True

    def before_epoch(self):
        if (
            self.cfg.TRAIN_MOD
            and self.epoch >= self.cfg.TRAINER.BANPL.STAGE_EPOCHS
            and not getattr(self, "_banpl_neg_loader_ready", False)
        ):
            self._build_neg_train_loader()
            
            
    @torch.no_grad()
    def select_hardest_negatives(self, classnames,batch_size=None):
        if batch_size is None:
            batch_size = self.cfg.TRAINER.BANPL.NEG_SELECT_BATCH_SIZE

        # 1) 定义与训练相同的预处理 transform
        transform = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711]
                )
            ])

        # 2) 构造原始负样本集
        raw_neg = ImageDataset(
            dir=resolve_neg_dir(self.cfg),
            neg=True,
            transform=transform
        )

        neg_dataset = raw_neg

        mode = getattr(self.cfg.TRAINER.BANPL, "HARD_NEG_MODE", "low_prob_fraction")

        def image_group_key(sample):
            name = osp.basename(sample["img_path"])
            match = re.match(r"(img_\d+_cls\d+).*?_neg\d+\.", name)
            if match:
                return match.group(1)
            return osp.splitext(name)[0]

        groups = None
        if mode == "per_image_high_prob":
            groups = {}
            for idx, sample in enumerate(neg_dataset.samples):
                groups.setdefault(image_group_key(sample), []).append(idx)

            # For k=1 negative generation, each ID image already contributes one
            # candidate. Running CLIP scoring would select the same set and only
            # adds several minutes of redundant preprocessing.
            if groups and all(len(idxs) == 1 for idxs in groups.values()):
                topk_indices = [idxs[0] for idxs in groups.values()]
                print(
                    "\n✅ Skip hard-negative scoring: one negative per ID image "
                    f"already present ({len(topk_indices)} samples)."
                )
                return topk_indices

        device = self.device
        clip_model = self.model.clip_model.to(device).eval()

        prompts = [f"a photo of a {name}" for name in classnames]
        prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(device)
        with torch.no_grad():
            text_features = clip_model.encode_text(prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # 7) 计算所有负样本的「真实类概率」
        def compute_probs(dataset):
            loader = DataLoader(dataset, batch_size=batch_size,
                                shuffle=False,
                                num_workers=self.cfg.TRAINER.BANPL.NEG_LOADER_NUM_WORKERS)
            all_probs = []
            all_labels = []
            for batch in tqdm(loader, desc="Processing Negative"):
                images = batch["img"].to(device)
                labels = batch["label"].to(device)
                # 图像特征
                img_feats = clip_model.encode_image(images)
                img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
                # logits & softmax
                logits = clip_model.logit_scale * img_feats @ text_features.T
                probs = F.softmax(logits, dim=-1)
                # 取每个样本的真实类概率
                prob_targets = probs[torch.arange(probs.size(0)), labels]
                all_probs.append(prob_targets.cpu())
                all_labels.append(labels.cpu())
            return torch.cat(all_probs), torch.cat(all_labels)

        all_probs_neg, all_labels_neg = compute_probs(neg_dataset)

        topk_indices = []

        if mode == "per_image_high_prob":
            if groups is None:
                groups = {}
                for idx, sample in enumerate(neg_dataset.samples):
                    groups.setdefault(image_group_key(sample), []).append(idx)
            for idxs in groups.values():
                probs = all_probs_neg[idxs]
                best = idxs[int(torch.argmax(probs).item())]
                topk_indices.append(best)
            print(f"\n✅ Selected one CLIP-confusing negative per ID image: {len(topk_indices)}")
            return topk_indices

        print("\n=== Hardest Negative Samples (per class) ===")
        n_cls = len(classnames)
        for cls in range(n_cls):
            mask = (all_labels_neg == cls)
            if mask.sum() == 0:
                continue
            probs_cls = all_probs_neg[mask]
            idxs_cls = torch.nonzero(mask, as_tuple=False).squeeze(1)

            descending = mode == "high_prob_fraction"
            sorted_probs, order = torch.sort(probs_cls, descending=descending)
            fraction = self.cfg.TRAINER.BANPL.HARD_NEG_FRACTION
            k = max(1, int(len(order) * fraction))
            selected = idxs_cls[order[:k]]

            # # === 打印每类最难样本信息 ===
            # print(f"\n[Class {cls:02d}] {classnames[cls]} | total={len(order)} | selected={k}")
            # for i, idx in enumerate(selected.tolist()):
            #     sample = neg_dataset.samples[idx]
            #     prob = sorted_probs[i].item()
            #     print(f"  -> idx={idx:<6d}  prob={prob:.4f}  path={sample['img_path']}")

            topk_indices.extend(selected.tolist())

        print("\n✅ Total hardest negatives selected:", len(topk_indices))
        return topk_indices
    
    def forward_backward(self, batch):
        image = batch["img"]
        label = batch["label"]
        is_neg = batch["is_neg"]
        
        image = image.to(self.device)
        label = label.to(self.device)
        is_neg = is_neg.to(self.device)
        prec = self.cfg.TRAINER.BANPL.PREC
        
        current_epoch = self.epoch
        total_epochs = self.cfg.OPTIM.MAX_EPOCH
        
        stage_epochs = self.cfg.TRAINER.BANPL.STAGE_EPOCHS

        # 阶段判断
        if current_epoch < stage_epochs:
            train_mode = 'pos'
        elif current_epoch >= stage_epochs:
            train_mode = 'neg'
             
        if prec == "amp":
            with autocast():
                if train_mode == 'pos':
                    self.model.prompt_learner.ctx.requires_grad_(True)
                    self.model.prompt_learner.neg_ctx.requires_grad_(False)
                    logits, neg_logits, text_features, neg_text_features, _ = self.model(image, return_extra=True)
                    if (~is_neg).sum() == 0:
                        loss_id = torch.tensor(0.0, device=logits.device, requires_grad=True)
                        loss = loss_id
                    else:
                        loss_id = F.cross_entropy(logits[~is_neg], label[~is_neg])
                        loss = loss_id
                        self.optim.zero_grad()
                        self.scaler.scale(loss).backward()
                        self.scaler.step(self.optim)
                        self.scaler.update()
                    

                elif train_mode == 'neg':
                    self.model.prompt_learner.ctx.requires_grad_(False)
                    self.model.prompt_learner.neg_ctx.requires_grad_(True)
                    logits, neg_logits, text_features, neg_text_features, _ = self.model(image, return_extra=True)

                    row = torch.arange(label.size(0), device=label.device)
                    target_pair_logits = torch.stack(
                        [logits[row, label], neg_logits[row, label]],
                        dim=1,
                    )

                    if (~is_neg).sum() > 0:
                        pos_targets = torch.zeros((~is_neg).sum(), dtype=torch.long, device=label.device)
                        pos_loss = F.cross_entropy(target_pair_logits[~is_neg], pos_targets)
                    else:
                        pos_loss = torch.tensor(0., device=logits.device, requires_grad=True)

                    if is_neg.sum() > 0:
                        neg_targets = torch.ones(is_neg.sum(), dtype=torch.long, device=label.device)
                        neg_loss = F.cross_entropy(target_pair_logits[is_neg], neg_targets)
                    else:
                        neg_loss = torch.tensor(0., device=logits.device, requires_grad=True)

                    # Keep positive/negative prompts from collapsing, but stop once they are not aligned.
                    cos_pos_neg = F.cosine_similarity(text_features, neg_text_features, dim=-1)
                    loss_prompt_separation = self.cfg.TRAINER.BANPL.SEP_WEIGHT * F.relu(cos_pos_neg).mean()
                    loss = pos_loss + neg_loss + loss_prompt_separation
                    self.optim.zero_grad()
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optim)
                    self.scaler.update()

        else:
            pass
        loss_summary = {"loss": loss.item()}
        if train_mode == 'pos':
            loss_summary["loss_id"] = loss_id.item()
            if (~is_neg).sum() == 0:
                loss_summary["acc"] = 0.0
            else:
                loss_summary["acc"] = compute_accuracy(logits[~is_neg], label[~is_neg])[0].item()
        else:
            loss_summary["loss_pos"] = pos_loss.item()
            loss_summary["loss_neg"] = neg_loss.item()
            loss_summary["loss_prompt_separation"] = loss_prompt_separation.item()
        
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




    @torch.no_grad()
    def test_ood(self, data_loader, is_list=False):
        to_np = lambda x: x.data.cpu().numpy()

        self.set_model_mode("eval")
        self.model.eval()
        self.evaluator.reset()

        scores = {
            'score1': [],
            'score2': [],
            'score3': [],
            'score4': []
        }

        for batch in tqdm(data_loader):
            if is_list: 
                images = batch[0].cuda()
            else: 
                images = batch["img"].cuda()
            logits, logits_no = self.model(images)
            logits = logits / 100
            logits_no = logits_no / 100

            pos_exp = torch.exp(logits )  # [B, C]
            neg_exp = torch.exp(logits_no )  # [B, C]
            denominator = pos_exp.sum(dim=1, keepdim=True) + neg_exp.sum(dim=1, keepdim=True)  # [B, 1]
            final_probs = pos_exp / denominator  # [B, C]
            topk_vals, _ = final_probs.topk(k=1, dim=1)
            score_topk = -topk_vals.mean(dim=1)
            score1 = to_np(score_topk)

            pos_exp = torch.exp(logits )  # [B, C]
            neg_exp = torch.exp(logits_no )  # [B, C]
            denominator = pos_exp.sum(dim=1, keepdim=True) + neg_exp.sum(dim=1, keepdim=True)  # [B, 1]
            final_probs = pos_exp / denominator  # [B, C]
            topk_vals, _ = final_probs.topk(k=2, dim=1)
            score_topk = -topk_vals.mean(dim=1)
            score2 = to_np(score_topk)

            pos_exp = torch.exp(logits)  # [B, C]
            neg_exp = torch.exp(logits_no)  # [B, C]
            denominator = pos_exp.sum(dim=1, keepdim=True) + neg_exp.sum(dim=1, keepdim=True)  # [B, 1]
            final_probs = pos_exp / denominator  # [B, C]
            topk_vals, _ = final_probs.topk(k=3, dim=1)
            score_topk = -(topk_vals).mean(dim=1)
            score3 = to_np(score_topk)


            pos_exp = torch.exp(logits)  # [B, C]
            neg_exp = torch.exp(logits_no)  # [B, C]
            denominator = pos_exp.sum(dim=1, keepdim=True) + neg_exp.sum(dim=1, keepdim=True)  # [B, 1]
            final_probs = pos_exp / denominator  # [B, C]
            topk_vals, _ = final_probs.topk(k=5, dim=1)
            score_topk = -(topk_vals).mean(dim=1)
            score4 = to_np(score_topk)

            scores['score1'].append(score1)
            scores['score2'].append(score2)
            scores['score3'].append(score3)
            scores['score4'].append(score4)
        n = len(data_loader.dataset)
        scores_array = (
            np.concatenate(scores['score1'])[:n].copy(),
            np.concatenate(scores['score2'])[:n].copy(),
            np.concatenate(scores['score3'])[:n].copy(),
            np.concatenate(scores['score4'])[:n].copy()
        )


        return scores_array


    def test(self):
        """A custom testing function."""
        self.set_model_mode("eval")
        self.model.eval()
        self.evaluator.reset()

        for batch in tqdm(self.test_loader):
            input = batch["img"]
            label = batch["label"]
            input = input.to(self.device)
            label = label.to(self.device)

            # 只使用分类 logits 进行评估
            with torch.no_grad():
                cls_logits,_= self.model(input)  # 忽略 ood_logits
                self.evaluator.process(cls_logits, label)

        results = self.evaluator.evaluate()

   
