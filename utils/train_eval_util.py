import os
import torch
from torchvision import datasets
import torchvision.transforms as transforms
import clip_w_local
from torch.utils.data import Subset


def set_model_clip(args):
    model, _ = clip_w_local.load(args.CLIP_ckpt)

    model = model.cuda()
    normalize = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                         std=(0.26862954, 0.26130258, 0.27577711))  # for CLIP
    val_preprocess = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])
    return model, val_preprocess


def set_val_loader(args, preprocess=None, folders=None):
    if preprocess is None:
        normalize = transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711)
        )
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])

    if getattr(args, "in_dataset", "imagenet") != "imagenet":
        raise ValueError("This release evaluates ImageNet-1K only")
    root_val = os.path.join(args.root, "imagenet", "val")
    # 1. 加载全量 ImageNet-1K 验证集
    full_val = datasets.ImageFolder(root_val, transform=preprocess)

    # 2. 如果传入 folders 列表，就只保留那些类
    if folders is not None:
        subset_classes = set(folders)
        # full_val.classes: list of folder names 对应 label
        subset_indices = [
            idx for idx, (_, label) in enumerate(full_val.samples)
            if full_val.classes[label] in subset_classes
        ]
        val_dataset = Subset(full_val, subset_indices)
    else:
        val_dataset = full_val

    # 3. 构建 DataLoader 并返回
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=getattr(args, 'batch_size', 1024+512),
        shuffle=False,
        pin_memory=False,
        num_workers=getattr(args, 'num_workers', 8)
        
    )
    return val_loader


import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from PIL import Image
import os

class FlexibleImageFolder(Dataset):
    """
    Works like ImageFolder, but also supports flat folders (no subdirectories).
    Always returns (Tensor, label), label=-1 for flat folders.
    """
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform

        subdirs = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
        if len(subdirs) > 0:
            # 有子文件夹 → 直接用 ImageFolder
            self.dataset = datasets.ImageFolder(root=root, transform=transform)
            self.is_flat = False
        else:
            self.samples = [os.path.join(root, f) for f in sorted(os.listdir(root))
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"))]
            self.is_flat = True

    def __len__(self):
        return len(self.dataset) if not self.is_flat else len(self.samples)

    def __getitem__(self, idx):
        if self.is_flat:
            path = self.samples[idx]
            img = Image.open(path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            return img, -1
        else:
            return self.dataset[idx]   # (Tensor, label)


def set_ood_loader_ImageNet(args, out_dataset, preprocess=None, batch_size=1024+512):
    '''
    Set OOD loaders for ImageNet-scale datasets, supporting both classic benchmarks
    and OpenOOD v1.5 benchmarks.
    '''
    if preprocess is None:
        normalize = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                         std=(0.26862954, 0.26130258, 0.27577711))
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])

    from protocols import FOLDERS
    dataset_roots = {name: os.path.join(args.root, folder) for name, folder in FOLDERS.items()}
    dataset_name = out_dataset
    if dataset_name not in dataset_roots:
        raise ValueError(f"Unknown out_dataset: {dataset_name}. "
                            f"Available: {list(dataset_roots.keys())}")
    if dataset_name in ["iNaturalist", "Places", "Textures"]:
        testsetout = datasets.ImageFolder(root=dataset_roots[dataset_name], transform=preprocess)
    else:
        testsetout = FlexibleImageFolder(root=dataset_roots[dataset_name],
                                        transform=preprocess)

    batch_size = getattr(args, 'batch_size', batch_size)

    loader = torch.utils.data.DataLoader(
        testsetout,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=False,
        num_workers=getattr(args, 'num_workers', 8)
    )

    return loader
