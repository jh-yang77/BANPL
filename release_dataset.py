"""Deterministic unlabeled OOD images, including flat and nested directories."""
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset

class OODImages(Dataset):
    def __init__(self, root, transform=None):
        self.transform=transform
        root=Path(root)
        if not root.is_dir():raise FileNotFoundError(root)
        self.samples=sorted(p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in {'.jpg','.jpeg','.png','.bmp','.tif','.tiff','.webp'})
        if not self.samples:raise ValueError(f'No images in {root}')
    def __len__(self):return len(self.samples)
    def __getitem__(self,index):
        with Image.open(self.samples[index]) as image: image=image.convert('RGB')
        return (self.transform(image) if self.transform else image),-1
