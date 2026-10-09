import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader

class ImageDataset(Dataset):
    def __init__(self, dir, neg, transform=None):
        self.dir = dir
        self.transform = transform
        self.neg = neg
        self.labels = self._read_labels(os.path.join(dir, "labels.txt"))

        # 获取所有负样本图像路径和标签
        self.samples = []
        for img_name, label in self.labels.items():
            if self.neg:
                self.samples.append({
                    "img_path": os.path.join(dir, img_name),
                    "label": label,
                    "is_neg": True  # 所有样本都是负样本
                })
            else:
                self.samples.append({
                    "img_path": os.path.join(dir, img_name),
                    "label": label,
                    "is_neg": False  # 所有样本都是负样本
                })

    def _read_labels(self, label_file):
        """读取标签文件"""
        labels = {}
        with open(label_file, 'r') as f:
            for line in f:
                img_name, label = line.strip().split(',')
                labels[img_name] = int(label)
        return labels

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample["img_path"]).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return {
            "img": image,
            "label": sample["label"],
            "is_neg": sample["is_neg"]
        }
        # return image, sample["label"], sample["is_neg"]

class CombinedDataset(Dataset):
    """组合原始数据集和负样本数据集"""
    def __init__(self, orig_dataset, neg_dataset, transform=None):
        self.orig_dataset = orig_dataset
        self.neg_dataset = neg_dataset
        self.transform = transform
        self.orig_len = len(orig_dataset)
        self.neg_len = len(neg_dataset)
        
    def __len__(self):
        return self.orig_len + self.neg_len
    
    def __getitem__(self, idx):
        if idx < self.orig_len:
            # 从原始数据集获取数据
            datum = self.orig_dataset[idx]
            # 加载图像
            image = Image.open(datum.impath).convert('RGB')
            if self.transform:
                image = self.transform(image)
            # 返回字典
            return {
                "img": image,
                "label": datum.label,
                "is_neg": False
            }
        else:
            # 从负样本数据集获取数据
            return self.neg_dataset[idx - self.orig_len]

class NegativeImageDataset(Dataset):
    """用于加载负样本图像的数据集类"""
    def __init__(self, neg_dir, transform=None):
        """
        Args:
            neg_dir: 负样本图像目录
            transform: 图像转换
        """
        self.neg_dir = neg_dir
        self.transform = transform

        # 读取标签文件，并添加 '_inpainted.png' 后缀
        self.neg_labels = self._read_labels(os.path.join(neg_dir, "labels.txt"))

        # 获取所有负样本图像路径和标签
        self.samples = []
        for img_name, label in self.neg_labels.items():
            # 加上 _inpainted.png 后缀
            inpainted_name = img_name.replace(".png", "_inpainted.png")
            self.samples.append({
                "img_path": os.path.join(neg_dir, inpainted_name),
                "label": label,
                "is_neg": True  # 所有样本都是负样本
            })

    def _read_labels(self, label_file):
        """读取标签文件"""
        labels = {}
        with open(label_file, 'r') as f:
            for line in f:
                img_name, label = line.strip().split(',')
                labels[img_name] = int(label)
        return labels

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample["img_path"]).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return {
            "img": image,
            "label": sample["label"],
            "is_neg": True
        }

