"""
数据加载、预处理、batch整理与解码工具
支持变长宽度文本行图片，通过collate_fn实现mini-batch动态padding
"""
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from typing import List, Tuple


class TextLineDataset(Dataset):
    """手写文本行数据集"""

    def __init__(self, image_dir: str, label_file: str, alphabet: str,
                 img_height: int = 64, is_train: bool = True):
        self.image_dir = image_dir
        self.img_height = img_height
        self.is_train = is_train

        # 读取图片路径与标签
        self.samples = self._load_labels(label_file)

        # 建立字符<->索引映射，索引0留给CTC的blank
        self.char2idx = {ch: i + 1 for i, ch in enumerate(alphabet)}
        self.char2idx['<blank>'] = 0
        self.idx2char = {i + 1: ch for i, ch in enumerate(alphabet)}
        self.idx2char[0] = ''  # blank对应空字符串

    def _load_labels(self, label_file: str) -> List[Tuple[str, str]]:
        """读取标注文件（每行：图片名\t文本）"""
        samples = []
        with open(label_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    # 去除文本字段可能存在的引号
                    text = parts[1].strip('"').strip("'")
                    samples.append((parts[0], text))
        return samples

    def encode_text(self, text: str) -> torch.Tensor:
        """将文本转换为数字索引（跳过不在字母表中的字符）"""
        return torch.tensor([self.char2idx[c] for c in text if c in self.char2idx],
                            dtype=torch.long)

    def decode_text(self, indices: torch.Tensor) -> str:
        """将模型输出的索引序列解码为字符串"""
        decoded = []
        for i in indices:
            if i != 0:  # 跳过blank
                decoded.append(self.idx2char[i.item()])
        return ''.join(decoded)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_name, text = self.samples[idx]
        # 以灰度图方式读取
        img_path = f"{self.image_dir}/{img_name}"
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(
                f"Image not found: {img_path}\n"
                f"Make sure the images exist in '{self.image_dir}'. "
                f"Run generate_data.py first, or let train.py auto-generate data."
            )

        # 缩放到统一高度，保持宽高比
        h, w = img.shape
        scale = self.img_height / h
        new_w = max(int(w * scale), 1)  # 宽度至少为1
        img = cv2.resize(img, (new_w, self.img_height))

        # 数据增强（仅训练时）
        if self.is_train:
            # 随机亮度和对比度
            if np.random.rand() < 0.3:
                img = cv2.convertScaleAbs(img, alpha=1.2, beta=10)
            # 随机小角度旋转
            if np.random.rand() < 0.3:
                center = (new_w / 2, self.img_height / 2)
                angle = np.random.uniform(-3, 3)
                M = cv2.getRotationMatrix2D(center, angle, 1)
                img = cv2.warpAffine(img, M, (new_w, self.img_height))

        # 归一化，增加channel维度 (1, H, W)
        img = img.astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img).unsqueeze(0)
        label_tensor = self.encode_text(text)
        return img_tensor, label_tensor


def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, list]:
    """
    批次整理函数：将图像padding到批次内最大宽度，标签保持列表形式（CTC需要不等长）
    """
    imgs, labels = zip(*batch)
    widths = [img.shape[2] for img in imgs]
    max_w = max(widths)

    padded_imgs = []
    for img in imgs:
        pad_w = max_w - img.shape[2]
        # 在宽度维度右侧补零（参数顺序：左、右、上、下）
        padded = torch.nn.functional.pad(img, (0, pad_w, 0, 0), value=0)
        padded_imgs.append(padded)
    imgs = torch.stack(padded_imgs, 0)
    return imgs, labels  # labels是Tensor列表，不等长


def decode_predictions(log_probs: torch.Tensor, idx2char: dict, blank: int = 0) -> List[str]:
    """
    CTC贪婪解码：取每步最大概率的字符，去重去blank，返回字符串列表
    log_probs: (B, T, C) 对数概率
    idx2char: 数字到字符的映射字典
    """
    preds = log_probs.argmax(dim=-1)  # (B, T)
    decoded_texts = []
    for b in range(preds.size(0)):
        prev = blank
        seq = []
        for t in preds[b]:
            token = t.item()
            if token != blank and token != prev:
                seq.append(token)
            prev = token
        text = ''.join([idx2char.get(token, '') for token in seq])
        decoded_texts.append(text)
    return decoded_texts