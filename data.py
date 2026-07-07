"""
数据加载、预处理、batch整理与解码工具
支持变长宽度文本行图片，通过collate_fn实现mini-batch动态padding
"""
import math
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import cv2
import numpy as np
from typing import List, Tuple


class ElasticDistortion:
    """
    弹性变形增强：模拟手写文字在纸张上的自然变形
    使用高斯滤波平滑的随机位移场对图片进行变形
    """
    def __init__(self, alpha: float = 12.0, sigma: float = 2.5, p: float = 0.4):
        self.alpha = alpha
        self.sigma = sigma
        self.p = p

    def __call__(self, img):
        if torch.rand(1).item() > self.p:
            return img
        # img: PIL Image, 转numpy
        img_np = np.array(img, dtype=np.float32)
        h, w = img_np.shape[:2]
        # 生成随机位移场
        dx = np.random.randn(h, w).astype(np.float32) * self.sigma
        dy = np.random.randn(h, w).astype(np.float32) * self.sigma
        # 高斯平滑（用 cv2.GaussianBlur 代替 scipy，无需额外依赖）
        ksize = int(4 * self.sigma + 1) | 1  # 确保为奇数
        ksize = max(3, min(ksize, 31))       # 限制在 3~31 之间
        dx = cv2.GaussianBlur(dx, (ksize, ksize), self.sigma) * self.alpha
        dy = cv2.GaussianBlur(dy, (ksize, ksize), self.sigma) * self.alpha
        # 坐标映射
        x, y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        indices = (y + dy).astype(np.float32), (x + dx).astype(np.float32)
        # cv2.remap 实现变形
        img_out = cv2.remap(img_np, indices[1], indices[0], cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REPLICATE)
        from PIL import Image
        return Image.fromarray(img_out.astype(np.uint8) if img_np.dtype == np.uint8
                               else np.clip(img_out, 0, 255).astype(np.uint8))


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

        # 训练数据增强（增强策略加强，提升模型鲁棒性）
        if self.is_train:
            self.augmentation = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Grayscale(num_output_channels=3),    # L→RGB 兼容后续操作
                transforms.RandomAdjustSharpness(2, p=0.3),     # 锐化
                transforms.ColorJitter(brightness=0.35, contrast=0.35),  # 亮度/对比度（加强）
                transforms.RandomAffine(degrees=5, shear=8,      # 旋转±5° + 倾斜±8°（加强）
                                        scale=(0.9, 1.1)),       # 缩放±10%
                ElasticDistortion(alpha=12.0, sigma=2.5, p=0.4), # 弹性变形（新增）
                transforms.GaussianBlur(kernel_size=(3, 3),      # 高斯模糊（新增）
                                        sigma=(0.1, 1.5)),
                transforms.RandomPerspective(distortion_scale=0.12, p=0.3),  # 透视变换（新增）
                transforms.ToTensor(),
                transforms.RandomErasing(p=0.1, scale=(0.02, 0.1),  # 随机擦除（新增）
                                        ratio=(0.3, 3.3)),
                # 不转灰度，保持3通道伪RGB（ResNet34预训练权重需要3通道输入）
            ])
        else:
            self.augmentation = None

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

        # 归一化
        img_norm = img.astype(np.float32) / 255.0

        # 数据增强（仅训练时）
        if self.augmentation is not None:
            img_tensor = self.augmentation(img)  # (3, H, W) 伪RGB
        else:
            # 灰度图复制为3通道伪RGB（适配ResNet34预训练权重）
            img_tensor = torch.from_numpy(img_norm).unsqueeze(0).repeat(3, 1, 1)

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


def log_sum_exp(a: float, b: float) -> float:
    """数值稳定的 log-sum-exp"""
    if a == float('-inf'):
        return b
    if b == float('-inf'):
        return a
    if a > b:
        return a + math.log(1 + math.exp(b - a))
    else:
        return b + math.log(1 + math.exp(a - b))


def ctc_beam_search(log_probs: torch.Tensor, beam_width: int = 5,
                    blank: int = 0) -> List[int]:
    """
    CTC 前缀束搜索（单条序列）
    log_probs: (T, C) 对数概率
    返回最佳 token 序列
    """
    T, C = log_probs.shape

    # 每个 beam: (prefix_tuple, prob_blank, prob_non_blank)
    # prob_blank: 前缀以 blank 结尾的概率（对数空间）
    # prob_non_blank: 前缀以非 blank 结尾的概率（对数空间）
    NEG_INF = float('-inf')
    beams = {(): [0.0, NEG_INF]}  # {prefix: [pb, pnb]}

    for t in range(T):
        new_beams = {}

        for prefix, (pb, pnb) in beams.items():
            for c in range(C):
                p_c = log_probs[t, c].item()

                if c == blank:
                    # 添加 blank：前缀不变
                    old_pb, old_pnb = new_beams.get(prefix, [NEG_INF, NEG_INF])
                    new_beams[prefix] = [
                        log_sum_exp(old_pb, log_sum_exp(pb + p_c, pnb + p_c)),
                        old_pnb
                    ]
                else:
                    last_char = prefix[-1] if len(prefix) > 0 else None

                    if c == last_char:
                        # 重复字符：可能不扩展（当前字符是之前路径的重叠）
                        old_pb, old_pnb = new_beams.get(prefix, [NEG_INF, NEG_INF])
                        # pnb + p_c: 前一步非 blank，当前字符合并到前一个
                        new_beams[prefix] = [
                            old_pb,
                            log_sum_exp(old_pnb, pnb + p_c)
                        ]

                    # 扩展新字符
                    new_key = prefix + (c,)
                    old_pb, old_pnb = new_beams.get(new_key, [NEG_INF, NEG_INF])
                    # pb + p_c: 前一步是 blank，当前字符是新的
                    # pnb + p_c: 前一步非 blank（且不是重复），当前字符是新的
                    new_beams[new_key] = [
                        old_pb,
                        log_sum_exp(old_pnb, log_sum_exp(pb + p_c, pnb + p_c))
                    ]

        # 保留概率最高的 beam_width 个前缀
        scored = [(k, log_sum_exp(v[0], v[1])) for k, v in new_beams.items()]
        scored.sort(key=lambda x: x[1], reverse=True)
        beams = {k: new_beams[k] for k, _ in scored[:beam_width]}

    # 返回最优路径
    best = max(beams.items(), key=lambda x: log_sum_exp(x[1][0], x[1][1]))
    return list(best[0])


def decode_predictions(log_probs: torch.Tensor, idx2char: dict,
                       blank: int = 0, use_beam: bool = False,
                       beam_width: int = 5) -> List[str]:
    """
    CTC 解码：支持贪心解码和束搜索解码
    log_probs: (B, T, C) 对数概率
    idx2char: 数字到字符的映射字典
    """
    decoded_texts = []

    for b in range(log_probs.size(0)):
        single_probs = log_probs[b]  # (T, C)

        if use_beam:
            tokens = ctc_beam_search(single_probs, beam_width, blank)
        else:
            # 贪心解码
            preds = single_probs.argmax(dim=-1)
            prev = blank
            tokens = []
            for t in preds:
                token = t.item()
                if token != blank and token != prev:
                    tokens.append(token)
                prev = token

        text = ''.join([idx2char.get(token, '') for token in tokens])
        decoded_texts.append(text)

    return decoded_texts
