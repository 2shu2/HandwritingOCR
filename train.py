"""
模型训练主程序
支持混合精度、checkpoint保存、验证集评估
自动从 data/trdg/labels.txt 按比例拆分训练集/验证集

改进点：
- 验证指标：从逐位对齐比较改为CER（Character Error Rate / 编辑距离）
- 验证解码：启用CTC Beam Search，与推理时一致
- 优化器：AdamW + 权重衰减，防止过拟合
- 学习率调度：CosineAnnealingWarmRestarts，周期性重启避免陷入局部最优
- 梯度裁剪：防止CTC损失偶尔产生的梯度爆炸
"""
import os
import random
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from data import TextLineDataset, collate_fn, decode_predictions
from model import CRNN


def levenshtein_distance(s1: str, s2: str) -> int:
    """计算两个字符串的编辑距离（Levenshtein Distance），O(n*m)"""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            insert = prev[j + 1] + 1
            delete = curr[j] + 1
            sub = prev[j] + (0 if c1 == c2 else 1)
            curr.append(min(insert, delete, sub))
        prev = curr
    return prev[-1]


def split_train_val(label_file: str, train_file: str, val_file: str,
                    val_ratio: float = 0.2, seed: int = 42):
    """将 labels.txt 按比例拆分为 train_labels.txt 和 val_labels.txt"""
    with open(label_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    random.seed(seed)
    random.shuffle(lines)
    split_idx = int(len(lines) * (1 - val_ratio))

    os.makedirs(os.path.dirname(train_file) or '.', exist_ok=True)
    os.makedirs(os.path.dirname(val_file) or '.', exist_ok=True)

    with open(train_file, 'w', encoding='utf-8') as f:
        f.writelines(lines[:split_idx])
    with open(val_file, 'w', encoding='utf-8') as f:
        f.writelines(lines[split_idx:])

    print(f"Split {len(lines)} samples -> {split_idx} train + {len(lines) - split_idx} val")


def auto_generate_data(config: dict):
    """如果标签文件不存在，自动调用 generate_data 逻辑生成并拆分数据"""
    import glob
    from trdg.generators import GeneratorFromStrings

    source_label = "data/trdg/labels.txt"
    image_dir = config['data']['train_image_dir']
    train_label = config['data']['train_label_file']
    val_label = config['data']['val_label_file']

    # 如果训练/验证标签文件已存在，直接返回
    if os.path.exists(train_label) and os.path.exists(val_label):
        return

    # 如果源 labels.txt 存在但未拆分，直接拆分
    if os.path.exists(source_label):
        split_train_val(source_label, train_label, val_label)
        return

    # 否则自动生成数据
    print("No data found, auto-generating with trdg...")
    os.makedirs(image_dir, exist_ok=True)

    # 用随机英文单词组合生成多样化的文本（不再局限于20句）
    WORDS = [
        "the", "a", "an", "is", "are", "was", "were", "have", "has", "had",
        "can", "will", "would", "could", "should", "may", "might", "must",
        "time", "year", "people", "way", "day", "man", "woman", "child",
        "world", "life", "hand", "part", "place", "case", "week", "company",
        "system", "program", "question", "work", "government", "number",
        "night", "point", "home", "water", "room", "mother", "area", "money",
        "story", "fact", "month", "lot", "right", "study", "book", "eye",
        "job", "word", "business", "issue", "side", "kind", "head", "house",
        "service", "friend", "father", "power", "hour", "game", "line",
        "city", "community", "name", "president", "team", "minute", "idea",
        "body", "information", "back", "parent", "face", "others", "level",
        "office", "door", "health", "person", "art", "war", "history",
        "party", "result", "morning", "reason", "research", "girl", "guy",
        "moment", "air", "teacher", "force", "education", "run", "walk",
        "say", "go", "come", "know", "get", "make", "see", "think", "look",
        "want", "give", "use", "find", "tell", "ask", "try", "leave",
        "call", "keep", "let", "begin", "seem", "help", "show", "hear",
        "play", "move", "live", "believe", "hold", "bring", "happen",
        "write", "provide", "sit", "stand", "lose", "pay", "meet", "include",
        "continue", "set", "learn", "change", "lead", "understand", "watch",
        "follow", "stop", "create", "speak", "read", "allow", "add",
        "spend", "grow", "open", "walk", "win", "offer", "remember", "love",
        "consider", "appear", "buy", "wait", "serve", "die", "send",
        "expect", "build", "stay", "fall", "cut", "reach", "kill", "remain",
        "suggest", "raise", "pass", "sell", "require", "report", "decide",
        "pull", "new", "good", "high", "old", "great", "big", "small",
        "large", "long", "own", "other", "right", "different", "next",
        "early", "young", "important", "public", "bad", "same", "able",
        "simple", "sure", "strong", "real", "full", "special", "clear",
        "easy", "late", "hard", "major", "better", "economic", "strong",
        "possible", "whole", "free", "military", "true", "federal",
        "international", "certain", "personal", "happy", "final", "main",
        "nice", "hard", "short", "left", "dead", "ready", "common",
    ]

    def random_text(min_words=2, max_words=6):
        import random as _random
        n = _random.randint(min_words, max_words)
        return " ".join(_random.choice(WORDS) for _ in range(n))

    texts = [random_text() for _ in range(500)]  # 生成500句随机短语

    font_dir = "fonts/"
    fonts = []
    if os.path.exists(font_dir):
        fonts = glob.glob(os.path.join(font_dir, "*.ttf"))
        fonts += glob.glob(os.path.join(font_dir, "*.otf"))

    num_samples = 5000  # 数据量从2000增加到5000
    generator = GeneratorFromStrings(
        strings=texts, count=num_samples, fonts=fonts, language='en',
        size=config['data']['img_height'],
        skewing_angle=8, random_skew=True, blur=2, random_blur=True,
        background_type=1, distorsion_type=2, distorsion_orientation=2,
        text_color="#282828", alignment=1, image_dir=image_dir, fit=True,
    )

    with open(source_label, 'w', encoding='utf-8') as f:
        for i, (img, text) in enumerate(generator):
            if i >= num_samples:
                break
            img_name = f"sample_{i:05d}.png"
            img.save(os.path.join(image_dir, img_name))
            f.write(f"{img_name}\t{text}\n")
            if (i + 1) % 500 == 0:
                print(f"  Generated {i + 1}/{num_samples} images")
    print(f"Data generation complete: {num_samples} images")
    split_train_val(source_label, train_label, val_label)


def train(config: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---------- 自动准备数据 ----------
    auto_generate_data(config)

    # ---------- 数据准备 ----------
    alphabet = config['data']['alphabet']
    num_classes = len(alphabet) + 1  # 包含blank占位符

    train_dataset = TextLineDataset(
        config['data']['train_image_dir'],
        config['data']['train_label_file'],
        alphabet,
        config['data']['img_height'],
        is_train=True
    )
    val_dataset = TextLineDataset(
        config['data']['val_image_dir'],
        config['data']['val_label_file'],
        alphabet,
        config['data']['img_height'],
        is_train=False
    )

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['train']['batch_size'],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config['train']['num_workers']
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['train']['batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config['train']['num_workers']
    )

    # ---------- 模型、损失、优化器 ----------
    model = CRNN(
        num_classes,
        config['model']['backbone'],
        config['model']['hidden_size'],
        config['model']['num_layers'],
        img_height=config['data']['img_height']
    ).to(device)

    ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)  # blank索引为0

    # AdamW: 带解耦权重衰减的Adam，比Adam更不容易过拟合
    wd = config['train'].get('weight_decay', 1e-4)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config['train']['learning_rate'],
        weight_decay=wd
    )
    scaler = GradScaler(enabled=config['train']['mixed_precision'])

    # CosineAnnealingWarmRestarts: 余弦退火 + 周期性重启
    # T_0=10 表示第一个周期10个epoch，每个周期结束后学习率重置并继续
    # T_mult=2 表示每个周期长度翻倍（第2周期20 epoch，第3周期40 epoch...）
    lr_config = config['train'].get('lr_scheduler', {})
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=lr_config.get('T_0', 10),
        T_mult=lr_config.get('T_mult', 2),
        eta_min=lr_config.get('eta_min', 1e-6)
    )

    # ---------- 训练循环 ----------
    best_acc = 0.0
    for epoch in range(config['train']['epochs']):
        model.train()
        total_loss = 0.0
        for batch_idx, (imgs, labels) in enumerate(train_loader):
            imgs = imgs.to(device)
            optimizer.zero_grad()

            with autocast(enabled=config['train']['mixed_precision']):
                log_probs = model(imgs)  # (B, T, num_classes)
                input_lengths = torch.full(
                    (log_probs.size(0),), log_probs.size(1), dtype=torch.long
                ).to(device)
                target_lengths = torch.tensor(
                    [len(lbl) for lbl in labels], dtype=torch.long
                ).to(device)
                # CTCLoss: targets保持CPU，避免某些PyTorch版本的兼容性问题
                labels_concat = torch.cat(labels)
                loss = ctc_loss(
                    log_probs.permute(1, 0, 2),  # (T, B, C)
                    labels_concat,
                    input_lengths,
                    target_lengths
                )

            # 反向传播，支持混合精度 + 梯度裁剪
            scaler.scale(loss).backward()
            # 梯度裁剪：防止CTC损失偶尔产生的梯度爆炸
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            if batch_idx % config['train']['log_interval'] == 0:
                print(f"Epoch [{epoch+1}/{config['train']['epochs']}] "
                      f"Batch [{batch_idx}] Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch+1}] Average Loss: {avg_loss:.4f}  LR: {current_lr:.6f}")

        # ---------- 验证 ----------
        model.eval()
        total_cer = 0.0   # Character Error Rate（编辑距离 / 真实字符数）
        total_samples = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                log_probs = model(imgs)
                # 验证时使用Beam Search解码，与推理时保持一致
                pred_texts = decode_predictions(
                    log_probs, val_dataset.idx2char, use_beam=True, beam_width=5
                )
                for i, pred_str in enumerate(pred_texts):
                    true_str = ''.join([val_dataset.idx2char.get(idx.item(), '')
                                        for idx in labels[i] if idx.item() != 0])
                    # 使用编辑距离（CER）计算错误率
                    dist = levenshtein_distance(pred_str, true_str)
                    total_cer += dist / max(len(true_str), 1)
                    total_samples += 1

        avg_cer = total_cer / total_samples if total_samples > 0 else 1.0
        char_accuracy = 1.0 - avg_cer  # 将CER转换为准确率
        print(f"Epoch [{epoch+1}] Validation CER: {avg_cer:.4f}  "
              f"Char Accuracy: {char_accuracy:.4f}")

        # 学习率调度（余弦退火按epoch调用，不需要传入metrics）
        scheduler.step(epoch)

        # 保存最佳模型
        if char_accuracy > best_acc:
            best_acc = char_accuracy
            os.makedirs(config['train']['save_dir'], exist_ok=True)
            save_path = os.path.join(config['train']['save_dir'], "best_model.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Best model saved to {save_path} (acc: {best_acc:.4f})")


if __name__ == "__main__":
    # 切换到脚本所在目录，确保相对路径（config.yaml、data/ 等）正确解析
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    with open("config.yaml", 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    train(config)