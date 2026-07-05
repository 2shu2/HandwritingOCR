"""
使用 trdg 生成多样化文本行图片（适用于 OCR / 手写体识别训练）
每行生成一张图片，同时生成 labels.txt

新版 trdg (1.8.0) API 适配：
- load_fonts_from_folder() 已移除，改用 glob 扫描 .ttf/.otf 文件
- extension 参数已移除
- fonts 参数传 [] 让 trdg 自动加载内置字体；传字体路径列表使用自定义字体
- is_handwritten 依赖外部 TF1 模型（下载/兼容性不稳定），这里通过字体 + 变形增强替代
"""
import os
import glob
from trdg.generators import GeneratorFromStrings

# 配置参数
OUTPUT_DIR = "data/trdg/images"
LABEL_FILE = "data/trdg/labels.txt"
NUM_SAMPLES = 2000          # 生成图片数量
IMG_HEIGHT = 64             # 固定高度（与 config.yaml 一致）
FONT_DIR = "fonts/"         # 存放手写风格 .ttf/.otf 的文件夹（可选）

# 准备英文句子作为文本来源
texts = [
    "Hello world",
    "Machine learning is fun",
    "Handwriting recognition with deep learning",
    "Artificial intelligence and neural networks",
    "Python programming language",
    "Natural language processing",
    "Convolutional recurrent neural network",
    "Sequence to sequence models",
    "Optical character recognition system",
    "Data augmentation for better accuracy",
    "The quick brown fox jumps over the lazy dog",
    "All good things come to those who wait",
    "Practice makes perfect",
    "Knowledge is power",
    "Actions speak louder than words",
    "A picture is worth a thousand words",
    "Better late than never",
    "Every cloud has a silver lining",
    "Great minds think alike",
    "No pain no gain",
]

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 加载自定义字体（如果有 fonts/ 目录）
fonts = []
if os.path.exists(FONT_DIR):
    fonts = glob.glob(os.path.join(FONT_DIR, "*.ttf"))
    fonts += glob.glob(os.path.join(FONT_DIR, "*.otf"))
    if fonts:
        print(f"Loaded {len(fonts)} custom font(s) from {FONT_DIR}")
    else:
        print(f"No .ttf/.otf fonts found in {FONT_DIR}, will use trdg built-in fonts")

# 生成器：随机选择文本 + 字体 + 变形（倾斜 / 模糊 / 扭曲 / 背景噪声）
generator = GeneratorFromStrings(
    strings=texts,
    count=NUM_SAMPLES,
    fonts=fonts,                     # [] 时自动加载 trdg 内置字体
    language='en',
    size=IMG_HEIGHT,
    skewing_angle=8,                 # 最大倾斜角度
    random_skew=True,
    blur=2,                          # 模糊半径
    random_blur=True,
    background_type=1,               # 0-高斯噪声, 1-纯色, 2-线条, 3-图片背景
    distorsion_type=2,               # 0-无, 1-正弦, 2-余弦, 3-随机扭曲
    distorsion_orientation=2,        # 扭曲方向: 0-垂直, 1-水平, 2-两者
    text_color="#282828",            # 深灰色（不同字体/粗细更自然）
    alignment=1,                     # 0-左对齐, 1-居中
    image_dir=OUTPUT_DIR,
    fit=True,
)

# 生成图片并记录标注
with open(LABEL_FILE, 'w', encoding='utf-8') as f:
    for i, (img, text) in enumerate(generator):
        if i >= NUM_SAMPLES:
            break
        # 文件名格式：sample_00000.png
        img_name = f"sample_{i:05d}.png"
        img.save(os.path.join(OUTPUT_DIR, img_name))
        f.write(f"{img_name}\t{text}\n")
        if (i+1) % 500 == 0:
            print(f"Generated {i+1}/{NUM_SAMPLES} images")

print(f"Dataset generated: {NUM_SAMPLES} images in {OUTPUT_DIR}")
print(f"Labels saved to {LABEL_FILE}")