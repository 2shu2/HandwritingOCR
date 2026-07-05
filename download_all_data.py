"""
下载所有可用的手写英文数据集，整合为统一训练集

数据来源:
  1. Teklia/IAM-line        — 10,373 行 (HuggingFace，已下载)
  2. CATMuS/modern 英文手写  — ~432 行英文手写 (HuggingFace)
  3. GNHK handwriting        — 真实手写单词 (HuggingFace)
  4. trdg 大规模合成         — 生成 N 张多样化手写图片

用法:
  python download_all_data.py               # 下载所有 + 生成 3万合成图
  python download_all_data.py --synth 50000  # 生成 5万合成图
"""
import os
import sys
import glob
import ssl
import random
import argparse
import urllib.request
from tqdm import tqdm

# ===== 配置 =====
OUTPUT_DIR = "data/all_dataset/images"
LABEL_FILE = "data/all_dataset/labels.txt"
IMG_HEIGHT = 64

# 绕过 SSL 证书问题
ssl._create_default_https_context = ssl._create_unverified_context


def download_iam_lines():
    """下载 IAM 行级数据 (Teklia/IAM-line) — 约 10,373 行"""
    print("=" * 55)
    print("[1/4] IAM 行级数据 (Teklia/IAM-line)")
    print("=" * 55)

    from datasets import load_dataset

    existing = glob.glob(os.path.join(OUTPUT_DIR, "iam_line_*.png"))
    if len(existing) > 5000:
        print(f"  已有 {len(existing)} 张，跳过下载")
        return len(existing)

    dataset = load_dataset("Teklia/IAM-line")
    count = 0

    for split_name in ["train", "validation", "test"]:
        if split_name not in dataset:
            continue
        for sample in tqdm(dataset[split_name], desc=f"  IAM {split_name}"):
            img_name = f"iam_line_{split_name}_{count:06d}.png"
            img_path = os.path.join(OUTPUT_DIR, img_name)
            sample["image"].save(img_path)
            text = sample["text"].replace("\n", " ").replace("\t", " ")
            with open(LABEL_FILE, "a", encoding="utf-8") as f:
                f.write(f"{img_name}\t{text}\n")
            count += 1

    print(f"  Done: {count} images")
    return count


def download_catmus_english():
    """下载 CATMuS/modern 英文手写部分 — ~432 行"""
    print("\n" + "=" * 55)
    print("[2/4] CATMuS/modern 英文手写")
    print("=" * 55)

    from datasets import load_dataset

    existing = glob.glob(os.path.join(OUTPUT_DIR, "catmus_*.png"))
    if len(existing) > 200:
        print(f"  已有 {len(existing)} 张，跳过下载")
        return len(existing)

    try:
        dataset = load_dataset("CATMuS/modern")
    except Exception as e:
        print(f"  下载失败: {e}")
        return 0

    count = 0
    for split_name in ["train", "validation", "test"]:
        if split_name not in dataset:
            continue
        for sample in tqdm(dataset[split_name], desc=f"  CATMuS {split_name}"):
            # 只保留英文手写
            lang = sample.get("language", "")
            wtype = sample.get("writing_type", "")
            if "english" not in lang.lower() or "handwrit" not in wtype.lower():
                continue

            img_name = f"catmus_{split_name}_{count:06d}.png"
            img_path = os.path.join(OUTPUT_DIR, img_name)
            sample["im"].save(img_path)
            text = sample["text"].replace("\n", " ").replace("\t", " ")
            with open(LABEL_FILE, "a", encoding="utf-8") as f:
                f.write(f"{img_name}\t{text}\n")
            count += 1

    print(f"  Done: {count} images")
    return count


def download_iam_words():
    """
    下载 IAM 单词级数据
    来源: Keras OCR 教程提供的预处理版本
    约 80,000+ 英文手写单词
    """
    print("\n" + "=" * 55)
    print("[3/4] IAM 单词级数据")
    print("=" * 55)

    existing = glob.glob(os.path.join(OUTPUT_DIR, "iam_word_*.png"))
    if len(existing) > 5000:
        print(f"  已有 {len(existing)} 张，跳过下载")
        return len(existing)

    # 使用 HuggingFace 上的 IAM 单词数据集
    import zipfile
    import io

    url = "https://github.com/sayakpaul/Handwriting-Recognizer-in-Keras/releases/download/v1.0.0/IAM_Words.zip"
    zip_path = "data/IAM_Words.zip"

    try:
        print("  Downloading IAM Words dataset (~350MB)...")
        urllib.request.urlretrieve(url, zip_path)
        print("  Extracting...")

        extract_dir = "data/iam_words"
        os.makedirs(extract_dir, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(extract_dir)

        # 查找 words.txt 标注文件
        words_file = None
        for root, dirs, files in os.walk(extract_dir):
            for f in files:
                if f == "words.txt":
                    words_file = os.path.join(root, f)
                    break

        if not words_file:
            print("  words.txt not found in extracted files")
            return 0

        count = 0
        with open(words_file, 'r', encoding='utf-8') as wf:
            lines = wf.readlines()

        # 找到图片目录
        img_base = os.path.dirname(words_file)

        for line in tqdm(lines, desc="  Saving words"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 9:
                continue

            # 格式: a01-000u-00-00 ok 154 408 768 27 51 AT A
            img_id = parts[0]
            status = parts[1]
            text = parts[-1]

            # 跳过有问题的标注
            if status != "ok":
                continue

            # 找对应的 word 图片
            # 文件名格式: a01-000u-00-00.png
            img_name = f"{img_id}.png"
            img_path = os.path.join(img_base, img_name)

            if not os.path.exists(img_path):
                # 尝试在子目录中查找
                found = False
                for root, dirs, files in os.walk(extract_dir):
                    if img_name in files:
                        img_path = os.path.join(root, img_name)
                        found = True
                        break
                if not found:
                    continue

            # 拷贝到统一目录
            import shutil
            new_name = f"iam_word_{count:06d}.png"
            new_path = os.path.join(OUTPUT_DIR, new_name)
            shutil.copy2(img_path, new_path)
            with open(LABEL_FILE, "a", encoding="utf-8") as f:
                f.write(f"{new_name}\t{text}\n")
            count += 1

        print(f"  Done: {count} word images")
        return count

    except Exception as e:
        print(f"  Download failed: {e}")
        print(f"  (This is expected — IAM Words requires manual download)")
        print(f"  Visit: https://fki.tic.heia-fr.ch/databases/iam-handwriting-database")
        return 0


def generate_synthetic(num_samples, fonts=None):
    """生成 trdg 合成数据"""
    from trdg.generators import GeneratorFromStrings

    # 大规模语料库
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
        "moment", "air", "teacher", "force", "education",
        "please", "thank", "hello", "welcome", "sorry",
        "today", "tomorrow", "yesterday", "always", "never",
        "new", "good", "high", "old", "great", "big", "small",
        "large", "long", "own", "other", "right", "different",
        "important", "public", "bad", "same", "able", "simple",
        "run", "walk", "say", "go", "come", "know", "get", "make",
        "see", "think", "look", "want", "give", "use", "find",
        "tell", "ask", "try", "leave", "call", "keep", "let",
        "begin", "help", "show", "hear", "play", "move", "live",
        "believe", "hold", "bring", "happen", "write", "provide",
        "sit", "stand", "lose", "pay", "meet", "include",
        "continue", "set", "learn", "change", "lead", "understand",
        "about", "above", "across", "after", "against", "along",
        "around", "before", "behind", "below", "between", "beyond",
        "inside", "outside", "through", "within", "without", "under",
    ]

    SENTENCES = [
        "The quick brown fox jumps over the lazy dog",
        "Pack my box with five dozen liquor jugs",
        "How vexingly quick daft zebras jump",
        "A very bad quack might jinx zippy fowls",
        "Waltz bad nymph for quick jigs vex",
        "To be or not to be that is the question",
        "All that glitters is not gold",
        "A journey of a thousand miles begins with a single step",
        "Genius is one percent inspiration and ninety nine percent perspiration",
        "The only thing we have to fear is fear itself",
        "It was the best of times it was the worst of times",
        "Knowledge is power and ignorance is bliss",
        "Actions speak louder than words",
        "Practice makes perfect",
        "Every cloud has a silver lining",
    ]

    def random_text():
        if random.random() < 0.15:
            return random.choice(SENTENCES)
        n = random.randint(2, 8)
        return " ".join(random.choice(WORDS) for _ in range(n))

    text_pool = [random_text() for _ in range(max(1000, num_samples // 20))]

    generator = GeneratorFromStrings(
        strings=text_pool,
        count=num_samples,
        fonts=fonts if fonts else [],
        language='en',
        size=IMG_HEIGHT,
        skewing_angle=12,
        random_skew=True,
        blur=3,
        random_blur=True,
        background_type=1,
        distorsion_type=2,
        distorsion_orientation=2,
        text_color="#282828",
        alignment=1,
        image_dir=OUTPUT_DIR,
        fit=True,
    )

    count = 0
    for img, text in tqdm(generator, total=num_samples, desc="  Generating"):
        if count >= num_samples:
            break
        img_name = f"synth_{count:06d}.png"
        img.save(os.path.join(OUTPUT_DIR, img_name))
        with open(LABEL_FILE, "a", encoding="utf-8") as f:
            f.write(f"{img_name}\t{text}\n")
        count += 1

    return count


def main():
    parser = argparse.ArgumentParser(description="下载所有可用手写英文数据")
    parser.add_argument('--synth', type=int, default=30000,
                        help='合成数据数量（默认 3万）')
    parser.add_argument('--no-synth', action='store_true',
                        help='跳过合成数据生成')
    parser.add_argument('--no-iam-words', action='store_true',
                        help='跳过 IAM 单词数据下载')
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 清空旧标签（如果存在）
    if os.path.exists(LABEL_FILE):
        os.remove(LABEL_FILE)

    total = 0

    # 1. IAM 行级数据
    n = download_iam_lines()
    total += n

    # 2. CATMuS 英文手写
    n = download_catmus_english()
    total += n

    # 3. IAM 单词数据
    if not args.no_iam_words:
        n = download_iam_words()
        total += n

    # 4. trdg 合成数据
    if not args.no_synth:
        print("\n" + "=" * 55)
        print(f"[4/4] trdg 合成数据 ({args.synth} 张)")
        print("=" * 55)

        # 扫描可用字体
        font_dirs = ["fonts/"]
        fonts = []
        for fd in font_dirs:
            if os.path.exists(fd):
                fonts += glob.glob(os.path.join(fd, "*.ttf"))
                fonts += glob.glob(os.path.join(fd, "*.otf"))
        if fonts:
            print(f"  使用 {len(fonts)} 个自定义字体")

        n = generate_synthetic(args.synth, fonts)
        total += n

    # 最终汇总
    print("\n" + "=" * 55)
    print(f"  总计: {total} 张训练图片")
    print(f"  输出: {OUTPUT_DIR}")
    print(f"  标签: {LABEL_FILE}")
    print("=" * 55)
    print()
    print("使用方法:")
    print(f"  1. 修改 config.yaml:")
    print(f"     train_image_dir: \"{OUTPUT_DIR}\"")
    print(f"     train_label_file: \"{LABEL_FILE}\"")
    print(f"  2. python train.py")


if __name__ == "__main__":
    main()
