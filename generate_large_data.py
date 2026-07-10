"""
大规模训练数据生成器
- 自动下载免费手写字体（Google Fonts 开源字体）
- 使用 trdg 生成多样化风格的手写图片
- 可生成 5万~10万+ 张训练数据
- 结合 IAM 真实数据一起训练

用法：
  python generate_large_data.py              # 生成 5万张（默认）
  python generate_large_data.py --count 100000  # 生成 10万张
"""
import os
import sys
import glob
import random
import argparse
import urllib.request
import zipfile
import shutil
from trdg.generators import GeneratorFromStrings

# ========== 配置 ==========
OUTPUT_DIR = "data/large_dataset/images"
LABEL_FILE = "data/large_dataset/labels.txt"
FONT_DIR = "fonts/"
DEFAULT_COUNT = 50000
IMG_HEIGHT = 64

# 免费手写字体下载列表（Google Fonts / 开源字体）
# 这些字体模拟不同书写风格，极大增加数据多样性
HANDWRITING_FONTS = [
    # Google Fonts 手写风格（开源免费）
    ("https://github.com/google/fonts/raw/main/ofl/caveat/static/Caveat-Regular.ttf", "Caveat-Regular.ttf"),
    ("https://github.com/google/fonts/raw/main/ofl/caveat/static/Caveat-Bold.ttf", "Caveat-Bold.ttf"),
    ("https://github.com/google/fonts/raw/main/ofl/shadowsintolight/ShadowsIntoLight.ttf", "ShadowsIntoLight.ttf"),
    ("https://github.com/google/fonts/raw/main/ofl/indieflower/IndieFlower.ttf", "IndieFlower.ttf"),
    ("https://github.com/google/fonts/raw/main/ofl/handlee/Handlee-Regular.ttf", "Handlee-Regular.ttf"),
    ("https://github.com/google/fonts/raw/main/ofl/architectsdaughter/ArchitectsDaughter.ttf", "ArchitectsDaughter.ttf"),
    ("https://github.com/google/fonts/raw/main/ofl/kalam/Kalam-Regular.ttf", "Kalam-Regular.ttf"),
    ("https://github.com/google/fonts/raw/main/ofl/kalam/Kalam-Bold.ttf", "Kalam-Bold.ttf"),
    ("https://github.com/google/fonts/raw/main/ofl/gloriahallelujah/GloriaHallelujah.ttf", "GloriaHallelujah.ttf"),
    ("https://github.com/google/fonts/raw/main/ofl/permanentmarker/PermanentMarker-Regular.ttf", "PermanentMarker-Regular.ttf"),
    ("https://github.com/google/fonts/raw/main/ofl/satisfy/Satisfy-Regular.ttf", "Satisfy-Regular.ttf"),
    ("https://github.com/google/fonts/raw/main/ofl/dancingscript/static/DancingScript-Regular.ttf", "DancingScript-Regular.ttf"),
    ("https://github.com/google/fonts/raw/main/ofl/dancingscript/static/DancingScript-Bold.ttf", "DancingScript-Bold.ttf"),
    ("https://github.com/google/fonts/raw/main/ofl/patrickhand/PatrickHand-Regular.ttf", "PatrickHand-Regular.ttf"),
    ("https://github.com/google/fonts/raw/main/ofl/schoolbell/Schoolbell-Regular.ttf", "Schoolbell-Regular.ttf"),
    ("https://github.com/google/fonts/raw/main/ofl/justanotherhand/JustAnotherHand-Regular.ttf", "JustAnotherHand-Regular.ttf"),
    ("https://github.com/google/fonts/raw/main/ofl/coveredbyyourgrace/CoveredByYourGrace.ttf", "CoveredByYourGrace.ttf"),
    ("https://github.com/google/fonts/raw/main/ofl/comingsoon/ComingSoon-Regular.ttf", "ComingSoon-Regular.ttf"),
    ("https://github.com/google/fonts/raw/main/ofl/homemadeapple/HomemadeApple-Regular.ttf", "HomemadeApple-Regular.ttf"),
    ("https://github.com/google/fonts/raw/main/ofl/rockalt/Rockalt-Regular.ttf", "Rockalt-Regular.ttf"),
]

# 大规模英文语料 —— 常见词汇随机组合，生成无数不同句子
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
    "spend", "grow", "open", "win", "offer", "remember", "love",
    "consider", "appear", "buy", "wait", "serve", "die", "send",
    "expect", "build", "stay", "fall", "cut", "reach", "kill", "remain",
    "suggest", "raise", "pass", "sell", "require", "report", "decide",
    "pull", "new", "good", "high", "old", "great", "big", "small",
    "large", "long", "own", "other", "right", "different", "next",
    "early", "young", "important", "public", "bad", "same", "able",
    "simple", "sure", "strong", "real", "full", "special", "clear",
    "easy", "late", "hard", "major", "better", "economic",
    "possible", "whole", "free", "military", "true", "federal",
    "international", "certain", "personal", "happy", "final", "main",
    "nice", "short", "left", "dead", "ready", "common",
    "please", "thank", "hello", "welcome", "sorry", "excuse",
    "today", "tomorrow", "yesterday", "always", "never", "sometimes",
    "often", "usually", "really", "very", "quite", "almost", "enough",
    "still", "already", "just", "only", "also", "even", "however",
    "therefore", "because", "although", "unless", "until", "during",
    "about", "above", "across", "after", "against", "along", "among",
    "around", "before", "behind", "below", "beneath", "beside",
    "between", "beyond", "inside", "outside", "through", "within",
    "without", "upon", "since", "toward", "under", "over",
]

# IAM 数据集中的真实句子（增加真实文本分布）
IAM_SENTENCES = [
    "The quick brown fox jumps over the lazy dog",
    "Pack my box with five dozen liquor jugs",
    "How vexingly quick daft zebras jump",
    "The five boxing wizards jump quickly",
    "Mr Jock TV quiz PhD bags few lynx",
    "A very bad quack might jinx zippy fowls",
    "Waltz bad nymph for quick jigs vex",
    "Glib jocks quiz nymph to vex dwarf",
    "Sphinx of black quartz judge my vow",
    "Jackdaws love my big sphinx of quartz",
    "We promptly judged antique ivory buckles for the next prize",
    "Sixty zippers were quickly picked from the woven jute bag",
    "Crazy Fredericka bought many very exquisite opal jewels",
    "Jump by vow of quick lazy strength in Oxford",
    "The explorer was frozen in his big kayak just after making discoveries",
    "Whenever the black fox jumped the squirrel gazed suspiciously",
    "A journey of a thousand miles begins with a single step",
    "To be or not to be that is the question",
    "All that glitters is not gold",
    "Ask not what your country can do for you",
    "Genius is one percent inspiration and ninety nine percent perspiration",
    "The only thing we have to fear is fear itself",
    "I have a dream that one day this nation will rise up",
    "In the beginning God created the heaven and the earth",
    "It was the best of times it was the worst of times",
    "Happy families are all alike every unhappy family is unhappy in its own way",
    "It is a truth universally acknowledged that a single man in possession",
    "Call me Ishmael some years ago never mind how long precisely",
    "You dont know about me without you have read a book",
    "The past is a foreign country they do things differently there",
]


def random_text(min_words=2, max_words=8):
    """随机生成英文短语/句子"""
    n = random.randint(min_words, max_words)
    return " ".join(random.choice(WORDS) for _ in range(n))


def random_sentence():
    """随机句子，80% 概率随机生成，20% 使用知名句子"""
    if random.random() < 0.2:
        return random.choice(IAM_SENTENCES)
    return random_text(3, 10)


def download_fonts():
    """下载免费手写字体到 fonts/ 目录"""
    os.makedirs(FONT_DIR, exist_ok=True)

    downloaded = 0
    for url, filename in HANDWRITING_FONTS:
        filepath = os.path.join(FONT_DIR, filename)
        if os.path.exists(filepath):
            downloaded += 1
            continue
        try:
            print(f"  Downloading {filename}...", end=" ")
            urllib.request.urlretrieve(url, filepath)
            print("OK")
            downloaded += 1
        except Exception as e:
            print(f"Failed: {e}")

    print(f"Fonts ready: {downloaded}/{len(HANDWRITING_FONTS)}")

    # 扫描可用字体
    fonts = glob.glob(os.path.join(FONT_DIR, "*.ttf"))
    fonts += glob.glob(os.path.join(FONT_DIR, "*.otf"))
    return fonts


def generate_data(num_samples, fonts):
    """生成训练数据"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 生成多样化的文本池
    print(f"Generating text corpus...")
    text_pool = []
    for _ in range(max(2000, num_samples // 10)):
        text_pool.append(random_sentence())
    print(f"  {len(text_pool)} unique sentences")

    # 如果字体太少，让 trdg 也用内置字体
    use_fonts = fonts if fonts else []

    print(f"Generating {num_samples} images with {len(use_fonts) or 'trdg built-in'} fonts...")

    generator = GeneratorFromStrings(
        strings=text_pool,
        count=num_samples,
        fonts=use_fonts,
        language='en',
        size=IMG_HEIGHT,
        skewing_angle=12,              # 倾斜角度加大 → 更多样
        random_skew=True,
        blur=3,                        # 模糊加大
        random_blur=True,
        background_type=1,             # 纯色背景（干净数据）
        distorsion_type=2,             # 余弦扭曲
        distorsion_orientation=2,       # 双向扭曲
        text_color="#282828",
        alignment=1,                   # 居中
        image_dir=OUTPUT_DIR,
        fit=True,
    )

    with open(LABEL_FILE, 'w', encoding='utf-8') as f:
        for i, (img, text) in enumerate(generator):
            if i >= num_samples:
                break
            img_name = f"synth_{i:06d}.png"
            img.save(os.path.join(OUTPUT_DIR, img_name))
            f.write(f"{img_name}\t{text}\n")
            if (i + 1) % 500 == 0:
                print(f"  Generated {i + 1}/{num_samples} images ({(i+1)*100//num_samples}%)")

    print(f"\nDone! {num_samples} images → {OUTPUT_DIR}")
    print(f"Labels → {LABEL_FILE}")


def merge_with_iam():
    """
    将 IAM 真实数据复制到大规模数据集文件夹中，
    同时保留原始标注，形成"合成+真实"混合训练集
    """
    iam_dir = "data/iam/images"
    iam_label = "data/iam/train_labels.txt"

    if not os.path.exists(iam_label):
        print("\nIAM data not found, skipping merge.")
        print("Run 'python download_iam.py' first to get real handwriting data.")
        return 0

    print(f"\nMerging IAM data into large_dataset...")
    count = 0
    with open(iam_label, 'r', encoding='utf-8') as src, \
         open(LABEL_FILE, 'a', encoding='utf-8') as dst:
        for line in src:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                img_name, text = parts[0], parts[1]
                src_path = os.path.join(iam_dir, img_name)
                dst_path = os.path.join(OUTPUT_DIR, f"iam_{img_name}")
                if os.path.exists(src_path):
                    shutil.copy2(src_path, dst_path)
                    dst.write(f"iam_{img_name}\t{text}\n")
                    count += 1

    print(f"  Merged {count} IAM images")
    return count


def main():
    parser = argparse.ArgumentParser(description="大规模手写数据生成")
    parser.add_argument('--count', type=int, default=DEFAULT_COUNT,
                        help=f'生成图片数量（默认 {DEFAULT_COUNT}）')
    parser.add_argument('--no-fonts', action='store_true',
                        help='跳过字体下载（使用 trdg 内置字体）')
    parser.add_argument('--no-merge', action='store_true',
                        help='不合并 IAM 数据')
    args = parser.parse_args()

    print("=" * 55)
    print("  大规模手写英文数据生成器")
    print("=" * 55)
    print()

    # 1. 下载字体
    if args.no_fonts:
        fonts = []
        print("Skipping font download.")
    else:
        print("[1/3] Downloading handwriting fonts...")
        fonts = download_fonts()
        print()

    # 2. 生成合成数据
    print(f"[2/3] Generating {args.count} synthetic images...")
    generate_data(args.count, fonts)

    # 3. 合并 IAM 真实数据
    iam_count = 0
    if not args.no_merge:
        print(f"[3/3] Merging real IAM data...")
        iam_count = merge_with_iam()

    total = args.count + iam_count
    print()
    print("=" * 55)
    print(f"  Total dataset: {total} images")
    print(f"    - Synthetic: {args.count}")
    if iam_count:
        print(f"    - IAM real:  {iam_count}")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"  Labels: {LABEL_FILE}")
    print("=" * 55)
    print()
    print("Next: Update config.yaml to use this data:")
    print(f"  train_image_dir: \"{OUTPUT_DIR}\"")
    print(f"  train_label_file: \"{LABEL_FILE}\"")


if __name__ == "__main__":
    main()
