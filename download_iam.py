"""
下载 IAM 手写英文数据集 (Teklia/IAM-line)
来源: https://huggingface.co/datasets/Teklia/IAM-line
约 10,000 张真实手写英文文本行图片
"""
import os
import yaml
from tqdm import tqdm

OUTPUT_DIR = "data/iam/images"
TRAIN_LABEL = "data/iam/train_labels.txt"
VAL_LABEL = "data/iam/val_labels.txt"
TEST_LABEL = "data/iam/test_labels.txt"


def download_iam():
    from datasets import load_dataset

    print("Downloading IAM Handwriting Dataset (Teklia/IAM-line)...")
    print("This may take a few minutes for the first download (~500MB).\n")

    # 加载数据集（自动下载并缓存到 ~/.cache/huggingface/）
    # SSL 证书问题绕过（仅用于下载）
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    dataset = load_dataset("Teklia/IAM-line")

    print(f"\nDataset loaded:")
    for split in dataset:
        print(f"  {split}: {len(dataset[split])} samples")

    # 保存图片和标签
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for split_name, label_file in [
        ("train", TRAIN_LABEL),
        ("validation", VAL_LABEL),
        ("test", TEST_LABEL),
    ]:
        if split_name not in dataset:
            print(f"  Split '{split_name}' not found, skipping...")
            continue

        os.makedirs(os.path.dirname(label_file), exist_ok=True)
        with open(label_file, "w", encoding="utf-8") as f:
            for i, sample in enumerate(tqdm(dataset[split_name], desc=f"  Saving {split_name}")):
                img_name = f"iam_{split_name}_{i:05d}.png"
                img_path = os.path.join(OUTPUT_DIR, img_name)
                # 保存 PIL Image
                sample["image"].save(img_path)
                text = sample["text"].replace("\n", " ").replace("\t", " ")
                f.write(f"{img_name}\t{text}\n")

    print(f"\nIAM dataset saved to {OUTPUT_DIR}/")
    print(f"Train labels: {TRAIN_LABEL}")
    print(f"Val labels:   {VAL_LABEL}")
    print(f"Test labels:  {TEST_LABEL}")


if __name__ == "__main__":
    download_iam()
