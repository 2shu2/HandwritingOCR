import os
import random
from sklearn.model_selection import train_test_split

DATA_DIR = "data/trdg"
LABEL_FILE = os.path.join(DATA_DIR, "labels.txt")
TRAIN_FILE = os.path.join(DATA_DIR, "train_labels.txt")
VAL_FILE = os.path.join(DATA_DIR, "val_labels.txt")

with open(LABEL_FILE, 'r', encoding='utf-8') as f:
    lines = f.readlines()

train_lines, val_lines = train_test_split(lines, test_size=0.2, random_state=42)

with open(TRAIN_FILE, 'w', encoding='utf-8') as f:
    f.writelines(train_lines)
with open(VAL_FILE, 'w', encoding='utf-8') as f:
    f.writelines(val_lines)

print(f"Train samples: {len(train_lines)}, Val samples: {len(val_lines)}")