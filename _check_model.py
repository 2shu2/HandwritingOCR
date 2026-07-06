"""Check model accuracy on 10 samples (uses CER / beam search)"""
import torch, cv2, yaml
from model import CRNN
from data import decode_predictions


def levenshtein_distance(s1: str, s2: str) -> int:
    """编辑距离"""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (0 if c1 == c2 else 1)))
        prev = curr
    return prev[-1]


# Load config
with open('config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

alphabet = config['data']['alphabet']
num_classes = len(alphabet) + 1
idx2char = {i + 1: ch for i, ch in enumerate(alphabet)}
idx2char[0] = ''

# 使用config中的模型参数
model = CRNN(
    num_classes,
    config['model']['backbone'],
    config['model']['hidden_size'],
    config['model']['num_layers'],
    img_height=config['data']['img_height']
)
model.load_state_dict(torch.load('checkpoints/best_model.pth', map_location='cpu'))
model.eval()

# Test first 10 samples
with open('data/trdg/labels.txt') as f:
    lines = f.readlines()[:10]

total_dist = 0
total_chars = 0
for line in lines:
    fname, true_text = line.strip().split('\t')
    img = cv2.imread(f'data/trdg/images/{fname}', cv2.IMREAD_GRAYSCALE)
    h, w = img.shape
    scale = 64 / h
    img = cv2.resize(img, (max(int(w * scale), 1), 64))
    img = img.astype('float32') / 255.0
    tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        log_probs = model(tensor)
    # 使用Beam Search解码
    pred = decode_predictions(log_probs, idx2char, use_beam=True, beam_width=5)[0]
    dist = levenshtein_distance(pred, true_text)
    total_dist += dist
    total_chars += len(true_text)
    status = "OK" if pred == true_text else "MISMATCH"
    cer = dist / max(len(true_text), 1)
    print(f'{status}: true="{true_text}" -> pred="{pred}" (CER: {cer:.2f})')

acc = 100 * (1 - total_dist / total_chars) if total_chars > 0 else 0
print(f'\nOverall CER: {total_dist}/{total_chars} = {total_dist/total_chars:.4f}')
print(f'Char Accuracy: {acc:.1f}%')
