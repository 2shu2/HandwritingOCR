"""Check model accuracy on 10 samples"""
import torch, cv2, yaml
from model import CRNN
from data import decode_predictions

# Load config for alphabet
with open('config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

alphabet = config['data']['alphabet']
num_classes = len(alphabet) + 1
idx2char = {i + 1: ch for i, ch in enumerate(alphabet)}
idx2char[0] = ''

model = CRNN(num_classes, 'resnet34', 256, 2, img_height=64)
model.load_state_dict(torch.load('checkpoints/best_model.pth', map_location='cpu'))
model.eval()

# Test first 10 samples
with open('data/trdg/labels.txt') as f:
    lines = f.readlines()[:10]

total_correct = 0
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
    pred = decode_predictions(log_probs, idx2char)[0]
    correct = sum(p == t for p, t in zip(pred, true_text))
    total_correct += correct
    total_chars += len(true_text)
    status = "OK" if pred == true_text else "MISMATCH"
    print(f'{status}: true="{true_text}" -> pred="{pred}" ({correct}/{len(true_text)} chars)')

print(f'\nOverall: {total_correct}/{total_chars} = {100*total_correct/total_chars:.1f}% char accuracy')
