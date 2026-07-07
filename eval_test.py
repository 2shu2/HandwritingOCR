"""验证集完整评估（贪心 + Beam Search）"""
import torch, yaml, sys
sys.path.insert(0, '.')
from model import CRNN
from data import TextLineDataset, collate_fn, decode_predictions
from torch.utils.data import DataLoader

def lev(s1, s2):
    if len(s1) < len(s2): return lev(s2, s1)
    if len(s2) == 0: return len(s1)
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            curr.append(min(prev[j+1]+1, curr[j]+1, prev[j]+(0 if c1==c2 else 1)))
        prev = curr
    return prev[-1]

with open('config.yaml', encoding='utf-8') as f:
    config = yaml.safe_load(f)

alphabet = config['data']['alphabet']
ncls = len(alphabet) + 1
i2c = {i+1: ch for i, ch in enumerate(alphabet)}
i2c[0] = ''

print('加载模型...')
device = torch.device('cuda')
model = CRNN(ncls, config['model']['backbone'],
             config['model']['hidden_size'], config['model']['num_layers'],
             img_height=config['data']['img_height']).to(device)
model.load_state_dict(torch.load('checkpoints/best_model.pth', map_location=device))
model.eval()

print('加载验证集...')
val_ds = TextLineDataset(config['data']['val_image_dir'],
    config['data']['val_label_file'], alphabet, config['data']['img_height'], is_train=False)

loader = DataLoader(val_ds, batch_size=16, shuffle=False, collate_fn=collate_fn)

print(f'评估 {len(val_ds)} 张图片...')
print()

# ---- 贪心解码 ----
total_cer = 0.0
cnt = 0
with torch.no_grad():
    for imgs, labels in loader:
        imgs = imgs.to(device)
        log_probs = model(imgs)
        preds = decode_predictions(log_probs, val_ds.idx2char, use_beam=False)
        for pred_str, lbl in zip(preds, labels):
            true_str = ''.join(val_ds.idx2char.get(x.item(), '') for x in lbl if x.item() != 0)
            total_cer += lev(pred_str, true_str) / max(len(true_str), 1)
            cnt += 1
        if cnt % 500 == 0:
            print(f'  贪心解码: {cnt}/{len(val_ds)}', flush=True)

acc_g = (1 - total_cer / cnt) * 100
print(f'[贪心解码]  CER = {total_cer/cnt:.4f}  准确率 = {acc_g:.2f}%')
print()

# ---- Beam Search ----
total_cer = 0.0
cnt = 0
with torch.no_grad():
    for imgs, labels in loader:
        imgs = imgs.to(device)
        log_probs = model(imgs)
        preds = decode_predictions(log_probs, val_ds.idx2char, use_beam=True, beam_width=5)
        for pred_str, lbl in zip(preds, labels):
            true_str = ''.join(val_ds.idx2char.get(x.item(), '') for x in lbl if x.item() != 0)
            total_cer += lev(pred_str, true_str) / max(len(true_str), 1)
            cnt += 1
        if cnt % 500 == 0:
            print(f'  Beam Search: {cnt}/{len(val_ds)}', flush=True)

acc_b = (1 - total_cer / cnt) * 100
print(f'[Beam Search]  CER = {total_cer/cnt:.4f}  准确率 = {acc_b:.2f}%')
print()
print(f'验证集总数: {cnt} 张')
