"""
HandwritingOCR — 手写英文识别 Gradio Demo
基于 CRNN (ResNet34 + 3层BiLSTM + CTC)，IAM 数据集训练
"""
import os
import numpy as np
import torch
import cv2
import yaml
import gradio as gr
from model import CRNN
from data import decode_predictions

# 切换到脚本目录
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 加载配置和模型
with open("config.yaml", 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

alphabet = config['data']['alphabet']
num_classes = len(alphabet) + 1
idx2char = {i + 1: ch for i, ch in enumerate(alphabet)}
idx2char[0] = ''

print("Loading model...")
model = CRNN(
    num_classes,
    config['model']['backbone'],
    config['model']['hidden_size'],
    config['model']['num_layers'],
    img_height=config['inference']['img_height']
)
model.load_state_dict(torch.load(config['inference']['model_path'], map_location='cpu'))
model.eval()
print("Model loaded!")


def recognize(image):
    """识别上传的手写图片"""
    if image is None:
        return "", ""

    # 转灰度
    if len(image.shape) == 3:
        img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        img = image

    # 缩放到统一高度，保持宽高比
    h, w = img.shape
    img_height = config['inference']['img_height']
    scale = img_height / h
    new_w = max(int(w * scale), 1)
    img = cv2.resize(img, (new_w, img_height))

    # 归一化 + 转为3通道伪RGB（适配 ResNet34 预训练权重）
    img = img.astype(np.float32) / 255.0
    tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    tensor = tensor.repeat(1, 3, 1, 1)  # (1, 3, H, W) 灰度复制3份

    # 推理
    with torch.no_grad():
        log_probs = model(tensor)

    # 置信度
    probs = torch.exp(log_probs)
    max_probs, _ = probs.max(dim=-1)
    confidence = max_probs.mean().item()

    # Beam Search 解码（更准确）
    pred_texts = decode_predictions(log_probs, idx2char, use_beam=True, beam_width=5)
    text = pred_texts[0]

    # OCR 后处理修正
    text = text.replace("''", '"').replace("  ", " ")

    return text, f"{confidence:.2%}"


# 示例图片
examples = [
    ["examples/sample1.png"] if os.path.exists("examples/sample1.png") else None,
]

demo = gr.Interface(
    fn=recognize,
    inputs=gr.Image(type="numpy", label="Upload handwritten English image"),
    outputs=[
        gr.Textbox(label="Recognition Result", placeholder="Waiting..."),
        gr.Textbox(label="Confidence"),
    ],
    title="HandwritingOCR — Handwritten English Recognition",
    description="""
**CRNN (ResNet34 + 3-layer BiLSTM + CTC)** architecture, trained on **IAM Handwriting Database**.

- Handwritten English text line recognition
- Model: ResNet34 + 3-layer BiLSTM (hidden=512) + CTC Loss
- Training data: IAM Handwriting Database (7,154 train + 1,789 val)
- Validation accuracy: **89.9%** (CER-based)
- Beam Search decoding for best accuracy

Upload a handwritten English image and click Submit to recognize.
    """,
    article="""
### Architecture
```
Input Image → Grayscale → Resize(128px) → ResNet34 → 3-layer BiLSTM → CTC Beam Search → Output Text
```

### Model Info
| Metric | Value |
|--------|-------|
| Backbone | ResNet34 (ImageNet pretrained) |
| RNN | 3-layer BiLSTM (hidden=512) |
| Parameters | ~46M |
| Epochs | 50 |
| Batch Size | 16 |
| Validation Acc | **89.9%** |
| Dataset | IAM Handwriting Database |

[GitHub](https://github.com/2shu2/HandwritingOCR)
    """,
    examples=examples if examples[0] else None,
    theme="soft",
)

if __name__ == "__main__":
    demo.launch()
