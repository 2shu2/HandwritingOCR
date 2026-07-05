"""
HuggingFace Spaces 手写英文识别 Demo
基于 CRNN + CTC，IAM 数据集训练
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

    # 归一化
    img = img.astype(np.float32) / 255.0
    tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)

    # 推理
    with torch.no_grad():
        log_probs = model(tensor)

    # 置信度
    probs = torch.exp(log_probs)
    max_probs, _ = probs.max(dim=-1)
    confidence = max_probs.mean().item()

    # 解码
    pred_texts = decode_predictions(log_probs, idx2char, use_beam=True, beam_width=5)
    text = pred_texts[0]

    # OCR 常见后处理修正
    text = text.replace("''", '"').replace("  ", " ")

    return text, f"{confidence:.2%}"


# 示例图片（训练集里的 IAM 真实手写样本）
examples = [
    ["examples/sample1.png"] if os.path.exists("examples/sample1.png") else None,
]

demo = gr.Interface(
    fn=recognize,
    inputs=gr.Image(type="numpy", label="上传手写英文图片"),
    outputs=[
        gr.Textbox(label="识别结果", placeholder="等待识别..."),
        gr.Textbox(label="置信度"),
    ],
    title="📝 HandwritingOCR — 手写英文识别",
    description="""
基于 **CRNN (CNN + BiLSTM + CTC)** 架构，使用 **IAM 真实手写数据集** 训练。

- 支持手写英文行识别
- 模型: Custom CNN + 2层 BiLSTM + CTC Loss
- 训练数据: IAM Handwriting Database (6,482 张)
- 验证准确率: **68.2%**
- 置信度越高越可靠

📌 上传手写英文图片，点击 Submit 即可识别。
📌 支持直接拍照上传、拖拽图片到框内。
    """,
    article="""
### 技术架构
```
输入图片 → 灰度化 → 缩放(64px高) → CRNN Model → CTC贪心解码 → 输出文字
```

### 模型信息
| 指标 | 数值 |
|------|------|
| Backbone | Custom CNN (7层) |
| RNN | 2层 BiLSTM (hidden=256) |
| 参数量 | 8.5M |
| 训练轮数 | 30 epochs |
| 验证准确率 | 68.2% |
| 数据集 | IAM Handwriting Database |

🔗 [GitHub](https://github.com/your-username/HandwritingOCR)
    """,
    examples=examples if examples[0] else None,
    theme="soft",
)

if __name__ == "__main__":
    demo.launch()
