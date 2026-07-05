"""
FastAPI 在线手写文字识别服务
启动方式: uvicorn app:app --host 0.0.0.0 --port 8000
"""
import io
import os
import numpy as np
import torch
import cv2
import yaml
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import logging

from model import CRNN
from data import decode_predictions

# 日志配置
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("handwriting_ocr")

app = FastAPI(title="手写文字识别API", version="1.0.0")

# 切换到脚本所在目录，确保相对路径正确
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 加载配置和模型
with open("config.yaml", 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)
alphabet = config['data']['alphabet']
num_classes = len(alphabet) + 1
model = CRNN(num_classes, config['model']['backbone'],
             config['model']['hidden_size'], config['model']['num_layers'],
             img_height=config['inference']['img_height'])
model.load_state_dict(torch.load(config['inference']['model_path'], map_location='cpu'))
model.eval()
# 构建字符映射
idx2char = {i+1: ch for i, ch in enumerate(alphabet)}
idx2char[0] = ''


class PredictionResult(BaseModel):
    text: str
    confidence: float


def preprocess_bytes(image_bytes: bytes, img_height: int) -> torch.Tensor:
    """将上传的图片字节流转为模型输入张量"""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("无法解码图片")
    h, w = img.shape
    scale = img_height / h
    new_w = max(int(w * scale), 1)
    img = cv2.resize(img, (new_w, img_height))
    img = img.astype(np.float32) / 255.0
    return torch.from_numpy(img).unsqueeze(0).unsqueeze(0)


@app.post("/predict", response_model=PredictionResult)
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="仅支持图片文件")
    try:
        contents = await file.read()
        img_tensor = preprocess_bytes(contents, config['inference']['img_height'])
        with torch.no_grad():
            log_probs = model(img_tensor)
        # 置信度：取每个时间步最大概率的平均值
        probs = torch.exp(log_probs)
        max_probs, _ = probs.max(dim=-1)
        confidence = max_probs.mean().item()
        pred_texts = decode_predictions(log_probs, idx2char)
        text = pred_texts[0]
        logger.info(f"识别结果: {text}, 置信度: {confidence:.3f}")
        return PredictionResult(text=text, confidence=confidence)
    except Exception as e:
        logger.error(f"推理错误: {str(e)}")
        raise HTTPException(status_code=500, detail="内部服务器错误")