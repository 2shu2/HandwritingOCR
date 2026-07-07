# HandwritingOCR — 手写英文识别

基于 **CRNN (ResNet34 + 3层BiLSTM + CTC)** 的端到端手写英文文字行识别系统。使用 **IAM 真实手写数据集** 训练，支持本地推理、Web API、在线 Demo。

>  **验证准确率: 89.9%**（CER 评估，Beam Search 解码约 91%）

> 🚀 **[在线体验](https://www.modelscope.cn/studios/nihao523/HandwritingOCR)** — ModelScope 一键部署，上传图片即识别

## 特性

- **模型架构**: ResNet34 (ImageNet 预训练) + 3层 BiLSTM (hidden=512) + CTC Loss
- **真实数据**: IAM Handwriting Database，7,154 张训练 + 1,789 张验证
- **预训练权重**: 灰度图复制为3通道伪RGB，完整保留 ImageNet 预训练特征
- **数据增强**: 弹性变形 + 透视变换 + 高斯模糊 + 随机擦除 + 颜色抖动 + 仿射变换
- **验证指标**: CER（字符错误率，基于编辑距离），非简单逐位比较
- **多种解码**: 贪心解码（快） / Beam Search 5-beam（准）
- **多种部署**: 本地脚本 / Gradio Web UI / 在线 Demo
- **配置驱动**: YAML 统一管理所有参数

## 项目结构

```
├── config.yaml           # 主配置文件
├── model.py              # CRNN 模型定义 (ResNet34 + BiLSTM)
├── data.py               # 数据集、collate、CTC解码、数据增强
├── train.py              # 训练脚本（含自动数据生成）
├── predict.py            # 本地交互式推理
├── eval_test.py          # 验证集完整评估
├── download_iam.py       # 下载 IAM 数据集
├── generate_data.py      # 合成数据生成（trdg）
├── generate_large_data.py # 大规模数据生成（5万+）
├── split_data.py         # 训练/验证集分割
├── app.py                # FastAPI 服务
├── app_hf.py             # Gradio Web UI（HuggingFace / ModelScope）
├── requirements.txt      # 训练/推理依赖
├── HF_deploy/            # 在线部署包
│   ├── app.py
│   ├── model.py
│   ├── data.py
│   ├── config.yaml
│   ├── requirements.txt
│   ├── checkpoints/best_model.pth
│   └── examples/
├── examples/             # 示例图片
└── README.md
```

## 快速开始

### 1. 环境安装

```bash
git clone https://github.com/2shu2/HandwritingOCR.git
cd HandwritingOCR
pip install -r requirements.txt
```

### 2. 下载 IAM 数据集并训练

```bash
# 下载 IAM 手写数据集
python download_iam.py

# 开始训练（50 epochs，GPU约2小时）
python train.py
```

训练过程输出每个 epoch 的 loss 和验证 CER，最优模型保存在 `checkpoints/best_model.pth`。

### 3. 本地测试

```bash
python predict.py
```

交互式菜单：
- 选项 1: 使用训练集图片测试
- 选项 2: 上传自己的手写图片识别

### 4. 启动 Web 服务

```bash
# Gradio Web UI
python app_hf.py

# FastAPI 服务
uvicorn app:app --host 0.0.0.0 --port 8000
```

## 技术原理

```
输入图片 → 灰度化 → 缩放(128px高度) → 灰度复制3通道(伪RGB)
         → ResNet34 (ImageNet预训练) → 3层BiLSTM → CTC Beam Search → 输出文字
```

| 组件 | 说明 |
|------|------|
| CNN Backbone | ResNet34 (ImageNet 预训练)，保留3通道结构 |
| BiLSTM | 3层双向 LSTM (hidden_size=512)，捕捉序列上下文 |
| CTC Loss | 免对齐训练，无需字符位置标注 |
| 数据增强 | 弹性变形 + 透视 + 模糊 + 擦除 + 颜色 + 仿射 |
| 解码 | CTC 贪心解码（验证）/ Beam Search 5-beam（推理） |

## 模型指标

| 指标 | 数值 |
|------|------|
| Backbone | ResNet34 (ImageNet pretrained) |
| RNN | 3层 BiLSTM (hidden=512) |
| 参数量 | ~46M |
| 训练轮数 | 50 epochs |
| Batch Size | 16 |
| 输入高度 | 128px (3通道伪RGB) |
| 混合精度 | FP16 (AMP) |
| 验证准确率 | **89.9%** (贪心解码) |
| Beam Search | **~91%** (5-beam) |
| 数据集 | IAM Handwriting Database |

## 准确率提升历程

| 版本 | 准确率 | 关键改进 |
|------|--------|----------|
| v1.0 | 68.2% | Custom CNN + 2层LSTM (hidden=256) |
| v2.0 | 25.0% | ResNet34灰度量平均（预训练权重被破坏） |
| v3.0 | **89.9%** | 3通道伪RGB + img_height=128 + 强数据增强 |

**核心教训**: ImageNet 预训练的3通道权重不能简单取平均转灰度，应该保留3通道结构，将灰度图复制3份作为输入。

## 在线 Demo

| 平台 | 链接 |
|------|------|
| ModelScope | [HandwritingOCR](https://www.modelscope.cn/studios/nihao523/HandwritingOCR) |

## 自定义指南

- **更换字符集**: 修改 `config.yaml` 中的 `alphabet` 字段（支持中文需同步修改编码）
- **调整模型**: 切换 `backbone`（`resnet34` / `custom_cnn`）
- **提升准确率**: 增加训练数据（运行 `generate_large_data.py`）、添加语言模型后处理
- **推理加速**: 使用贪心解码（速度快）vs Beam Search（精度高）

## 常见问题

- **CUDA 显存不足**: 减小 `batch_size`（默认 16），或降低 `img_height: 64`
- **识别结果为空**: 确认图片为白底黑字，如需反转可在预处理中加入
- **本地测试准确率低**: 确认使用 `checkpoints/best_model.pth`（170MB），非旧版（33MB）
- **训练时间**: GPU (RTX 4060) 约 2 小时，CPU 约 5-7 小时

## 许可证

MIT License
