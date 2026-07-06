# HandwritingOCR — 手写英文识别

基于 **CRNN (CNN + BiLSTM + CTC)** 的端到端手写英文文字行识别系统。使用 **IAM 真实手写数据集** 训练，支持本地推理、Web API、在线 Demo。

> 🚀 **[在线体验](https://www.modelscope.cn/studios/nihao523/HandwritingOCR)** — ModelScope 一键部署，上传图片即识别

## 演示

<video src="./demo.mp4" controls width="100%"></video>

## 特性

- **模型架构**: Custom CNN (7层) + 2层 BiLSTM (hidden=256) + CTC Loss
- **真实数据**: IAM Handwriting Database，6,482 张训练 + 976 张验证
- **轻量高效**: 仅 8.5M 参数，33MB 模型文件
- **多种部署**: 本地脚本 / FastAPI 服务 / Gradio Web UI
- **配置驱动**: YAML 统一管理所有参数

## 项目结构

```
├── config.yaml           # 主配置文件
├── model.py              # CRNN 模型定义
├── data.py               # 数据集、collate、CTC解码
├── train.py              # 训练脚本（含自动数据生成）
├── predict.py            # 本地交互式推理
├── download_iam.py       # 下载 IAM 数据集
├── generate_data.py      # 合成数据生成（trdg）
├── split_data.py         # 训练/验证集分割
├── app.py                # FastAPI 服务
├── app_hf.py             # Gradio Web UI（HuggingFace / ModelScope）
├── requirements.txt      # 训练/推理依赖
├── requirements_hf.txt   # 在线部署依赖（精简）
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
git clone https://github.com/nihao523/HandwritingOCR.git
cd HandwritingOCR
pip install -r requirements.txt
```

### 2. 下载 IAM 数据集并训练

```bash
# 下载 IAM 手写数据集
python download_iam.py

# 开始训练
python train.py
```

训练过程会输出每个 epoch 的 loss 和验证准确率，最优模型保存在 `checkpoints/best_model.pth`。

### 3. 本地测试

```bash
python predict.py
```

交互式菜单：
- 选项 1: 使用训练集图片测试
- 选项 2: 上传自己的手写图片识别

### 4. 启动 Web 服务

```bash
# FastAPI 服务
uvicorn app:app --host 0.0.0.0 --port 8000

# Gradio Web UI
python app_hf.py
```

## 技术原理

```
输入图片 → 灰度化 → 缩放(64px高度) → CRNN Model → CTC贪心解码 → 输出文字
```

| 组件 | 说明 |
|------|------|
| CNN Backbone | 7层自定义卷积网络，逐层下采样提取特征 |
| BiLSTM | 2层双向 LSTM (hidden_size=256)，捕捉序列上下文 |
| CTC Loss | 免对齐训练，无需字符位置标注 |
| 解码 | CTC 贪心解码 + 后处理修正 |

## 模型指标

| 指标 | 数值 |
|------|------|
| Backbone | Custom CNN (7层) |
| RNN | 2层 BiLSTM (hidden=256) |
| 参数量 | 8.5M |
| 训练轮数 | 30 epochs |
| Batch Size | 8 |
| 验证准确率 | **68.2%** |
| 数据集 | IAM Handwriting Database |

## 在线 Demo

| 平台 | 链接 |
|------|------|
| ModelScope | [HandwritingOCR](https://www.modelscope.cn/studios/nihao523/HandwritingOCR) |

## 自定义指南

- **更换字符集**: 修改 `config.yaml` 中的 `alphabet` 字段
- **调整模型**: 切换 `backbone`（`custom_cnn` / `resnet34`）
- **提升准确率**: 增加训练数据、添加语言模型后处理、调大 hidden_size

## 常见问题

- **CUDA 显存不足**: 减小 `batch_size`（默认 8）
- **识别结果为空**: 确认图片为白底黑字，如需反转可在预处理中加入
- **字符准确率低**: 当前模型在 IAM 数据集上准确率 68.2%，手写识别本身难度较高

## 许可证

MIT License
