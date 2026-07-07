"""
CRNN 手写文字识别模型
CNN(ResNet34) 提取视觉特征 -> LSTM 建模序列关系 -> 全连接分类
输出对数概率，配合 CTC 损失进行训练
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class CRNN(nn.Module):
    """卷积循环神经网络，用于文本行识别"""

    def __init__(self, num_classes: int, backbone: str = 'resnet34',
                 hidden_size: int = 256, num_layers: int = 2,
                 img_height: int = 64):
        super().__init__()

        if backbone == 'resnet34':
            # 使用ResNet34，保留原始3通道结构和ImageNet预训练权重
            # 输入是3通道伪RGB（灰度图复制3份），不破坏预训练特征
            resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
            self.cnn = nn.Sequential(
                resnet.conv1,   # 3→64, stride 2
                resnet.bn1,
                resnet.relu,
                resnet.maxpool,  # /2
                resnet.layer1,   # 64
                resnet.layer2,   # 128
                resnet.layer3,   # 256
                resnet.layer4    # 512
            )
            cnn_out_channels = 512
        elif backbone == 'custom_cnn':
            # 一个简单的CNN，适合快速实验
            self.cnn = nn.Sequential(
                nn.Conv2d(1, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(True),
                nn.MaxPool2d(2, 2),          # 64x H/2 x W/2
                nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(True),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(True),
                nn.Conv2d(256, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(True),
                nn.MaxPool2d((2, 1), (2, 1)), # 高度减半，宽度不变
                nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(True),
                nn.MaxPool2d((2, 1), (2, 1))
            )
            cnn_out_channels = 512
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # 用 dummy 输入计算 CNN 输出高度，确保 LSTM 输入维度正确
        with torch.no_grad():
            dummy = torch.zeros(1, 3, img_height, 128)  # 3通道输入
            cnn_out = self.cnn(dummy)
            cnn_out_height = cnn_out.shape[2]  # H' 维度

        lstm_input_size = cnn_out_channels * cnn_out_height
        self.rnn = nn.LSTM(lstm_input_size, hidden_size, num_layers,
                           batch_first=True, bidirectional=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # 双向拼接

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 3, H, W) 伪RGB图（灰度复制3通道）
        返回: (B, T, num_classes) 对数softmax概率
        """
        # CNN特征提取
        feat = self.cnn(x)                     # (B, C, H', W')
        b, c, h, w = feat.size()
        # 将高度与通道维度合并，变为宽度方向上的序列
        feat = feat.permute(0, 3, 1, 2)        # (B, W', C, H')
        feat = feat.reshape(b, w, c * h)       # (B, W', C*H)
        # RNN序列建模
        out, _ = self.rnn(feat)                # (B, W', hidden_size*2)
        logits = self.fc(out)                  # (B, W', num_classes)
        return F.log_softmax(logits, dim=-1)