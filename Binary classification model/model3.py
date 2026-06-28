import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """通道注意力机制（简化版，适配小数据集）"""

    def __init__(self, in_channels, reduction_ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # 全局最大池化

        # 轻量级MLP，防止通道数过小导致性能下降
        hidden_channels = max(in_channels // reduction_ratio, 8)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 1, bias=False),
            nn.ReLU(inplace=True),  # 原地激活，节省内存
            nn.Conv2d(hidden_channels, in_channels, 1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 融合平均池化（80%）和最大池化（20%），侧重全局特征
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        out = 0.8 * avg_out + 0.2 * max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """空间注意力机制（小卷积核适配小特征图）"""

    def __init__(self, kernel_size=5):
        super(SpatialAttention, self).__init__()
        padding = kernel_size // 2  # 保持尺寸不变
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 通道维度的平均和最大池化，提取空间特征
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    """卷积块注意力模块（轻量级版本）"""

    def __init__(self, in_channels, reduction_ratio=8, kernel_size=5):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        # 先通道注意力（筛选重要通道），再空间注意力（定位重要区域）
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x


class ImageClassifier(nn.Module):
    """带CBAM注意力的轻量级分类器（适配210×140、3通道、2分类）"""

    def __init__(self,  input_channels=3, dropout_rate=0.35):
        super().__init__()

        # 卷积块1：3→16通道，210×140 → 105×70（2倍下采样）
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

        # 卷积块2：16→32通道，105×70 → 52×35（2倍下采样）
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

        # 卷积块3：32→64→96通道，52×35 → 26×17（2倍下采样）
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 96, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

        # CBAM注意力模块（仅在96通道层添加，平衡性能和计算量）
        self.cbam = CBAM(96, reduction_ratio=8, kernel_size=5)

        # 卷积块4：96→32通道，自适应池化到2×2（统一尺寸）
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(96, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),  # 缓解梯度消失
            nn.AdaptiveMaxPool2d((2, 2))
        )

        # 全连接层（极轻量，适配小数据集）
        self.fc_layers = nn.Sequential(
            nn.Flatten(),  # 替代x.view，更直观
            nn.Linear(32 * 2 * 2, 32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout_rate + 0.3),  # 0.65 dropout，强化过拟合抑制
            nn.Linear(32, 12),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),  # 0.35 dropout
            nn.Linear(12, 1)  # 2分类输出
        )

    def forward(self, x):
        # 前向传播
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.cbam(x)  # 应用注意力机制
        x = self.conv_block4(x)
        x = self.fc_layers(x)
        return x

    def get_param_count(self):
        """计算模型参数量，返回可读格式"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "总参数量": f"{total_params:,}",
            "可训练参数量": f"{trainable_params:,}"
        }


if __name__ == "__main__":
    model = ImageClassifier(input_channels=3, dropout_rate=0.35)

    input_tensor = torch.randn(8, 3, 210, 140)
    logits = model(input_tensor)

    print(f"输入形状: {input_tensor.shape}")
    print(f"输出形状: {logits.shape}")  # (8, 1)

    probs = torch.sigmoid(logits)
    print(f"Sigmoid后示例: {probs[:5].view(-1)}")

    param_info = model.get_param_count()
    print(f"模型总参数量: {param_info['总参数量']}")

    # 验证各层输出维度（调试用）
    with torch.no_grad():
        x = model.conv_block1(input_tensor)
        print(f"conv_block1输出: {x.shape}")  # (8,16,105,70)
        x = model.conv_block2(x)
        print(f"conv_block2输出: {x.shape}")  # (8,32,52,35)
        x = model.conv_block3(x)
        print(f"conv_block3输出: {x.shape}")  # (8,96,26,17)
        x = model.cbam(x)
        print(f"CBAM输出: {x.shape}")  # (8,96,26,17)
        x = model.conv_block4(x)
        print(f"conv_block4输出: {x.shape}")  # (8,32,2,2)