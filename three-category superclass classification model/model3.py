import torch
import torch.nn as nn
from torchvision import models


class ImageClassifier(nn.Module):
    """
    基于预训练 ResNet18 的三分类器（SS / CMS / AIS）
    小数据集最优配置：
    - ResNet18 全层可训练（不冻结，但主干用小 lr）
    - 分类头：512 -> 256 -> 3，单 Dropout
    """

    def __init__(self, input_channels=3, dropout_rate=0.5):
        super().__init__()

        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        self.conv1   = backbone.conv1
        self.bn1     = backbone.bn1
        self.relu    = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1  = backbone.layer1
        self.layer2  = backbone.layer2
        self.layer3  = backbone.layer3
        self.layer4  = backbone.layer4
        self.avgpool = backbone.avgpool

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 3)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x

    def get_param_count(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"总参数量": f"{total:,}", "可训练参数量": f"{trainable:,}"}


if __name__ == "__main__":
    model = ImageClassifier(dropout_rate=0.5)
    x = torch.randn(4, 3, 224, 224)
    print(model(x).shape)
    print(model.get_param_count())

