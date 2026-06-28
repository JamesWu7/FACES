import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights


class ImageClassifier(nn.Module):
    """
    基于预训练 ResNet50 的迁移学习分类器。
    freeze_layers=0：全部解冻，全量微调
    使用极小 lr 避免破坏预训练特征
    """

    def __init__(self, num_classes: int = 10, dropout_rate: float = 0.5, freeze_layers: int = 0):
        super().__init__()
        self.num_classes = int(num_classes)

        backbone = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

        # freeze_layers=0: 不冻结任何层，全量微调
        if freeze_layers > 0:
            frozen_modules = [backbone.conv1, backbone.bn1, backbone.layer1, backbone.layer2]
            for module in frozen_modules[:2 + freeze_layers]:
                for param in module.parameters():
                    param.requires_grad = False

        self.conv1   = backbone.conv1
        self.bn1     = backbone.bn1
        self.relu    = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1  = backbone.layer1
        self.layer2  = backbone.layer2
        self.layer3  = backbone.layer3
        self.layer4  = backbone.layer4
        self.avgpool = backbone.avgpool

        # 替换分类头
        in_features = backbone.fc.in_features  # 2048
        self.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate * 0.5),
            nn.Linear(512, num_classes),
        )

        nn.init.kaiming_normal_(self.fc[1].weight, mode='fan_out')
        nn.init.zeros_(self.fc[1].bias)
        nn.init.kaiming_normal_(self.fc[4].weight, mode='fan_out')
        nn.init.zeros_(self.fc[4].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

    def get_param_count(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable
        return {
            "总参数量": f"{total:,}",
            "可训练参数量": f"{trainable:,}",
            "冻结参数量": f"{frozen:,}",
        }


if __name__ == "__main__":
    model = ImageClassifier(num_classes=10, dropout_rate=0.5, freeze_layers=0)
    x = torch.randn(4, 3, 224, 224)
    out = model(x)
    info = model.get_param_count()
    print(f"输入: {x.shape} -> 输出: {out.shape}")
    for k, v in info.items():
        print(f"{k}: {v}")
