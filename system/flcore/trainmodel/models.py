import torch
import torch.nn as nn

from torchvision import models
batch_size = 16

class LocalModel(nn.Module):
    def __init__(self, feature_extractor, head):
        super(LocalModel, self).__init__()

        self.feature_extractor = feature_extractor
        self.head = head
        
    def forward(self, x, feat=False):
        out = self.feature_extractor(x)
        if feat:
            return out
        else:
            out = self.head(out)
            return out


# https://github.com/FengHZ/KD3A/blob/master/model/amazon.py
class AmazonMLP(nn.Module):
    def __init__(self):
        super(AmazonMLP, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(5000, 1000), 
            nn.ReLU(), 
            nn.Linear(1000, 500), 
            nn.ReLU(),
            nn.Linear(500, 100), 
            nn.ReLU()
        )
        self.fc = nn.Linear(100, 2)

    def forward(self, x):
        out = self.encoder(x)
        out = self.fc(out)
        return out


class FedAvgCNN(nn.Module):
    def __init__(self, in_features=1, num_classes=10, dim=1024, dim1=512):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_features,
                        32,
                        kernel_size=5,
                        padding=0,
                        stride=1,
                        bias=True),
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32,
                        64,
                        kernel_size=5,
                        padding=0,
                        stride=1,
                        bias=True),
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.fc1 = nn.Sequential(
            nn.Linear(dim, dim1), 
            nn.ReLU(inplace=True)
        )
        self.fc = nn.Linear(dim1, num_classes)

    def forward(self, x):
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("NaN/Inf after conv0!")
        out = self.conv1(x)
        if torch.isnan(out).any() or torch.isinf(out).any():
            print("NaN/Inf after conv1!")
        out = self.conv2(out)
        if torch.isnan(out).any() or torch.isinf(out).any():
            print("NaN/Inf after conv2!")
        out = torch.flatten(out, 1)
        if torch.isnan(out).any() or torch.isinf(out).any():
            print("NaN/Inf after conv3!")
        out = self.fc1(out)
        if torch.isnan(out).any() or torch.isinf(out).any():
            print("NaN/Inf after conv4!")
        out = self.fc(out)
        if torch.isnan(out).any() or torch.isinf(out).any():
            print("NaN/Inf after conv5!")
        return out


class fastText(nn.Module):
    def __init__(self, hidden_dim, padding_idx=0, vocab_size=98635, num_classes=10):
        super(fastText, self).__init__()
        
        # Embedding Layer
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx)
        
        # Hidden Layer
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        
        # Output Layer
        self.fc = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        text, text_lengths = x

        embedded_sent = self.embedding(text)
        h = self.fc1(embedded_sent.mean(1))
        z = self.fc(h)
        out = z

        return out

class FedAvgResNet18(nn.Module):
    """
    与 FedAvgCNN 类似的写法，使用 ResNet18 作为主干网络。
    这里 in_features 表示输入通道数，如 3 表示 RGB 彩色图像。
    """
    def __init__(self, in_features=3, num_classes=200, dim=1600):
        super(FedAvgResNet18, self).__init__()

        # 1. 加载 torchvision 中的 resnet18 (无预训练)
        self.model = models.resnet18(pretrained=False)

        # 2. 若 in_features != 3，则需要替换第一层的卷积核
        #    TinyImageNet 通常是 RGB, 所以 in_features=3，若你需要灰度图 (1 通道)，就需替换
        if in_features != 3:
            self.model.conv1 = nn.Conv2d(
                in_features, 64, kernel_size=7, stride=2, padding=3, bias=False
            )

        # 3. 替换最后一层，全连接层输出改为 num_classes（TinyImageNet 默认 200 类）
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

        # 如果你希望用到 dim 参数做额外处理（比如 Flatten 后特征维度 = 1600），可在此添加自定义层。
        # 这里先不额外用 dim，仅保留原版 ResNet18 结构
        # 例如:
        # self.linear_out = nn.Linear(dim, num_classes)  # 你可以酌情添加

    def forward(self, x):
        return self.model(x)
