import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 创建保存数据的文件夹
dataset_folder = "dataset"
os.makedirs(dataset_folder, exist_ok=True)

# 定义数据转换
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # 归一化
])

# 下载和加载训练数据
train_dataset = datasets.CIFAR100(
    root=dataset_folder,
    train=True,
    download=True,
    transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# 下载和加载测试数据
test_dataset = datasets.CIFAR100(
    root=dataset_folder,
    train=False,
    download=True,
    transform=transform
)

test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

print(f"Loaded {len(train_dataset)} training images and {len(test_dataset)} testing images.")
