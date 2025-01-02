import os
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

# CIFAR-10 常用均值和标准差
CIFAR10_MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
CIFAR10_STD = np.array([0.247, 0.243, 0.261], dtype=np.float32)

# 设置德拉克雷分布的 alpha 值和客户端数量
alpha = 0.05
num_clients = 20

# 加载 CIFAR-10 数据集
def load_cifar10(data_dir):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)

    train_data = train_dataset.data  # shape: (50000, 32, 32, 3)
    train_labels = np.array(train_dataset.targets)

    test_data = test_dataset.data  # shape: (10000, 32, 32, 3)
    test_labels = np.array(test_dataset.targets)

    return train_data, train_labels, test_data, test_labels

# 数据归一化
def normalize_images(images):
    images = images.astype(np.float32) / 255.0
    for c in range(3):  # 每个通道分别归一化
        images[..., c] = (images[..., c] - CIFAR10_MEAN[c]) / CIFAR10_STD[c]
    return images

# 根据德拉克雷分布分配数据（不变）
def split_data_by_dirichlet(data, labels, num_clients, alpha):
    client_data = {i: {'data': [], 'labels': []} for i in range(num_clients)}
    num_classes = len(np.unique(labels))

    for c in range(num_classes):
        idx = np.where(labels == c)[0]
        np.random.shuffle(idx)
        proportions = np.random.dirichlet(alpha * np.ones(num_clients))
        proportions = (proportions * len(idx)).astype(int)

        start_idx = 0
        for client_id, proportion in enumerate(proportions):
            client_data[client_id]['data'].extend(data[idx[start_idx:start_idx + proportion]])
            client_data[client_id]['labels'].extend(labels[idx[start_idx:start_idx + proportion]])
            start_idx += proportion

    return client_data

# 保存数据到 .npz 文件（保持不变）
def save_client_data(client_data, output_dir, prefix):
    os.makedirs(output_dir, exist_ok=True)

    for client_id, content in client_data.items():
        X = normalize_images(np.array(content['data'], dtype=np.float32))  # 归一化
        X = X.transpose(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        Y = np.array(content['labels'], dtype=np.int64)

        client_path = os.path.join(output_dir, f"{prefix}{client_id}_.npz")
        np.savez(client_path, data={'x': X, 'y': Y})

# 主程序
if __name__ == "__main__":
    output_dir = 'cifar10_data'
    output_partition_dir = 'partitioned_cifar10'

    # 1. 加载 CIFAR-10 数据
    train_data, train_labels, test_data, test_labels = load_cifar10(output_dir)

    # 2. 分割训练和测试数据 (Dirichlet)
    train_client_data = split_data_by_dirichlet(train_data, train_labels, num_clients, alpha)
    test_client_data = split_data_by_dirichlet(test_data, test_labels, num_clients, alpha)

    # 3. 保存训练和测试数据到 npz
    save_client_data(train_client_data, output_partition_dir, prefix='train')
    save_client_data(test_client_data, output_partition_dir, prefix='test')

    # 4. 打印每个客户端的类别分布
    print(f"数据已保存到 {output_partition_dir} 文件夹下！")
