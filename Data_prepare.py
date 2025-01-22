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
alpha = 1
num_clients = 20


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


def normalize_images(images):
    images = images.astype(np.float32) / 255.0
    for c in range(3):  # 每个通道分别归一化
        images[..., c] = (images[..., c] - CIFAR10_MEAN[c]) / CIFAR10_STD[c]
    return images


def gen_class_proportions_for_clients(num_classes, num_clients, alpha):
    """
    使用 Dirichlet 为每个类别生成对各个客户端的分配比例。
    返回一个 shape=(num_classes, num_clients) 的数组 proportions[c, k] 表示
    类 c 分配到客户端 k 的比例（所有 k 之和为 1）。
    """
    # 对每个类别 c，都采样一次 Dirichlet(num_clients, alpha)
    # 这样我们得到 num_classes 行, each row is p_c = (p_c1, p_c2, ... p_cK)
    proportions = np.zeros((num_classes, num_clients), dtype=np.float32)
    for c in range(num_classes):
        # 采样 Dirichlet
        p = np.random.dirichlet([alpha] * num_clients)
        proportions[c] = p
    return proportions


def split_data_with_given_proportions(data, labels, proportions):
    """
    根据已有的 proportions 数组 (num_classes, num_clients),
    把 data/labels 分割到各个客户端。

    data: shape=(N, 32, 32, 3)
    labels: shape=(N,)
    proportions: shape=(num_classes, num_clients)
      proportions[c, k] 表示 类 c 分配到客户端 k 的比例

    返回: client_data (字典)
        client_data[k] = {'data': [...], 'labels': [...]}
    """
    num_clients = proportions.shape[1]
    client_data = {i: {'data': [], 'labels': []} for i in range(num_clients)}
    unique_classes = np.unique(labels)

    for c in unique_classes:
        idx_c = np.where(labels == c)[0]
        np.random.shuffle(idx_c)
        n_c = len(idx_c)
        # 计算该类别在各客户端上要分多少样本(四舍五入/向下取整都行)
        # 先根据 proportions[c, k] * n_c 得到每个客户端分到多少该类的样本
        # 再取整以保证样本总数一致
        counts = np.floor(proportions[c] * n_c).astype(int)

        # 有可能出现加和 < n_c 的情况（由于floor）
        # 我们可以将余数再分配给若干客户端
        sum_counts = np.sum(counts)
        remainder = n_c - sum_counts
        # 随机给 remainder 个客户端 +1
        if remainder > 0:
            # 选 remainder 个客户端，让其 counts[k] += 1
            inc_indices = np.random.choice(num_clients, remainder, replace=False)
            for inc_k in inc_indices:
                counts[inc_k] += 1

        start = 0
        for k in range(num_clients):
            num_k = counts[k]
            if num_k > 0:
                client_data[k]['data'].extend(data[idx_c[start:start + num_k]])
                client_data[k]['labels'].extend(labels[idx_c[start:start + num_k]])
                start += num_k

    return client_data

def shuffle_client_data(client_data):
    """
    对每个客户端的数据进行随机打乱。
    """
    for client_id, content in client_data.items():
        if len(content['data']) == 0:  # 检查数据是否为空
            continue
        # 将数据和标签一起打乱
        indices = np.arange(len(content['data']))
        np.random.shuffle(indices)
        client_data[client_id]['data'] = np.array(content['data'])[indices].tolist()
        client_data[client_id]['labels'] = np.array(content['labels'])[indices].tolist()


def save_client_data(client_data, output_dir, prefix):
    """
    保存客户端数据到文件，数据已归一化并转置为 (N, C, H, W) 格式。
    """
    os.makedirs(output_dir, exist_ok=True)
    for client_id, content in client_data.items():
        if len(content['data']) == 0:  # 检查数据是否为空
            print(f"Skipping client {client_id} due to no data.")
            continue

        X = normalize_images(np.array(content['data'], dtype=np.float32))  # 归一化
        X = X.transpose(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        Y = np.array(content['labels'], dtype=np.int64)

        client_path = os.path.join(output_dir, f"{prefix}{client_id}_.npz")
        np.savez(client_path, data={'x': X, 'y': Y})


if __name__ == "__main__":
    output_dir = 'cifar10_data'
    output_partition_dir = 'partitioned_cifar10'

    # 1. 加载 CIFAR-10 数据
    train_data, train_labels, test_data, test_labels = load_cifar10(output_dir)
    num_classes = len(np.unique(train_labels))  # 对 CIFAR-10 而言是10

    # 2. 生成 Dirichlet 分配比例
    proportions = gen_class_proportions_for_clients(num_classes, num_clients, alpha)

    # 3. 切分训练和测试数据
    train_client_data = split_data_with_given_proportions(train_data, train_labels, proportions)
    test_client_data = split_data_with_given_proportions(test_data, test_labels, proportions)

    # 4. 对每个客户端的数据进行 shuffle
    shuffle_client_data(train_client_data)
    shuffle_client_data(test_client_data)

    # 5. 保存训练和测试数据
    save_client_data(train_client_data, output_partition_dir, prefix='train')
    save_client_data(test_client_data, output_partition_dir, prefix='test')

    print(f"数据已保存到 {output_partition_dir} 文件夹下，并已随机打乱！")
