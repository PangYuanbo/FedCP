import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

# CIFAR-10 常用均值和标准差（3通道）
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

# 设置 Dirichlet 分布的 alpha 值
alpha = 1
# 正常训练的客户端数量
num_clients_normal = 20
# 影子（用于成员攻击）训练的客户端数量
num_clients_shadow = 20

def load_cifar10(data_dir):
    """
    加载 CIFAR-10 数据集，返回：
    train_data, train_labels, test_data, test_labels
    其中：
      - train_data.shape = (50000, 32, 32, 3)
      - train_labels.shape = (50000,)
      - test_data.shape  = (10000, 32, 32, 3)
      - test_labels.shape = (10000,)
    """
    transform = transforms.Compose([
        transforms.ToTensor()  # 这里仅用于确保下载后的数据格式正确，但后续使用原始 .data 属性
    ])
    train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)

    train_data = train_dataset.data  # shape: (50000, 32, 32, 3)
    train_labels = np.array(train_dataset.targets)  # shape: (50000,)

    test_data = test_dataset.data  # shape: (10000, 32, 32, 3)
    test_labels = np.array(test_dataset.targets)  # shape: (10000,)

    return train_data, train_labels, test_data, test_labels

def normalize_images(images):
    """
    对 CIFAR-10 图像进行归一化：
    - 将图像转换到 [0,1]
    - 对每个通道分别减去均值再除以标准差
    images: numpy 数组，形状为 (N, H, W, 3)
    """
    images = images.astype(np.float32) / 255.0
    # 将均值和标准差调整为可广播的形状
    mean = np.array(CIFAR10_MEAN).reshape(1, 1, 1, 3)
    std = np.array(CIFAR10_STD).reshape(1, 1, 1, 3)
    images = (images - mean) / std
    return images

def gen_class_proportions_for_clients(num_classes, num_clients, alpha):
    """
    使用 Dirichlet 为每个类别生成对各个客户端的分配比例。
    返回形状为 (num_classes, num_clients) 的数组 proportions[c, k] 表示
    类 c 分配到客户端 k 的比例（所有 k 之和为 1）。
    """
    proportions = np.zeros((num_classes, num_clients), dtype=np.float32)
    for c in range(num_classes):
        p = np.random.dirichlet([alpha] * num_clients)
        proportions[c] = p
    return proportions

def split_data_with_given_proportions(data, labels, proportions):
    """
    根据已有的 proportions 数组 (num_classes, num_clients),
    把 data/labels 分配到各个客户端。

    data: shape=(N, H, W, 3)
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

        # 计算该类别在各客户端上要分多少样本
        counts = np.floor(proportions[c] * n_c).astype(int)
        sum_counts = np.sum(counts)
        remainder = n_c - sum_counts

        # 将余数再随机分配给若干客户端
        if remainder > 0:
            inc_indices = np.random.choice(num_clients, remainder, replace=False)
            for inc_k in inc_indices:
                counts[inc_k] += 1

        start = 0
        for k in range(num_clients):
            num_k = counts[k]
            if num_k > 0:
                # 分配给客户端 k
                client_data[k]['data'].extend(data[idx_c[start:start + num_k]])
                client_data[k]['labels'].extend(labels[idx_c[start:start + num_k]])
                start += num_k

    return client_data

def shuffle_client_data(client_data):
    """
    对每个客户端的数据进行随机打乱。
    """
    for client_id, content in client_data.items():
        if len(content['data']) == 0:
            continue
        indices = np.arange(len(content['data']))
        np.random.shuffle(indices)
        client_data[client_id]['data'] = np.array(content['data'])[indices].tolist()
        client_data[client_id]['labels'] = np.array(content['labels'])[indices].tolist()

def save_client_data(client_data, output_dir, prefix):
    """
    保存客户端数据到文件，数据已归一化并转换为 (N, C, H, W) 格式（C=3）。
    """
    os.makedirs(output_dir, exist_ok=True)
    for client_id, content in client_data.items():
        if len(content['data']) == 0:
            print(f"Skipping client {client_id} due to no data.")
            continue

        # 转换为 numpy 数组
        X = np.array(content['data'], dtype=np.float32)  # shape: (N, 32, 32, 3)
        Y = np.array(content['labels'], dtype=np.int64)

        # 归一化
        X = normalize_images(X)  # shape: (N, 32, 32, 3)
        # 转换为 (N, 3, 32, 32)
        X = np.transpose(X, (0, 3, 1, 2))

        client_path = os.path.join(output_dir, f"{prefix}{client_id}_.npz")
        np.savez(client_path, data={'x': X, 'y': Y})

def create_per_client_attack_dataset(
        train_clients,
        test_clients,
        output_dir="attack_data_per_client",
        prefix="client"
):
    """
    为每个客户端分别抽取「成员」(member) 与「非成员」(non-member) 攻击数据集。
    - 从 train_clients[cid] 中抽取 M 条作为 member
    - 从 test_clients[cid] 中抽取 M 条作为 non-member
    并将其保存到本地。

    参数：
    ----------
    train_clients, test_clients : dict
        形如 {cid: {'data': [...], 'labels': [...]}} 的数据格式
    output_dir : str
        输出攻击数据目录
    prefix : str
        保存文件名的前缀
    """
    os.makedirs(output_dir, exist_ok=True)

    for cid in train_clients.keys():
        train_data = np.array(train_clients[cid]['data'])
        train_labels = np.array(train_clients[cid]['labels'])

        test_data = np.array(test_clients[cid]['data'])
        test_labels = np.array(test_clients[cid]['labels'])

        # 如果该客户端训练或测试数据为空，跳过
        if len(train_data) == 0 or len(test_data) == 0:
            print(f"[Client {cid}] has no data in train/test, skip.")
            continue

        # 取最小值，以防止测试数据比训练数据少
        M = min(len(train_data), len(test_data), 500)

        # 打乱
        idx_train = np.random.permutation(len(train_data))
        idx_test = np.random.permutation(len(test_data))

        train_data = train_data[idx_train]
        train_labels = train_labels[idx_train]
        test_data = test_data[idx_test]
        test_labels = test_labels[idx_test]

        # 抽取前 M 条
        member_x = train_data[:M]
        member_y = train_labels[:M]
        nonmember_x = test_data[:M]
        nonmember_y = test_labels[:M]

        # 归一化并转换为 (N, 3, 32, 32)
        member_x = normalize_images(member_x)
        member_x = np.transpose(member_x, (0, 3, 1, 2))

        nonmember_x = normalize_images(nonmember_x)
        nonmember_x = np.transpose(nonmember_x, (0, 3, 1, 2))

        # 保存 npz
        member_path = os.path.join(output_dir, f"{prefix}_{cid}_member.npz")
        nonmember_path = os.path.join(output_dir, f"{prefix}_{cid}_nonmember.npz")

        np.savez(member_path, data={'x': member_x, 'y': member_y})
        np.savez(nonmember_path, data={'x': nonmember_x, 'y': nonmember_y})

        print(f"[Client {cid}] Attack data saved: member={member_x.shape}, nonmember={nonmember_x.shape}")

if __name__ == "__main__":
    # ========== 1. 加载并打乱 CIFAR-10 数据 ==========
    data_dir = 'cifar_data'
    train_data, train_labels, test_data, test_labels = load_cifar10(data_dir)

    perm_train = np.random.permutation(len(train_data))
    train_data = train_data[perm_train]
    train_labels = train_labels[perm_train]

    perm_test = np.random.permutation(len(test_data))
    test_data = test_data[perm_test]
    test_labels = test_labels[perm_test]

    # ========== 2. 拆分出 normal / shadow 两部分 (示例) ==========

    train_size = len(train_data)
    test_size = len(test_data)

    half_train = train_size // 2
    half_test = test_size // 2

    # normal 部分 (各取一半)
    normal_train_data = train_data[:half_train]
    normal_train_labels = train_labels[:half_train]
    normal_test_data = test_data[:half_test]
    normal_test_labels = test_labels[:half_test]

    # ------------------------------------------------
    # 关键改动：shadow 部分在原先基础上，再额外加入 10% 的真实子集
    # ------------------------------------------------

    # 先确定 10% 的数量
    ten_percent_train = int(train_size * 0.1)
    ten_percent_test = int(test_size * 0.1)

    # 从训练集前面抽 10% 作为真实子集（示例中直接取前 10%，可自行换成别的方法）
    real_subset_train_data = train_data[:ten_percent_train]
    real_subset_train_labels = train_labels[:ten_percent_train]

    # 从测试集前面抽 10% 作为真实子集
    real_subset_test_data = test_data[:ten_percent_test]
    real_subset_test_labels = test_labels[:ten_percent_test]

    # shadow 部分：原先后半部分 + 10% 的真实子集
    shadow_train_data = np.concatenate([train_data[half_train:], real_subset_train_data], axis=0)
    shadow_train_labels = np.concatenate([train_labels[half_train:], real_subset_train_labels], axis=0)

    shadow_test_data = np.concatenate([test_data[half_test:], real_subset_test_data], axis=0)
    shadow_test_labels = np.concatenate([test_labels[half_test:], real_subset_test_labels], axis=0)
    # ------------------------------------------------

    # ========== 3. 对 normal / shadow 数据使用 Dirichlet 划分并 Shuffle ==========
    num_classes = 10

    # normal clients
    normal_proportions = gen_class_proportions_for_clients(num_classes, num_clients_normal, alpha)
    normal_train_clients = split_data_with_given_proportions(normal_train_data, normal_train_labels, normal_proportions)
    normal_test_clients = split_data_with_given_proportions(normal_test_data, normal_test_labels, normal_proportions)

    # shadow clients
    shadow_proportions = gen_class_proportions_for_clients(num_classes, num_clients_shadow, alpha)
    shadow_train_clients = split_data_with_given_proportions(shadow_train_data, shadow_train_labels, shadow_proportions)
    shadow_test_clients = split_data_with_given_proportions(shadow_test_data, shadow_test_labels, shadow_proportions)

    # Shuffle
    shuffle_client_data(normal_train_clients)
    shuffle_client_data(normal_test_clients)
    shuffle_client_data(shadow_train_clients)
    shuffle_client_data(shadow_test_clients)

    # ========== 4. 保存分好客户端的数据 (可选) ==========
    output_partition_dir_normal = 'partitioned_cifar_normal'
    output_partition_dir_shadow = 'partitioned_cifar_shadow'
    save_client_data(normal_train_clients, output_partition_dir_normal, prefix='train')
    save_client_data(normal_test_clients, output_partition_dir_normal, prefix='test')
    save_client_data(shadow_train_clients, output_partition_dir_shadow, prefix='train')
    save_client_data(shadow_test_clients, output_partition_dir_shadow, prefix='test')

    print("[Info] Finished partitioning normal/shadow data.")

    # ========== 5. 为每个客户端生成攻击数据集 (member vs non-member) ==========
    # attack_out_dir = "attack_data_per_client_normal"
    # create_per_client_attack_dataset(
    #     train_clients=normal_train_clients,
    #     test_clients=normal_test_clients,
    #     output_dir=attack_out_dir,
    #     prefix="normal_client"
    # )
    print("[Info] Per-client attack dataset generation done.")
