import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
import requests
import zipfile
from tqdm import tqdm
from PIL import Image  # 用于读取图片像素

# Tiny-ImageNet 常用均值和标准差 (与 ImageNet 略有差别，以下仅供参考)
TINY_IMAGENET_MEAN = np.array([0.4802, 0.4481, 0.3975], dtype=np.float32)
TINY_IMAGENET_STD  = np.array([0.2302, 0.2265, 0.2262], dtype=np.float32)

# 设置德拉克雷分布的 alpha 值和客户端数量
alpha = 0.05
num_clients = 20

# 下载 Tiny ImageNet 数据集
def download_and_extract_tiny_imagenet(output_dir):
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    filename = os.path.join(output_dir, "tiny-imagenet-200.zip")
    extracted_dir = os.path.join(output_dir, "tiny-imagenet-200")

    if not os.path.exists(extracted_dir):
        os.makedirs(output_dir, exist_ok=True)
        print("Downloading Tiny ImageNet dataset...")
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(filename, "wb") as f:
                for chunk in tqdm(r.iter_content(chunk_size=8192), desc="Downloading"):
                    f.write(chunk)

        print("Extracting Tiny ImageNet dataset...")
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
    else:
        print("Tiny ImageNet dataset already exists.")

    return extracted_dir


# 加载 Tiny ImageNet 数据
def load_tiny_imagenet(data_dir):
    def load_images_labels(split_dir):
        data = []
        labels = []
        label_map = {}
        class_id = 0

        # 训练集
        if split_dir.endswith("train"):
            for class_folder in os.listdir(split_dir):
                class_path = os.path.join(split_dir, class_folder)
                if os.path.isdir(class_path):
                    if class_folder not in label_map:
                        label_map[class_folder] = class_id
                        class_id += 1
                    for img_file in os.listdir(os.path.join(class_path, "images")):
                        img_path = os.path.join(class_path, "images", img_file)
                        data.append(img_path)
                        labels.append(label_map[class_folder])

        # 验证集
        elif split_dir.endswith("val"):
            val_annotations_path = os.path.join(split_dir, "val_annotations.txt")
            with open(val_annotations_path, "r") as f:
                annotations = f.readlines()
            for line in annotations:
                img_name, class_name = line.split("\t")[:2]
                if class_name not in label_map:
                    label_map[class_name] = class_id
                    class_id += 1
                img_path = os.path.join(split_dir, "images", img_name)
                data.append(img_path)
                labels.append(label_map[class_name])

        return np.array(data), np.array(labels)

    train_data, train_labels = load_images_labels(os.path.join(data_dir, "train"))
    val_data, val_labels = load_images_labels(os.path.join(data_dir, "val"))
    return train_data, train_labels, val_data, val_labels


# 根据德拉克雷分布分配数据
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

    # 确保每个客户端至少包含两个类别
    for client_id, content in client_data.items():
        unique_labels = np.unique(content['labels'])
        if len(unique_labels) < 2:
            print(f"Warning: Client {client_id} has only one class: {unique_labels}. Adding extra data.")
            extra_idx = np.random.choice(len(labels), size=10, replace=False)
            client_data[client_id]['data'].extend(data[extra_idx])
            client_data[client_id]['labels'].extend(labels[extra_idx])

    return client_data

# 打印每个客户端的类别数量
def print_client_class_distribution(client_data, split_name):
    print(f"\n===== {split_name} Client Class Distribution =====")
    for client_id, content in client_data.items():
        unique_classes = np.unique(content['labels'])
        print(f"Client {client_id}: {len(unique_classes)} unique classes")


def save_client_data(client_data, output_dir, prefix):
    """
    保存客户端数据时：
      1. 将图像转为 float32 / [0,1]
      2. 按 Tiny-ImageNet 的统计量做归一化 (可选)
      3. 调整通道顺序为 [C, H, W]
      4. 最终保存到 .npz
    """
    os.makedirs(output_dir, exist_ok=True)

    for client_id, content in client_data.items():
        X = []
        for image_path in content['data']:
            # 1) 读取图像为 RGB
            img = Image.open(image_path).convert('RGB')
            # 2) 转 float32, 归一化到 [0,1]
            img_array = np.array(img, dtype=np.float32) / 255.0  # shape: (64, 64, 3)

            # 3) 按通道减均值除标准差
            #    mean/std shape => (3,)，需要广播到 (64,64,3)，可手动循环或 reshape
            for c in range(3):
                img_array[..., c] = (img_array[..., c] - TINY_IMAGENET_MEAN[c]) / TINY_IMAGENET_STD[c]

            # 4) (H,W,C) -> (C,H,W)
            img_array = np.transpose(img_array, (2, 0, 1))  # shape: (3,64,64)

            X.append(img_array)

        # 转成 (N,3,64,64)
        X = np.array(X, dtype=np.float32)
        Y = np.array(content['labels'], dtype=np.int64)

        client_path = os.path.join(output_dir, f"{prefix}{client_id}_.npz")
        np.savez(client_path, data={'x': X, 'y': Y})


def plot_bubble_distribution(client_data, num_classes, title):
    bubble_sizes = np.zeros((len(client_data), num_classes))
    for client_id, content in client_data.items():
        labels = np.array(content['labels'])
        for c in range(num_classes):
            bubble_sizes[client_id, c] = np.sum(labels == c)

    plt.figure(figsize=(12, 8))
    for client_id in range(len(client_data)):
        plt.scatter(
            [client_id] * num_classes,
            range(num_classes),
            s=bubble_sizes[client_id] * 5,  # 调整气泡大小
            alpha=0.6,
            color='red'
        )

    plt.xlabel("Client IDs")
    plt.ylabel("Class IDs")
    plt.title(title)
    plt.grid(True)
    plt.show()


# 主程序
if __name__ == "__main__":
    output_dir = 'tiny_imagenet_data'          # 数据集保存路径
    output_partition_dir = 'partitioned_tiny_imagenet'  # 数据划分保存路径

    # 1. 下载并解压 Tiny ImageNet
    data_dir = download_and_extract_tiny_imagenet(output_dir)

    # 2. 加载数据路径 + 标签
    train_data, train_labels, val_data, val_labels = load_tiny_imagenet(data_dir)

    # 3. 分割训练和测试数据 (Dirichlet)
    train_client_data = split_data_by_dirichlet(train_data, train_labels, num_clients, alpha)
    val_client_data   = split_data_by_dirichlet(val_data,  val_labels,  num_clients, alpha)

    # 4. 打印训练/验证集上每个客户端的类别数量
    print_client_class_distribution(train_client_data, split_name="Train")
    print_client_class_distribution(val_client_data,   split_name="Validation")

    print("===== Train Set Distribution =====")
    for client_id, content in train_client_data.items():
        print(f"Client {client_id}: {len(content['data'])} images")

    print("\n===== Validation Set Distribution =====")
    for client_id, content in val_client_data.items():
        print(f"Client {client_id}: {len(content['data'])} images")

    # 5. 保存训练和测试数据到 npz (此时已做归一化 + [C,H,W] 通道变换)
    save_client_data(train_client_data, output_partition_dir, prefix='train')
    save_client_data(val_client_data,   output_partition_dir, prefix='test')

    # 6. 绘制气泡图
    plot_bubble_distribution(train_client_data, num_classes=200, title=r"Train Set Distribution ($\alpha = 0.05$)")
    plot_bubble_distribution(val_client_data,   num_classes=200, title=r"Validation Set Distribution ($\alpha = 0.05$)")

    print(f"数据已保存到 {output_partition_dir} 文件夹下！")
