import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image

# Tiny-ImageNet 的均值和标准差（可根据自己统计值进行调整，这里仅作示例）
TINY_IMAGENET_MEAN = np.array([0.4802, 0.4481, 0.3975], dtype=np.float32)
TINY_IMAGENET_STD  = np.array([0.2302, 0.2265, 0.2262], dtype=np.float32)

# 设置德拉克雷分布的 alpha 值和客户端数量
alpha = 0.5
num_clients = 20

##############################################################################
#                        1.  加载 Tiny-ImageNet 数据                          #
##############################################################################
def load_tiny_imagenet(data_dir):
    """
    假设 data_dir 下存在 tiny-imagenet-200 文件夹，
    内含 train/val 等子目录结构。
    """
    tiny_imagenet_root = os.path.join(data_dir, "tiny-imagenet-200")
    if not os.path.exists(tiny_imagenet_root):
        raise FileNotFoundError(f"未找到 {tiny_imagenet_root} 文件夹，请先下载解压 Tiny-ImageNet.")

    def load_images_labels(split_dir, is_train=True):
        data = []
        labels = []
        label_map = {}
        class_id = 0

        if is_train:
            # 训练集的目录结构：train/<class>/images/*.JPEG
            for class_folder in os.listdir(split_dir):
                class_path = os.path.join(split_dir, class_folder)
                if not os.path.isdir(class_path):
                    continue
                if class_folder not in label_map:
                    label_map[class_folder] = class_id
                    class_id += 1

                img_dir = os.path.join(class_path, "images")
                for img_file in os.listdir(img_dir):
                    img_path = os.path.join(img_dir, img_file)
                    data.append(img_path)
                    labels.append(label_map[class_folder])
        else:
            # 验证集的目录结构：val/images/*.JPEG  + val_annotations.txt
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

    # 分别加载 train 和 val
    train_dir = os.path.join(tiny_imagenet_root, "train")
    val_dir = os.path.join(tiny_imagenet_root, "val")

    train_data, train_labels = load_images_labels(train_dir, is_train=True)
    val_data, val_labels     = load_images_labels(val_dir,  is_train=False)

    return train_data, train_labels, val_data, val_labels


##############################################################################
#             2.  归一化图片数据 + 生成 Dirichlet 拆分函数等                   #
##############################################################################
def normalize_images(images):
    """
    images: (N, 64, 64, 3)，这里与 CIFAR-10 不同，大小是 64x64。
    先将 [0,255] -> [0,1]，再减去均值，除以标准差。
    """
    images = images.astype(np.float32) / 255.0
    for c in range(3):
        images[..., c] = (images[..., c] - TINY_IMAGENET_MEAN[c]) / TINY_IMAGENET_STD[c]
    return images

def gen_class_proportions_for_clients(num_classes, num_clients, alpha):
    """
    使用 Dirichlet 为每个类别生成对各个客户端的分配比例。
    返回 shape=(num_classes, num_clients) 的数组 p[c, k] 表示类 c 分配到客户端 k 的比例。
    """
    proportions = np.zeros((num_classes, num_clients), dtype=np.float32)
    for c in range(num_classes):
        p = np.random.dirichlet([alpha] * num_clients)
        proportions[c] = p
    return proportions

def split_data_with_given_proportions(data_paths, labels, proportions):
    """
    data_paths: 图片路径列表 (N,)
    labels: 标签数组 (N,)
    proportions: shape=(num_classes, num_clients)
    返回 client_data: {k: {'data': [...], 'labels': [...]}}
    """
    num_clients = proportions.shape[1]
    client_data = {i: {'data': [], 'labels': []} for i in range(num_clients)}
    unique_classes = np.unique(labels)

    for c in unique_classes:
        idx_c = np.where(labels == c)[0]
        np.random.shuffle(idx_c)
        n_c = len(idx_c)

        # 对应类别 c 在每个客户端应分到的样本数
        counts = np.floor(proportions[c] * n_c).astype(int)

        # 处理 floor 后的余数
        sum_counts = np.sum(counts)
        remainder = n_c - sum_counts
        if remainder > 0:
            inc_indices = np.random.choice(num_clients, remainder, replace=False)
            for inc_k in inc_indices:
                counts[inc_k] += 1

        start = 0
        for k in range(num_clients):
            num_k = counts[k]
            if num_k > 0:
                client_data[k]['data'].extend(data_paths[idx_c[start:start + num_k]])
                client_data[k]['labels'].extend(labels[idx_c[start:start + num_k]])
                start += num_k

    return client_data

def shuffle_client_data(client_data):
    """对每个客户端的数据进行随机打乱。"""
    for client_id, content in client_data.items():
        data_len = len(content['data'])
        if data_len == 0:
            continue
        indices = np.arange(data_len)
        np.random.shuffle(indices)
        client_data[client_id]['data']   = np.array(content['data'])[indices].tolist()
        client_data[client_id]['labels'] = np.array(content['labels'])[indices].tolist()


##############################################################################
#                      3.  读取图片、保存为 .npz 文件                         #
##############################################################################
def save_client_data(client_data, output_dir, prefix):
    """
    1) 根据图片路径读取图像、归一化
    2) 存为 .npz (内含 data['x'], data['y'])
       其中 x shape: (N, 3, 64, 64)
           y shape: (N,)
    """
    os.makedirs(output_dir, exist_ok=True)
    for client_id, content in client_data.items():
        if len(content['data']) == 0:
            print(f"Skipping client {client_id} due to no data.")
            continue

        X_list = []
        for img_path in content['data']:
            # 打开图片并转换为 RGB
            img = Image.open(img_path).convert('RGB')
            img_array = np.array(img, dtype=np.float32)  # (64,64,3)
            X_list.append(img_array)

        # (N, 64, 64, 3)
        X = np.array(X_list, dtype=np.float32)
        # 归一化
        X = normalize_images(X)
        # (N, 3, 64, 64)
        X = X.transpose(0, 3, 1, 2)

        Y = np.array(content['labels'], dtype=np.int64)

        client_path = os.path.join(output_dir, f"{prefix}{client_id}_.npz")
        np.savez(client_path, data={'x': X, 'y': Y})


##############################################################################
#                               4. 主流程                                     #
##############################################################################
if __name__ == "__main__":
    # 目录配置
    output_dir = 'tiny_imagenet_data'         # 下载或解压后保存 Tiny-ImageNet 的根目录
    output_partition_dir = 'partitioned_tiny_imagenet'  # 数据划分保存路径

    # 1. 加载 Tiny-ImageNet（train + val）
    train_data_paths, train_labels, val_data_paths, val_labels = load_tiny_imagenet(output_dir)

    # Tiny-ImageNet 原本有 200 个类别
    num_classes = len(np.unique(train_labels))

    # 2. 生成 Dirichlet 分配比例
    proportions = gen_class_proportions_for_clients(num_classes, num_clients, alpha)

    # 3. 对 train、val 分别进行拆分
    train_client_data = split_data_with_given_proportions(train_data_paths, train_labels, proportions)
    val_client_data   = split_data_with_given_proportions(val_data_paths,   val_labels,   proportions)

    # 4. 随机打乱各客户端数据
    shuffle_client_data(train_client_data)
    shuffle_client_data(val_client_data)

    # 5. 保存拆分后的数据
    save_client_data(train_client_data, output_partition_dir, prefix='train')
    save_client_data(val_client_data,   output_partition_dir, prefix='val')

    print(f"\n数据已保存到 {output_partition_dir} 文件夹下，并已随机打乱！")
