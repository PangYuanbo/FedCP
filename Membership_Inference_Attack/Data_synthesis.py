import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random


# ============ 1. 定义辅助函数 ============

def rand_record_mnist():
    """
    随机初始化一张“伪”MNIST图像，返回 shape = (1, 28, 28) 的张量，取值在 [0,1]。
    注意输出是 (1,28,28)，以便和后续 PyTorch 模型的输入格式对齐。
    """
    return torch.rand(1, 28, 28)


def randomize_k_features(x, k=50):
    """
    在给定图像 x (shape=[1,28,28]) 中随机挑选 k 个像素，重新赋为 [0,1] 的新随机值。
    x 会被就地修改并返回。
    """
    # 先打平(784维)，随机选 k 个索引
    x_flat = x.view(-1)
    indices = np.random.choice(len(x_flat), size=k, replace=False)
    for idx in indices:
        x_flat[idx] = random.random()
    return x


def dummy_target_model(x):
    """
    演示用的“假”目标模型 (dummy)，
    x: shape = [batch_size, 1, 28, 28]
    返回 shape = [batch_size, 10] 的预测分布（每个样本对应10个类别的概率）。
    这里仅用随机生成来模拟输出分布。
    在实际应用中，请替换为你真实的 target model。
    """
    batch_size = x.size(0)
    # 随机生成 [batch_size, 10] 的概率向量
    out = torch.rand(batch_size, 10)
    out = out / out.sum(dim=1, keepdim=True)  # 归一化到 [0,1]
    return out


def synthesize(
        target_model,  # 目标模型，输入 (N,1,28,28) -> 输出(N,10)
        c,  # 目标类别 (0~9)
        k_max=50,
        k_min=5,
        conf_min=0.9,
        rej_max=30,
        iter_max=200
):
    """
    使用目标模型 target_model，为指定类别 c 合成一张“MNIST风格”样本。
    返回值: 若合成成功，返回 shape=[1,28,28] 的张量；否则返回 None。
    """
    # 1. 随机初始化 x
    x = rand_record_mnist()  # shape = (1,28,28)

    # 当前最佳样本及其置信度
    y_c_star = 0.0
    x_star = x.clone()

    # 连续拒绝计数
    j = 0
    k = k_max

    for iteration in range(iter_max):
        # 2. 查询目标模型：f_target(x)，输出 shape=[1,10]
        with torch.no_grad():
            y = target_model(x)  # shape=[1,10]

        # 当前目标类别置信度
        y_c = y[0, c].item()

        # 3. 若 y_c >= 历史最佳，则接受并尝试采样
        if y_c >= y_c_star:
            # 检查是否超过置信度阈值，并且 argmax 为 c
            pred_class = torch.argmax(y, dim=1).item()
            if y_c > conf_min and pred_class == c:
                # 以 y_c 的概率“采样成功”
                if random.random() < y_c:
                    return x  # shape = [1,28,28]

            # 更新“最佳”
            x_star = x.clone()
            y_c_star = y_c
            # 拒绝计数清零
            j = 0
        else:
            # 否则计一次拒绝
            j += 1
            if j > rej_max:
                # 缩减 k
                k = max(k_min, k // 2)
                j = 0

        # 4. 基于当前 x_star，再随机修改 k 个像素
        x = x_star.clone()
        x = randomize_k_features(x, k)

    # 若 iter_max 次迭代还没成功，返回 None
    return None


# ============ 2. 批量合成 5000 条数据并保存 ============

def generate_synthetic_dataset(
        target_model,
        total_samples=5000,
        per_class=500,
        output_path='synthetic_mnist.pt'
):
    """
    - 为 10 个类别 (0~9) 各合成 per_class 条，总计 total_samples 条数据。
    - 拆分为 train(4000) + test(1000)，并保存到 output_path 中。
    """
    all_data = []
    all_labels = []

    # 如果要保证合计数 == total_samples，假设 total_samples=5000, per_class=500，
    # 那么 10*500=5000。若要改动，请确保对应数量匹配。

    for c in range(10):  # 对每个类别 c
        print(f"=== 开始合成类别 {c} 的数据 ===")
        count = 0
        while count < per_class:
            x_synth = synthesize(target_model, c,
                                 k_max=50, k_min=5,
                                 conf_min=0.9, rej_max=30, iter_max=200)
            if x_synth is not None:
                all_data.append(x_synth.squeeze(0))  # 存入 shape=[28,28]
                all_labels.append(c)
                count += 1
                if count % 50 == 0:
                    print(f"类别 {c}: 已合成 {count}/{per_class} 条")

    # 拼接成 Tensor
    data_tensor = torch.stack(all_data, dim=0)  # shape=[5000, 28, 28]
    labels_tensor = torch.tensor(all_labels)  # shape=[5000]

    # 随机打乱
    perm = torch.randperm(len(data_tensor))
    data_tensor = data_tensor[perm]
    labels_tensor = labels_tensor[perm]

    # 拆分 train / test
    train_data = data_tensor[:4000]
    train_labels = labels_tensor[:4000]
    test_data = data_tensor[4000:]
    test_labels = labels_tensor[4000:]

    # 保存到文件
    torch.save({
        'train_data': train_data,
        'train_labels': train_labels,
        'test_data': test_data,
        'test_labels': test_labels
    }, output_path)

    print(f"合成完毕，共生成 {len(data_tensor)} 条数据，已保存到 {output_path}。")
    print(f"其中训练集 {len(train_data)} 条，测试集 {len(test_data)} 条。")


if __name__ == '__main__':
    # 用 dummy_target_model 演示：你应替换为你真正的 target_model
    generate_synthetic_dataset(
        target_model=dummy_target_model,
        total_samples=5000,  # 总数=5000
        per_class=500,  # 每个类别 500
        output_path='synthetic_mnist.pt'
    )