import copy

import torch
import torch.optim as optim
import os
from torch.utils.data import DataLoader, TensorDataset, Subset
from collections import OrderedDict
import numpy as np
from sklearn.metrics import accuracy_score
from model import *

# ==============================
# 一些超参数 & 配置
# ==============================
BATCH_SIZE = 10
EPOCHS = 800  # 为了演示，训练回合数较小，实际可调大一些
LR = 1e-3

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# ===================================================
# 1. 定义一个简单的 MLP，用于多分类 (target/shadow)
# ===================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FedAvgCNN(in_features=1, num_classes=10, dim=1024).to(DEVICE)
model_head = copy.deepcopy(model.fc)
model.fc = nn.Identity()
model = LocalModel(model, model_head)
in_dim = list(model.head.parameters())[0].shape[1]
cs = ConditionalSelection(in_dim, in_dim).to(DEVICE)

model = Ensemble(
    model=copy.deepcopy(model),
    cs=copy.deepcopy(cs),
    head_g=copy.deepcopy(model.head),  # head is the global head
    feature_extractor=copy.deepcopy(model.feature_extractor)
    # feature_extractor is the global feature_extractor
)
target_model = copy.deepcopy(model)
shadow_model = copy.deepcopy(model)


# ===================================================
# 2. 定义一个简单的 MLP，用于二分类 (attack model)
# ===================================================
class AttackMLP(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=32, output_dim=2):
        super(AttackMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x  # logits


# ===================================================
# 3. 训练/评估函数
# ===================================================
def train_one_epoch(model, dataloader, optimizer, device=DEVICE):
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    for batch_x, batch_y in dataloader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        # 前向
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        # 反向传播
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def evaluate(model, dataloader, device=DEVICE):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            logits = model(batch_x)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == batch_y).sum().item()
            total += len(batch_y)
    return correct / total


# ================================
# 5. 使用合成数据训练影子模型
# ================================
# 这里模拟一个 784 维，10 类的合成数据集

def read_data(is_train=True, is_shadow=True):
    """
    读取 train0_.npz ~ train19_.npz 或 test0_.npz ~ test19_.npz，
    并将它们的 'data'（字典）按键进行拼接，最终返回一个“字典”结构。
    可以像原来那样使用 train_data['x'] 获取合并后的数据。
    """
    # 用于存放合并后数据的容器，格式：{key1: [array1, array2, ...], key2: [...], ...}
    merged_dict = {}

    for i in range(20):
        if is_shadow:
            file_name = f"{'train_shadow' if is_train else 'test_shadow'}{i}_.npz"
        else:
            file_name = f"{'train' if is_train else 'test'}{i}_.npz"
        if not os.path.exists(file_name):
            raise FileNotFoundError(f"File {file_name} not found.")

        # 1. 读取单个文件中的字典
        with open(file_name, 'rb') as f:
            single_data = np.load(f, allow_pickle=True)['data'].tolist()
            # single_data 应该是一个字典，如 {'x': np.array(...), 'y': np.array(...), ...}

        # 2. 将 single_data 的键值，合并到 merged_dict 中
        for key, value in single_data.items():
            # 若在 merged_dict 中还没有这个 key，就初始化为一个空列表
            if key not in merged_dict:
                merged_dict[key] = []
            # 把当前文件的 value 追加进列表
            merged_dict[key].append(value)

    # 3. 把每个 key 对应的列表都做一次拼接 (np.concatenate)，得到单个数组
    final_dict = {}
    for key, list_of_arrays in merged_dict.items():
        # 假设这些数组的形状在第 0 维可以拼接
        # 如果有的键是标量或不需拼接，需自己定制逻辑
        final_dict[key] = np.concatenate(list_of_arrays, axis=0)

    return final_dict


def read_client_data(is_train=True, is_shadow=True):
    # 如果 dataset 中包含其他情况，比如 News / Shakespeare，需要你自己实现
    # 这里只演示默认读取
    if is_train:
        train_data = read_data(is_train=True, is_shadow=is_shadow)
        X_train = torch.Tensor(train_data['x']).type(torch.float32)
        y_train = torch.Tensor(train_data['y']).type(torch.int64)
        train_data = [(x, y) for x, y in zip(X_train, y_train)]
        return TensorDataset(X_train, y_train)
    else:
        test_data = read_data(is_train=False, is_shadow=is_shadow)
        X_test = torch.Tensor(test_data['x']).type(torch.float32)
        y_test = torch.Tensor(test_data['y']).type(torch.int64)
        test_data = [(x, y) for x, y in zip(X_test, y_test)]
        return TensorDataset(X_test, y_test)


shadow_train_dataset = read_client_data(is_train=True, is_shadow=True)
shadow_holdout_dataset = read_client_data(is_train=False, is_shadow=True)
shadow_train_loader = DataLoader(shadow_train_dataset, drop_last=True, batch_size=BATCH_SIZE, shuffle=False)
shadow_holdout_loader = DataLoader(shadow_holdout_dataset, drop_last=True, batch_size=BATCH_SIZE, shuffle=False)

# 影子模型 (与目标模型结构相同)

checkpoint = torch.load('results_mnist-0.1-normal_client0_1000_0.0050.pt', map_location=DEVICE)
partial_state_dict = {}
for k, v in checkpoint.items():
    if "gate" in k:
        partial_state_dict[k] = v
shadow_model.load_state_dict(partial_state_dict, strict=False)

for param in shadow_model.gate.parameters():
    param.requires_grad = False
optimizer_s = optim.SGD(shadow_model.parameters(), lr=LR)

# 训练影子模型
# for epoch in range(EPOCHS):
#     loss_s = train_one_epoch(shadow_model, shadow_train_loader, optimizer_s, device=DEVICE)
#     acc_s_train = evaluate(shadow_model, shadow_train_loader, device=DEVICE)
#     acc_s_holdout = evaluate(shadow_model, shadow_holdout_loader, device=DEVICE)
#     print(f"[Shadow Model] Epoch {epoch+1}/{EPOCHS}, Loss: {loss_s:.4f}, "
#           f"Train Acc: {acc_s_train:.4f}, Shadow-Holdout Acc: {acc_s_holdout:.4f}")
# torch.save(shadow_model.state_dict(), 'shadow_model.pth')


checkpoint2 = torch.load('shadow_model.pth', map_location=DEVICE)
shadow_model.load_state_dict(checkpoint2)


# ================================
# 6. 构造攻击模型 (Attack Model) 的训练数据
# ================================
def filter_by_label(dataset, target_label):
    """
    过滤数据集中的样本，仅保留标签为 target_label 的数据。
    """
    filtered_indices = [i for i, (_, label) in enumerate(dataset) if label == target_label]
    return Subset(dataset, filtered_indices)


def get_model_outputs_with_labels(model, dataloader):
    """
    获取模型对数据的预测向量和对应的真实标签。
    """
    model.eval()
    outputs_list = []
    labels_list = []
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(DEVICE)
            logits = model(batch_x)  # (batch_size, 10)
            probs = logits  # 未经过 softmax，保留 logits
            outputs_list.append(probs.cpu().numpy())
            labels_list.append(batch_y.cpu().numpy())  # 获取真实标签
    return np.concatenate(outputs_list, axis=0), np.concatenate(labels_list, axis=0)
def build_attack_loader(data, batch_size):
    """
    构造 PyTorch DataLoader

    每条数据格式为：[pred_vector, true_label, membership_label]
    其中 pred_vector 是预测向量，true_label 是真实标签，
    membership_label 表示成员/非成员标签。

    将 pred_vector 与 true_label 拼接成一个特征向量。

    参数:
        data: list，每个元素为一条记录
        batch_size: 批次大小

    返回:
        DataLoader 对象
    """
    # 提取特征向量：拼接预测向量和真实标签
    X = np.array([np.concatenate([record[0], [record[1]]]) for record in data])
    # 提取成员/非成员标签
    y = np.array([record[2] for record in data])
    # 构造 TensorDataset 和 DataLoader
    dataset = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).long())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader

def train_attack_model(shadow_model, shadow_train_dataset, shadow_holdout_dataset, target_label):
    # 过滤 shadow_train_dataset 和 shadow_holdout_dataset，仅保留标签为 7 的样本

    shadow_train_dataset_filtered = filter_by_label(shadow_train_dataset, target_label)
    shadow_holdout_dataset_filtered = filter_by_label(shadow_holdout_dataset, target_label)


    shadow_train_indices = list(range(len(shadow_holdout_dataset_filtered)))  # 定义需要加载的索引范围
    shadow_train_dataset = Subset(shadow_train_dataset_filtered, shadow_train_indices)
    shadow_train_loader = DataLoader(shadow_train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    shadow_holdout_loader = DataLoader(shadow_holdout_dataset_filtered, batch_size=BATCH_SIZE, shuffle=False)
    # 获取影子模型的预测概率和真实标签
    shadow_train_probs, shadow_train_true_labels = get_model_outputs_with_labels(shadow_model, shadow_train_loader)
    shadow_holdout_probs, shadow_holdout_true_labels = get_model_outputs_with_labels(shadow_model,
                                                                                     shadow_holdout_loader)

    # 构造攻击数据格式 (y_pred, y_true, in/out)
    shadow_train_data = [
        (y_pred, y_true, 1) for y_pred, y_true in zip(shadow_train_probs, shadow_train_true_labels)
    ]
    shadow_holdout_data = [
        (y_pred, y_true, 0) for y_pred, y_true in zip(shadow_holdout_probs, shadow_holdout_true_labels)
    ]

    # 合并成员和非成员数据
    attack_data = shadow_train_data + shadow_holdout_data

    # 转换为 NumPy 格式
    X_attack_train = np.array([np.concatenate([record[0], [record[1]]]) for record in attack_data])  # 预测向量 + 真实标签
    y_attack_train = np.array([record[2] for record in attack_data])  # 成员/非成员标签

    # 构造 PyTorch 数据集
    attack_train_dataset = TensorDataset(
        torch.from_numpy(X_attack_train).float(),
        torch.from_numpy(y_attack_train).long()
    )
    attack_train_loader = DataLoader(attack_train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    EPOCHS = 1000

    # 攻击模型 (二分类)
    attack_model = AttackMLP(input_dim=11, hidden_dim=32, output_dim=2).to(DEVICE)
    optimizer_a = optim.Adam(attack_model.parameters(), lr=LR)


    attack_model.train()
    # 训练攻击模型 (二分类)
    for epoch in range(EPOCHS):
        loss_a = train_one_epoch(attack_model, attack_train_loader, optimizer_a, device=DEVICE)
        # 这里简单地用训练精度衡量
        acc_a = evaluate(attack_model, attack_train_loader, device=DEVICE)
        print(f"[Attack Model] Epoch {epoch + 1}/{EPOCHS}, Loss: {loss_a:.4f}, Train Acc: {acc_a:.4f}")
        # 将训练数据（成员数据）和保留数据（非成员数据）合并为总数据集
        test_data = shadow_train_data + shadow_holdout_data
        # TPS 数据：成员数据
        test_TPS = shadow_train_data
        # FPS 数据：非成员数据
        test_FPS = shadow_holdout_data

        # 构造对应的 DataLoader
        attack_test_loader = build_attack_loader(test_data, BATCH_SIZE)
        attack_TPS_loader = build_attack_loader(test_TPS, BATCH_SIZE)
        attack_FPS_loader = build_attack_loader(test_FPS, BATCH_SIZE)

        # 调用示例：

        # 1. 对合并后的数据集 (成员 + 非成员) 进行评估
        attack_acc = evaluate(attack_model, attack_test_loader, DEVICE)
        print(f"[Membership Inference Attack] Accuracy: {attack_acc:.4f}")

        # 2. 对 TPS 数据（成员数据）进行评估
        TPS_acc = evaluate(attack_model, attack_TPS_loader, DEVICE)
        print(f"[Membership Inference Attack(TPS)] Accuracy: {TPS_acc:.4f}")

        # 3. 对 FPS 数据（非成员数据）进行评估
        # 注意：原代码中 FPS 的指标为 Error = 1.0 - accuracy
        FPS_acc = evaluate(attack_model, attack_FPS_loader, DEVICE)
        FPS_error = 1.0 - FPS_acc
        print(f"[Membership Inference Attack(FPS)] Error: {FPS_error:.4f}")

    torch.save(attack_model.state_dict(), f'attack_model{target_label}.pth')
    attack_model.load_state_dict(torch.load(f'attack_model{target_label}.pth'))

    shadow_train_dataset_filtered = filter_by_label(shadow_train_dataset, target_label)
    shadow_holdout_dataset_filtered = filter_by_label(shadow_holdout_dataset, target_label)

    shadow_train_indices = list(range(len(shadow_holdout_dataset_filtered)))  # 定义需要加载的索引范围
    shadow_train_dataset = Subset(shadow_train_dataset_filtered, shadow_train_indices)
    shadow_train_loader = DataLoader(shadow_train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    shadow_holdout_loader = DataLoader(shadow_holdout_dataset_filtered, batch_size=BATCH_SIZE, shuffle=False)
    # 获取影子模型的预测概率和真实标签
    shadow_train_probs, shadow_train_true_labels = get_model_outputs_with_labels(shadow_model, shadow_train_loader)
    shadow_holdout_probs, shadow_holdout_true_labels = get_model_outputs_with_labels(shadow_model,
                                                                                     shadow_holdout_loader)

    # 构造攻击数据格式 (y_pred, y_true, in/out)
    shadow_train_data = [
        (y_pred, y_true, 1) for y_pred, y_true in zip(shadow_train_probs, shadow_train_true_labels)
    ]
    shadow_holdout_data = [
        (y_pred, y_true, 0) for y_pred, y_true in zip(shadow_holdout_probs, shadow_holdout_true_labels)
    ]

    # 假设 train_data 和 holdout_data 已经定义
    # 将训练数据（成员数据）和保留数据（非成员数据）合并为总数据集
    test_data = shadow_train_data + shadow_holdout_data
    # TPS 数据：成员数据
    test_TPS = shadow_train_data
    # FPS 数据：非成员数据
    test_FPS = shadow_holdout_data

    # 构造对应的 DataLoader
    attack_test_loader = build_attack_loader(test_data, BATCH_SIZE)
    attack_TPS_loader = build_attack_loader(test_TPS, BATCH_SIZE)

    attack_FPS_loader = build_attack_loader(test_FPS, BATCH_SIZE)
    shadow_train_dataset_filtered = filter_by_label(shadow_train_dataset, target_label)
    shadow_holdout_dataset_filtered = filter_by_label(shadow_holdout_dataset, target_label)

    shadow_train_indices = list(range(len(shadow_holdout_dataset_filtered)))  # 定义需要加载的索引范围
    shadow_train_dataset = Subset(shadow_train_dataset_filtered, shadow_train_indices)
    shadow_train_loader = DataLoader(shadow_train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    shadow_holdout_loader = DataLoader(shadow_holdout_dataset_filtered, batch_size=BATCH_SIZE, shuffle=False)
    # 获取影子模型的预测概率和真实标签
    shadow_train_probs, shadow_train_true_labels = get_model_outputs_with_labels(shadow_model, shadow_train_loader)
    shadow_holdout_probs, shadow_holdout_true_labels = get_model_outputs_with_labels(shadow_model,
                                                                                     shadow_holdout_loader)

    # 构造攻击数据格式 (y_pred, y_true, in/out)
    shadow_train_data = [
        (y_pred, y_true, 1) for y_pred, y_true in zip(shadow_train_probs, shadow_train_true_labels)
    ]
    shadow_holdout_data = [
        (y_pred, y_true, 0) for y_pred, y_true in zip(shadow_holdout_probs, shadow_holdout_true_labels)
    ]

    # 假设 train_data 和 holdout_data 已经定义
    # 将训练数据（成员数据）和保留数据（非成员数据）合并为总数据集
    test_data = shadow_train_data + shadow_holdout_data
    # TPS 数据：成员数据
    test_TPS = shadow_train_data
    # FPS 数据：非成员数据
    test_FPS = shadow_holdout_data

    # 构造对应的 DataLoader
    attack_test_eval_loader = build_attack_loader(test_data, BATCH_SIZE)
    # 调用示例：

    # 1. 对合并后的数据集 (成员 + 非成员) 进行评估
    attack_acc = evaluate(attack_model, attack_test_loader, DEVICE)
    print(f"[Membership Inference Attack] Accuracy: {attack_acc:.4f}")

    # 2. 对 TPS 数据（成员数据）进行评估
    TPS_acc = evaluate(attack_model, attack_TPS_loader, DEVICE)
    print(f"[Membership Inference Attack(TPS)] Accuracy: {TPS_acc:.4f}")

    # 3. 对 FPS 数据（非成员数据）进行评估
    # 注意：原代码中 FPS 的指标为 Error = 1.0 - accuracy
    FPS_acc = evaluate(attack_model, attack_FPS_loader, DEVICE)
    FPS_error = 1.0 - FPS_acc
    print(f"[Membership Inference Attack(FPS)] Error: {FPS_error:.4f}")

    # 计算最终综合指标：
    # 注意：1.0 - FPS_error = FPS_acc，因此可等价写为 (len(test_TPS)*TPS_acc + len(test_FPS)*FPS_acc) / len(test_data)
    combined_metric = (len(test_TPS) * TPS_acc + len(test_FPS) * FPS_acc) / len(test_data)
    print(f"Combined Metric: {combined_metric:.4f}")
    for epoch in range(EPOCHS):
        loss_a = train_one_epoch(attack_model, attack_test_loader, optimizer_a, device=DEVICE)
        # 这里简单地用训练精度衡量
        acc_a = evaluate(attack_model, attack_test_eval_loader, device=DEVICE)
        print(f"[Attack Model] Epoch {epoch + 1}/{EPOCHS}, Loss: {loss_a:.4f}, Train Acc: {acc_a:.4f}")
    return attack_model,attack_test_loader


# for i in range(10):
#     train_attack_model(shadow_model, shadow_train_dataset,shadow_holdout_dataset,i)
# ================================
# 7. 使用攻击模型对目标模型实施成员推断
# ================================
target_label = 1
attack_model,attack_test_loader=train_attack_model(shadow_model, shadow_train_dataset,shadow_holdout_dataset,target_label)
acc=evaluate(attack_model,attack_test_loader,DEVICE)
print(acc)
# attack_model(single) = AttackMLP(input_dim=11, hidden_dim=32, output_dim=2).to(DEVICE)
# attack_model(single).load_state_dict(torch.load(f'attack_model(single){target_label}.pth'))
# target_model = copy.deepcopy(shadow_model)
# target_model.load_state_dict(torch.load('shadow_model.pth', map_location=DEVICE))


shadow_train_dataset_filtered = filter_by_label(shadow_train_dataset, target_label)
shadow_holdout_dataset_filtered = filter_by_label(shadow_holdout_dataset, target_label)

shadow_train_indices = list(range(len(shadow_holdout_dataset_filtered)))  # 定义需要加载的索引范围
shadow_train_dataset = Subset(shadow_train_dataset_filtered, shadow_train_indices)
shadow_train_loader = DataLoader(shadow_train_dataset, batch_size=BATCH_SIZE, shuffle=False)
shadow_holdout_loader = DataLoader(shadow_holdout_dataset_filtered, batch_size=BATCH_SIZE, shuffle=False)
# 获取影子模型的预测概率和真实标签
shadow_train_probs, shadow_train_true_labels = get_model_outputs_with_labels(shadow_model, shadow_train_loader)
shadow_holdout_probs, shadow_holdout_true_labels = get_model_outputs_with_labels(shadow_model, shadow_holdout_loader)

# 构造攻击数据格式 (y_pred, y_true, in/out)
shadow_train_data = [
    (y_pred, y_true, 1) for y_pred, y_true in zip(shadow_train_probs, shadow_train_true_labels)
]
shadow_holdout_data = [
    (y_pred, y_true, 0) for y_pred, y_true in zip(shadow_holdout_probs, shadow_holdout_true_labels)
]





# 假设 train_data 和 holdout_data 已经定义
# 将训练数据（成员数据）和保留数据（非成员数据）合并为总数据集
test_data = shadow_train_data + shadow_holdout_data
# TPS 数据：成员数据
test_TPS = shadow_train_data
# FPS 数据：非成员数据
test_FPS = shadow_holdout_data

# 构造对应的 DataLoader
attack_test_loader = build_attack_loader(test_data, BATCH_SIZE)
attack_TPS_loader = build_attack_loader(test_TPS, BATCH_SIZE)
attack_FPS_loader = build_attack_loader(test_FPS, BATCH_SIZE)





# 调用示例：

# 1. 对合并后的数据集 (成员 + 非成员) 进行评估
attack_acc = evaluate(attack_model,attack_test_loader,  DEVICE)
print(f"[Membership Inference Attack] Accuracy: {attack_acc:.4f}")

# 2. 对 TPS 数据（成员数据）进行评估
TPS_acc = evaluate(attack_model,attack_TPS_loader, DEVICE)
print(f"[Membership Inference Attack(TPS)] Accuracy: {TPS_acc:.4f}")

# 3. 对 FPS 数据（非成员数据）进行评估
# 注意：原代码中 FPS 的指标为 Error = 1.0 - accuracy
FPS_acc = evaluate(attack_model,attack_FPS_loader, DEVICE)
FPS_error = 1.0 - FPS_acc
print(f"[Membership Inference Attack(FPS)] Error: {FPS_error:.4f}")

# 计算最终综合指标：
# 注意：1.0 - FPS_error = FPS_acc，因此可等价写为 (len(test_TPS)*TPS_acc + len(test_FPS)*FPS_acc) / len(test_data)
combined_metric = (len(test_TPS) * TPS_acc + len(test_FPS) * FPS_acc) / len(test_data)
print(f"Combined Metric: {combined_metric:.4f}")
