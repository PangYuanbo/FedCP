import copy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score
from Membership_Inference_Attack.model import *

# ==============================
# 一些超参数 & 配置
# ==============================
BATCH_SIZE = 64
EPOCHS = 3   # 为了演示，训练回合数较小，实际可调大一些
LR = 1e-3
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ===================================================
# 1. 定义一个简单的 MLP，用于多分类 (target/shadow)
# ===================================================
class SimpleMLP(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=128, output_dim=10):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x shape: (batch_size, 784)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x  # logits

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
        optimizer.zero_grad()
        # 前向
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        # 反向传播
        loss.backward()
        optimizer.step()
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

# ===================================================
# 4. 加载 MNIST，准备目标模型数据
# ===================================================
transform = transforms.Compose([
    transforms.ToTensor(),  # [0,1]
    transforms.Normalize((0.1307,), (0.3081,))  # 常规 MNIST 均值/方差
])
mnist_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# 为简化演示，只取前 5000 个样本
sub_data, sub_targets = mnist_dataset.data[:5000], mnist_dataset.targets[:5000]
# sub_data shape: (5000, 28, 28), sub_targets: (5000,)

# 转换为 (N,784) 的浮点张量
sub_data = sub_data.view(-1, 28*28).float() / 255.0

# 拆分 => 目标训练集 (2500) + 目标 holdout (2500)
X_t_train, X_t_holdout, y_t_train, y_t_holdout = train_test_split(
    sub_data.numpy(), sub_targets.numpy(), test_size=0.5, random_state=42
)
# 转成 TensorDataset
train_dataset = TensorDataset(
    torch.from_numpy(X_t_train),
    torch.from_numpy(y_t_train).long()
)
holdout_dataset = TensorDataset(
    torch.from_numpy(X_t_holdout),
    torch.from_numpy(y_t_holdout).long()
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
holdout_loader = DataLoader(holdout_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ================================
# 训练目标模型 (Target Model)
# ================================
model = FedAvgCNN(in_features=1, num_classes=10, dim=1024).to(DEVICE)
model_head=copy.deepcopy(model.fc)
model=LocalModel(model,model_head)
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
optimizer_t = optim.SGD(target_model.parameters(), lr=LR)

for epoch in range(EPOCHS):
    loss_t = train_one_epoch(target_model, train_loader, optimizer_t, device=DEVICE)
    acc_train = evaluate(target_model, train_loader, device=DEVICE)
    acc_holdout = evaluate(target_model, holdout_loader, device=DEVICE)
    print(f"[Target Model] Epoch {epoch+1}/{EPOCHS}, Loss: {loss_t:.4f}, "
          f"Train Acc: {acc_train:.4f}, Holdout Acc: {acc_holdout:.4f}")

# ================================
# 5. 使用合成数据训练影子模型
# ================================
# 这里模拟一个 784 维，10 类的合成数据集
n_samples_shadow = 4000
X_synthetic, y_synthetic = make_classification(
    n_samples=n_samples_shadow,
    n_features=784,
    n_informative=50,
    n_redundant=0,
    n_classes=10,
    random_state=2023
)

# 拆分 => 影子训练集 (2000) + 影子 Holdout (2000)
X_s_train, X_s_holdout, y_s_train, y_s_holdout = train_test_split(
    X_synthetic, y_synthetic, test_size=0.5, random_state=42
)

shadow_train_dataset = TensorDataset(
    torch.from_numpy(X_s_train).float(),
    torch.from_numpy(y_s_train).long()
)
shadow_holdout_dataset = TensorDataset(
    torch.from_numpy(X_s_holdout).float(),
    torch.from_numpy(y_s_holdout).long()
)

shadow_train_loader = DataLoader(shadow_train_dataset, batch_size=BATCH_SIZE, shuffle=True)
shadow_holdout_loader = DataLoader(shadow_holdout_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 影子模型 (与目标模型结构相同)
shadow_model = copy.deepcopy(model)
shadow_model.cs=copy.deepcopy(target_model.cs)
for param in model.cs.parameters():
    param.requires_grad = False
optimizer_s = optim.SGD(shadow_model.parameters(), lr=LR)

# 训练影子模型
for epoch in range(EPOCHS):
    loss_s = train_one_epoch(shadow_model, shadow_train_loader, optimizer_s, device=DEVICE)
    acc_s_train = evaluate(shadow_model, shadow_train_loader, device=DEVICE)
    acc_s_holdout = evaluate(shadow_model, shadow_holdout_loader, device=DEVICE)
    print(f"[Shadow Model] Epoch {epoch+1}/{EPOCHS}, Loss: {loss_s:.4f}, "
          f"Train Acc: {acc_s_train:.4f}, Shadow-Holdout Acc: {acc_s_holdout:.4f}")

# ================================
# 6. 构造攻击模型 (Attack Model) 的训练数据
# ================================
# 获取影子模型对影子训练集 (成员=1) 和 holdout (非成员=0) 的预测概率
def get_model_outputs(model, dataloader):
    model.eval()
    outputs_list = []
    with torch.no_grad():
        for batch_x, _ in dataloader:
            batch_x = batch_x.to(DEVICE)
            logits = model(batch_x)  # (batch_size, 10)
            probs = F.softmax(logits, dim=1)  # 转为概率
            outputs_list.append(probs.cpu().numpy())
    return np.concatenate(outputs_list, axis=0)

shadow_train_probs = get_model_outputs(shadow_model, shadow_train_loader)   # shape (2000, 10)
shadow_holdout_probs = get_model_outputs(shadow_model, shadow_holdout_loader)  # shape (2000, 10)

shadow_train_labels = np.ones(len(shadow_train_probs), dtype=np.int64)      # 成员=1
shadow_holdout_labels = np.zeros(len(shadow_holdout_probs), dtype=np.int64) # 非成员=0

X_attack_train = np.vstack([shadow_train_probs, shadow_holdout_probs])  # (4000, 10)
y_attack_train = np.concatenate([shadow_train_labels, shadow_holdout_labels])  # (4000,)

attack_train_dataset = TensorDataset(
    torch.from_numpy(X_attack_train).float(),
    torch.from_numpy(y_attack_train).long()
)
attack_train_loader = DataLoader(attack_train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# 攻击模型 (二分类)
attack_model = AttackMLP(input_dim=10, hidden_dim=32, output_dim=2).to(DEVICE)
optimizer_a = optim.Adam(attack_model.parameters(), lr=LR)

# 训练攻击模型 (二分类)
for epoch in range(EPOCHS):
    loss_a = train_one_epoch(attack_model, attack_train_loader, optimizer_a, device=DEVICE)
    # 这里简单地用训练精度衡量
    acc_a = evaluate(attack_model, attack_train_loader, device=DEVICE)
    print(f"[Attack Model] Epoch {epoch+1}/{EPOCHS}, Loss: {loss_a:.4f}, Train Acc: {acc_a:.4f}")

# ================================
# 7. 使用攻击模型对目标模型实施成员推断
# ================================
t_train_probs = get_model_outputs(target_model, train_loader)     # (2500, 10)
t_holdout_probs = get_model_outputs(target_model, holdout_loader) # (2500, 10)
t_train_labels = np.ones(len(t_train_probs), dtype=np.int64)    # 真实成员=1
t_holdout_labels = np.zeros(len(t_holdout_probs), dtype=np.int64)  # 非成员=0

X_attack_test = np.vstack([t_train_probs, t_holdout_probs])  # (5000, 10)
y_attack_test = np.concatenate([t_train_labels, t_holdout_labels]) # (5000,)

# 构建测试集 DataLoader（给攻击模型用）
attack_test_dataset = TensorDataset(
    torch.from_numpy(X_attack_test).float(),
    torch.from_numpy(y_attack_test).long()
)
attack_test_loader = DataLoader(attack_test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 利用攻击模型做预测
attack_model.eval()
preds_list = []
labels_list = []
with torch.no_grad():
    for batch_x, batch_y in attack_test_loader:
        batch_x = batch_x.to(DEVICE)
        logits = attack_model(batch_x)  # (batch_size, 2)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        preds_list.append(preds)
        labels_list.append(batch_y.numpy())

y_pred = np.concatenate(preds_list, axis=0)
y_true = np.concatenate(labels_list, axis=0)

attack_acc = accuracy_score(y_true, y_pred)
print(f"[Membership Inference Attack] Accuracy: {attack_acc:.4f}")
