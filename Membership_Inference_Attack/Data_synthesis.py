# import os
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
# import numpy as np
# import random
# from Membership_Inference_Attack.model import *
# import copy
#
# # ============ 1. 定义辅助函数 ============
#
# def rand_record_mnist():
#     """
#     随机初始化一张“伪”MNIST图像，返回 shape = (1, 28, 28) 的张量，取值在 [0,1]。
#     注意输出是 (1,28,28)，以便和后续 PyTorch 模型的输入格式对齐。
#     """
#     return torch.rand(1,1, 28, 28)
#
#
# def randomize_k_features(x, k=50):
#     """
#     在给定图像 x (shape=[1,28,28]) 中随机挑选 k 个像素，重新赋为 [0,1] 的新随机值。
#     x 会被就地修改并返回。
#     """
#     # 先打平(784维)，随机选 k 个索引
#     x_flat = x.view(-1)
#     indices = np.random.choice(len(x_flat), size=k, replace=False)
#     for idx in indices:
#         x_flat[idx] = random.random()
#     return x
#
#
#
#
#
# def synthesize(
#         target_model,  # 目标模型，输入 (N,1,28,28) -> 输出(N,10)
#         c,  # 目标类别 (0~9)
#         k_max=50,
#         k_min=5,
#         conf_min=0.8,
#         rej_max=30,
#         iter_max=200
# ):
#     """
#     使用目标模型 target_model，为指定类别 c 合成一张“MNIST风格”样本。
#     返回值: 若合成成功，返回 shape=[1,28,28] 的张量；否则返回 None。
#     """
#     # 1. 随机初始化 x
#     x = rand_record_mnist()  # shape = (1,28,28)
#     headw_ps = []
#     for name, mat in target_model.model.head.named_parameters():
#         if 'weight' in name:
#             headw_ps.append(mat.data)
#     headw_p = headw_ps[-1]
#     for mat in headw_ps[-2::-1]:
#         headw_p = torch.matmul(headw_p, mat)
#         print(1)
#     headw_p.detach_()
#     context = torch.sum(headw_p, dim=0, keepdim=True)
#     # 当前最佳样本及其置信度
#     y_c_star = 0.0
#     x_star = x.clone()
#
#     # 连续拒绝计数
#     j = 0
#     k = k_max
#
#     for iteration in range(iter_max):
#         # 2. 查询目标模型：f_target(x)，输出 shape=[1,10]
#         with torch.no_grad():
#             y = target_model(x, is_rep=False, context=context)  # shape=[1,10]
#         y = y[0]
#         y = F.softmax(y, dim=0)
#
#         # 当前目标类别置信度
#         y_c = y[ c].item()
#
#         # 3. 若 y_c >= 历史最佳，则接受并尝试采样
#         if y_c >= y_c_star:
#             # 检查是否超过置信度阈值，并且 argmax 为 c
#             pred_class = torch.argmax(y).item()
#             if y_c > conf_min and pred_class == c:
#                 # 以 y_c 的概率“采样成功”
#                 if random.random() < y_c:
#                     print(f"合成成功！iter={iteration}, y_c={y_c:.4f}")
#                     return x  # shape = [1,28,28]
#
#             # 更新“最佳”
#             x_star = x.clone()
#             y_c_star = y_c
#             # 拒绝计数清零
#             j = 0
#         else:
#             # 否则计一次拒绝
#             j += 1
#             if j > rej_max:
#                 # 缩减 k
#                 k = max(k_min, k // 2)
#                 j = 0
#
#         # 4. 基于当前 x_star，再随机修改 k 个像素
#         x = x_star.clone()
#         x = randomize_k_features(x, k)
#
#     # 若 iter_max 次迭代还没成功，返回 None
#     return None
#
#
# # ============ 2. 批量合成 5000 条数据并保存 ============
#
# def generate_synthetic_dataset(
#         target_model,
#         total_samples=5000,
#         per_class=500,
#         output_path='synthetic_mnist.pt'
# ):
#     """
#     - 为 10 个类别 (0~9) 各合成 per_class 条，总计 total_samples 条数据。
#     - 拆分为 train(4000) + test(1000)，并保存到 output_path 中。
#     """
#     all_data = []
#     all_labels = []
#
#
#     # 如果要保证合计数 == total_samples，假设 total_samples=5000, per_class=500，
#     # 那么 10*500=5000。若要改动，请确保对应数量匹配。
#
#     for c in range(10):  # 对每个类别 c
#         print(f"=== 开始合成类别 {c} 的数据 ===")
#         count = 0
#         while count < per_class:
#             x_synth = synthesize(target_model, c,
#                                  k_max=50, k_min=5,
#                                  conf_min=0.9, rej_max=30, iter_max=200)
#             if x_synth is not None:
#                 all_data.append(x_synth.squeeze(0))  # 存入 shape=[28,28]
#                 all_labels.append(c)
#                 count += 1
#                 if count % 50 == 0:
#                     print(f"类别 {c}: 已合成 {count}/{per_class} 条")
#
#     # 拼接成 Tensor
#     data_tensor = torch.stack(all_data, dim=0)  # shape=[5000, 28, 28]
#     labels_tensor = torch.tensor(all_labels)  # shape=[5000]
#
#     # 随机打乱
#     perm = torch.randperm(len(data_tensor))
#     data_tensor = data_tensor[perm]
#     labels_tensor = labels_tensor[perm]
#
#     # 拆分 train / test
#     train_data = data_tensor[:4000]
#     train_labels = labels_tensor[:4000]
#     test_data = data_tensor[4000:]
#     test_labels = labels_tensor[4000:]
#
#     # 保存到文件
#     torch.save({
#         'train_data': train_data,
#         'train_labels': train_labels,
#         'test_data': test_data,
#         'test_labels': test_labels
#     }, output_path)
#
#     print(f"合成完毕，共生成 {len(data_tensor)} 条数据，已保存到 {output_path}。")
#     print(f"其中训练集 {len(train_data)} 条，测试集 {len(test_data)} 条。")
#
#
# if __name__ == '__main__':
#     DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = FedAvgCNN(in_features=1, num_classes=10, dim=1024).to(DEVICE)
#     model_head = copy.deepcopy(model.fc)
#     model.fc = nn.Identity()
#     model = LocalModel(model, model_head)
#     in_dim = list(model.head.parameters())[0].shape[1]
#     cs = ConditionalSelection(in_dim, in_dim).to(DEVICE)
#
#     target_model = Ensemble(
#         model=copy.deepcopy(model),
#         cs=copy.deepcopy(cs),
#         head_g=copy.deepcopy(model.head),  # head is the global head
#         feature_extractor=copy.deepcopy(model.feature_extractor)
#         # feature_extractor is the global feature_extractor
#     )
#     # 打印模型所需的 state_dict 键
#     print("Keys required by the model:")
#     for key in target_model.state_dict().keys():
#         print(key)
#     # 加载文件并打印其键
#     loaded_weights = torch.load('../system/pretrain/results_mnist-0.1-npz_client0_1000_0.0050.pt', map_location=DEVICE)
#
#     # 打印加载文件的键
#     print("\nKeys in the loaded state_dict:")
#     for key in loaded_weights.keys():
#         print(key)
#
#
#     # 如果键名有规则，比如前缀不同
#     new_weights = {k.replace("old_prefix", "new_prefix"): v for k, v in loaded_weights.items()}
#     target_model.load_state_dict(new_weights)
#     target_model.load_state_dict(torch.load('../system/pretrain/results_mnist-0.1-npz_client0_1000_0.0050.pt', map_location=DEVICE))
#     # 用 dummy_target_model 演示：你应替换为你真正的 target_model
#     generate_synthetic_dataset(
#         target_model=target_model,
#         total_samples=5000,  # 总数=5000
#         per_class=500,  # 每个类别 500
#         output_path='synthetic_mnist.pt'
#     )
import matplotlib.pyplot as plt

# Data extracted from your input
target_labels = list(range(10))
accuracy_model1 = [0.5369, 0.5190, 0.4588, 0.5019, 0.4641, 0.4884, 0.4843, 0.5016, 0.5195, 0.5770]  # model.feature_extractor
accuracy_model2 = [0.5271, 0.5182, 0.4740, 0.5829, 0.5098, 0.3756, 0.5400, 0.4787, 0.5733, 0.4597]  # model.head

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(target_labels, accuracy_model1, color='blue', label='model.feature_extractor', marker='o', linestyle='-')
plt.plot(target_labels, accuracy_model2, color='red', label='model.head', marker='s', linestyle='--')

# Customize the plot
plt.xlabel('Target Label')
plt.ylabel('Membership Inference Attack Accuracy')
plt.title('Membership Inference Attack Accuracy by Target Label')
plt.xticks(target_labels)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

# Show the plot
plt.show()