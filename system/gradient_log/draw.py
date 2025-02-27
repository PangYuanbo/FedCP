import ast
import matplotlib.pyplot as plt

data = []
with open("gradient_log_client1_cifar-10-past_1000_0.0050.txt", "r") as f:
    for line in f:
        # 用 ast.literal_eval() 解析字典格式的字符串
        record = ast.literal_eval(line.strip())
        data.append(record)

rounds = [d['round'] for d in data]
grad_head = [d['grad_norm_head'] for d in data]
grad_feat = [d['grad_norm_feat'] for d in data]
grad_ensemble = [d['grad_norm_ensemble'] for d in data]

plt.figure(figsize=(8, 5))
plt.plot(rounds, grad_head,  label='grad_norm_head')
plt.plot(rounds, grad_feat,  label='grad_norm_feat')
# plt.plot(rounds, grad_ensemble,  label='grad_norm_ensemble')

plt.xlabel('Round')
plt.ylabel('Gradient Norm')
plt.title('Gradient Norms over Rounds')
plt.legend()
plt.grid(True)
plt.show()
