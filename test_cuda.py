import torch

# 检查 CUDA 是否可用
if torch.cuda.is_available():
    print("CUDA 可用")
    print(f"可用的 GPU 数量: {torch.cuda.device_count()}")
    print(f"当前 GPU 设备: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    print(f"CUDA 版本: {torch.version.cuda}")
else:
    print("CUDA 不可用")
