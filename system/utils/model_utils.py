import torch
import torch.nn as nn
import math


def clip_and_add_noise_with_privacy(layer, clip_value, epsilon, delta, device=None):
    """
    对指定层进行梯度裁剪并基于隐私预算 (epsilon, delta) 添加噪声
    :param layer: 要操作的层（如 model.fc1）
    :param clip_value: 梯度裁剪阈值
    :param epsilon: 隐私预算 (epsilon)
    :param delta: 隐私泄露概率 (delta)
    :param device: 设备（如 'cuda' 或 'cpu'）
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 计算噪声强度 sigma
    sigma = (clip_value * math.sqrt(2 * math.log(1.25 / delta))) / epsilon

    with torch.no_grad():  # 不需要计算图
        for param in layer.parameters():
            if param.grad is not None:  # 确保梯度存在
                # 计算梯度的 L2 范数
                grad_norm = torch.norm(param.grad, p=2)
                # 裁剪梯度
                if grad_norm > clip_value:
                    param.grad.mul_(clip_value / grad_norm)

                # 添加高斯噪声
                noise = torch.normal(0, sigma, size=param.grad.shape).to(device)
                param.grad.add_(noise)  # 将噪声添加到梯度


def add_noise_to_gradients(data_shape, s, sigma, device=None):
    """
    Gaussian noise
    """
    return torch.normal(0, sigma * s, data_shape).to(device)