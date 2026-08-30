#!/usr/bin/env python3
# 训练辅助工具：随机种子、设备迁移、FGM 对抗扰动和 EMA 参数平均。
# 当前由 Task3 训练脚本引用；参数平均用于评估和保存，评估后恢复训练参数。
import random

import numpy as np
import torch


# 统一设置 Python、NumPy 和 PyTorch 随机种子，减少实验随机差异。
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# 仅将张量迁移到 CPU/GPU，保留句子ID、坐标列表等非张量对象。
def move_batch(batch, device):
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


# 沿词嵌入梯度方向添加小扰动，额外反向传播后恢复原始参数。
class FGM:
    def __init__(self, model, epsilon=1.0):
        self.model = model
        self.epsilon = epsilon
        self.backup = {}

    # 备份词嵌入参数并沿归一化梯度方向添加扰动。
    def attack(self):
        self.backup = {}
        target_name = "bert.embeddings.word_embeddings"
        for name, parameter in self.model.named_parameters():
            if target_name not in name or parameter.grad is None:
                continue
            norm = torch.norm(parameter.grad)
            if not torch.isfinite(norm) or norm.item() == 0.0:
                continue
            self.backup[name] = parameter.data.clone()
            parameter.data.add_(self.epsilon * parameter.grad / norm)

    # 恢复扰动前备份的词嵌入参数。
    def restore(self):
        for name, parameter in self.model.named_parameters():
            if name in self.backup:
                parameter.data.copy_(self.backup[name])
        self.backup = {}


# 维护可训练参数的指数滑动平均副本，供评估和模型保存使用。
class ExponentialMovingAverage:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.num_updates = 0
        self.shadow = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        self.backup = {}

    # 用当前训练参数更新滑动平均副本，训练初期适当减小平滑系数。
    @torch.no_grad()
    def update(self, model):
        self.num_updates += 1
        decay = min(
            self.decay,
            (1.0 + self.num_updates) / (10.0 + self.num_updates),
        )
        for name, parameter in model.named_parameters():
            if name in self.shadow:
                self.shadow[name].mul_(decay).add_(
                    parameter.detach(), alpha=1.0 - decay
                )

    # 备份当前参数并临时替换为滑动平均参数，供开发集评估与保存。
    @torch.no_grad()
    def apply(self, model):
        self.backup = {}
        for name, parameter in model.named_parameters():
            if name in self.shadow:
                self.backup[name] = parameter.detach().clone()
                parameter.data.copy_(self.shadow[name])

    # 恢复评估前备份的训练参数，并清空临时备份。
    @torch.no_grad()
    def restore(self, model):
        for name, parameter in model.named_parameters():
            if name in self.backup:
                parameter.data.copy_(self.backup[name])
        self.backup = {}
