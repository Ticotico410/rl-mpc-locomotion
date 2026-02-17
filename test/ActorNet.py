import torch
import torch.nn as nn

# 定义网络（只指定特征维度）
actor = nn.Sequential(
    nn.Linear(48, 512),   # 只需要指定: 48 → 512
    nn.ELU(),
    nn.Linear(512, 256),  # 只需要指定: 512 → 256
    nn.ELU(),
    nn.Linear(256, 128),  # 只需要指定: 256 → 128
    nn.ELU(),
    nn.Linear(128, 12)    # 只需要指定: 128 → 12
)

# 不同batch大小都能工作：
x1 = torch.randn(48)          # shape: (48,)
x2 = torch.randn(1, 48)       # shape: (1, 48)
x3 = torch.randn(32, 48)      # shape: (32, 48)
x4 = torch.randn(192, 48)     # shape: (192, 48)

y1 = actor(x1)                # shape: (12,)
y2 = actor(x2)                # shape: (1, 12)
y3 = actor(x3)                # shape: (32, 12)
y4 = actor(x4)                # shape: (192, 12)

print(y1.shape)
print(y2.shape)
print(y3.shape)
print(y4.shape)
